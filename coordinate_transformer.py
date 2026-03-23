from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import yaml


ArrayLike2D = Sequence[float] | np.ndarray


@dataclass(frozen=True)
class InstrumentTransform:
    """Affine transform between one instrument coordinate system and sample coordinates."""

    name: str
    units: str
    matrix: np.ndarray
    inverse_matrix: np.ndarray
    calibration_points: tuple[dict[str, tuple[float, float]], ...]

    def transform_point(self, x: float, y: float) -> tuple[float, float]:
        vector = np.array([x, y, 1.0], dtype=float)
        out = self.matrix @ vector
        return float(out[0]), float(out[1])

    def inverse_transform_point(self, x: float, y: float) -> tuple[float, float]:
        vector = np.array([x, y, 1.0], dtype=float)
        out = self.inverse_matrix @ vector
        return float(out[0]), float(out[1])

    def transform_points(self, points: Iterable[ArrayLike2D]) -> np.ndarray:
        arr = np.asarray(list(points), dtype=float)
        if arr.ndim != 2 or arr.shape[1] != 2:
            raise ValueError("points must be an iterable of [x, y] pairs")
        ones = np.ones((arr.shape[0], 1), dtype=float)
        hom = np.hstack([arr, ones])
        out = hom @ self.matrix.T
        return out[:, :2]

    def inverse_transform_points(self, points: Iterable[ArrayLike2D]) -> np.ndarray:
        arr = np.asarray(list(points), dtype=float)
        if arr.ndim != 2 or arr.shape[1] != 2:
            raise ValueError("points must be an iterable of [x, y] pairs")
        ones = np.ones((arr.shape[0], 1), dtype=float)
        hom = np.hstack([arr, ones])
        out = hom @ self.inverse_matrix.T
        return out[:, :2]

    def calibration_residuals(self) -> np.ndarray:
        instrument_points = np.asarray([p["instrument"] for p in self.calibration_points], dtype=float)
        sample_points = np.asarray([p["sample"] for p in self.calibration_points], dtype=float)
        predicted = self.transform_points(instrument_points)
        return predicted - sample_points

    def max_calibration_error(self) -> float:
        residuals = self.calibration_residuals()
        return float(np.max(np.linalg.norm(residuals, axis=1)))


class CoordinateTransformer:
    """Load per-instrument affine transforms from YAML and map points into sample coordinates.

    The YAML is expected to contain, for each instrument, at least three calibration pairs of the form:
        instrument: [x_inst, y_inst]
        sample: [x_sample, y_sample]

    If more than three points are provided, the affine transform is fit by least squares.
    """

    def __init__(self, config: dict[str, Any]):
        self.config = config
        self.canonical_system = config.get("canonical_system", {})
        self._transforms: dict[str, InstrumentTransform] = {}

        instruments = config.get("instruments", [])
        if not instruments:
            raise ValueError("Config does not contain any instruments")

        for instrument in instruments:
            name = instrument["name"]
            units = instrument.get("units", self.canonical_system.get("units", "unknown"))
            calibration_points = instrument.get("calibration_points", [])
            matrix = self._fit_affine_matrix(calibration_points)
            inverse_matrix = np.linalg.inv(matrix)
            stored_points = tuple(
                {
                    "instrument": tuple(map(float, point["instrument"])),
                    "sample": tuple(map(float, point["sample"])),
                }
                for point in calibration_points
            )
            self._transforms[name] = InstrumentTransform(
                name=name,
                units=units,
                matrix=matrix,
                inverse_matrix=inverse_matrix,
                calibration_points=stored_points,
            )

    @classmethod
    def from_yaml(cls, path: str | Path) -> "CoordinateTransformer":
        path = Path(path)
        with path.open("r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
        return cls(config)

    @staticmethod
    def _fit_affine_matrix(calibration_points: Sequence[dict[str, Sequence[float]]]) -> np.ndarray:
        if len(calibration_points) < 3:
            raise ValueError("At least three calibration points are required for an affine transform")

        A_rows = []
        b_rows = []
        for point in calibration_points:
            x_inst, y_inst = map(float, point["instrument"])
            x_sample, y_sample = map(float, point["sample"])
            A_rows.append([x_inst, y_inst, 1.0, 0.0, 0.0, 0.0])
            A_rows.append([0.0, 0.0, 0.0, x_inst, y_inst, 1.0])
            b_rows.append(x_sample)
            b_rows.append(y_sample)

        A = np.asarray(A_rows, dtype=float)
        b = np.asarray(b_rows, dtype=float)

        coeffs, residuals, rank, singular_values = np.linalg.lstsq(A, b, rcond=None)
        if rank < 6:
            raise ValueError(
                "Calibration points are degenerate; could not determine a full affine transform"
            )

        matrix = np.array(
            [
                [coeffs[0], coeffs[1], coeffs[2]],
                [coeffs[3], coeffs[4], coeffs[5]],
                [0.0, 0.0, 1.0],
            ],
            dtype=float,
        )
        return matrix

    def instruments(self) -> list[str]:
        return sorted(self._transforms)

    def get_transform(self, instrument_name: str) -> InstrumentTransform:
        try:
            return self._transforms[instrument_name]
        except KeyError as exc:
            available = ", ".join(self.instruments())
            raise KeyError(f"Unknown instrument '{instrument_name}'. Available instruments: {available}") from exc

    def transform(self, instrument_name: str, x: float, y: float) -> tuple[float, float]:
        return self.get_transform(instrument_name).transform_point(x, y)

    def inverse_transform(self, instrument_name: str, x: float, y: float) -> tuple[float, float]:
        return self.get_transform(instrument_name).inverse_transform_point(x, y)

    def transform_points(self, instrument_name: str, points: Iterable[ArrayLike2D]) -> np.ndarray:
        return self.get_transform(instrument_name).transform_points(points)

    def inverse_transform_points(self, instrument_name: str, points: Iterable[ArrayLike2D]) -> np.ndarray:
        return self.get_transform(instrument_name).inverse_transform_points(points)

    def validate(self) -> dict[str, float]:
        return {
            name: transform.max_calibration_error()
            for name, transform in sorted(self._transforms.items())
        }

    def summary(self) -> dict[str, Any]:
        return {
            "canonical_system": self.canonical_system,
            "instruments": {
                name: {
                    "units": transform.units,
                    "matrix": transform.matrix.tolist(),
                    "max_calibration_error": transform.max_calibration_error(),
                }
                for name, transform in sorted(self._transforms.items())
            },
        }


def _build_cli() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Transform instrument coordinates into sample coordinates")
    parser.add_argument("config", type=Path, help="Path to the YAML configuration file")
    parser.add_argument("instrument", help="Instrument name, for example MAXIMA")
    parser.add_argument("x", type=float, help="Instrument x coordinate")
    parser.add_argument("y", type=float, help="Instrument y coordinate")
    parser.add_argument(
        "--inverse",
        action="store_true",
        help="Interpret x and y as sample coordinates and convert back into instrument coordinates",
    )
    parser.add_argument(
        "--show-matrix",
        action="store_true",
        help="Print the affine matrix for the selected instrument",
    )
    return parser


def main() -> None:
    parser = _build_cli()
    args = parser.parse_args()

    transformer = CoordinateTransformer.from_yaml(args.config)
    if args.inverse:
        x_out, y_out = transformer.inverse_transform(args.instrument, args.x, args.y)
        print(f"sample ({args.x:.6f}, {args.y:.6f}) -> {args.instrument} ({x_out:.6f}, {y_out:.6f})")
    else:
        x_out, y_out = transformer.transform(args.instrument, args.x, args.y)
        print(f"{args.instrument} ({args.x:.6f}, {args.y:.6f}) -> sample ({x_out:.6f}, {y_out:.6f})")

    if args.show_matrix:
        matrix = transformer.get_transform(args.instrument).matrix
        print("Affine matrix:")
        print(matrix)


if __name__ == "__main__":
    main()
