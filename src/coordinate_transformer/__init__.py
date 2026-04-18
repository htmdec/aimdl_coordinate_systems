from __future__ import annotations

import argparse
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

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
        arr = np.asarray(points, dtype=float)
        if arr.ndim != 2 or arr.shape[1] != 2:
            raise ValueError("points must be an iterable of [x, y] pairs")
        ones = np.ones((arr.shape[0], 1), dtype=float)
        hom = np.hstack([arr, ones])
        out = hom @ self.matrix.T
        return out[:, :2]

    def inverse_transform_points(self, points: Iterable[ArrayLike2D]) -> np.ndarray:
        arr = np.asarray(points, dtype=float)
        if arr.ndim != 2 or arr.shape[1] != 2:
            raise ValueError("points must be an iterable of [x, y] pairs")
        ones = np.ones((arr.shape[0], 1), dtype=float)
        hom = np.hstack([arr, ones])
        out = hom @ self.inverse_matrix.T
        return out[:, :2]

    def calibration_residuals(self) -> np.ndarray:
        instrument_points = np.asarray(
            [p["instrument"] for p in self.calibration_points], dtype=float
        )
        sample_points = np.asarray(
            [p["sample"] for p in self.calibration_points], dtype=float
        )
        predicted = self.transform_points(instrument_points)
        return predicted - sample_points

    def max_calibration_error(self) -> float:
        residuals = self.calibration_residuals()
        return float(np.max(np.linalg.norm(residuals, axis=1)))


@dataclass(frozen=True)
class VersionedInstrumentTransform:
    """An InstrumentTransform with a temporal validity range.

    Validity uses half-open intervals: valid_from is inclusive,
    valid_until is exclusive. None means unbounded in that direction.
    """

    version: str
    valid_from: datetime | None  # None = beginning of time (inclusive)
    valid_until: datetime | None  # None = still current (exclusive)
    transform: InstrumentTransform

    def is_valid_at(self, timestamp: datetime) -> bool:
        """Check whether this version is in effect at the given timestamp.

        Parameters
        ----------
        timestamp : datetime
            Must be timezone-aware. Raises ValueError if naive.

        Returns
        -------
        bool
        """
        if timestamp.tzinfo is None:
            raise ValueError(
                "timestamp must be timezone-aware; got naive datetime "
                f"{timestamp.isoformat()!r}. Use e.g. "
                "datetime.now(timezone.utc) or attach a tzinfo."
            )
        if self.valid_from is not None and timestamp < self.valid_from:
            return False
        return not (self.valid_until is not None and timestamp >= self.valid_until)


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
        self._transforms: dict[str, list[VersionedInstrumentTransform]] = {}

        instruments = config.get("instruments", [])
        if not instruments:
            raise ValueError("Config does not contain any instruments")

        for instrument in instruments:
            name = instrument["name"]
            units = instrument.get(
                "units", self.canonical_system.get("units", "unknown")
            )

            if "versions" in instrument:
                raw_versions = instrument["versions"]
            else:
                raw_versions = [
                    {
                        "version": "v1",
                        "valid_from": None,
                        "valid_until": None,
                        "calibration_points": instrument.get("calibration_points", []),
                    }
                ]

            version_list: list[VersionedInstrumentTransform] = []
            for raw_v in raw_versions:
                version_label = raw_v["version"]
                valid_from = self._parse_timestamp(raw_v.get("valid_from"))
                valid_until = self._parse_timestamp(raw_v.get("valid_until"))
                calibration_points = raw_v.get("calibration_points", [])
                matrix = self._fit_affine_matrix(calibration_points)
                inverse_matrix = np.linalg.inv(matrix)
                cond = np.linalg.cond(matrix[:2, :2])
                if cond > 1e10:
                    raise ValueError(
                        f"Calibration for '{name}/{version_label}' produces a near-singular affine matrix "
                        f"(condition number: {cond:.2e}). Check that calibration points are "
                        f"not nearly collinear."
                    )
                stored_points = tuple(
                    {
                        "instrument": tuple(map(float, point["instrument"])),
                        "sample": tuple(map(float, point["sample"])),
                    }
                    for point in calibration_points
                )
                it = InstrumentTransform(
                    name=f"{name}/{version_label}",
                    units=units,
                    matrix=matrix,
                    inverse_matrix=inverse_matrix,
                    calibration_points=stored_points,
                )
                version_list.append(
                    VersionedInstrumentTransform(
                        version=version_label,
                        valid_from=valid_from,
                        valid_until=valid_until,
                        transform=it,
                    )
                )

            version_list.sort(
                key=lambda v: (
                    v.valid_from
                    if v.valid_from is not None
                    else datetime.min.replace(tzinfo=timezone.utc)
                )
            )
            self._transforms[name] = version_list

    @classmethod
    def from_yaml(cls, path: str | Path) -> CoordinateTransformer:
        path = Path(path)
        with path.open("r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
        return cls(config)

    @staticmethod
    def _parse_timestamp(value: str | None) -> datetime | None:
        """Parse an ISO 8601 timestamp string into a timezone-aware datetime.

        Returns None if value is None (representing unbounded).
        Raises ValueError if the parsed datetime is naive (no timezone).
        """
        if value is None:
            return None
        dt = (
            value if isinstance(value, datetime) else datetime.fromisoformat(str(value))
        )
        if dt.tzinfo is None:
            raise ValueError(
                f"Timestamp {value!r} in YAML must include a timezone "
                f"(e.g., '2026-06-15T00:00:00Z' or "
                f"'2026-06-15T00:00:00-04:00')"
            )
        return dt

    @staticmethod
    def _fit_affine_matrix(
        calibration_points: Sequence[dict[str, Sequence[float]]],
    ) -> np.ndarray:
        if len(calibration_points) < 3:
            raise ValueError(
                "At least three calibration points are required for an affine transform"
            )

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

    def _resolve_version(
        self, instrument_name: str, timestamp: datetime | None = None
    ) -> InstrumentTransform:
        """Select the correct transform version for an instrument.

        Parameters
        ----------
        instrument_name : str
            Instrument name (e.g., "MAXIMA").
        timestamp : datetime or None
            Timezone-aware datetime of the data file. If None, returns the
            current version (the one with valid_until=None). This is the
            guardrail for callers that cannot determine a file timestamp.

        Returns
        -------
        InstrumentTransform

        Raises
        ------
        KeyError
            If instrument_name is not found.
        ValueError
            If timestamp is naive, or if no version matches, or if multiple
            versions match (overlapping ranges — a YAML authoring error).
        """
        try:
            versions = self._transforms[instrument_name]
        except KeyError as exc:
            available = ", ".join(self.instruments())
            raise KeyError(
                f"Unknown instrument '{instrument_name}'. "
                f"Available instruments: {available}"
            ) from exc

        if timestamp is None:
            # Return the current version (valid_until is None)
            current = [v for v in versions if v.valid_until is None]
            if len(current) == 1:
                return current[0].transform
            if len(current) == 0:
                # All versions have ended — return the most recent
                return versions[-1].transform
            # Multiple open-ended versions — YAML authoring error
            raise ValueError(
                f"Multiple open-ended versions for '{instrument_name}'. "
                f"Only one version should have valid_until: null."
            )

        # timestamp is not None — validate and find the matching version
        if timestamp.tzinfo is None:
            raise ValueError(
                "timestamp must be timezone-aware; got naive datetime "
                f"{timestamp.isoformat()!r}. Use e.g. "
                "datetime.now(timezone.utc) or attach a tzinfo."
            )

        matches = [v for v in versions if v.is_valid_at(timestamp)]
        if len(matches) == 1:
            return matches[0].transform
        if len(matches) == 0:
            ranges = "; ".join(
                f"{v.version}: "
                f"{v.valid_from.isoformat() if v.valid_from else 'null'}"
                f" → "
                f"{v.valid_until.isoformat() if v.valid_until else 'null'}"
                for v in versions
            )
            raise ValueError(
                f"No valid coordinate version for '{instrument_name}' at "
                f"{timestamp.isoformat()}. Available versions: {ranges}"
            )
        raise ValueError(
            f"Overlapping validity ranges for '{instrument_name}' at "
            f"{timestamp.isoformat()}: versions "
            f"{', '.join(m.version for m in matches)}"
        )

    def get_transform(
        self, instrument_name: str, timestamp: datetime | None = None
    ) -> InstrumentTransform:
        return self._resolve_version(instrument_name, timestamp)

    def transform(
        self,
        instrument_name: str,
        x: float,
        y: float,
        timestamp: datetime | None = None,
    ) -> tuple[float, float]:
        return self.get_transform(instrument_name, timestamp).transform_point(x, y)

    def inverse_transform(
        self,
        instrument_name: str,
        x: float,
        y: float,
        timestamp: datetime | None = None,
    ) -> tuple[float, float]:
        return self.get_transform(instrument_name, timestamp).inverse_transform_point(
            x, y
        )

    def transform_points(
        self,
        instrument_name: str,
        points: Iterable[ArrayLike2D],
        timestamp: datetime | None = None,
    ) -> np.ndarray:
        return self.get_transform(instrument_name, timestamp).transform_points(points)

    def inverse_transform_points(
        self,
        instrument_name: str,
        points: Iterable[ArrayLike2D],
        timestamp: datetime | None = None,
    ) -> np.ndarray:
        return self.get_transform(instrument_name, timestamp).inverse_transform_points(
            points
        )

    def list_versions(self, instrument_name: str) -> list[dict[str, Any]]:
        """Return metadata about all versions of an instrument's transform.

        Parameters
        ----------
        instrument_name : str
            Instrument name (e.g., "MAXIMA").

        Returns
        -------
        list of dict
            Each dict contains: version, valid_from (ISO string or None),
            valid_until (ISO string or None), calibration_points (count),
            max_calibration_error (float).

        Raises
        ------
        KeyError
            If instrument_name is not found.
        """
        try:
            versions = self._transforms[instrument_name]
        except KeyError as exc:
            available = ", ".join(self.instruments())
            raise KeyError(
                f"Unknown instrument '{instrument_name}'. "
                f"Available instruments: {available}"
            ) from exc
        return [
            {
                "version": v.version,
                "valid_from": v.valid_from.isoformat() if v.valid_from else None,
                "valid_until": v.valid_until.isoformat() if v.valid_until else None,
                "calibration_points": len(v.transform.calibration_points),
                "max_calibration_error": v.transform.max_calibration_error(),
            }
            for v in versions
        ]

    def validate_version_continuity(self, instrument_name: str) -> list[str]:
        """Check for gaps, overlaps, or other issues in version ranges.

        Returns a list of warning strings. Empty list means the versions
        are cleanly contiguous.

        Parameters
        ----------
        instrument_name : str
            Instrument name (e.g., "MAXIMA").

        Returns
        -------
        list of str
            Warning messages. Empty if no issues found.

        Raises
        ------
        KeyError
            If instrument_name is not found.
        """
        try:
            versions = self._transforms[instrument_name]
        except KeyError as exc:
            raise KeyError(f"Unknown instrument '{instrument_name}'") from exc

        warnings: list[str] = []

        open_ended = [v for v in versions if v.valid_until is None]
        if len(open_ended) > 1:
            labels = ", ".join(v.version for v in open_ended)
            warnings.append(
                f"Multiple open-ended versions: {labels}. "
                f"Only one should have valid_until: null."
            )
        if len(open_ended) == 0 and len(versions) > 0:
            warnings.append(
                "No current version (all versions have valid_until set). "
                "Callers without a timestamp will get the most recent "
                "expired version."
            )

        for i in range(len(versions) - 1):
            current = versions[i]
            nxt = versions[i + 1]
            if current.valid_until is None:
                if nxt.valid_from is not None:
                    warnings.append(
                        f"Version '{current.version}' has no end date "
                        f"but is followed by '{nxt.version}' starting "
                        f"{nxt.valid_from.isoformat()}."
                    )
                continue
            if nxt.valid_from is None:
                warnings.append(
                    f"Version '{nxt.version}' has no start date but "
                    f"follows '{current.version}' ending "
                    f"{current.valid_until.isoformat()}."
                )
                continue
            if current.valid_until < nxt.valid_from:
                gap = nxt.valid_from - current.valid_until
                warnings.append(
                    f"Gap of {gap} between '{current.version}' "
                    f"(ends {current.valid_until.isoformat()}) and "
                    f"'{nxt.version}' "
                    f"(starts {nxt.valid_from.isoformat()})."
                )
            elif current.valid_until > nxt.valid_from:
                overlap = current.valid_until - nxt.valid_from
                warnings.append(
                    f"Overlap of {overlap} between "
                    f"'{current.version}' "
                    f"(ends {current.valid_until.isoformat()}) and "
                    f"'{nxt.version}' "
                    f"(starts {nxt.valid_from.isoformat()})."
                )

        return warnings

    def validate(self) -> dict[str, float]:
        result = {}
        for name, versions in sorted(self._transforms.items()):
            for v in versions:
                key = f"{name}/{v.version}" if len(versions) > 1 else name
                result[key] = v.transform.max_calibration_error()
        return result

    def summary(self) -> dict[str, Any]:
        instruments_summary = {}
        for name, versions in sorted(self._transforms.items()):
            if len(versions) == 1:
                v = versions[0]
                instruments_summary[name] = {
                    "units": v.transform.units,
                    "matrix": v.transform.matrix.tolist(),
                    "max_calibration_error": v.transform.max_calibration_error(),
                }
            else:
                instruments_summary[name] = {
                    "versions": [
                        {
                            "version": v.version,
                            "valid_from": (
                                v.valid_from.isoformat() if v.valid_from else None
                            ),
                            "valid_until": (
                                v.valid_until.isoformat() if v.valid_until else None
                            ),
                            "units": v.transform.units,
                            "matrix": v.transform.matrix.tolist(),
                            "max_calibration_error": v.transform.max_calibration_error(),
                        }
                        for v in versions
                    ]
                }
        return {
            "canonical_system": self.canonical_system,
            "instruments": instruments_summary,
        }


def _build_cli() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Transform instrument coordinates into sample coordinates"
    )
    parser.add_argument("config", type=Path, help="Path to the YAML configuration file")
    parser.add_argument("instrument", help="Instrument name, for example MAXIMA")
    parser.add_argument(
        "x", type=float, nargs="?", default=0.0, help="Instrument x coordinate"
    )
    parser.add_argument(
        "y", type=float, nargs="?", default=0.0, help="Instrument y coordinate"
    )
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
    parser.add_argument(
        "--timestamp",
        type=str,
        default=None,
        help=(
            "ISO 8601 timestamp for version selection "
            "(e.g., '2026-01-15T12:00:00Z'). "
            "If omitted, uses the current coordinate version."
        ),
    )
    parser.add_argument(
        "--list-versions",
        action="store_true",
        help="List all coordinate versions for the given instrument and exit.",
    )
    return parser


def main() -> None:
    parser = _build_cli()
    args = parser.parse_args()

    transformer = CoordinateTransformer.from_yaml(args.config)

    if args.list_versions:
        versions = transformer.list_versions(args.instrument)
        for v in versions:
            current = " (current)" if v["valid_until"] is None else ""
            print(
                f"  {args.instrument} {v['version']}: "
                f"{v['valid_from'] or 'null'} → "
                f"{v['valid_until'] or 'null'}"
                f"  ({v['calibration_points']} cal points, "
                f"error={v['max_calibration_error']:.2e})"
                f"{current}"
            )
        return

    ts = None
    if args.timestamp:
        ts_str = args.timestamp
        if ts_str.endswith("Z"):
            ts_str = ts_str[:-1] + "+00:00"
        try:
            ts = datetime.fromisoformat(ts_str)
        except ValueError:
            print(
                "Error: --timestamp must be an ISO 8601 string with "
                "timezone (e.g., 2026-01-15T12:00:00Z)"
            )
            raise SystemExit(1) from None
        if ts.tzinfo is None:
            print(
                "Error: --timestamp must include timezone "
                "(e.g., 2026-01-15T12:00:00Z)"
            )
            raise SystemExit(1)

    if args.inverse:
        x_out, y_out = transformer.inverse_transform(
            args.instrument, args.x, args.y, timestamp=ts
        )
        print(
            f"sample ({args.x:.6f}, {args.y:.6f}) -> "
            f"{args.instrument} ({x_out:.6f}, {y_out:.6f})"
        )
    else:
        x_out, y_out = transformer.transform(
            args.instrument, args.x, args.y, timestamp=ts
        )
        print(
            f"{args.instrument} ({args.x:.6f}, {args.y:.6f}) -> "
            f"sample ({x_out:.6f}, {y_out:.6f})"
        )

    if args.show_matrix:
        matrix = transformer.get_transform(args.instrument, timestamp=ts).matrix
        print("Affine matrix:")
        print(matrix)


if __name__ == "__main__":
    main()
