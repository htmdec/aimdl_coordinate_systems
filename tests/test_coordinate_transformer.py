"""Tests for coordinate_transformer module."""

from __future__ import annotations

import json
import textwrap
from pathlib import Path
from subprocess import run as subprocess_run

import numpy as np
import pytest
import yaml

from coordinate_transformer import CoordinateTransformer, InstrumentTransform


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

SIMPLE_CONFIG = {
    "canonical_system": {
        "origin": "bottom-right",
        "x_positive": "left",
        "y_positive": "up",
        "units": "mm",
        "sample_width": 40.0,
        "sample_height": 40.0,
    },
    "instruments": [
        {
            "name": "IDENTITY",
            "units": "mm",
            "calibration_points": [
                {"instrument": [0, 0], "sample": [0, 0]},
                {"instrument": [1, 0], "sample": [1, 0]},
                {"instrument": [0, 1], "sample": [0, 1]},
            ],
        },
        {
            "name": "TRANSLATE",
            "units": "mm",
            "calibration_points": [
                {"instrument": [0, 0], "sample": [10, 20]},
                {"instrument": [1, 0], "sample": [11, 20]},
                {"instrument": [0, 1], "sample": [10, 21]},
            ],
        },
        {
            "name": "SCALE2X",
            "units": "mm",
            "calibration_points": [
                {"instrument": [0, 0], "sample": [0, 0]},
                {"instrument": [1, 0], "sample": [2, 0]},
                {"instrument": [0, 1], "sample": [0, 2]},
            ],
        },
    ],
}

ROTATION_90_CONFIG = {
    "canonical_system": {"units": "mm"},
    "instruments": [
        {
            "name": "ROT90",
            "units": "mm",
            "calibration_points": [
                {"instrument": [0, 0], "sample": [0, 0]},
                {"instrument": [1, 0], "sample": [0, 1]},
                {"instrument": [0, 1], "sample": [-1, 0]},
            ],
        }
    ],
}


@pytest.fixture
def simple_transformer():
    return CoordinateTransformer(SIMPLE_CONFIG)


@pytest.fixture
def yaml_path(tmp_path):
    path = tmp_path / "test_config.yaml"
    path.write_text(yaml.dump(SIMPLE_CONFIG), encoding="utf-8")
    return path


@pytest.fixture
def real_yaml_path():
    return Path(__file__).resolve().parent.parent / "instrument_coordinate_transforms.yaml"


# ---------------------------------------------------------------------------
# Construction and loading
# ---------------------------------------------------------------------------

class TestConstruction:
    def test_from_dict(self, simple_transformer):
        assert set(simple_transformer.instruments()) == {"IDENTITY", "SCALE2X", "TRANSLATE"}

    def test_from_yaml(self, yaml_path):
        t = CoordinateTransformer.from_yaml(yaml_path)
        assert "IDENTITY" in t.instruments()

    def test_from_real_yaml(self, real_yaml_path):
        t = CoordinateTransformer.from_yaml(real_yaml_path)
        assert set(t.instruments()) == {"HELIX", "MAXIMA", "SPHINX"}

    def test_empty_instruments_raises(self):
        with pytest.raises(ValueError, match="does not contain any instruments"):
            CoordinateTransformer({"instruments": []})

    def test_missing_instruments_key_raises(self):
        with pytest.raises(ValueError, match="does not contain any instruments"):
            CoordinateTransformer({"canonical_system": {}})

    def test_too_few_calibration_points_raises(self):
        bad = {
            "instruments": [
                {
                    "name": "BAD",
                    "calibration_points": [
                        {"instrument": [0, 0], "sample": [0, 0]},
                        {"instrument": [1, 0], "sample": [1, 0]},
                    ],
                }
            ]
        }
        with pytest.raises(ValueError, match="At least three"):
            CoordinateTransformer(bad)

    def test_collinear_points_raises(self):
        bad = {
            "instruments": [
                {
                    "name": "BAD",
                    "calibration_points": [
                        {"instrument": [0, 0], "sample": [0, 0]},
                        {"instrument": [1, 0], "sample": [1, 0]},
                        {"instrument": [2, 0], "sample": [2, 0]},
                    ],
                }
            ]
        }
        with pytest.raises(ValueError, match="degenerate"):
            CoordinateTransformer(bad)

    def test_unknown_instrument_raises(self, simple_transformer):
        with pytest.raises(KeyError, match="Unknown instrument"):
            simple_transformer.get_transform("NONEXISTENT")


# ---------------------------------------------------------------------------
# Identity transform
# ---------------------------------------------------------------------------

class TestIdentityTransform:
    def test_forward(self, simple_transformer):
        x, y = simple_transformer.transform("IDENTITY", 5.0, 7.0)
        assert x == pytest.approx(5.0)
        assert y == pytest.approx(7.0)

    def test_inverse(self, simple_transformer):
        x, y = simple_transformer.inverse_transform("IDENTITY", 5.0, 7.0)
        assert x == pytest.approx(5.0)
        assert y == pytest.approx(7.0)

    def test_matrix_is_identity(self, simple_transformer):
        matrix = simple_transformer.get_transform("IDENTITY").matrix
        np.testing.assert_allclose(matrix, np.eye(3), atol=1e-12)


# ---------------------------------------------------------------------------
# Translation transform
# ---------------------------------------------------------------------------

class TestTranslationTransform:
    def test_forward(self, simple_transformer):
        x, y = simple_transformer.transform("TRANSLATE", 0.0, 0.0)
        assert x == pytest.approx(10.0)
        assert y == pytest.approx(20.0)

    def test_forward_nonzero(self, simple_transformer):
        x, y = simple_transformer.transform("TRANSLATE", 3.0, 4.0)
        assert x == pytest.approx(13.0)
        assert y == pytest.approx(24.0)

    def test_inverse(self, simple_transformer):
        x, y = simple_transformer.inverse_transform("TRANSLATE", 10.0, 20.0)
        assert x == pytest.approx(0.0)
        assert y == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Scale transform
# ---------------------------------------------------------------------------

class TestScaleTransform:
    def test_forward(self, simple_transformer):
        x, y = simple_transformer.transform("SCALE2X", 3.0, 5.0)
        assert x == pytest.approx(6.0)
        assert y == pytest.approx(10.0)

    def test_inverse(self, simple_transformer):
        x, y = simple_transformer.inverse_transform("SCALE2X", 6.0, 10.0)
        assert x == pytest.approx(3.0)
        assert y == pytest.approx(5.0)


# ---------------------------------------------------------------------------
# Rotation transform
# ---------------------------------------------------------------------------

class TestRotationTransform:
    def test_90_degree_rotation(self):
        t = CoordinateTransformer(ROTATION_90_CONFIG)
        x, y = t.transform("ROT90", 1.0, 0.0)
        assert x == pytest.approx(0.0, abs=1e-12)
        assert y == pytest.approx(1.0, abs=1e-12)

    def test_90_degree_inverse(self):
        t = CoordinateTransformer(ROTATION_90_CONFIG)
        x, y = t.inverse_transform("ROT90", 0.0, 1.0)
        assert x == pytest.approx(1.0, abs=1e-12)
        assert y == pytest.approx(0.0, abs=1e-12)


# ---------------------------------------------------------------------------
# Round-trip consistency
# ---------------------------------------------------------------------------

class TestRoundTrip:
    @pytest.mark.parametrize("instrument", ["IDENTITY", "TRANSLATE", "SCALE2X"])
    def test_forward_inverse_roundtrip(self, simple_transformer, instrument):
        x_orig, y_orig = 7.3, -2.1
        x_sample, y_sample = simple_transformer.transform(instrument, x_orig, y_orig)
        x_back, y_back = simple_transformer.inverse_transform(instrument, x_sample, y_sample)
        assert x_back == pytest.approx(x_orig, abs=1e-10)
        assert y_back == pytest.approx(y_orig, abs=1e-10)

    @pytest.mark.parametrize("instrument", ["IDENTITY", "TRANSLATE", "SCALE2X"])
    def test_inverse_forward_roundtrip(self, simple_transformer, instrument):
        x_orig, y_orig = 15.0, 25.0
        x_inst, y_inst = simple_transformer.inverse_transform(instrument, x_orig, y_orig)
        x_back, y_back = simple_transformer.transform(instrument, x_inst, y_inst)
        assert x_back == pytest.approx(x_orig, abs=1e-10)
        assert y_back == pytest.approx(y_orig, abs=1e-10)


# ---------------------------------------------------------------------------
# Batch transform
# ---------------------------------------------------------------------------

class TestBatchTransform:
    def test_batch_matches_single(self, simple_transformer):
        points = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        batch_result = simple_transformer.transform_points("TRANSLATE", points)
        for i, (x_in, y_in) in enumerate(points):
            x_s, y_s = simple_transformer.transform("TRANSLATE", x_in, y_in)
            assert batch_result[i, 0] == pytest.approx(x_s)
            assert batch_result[i, 1] == pytest.approx(y_s)

    def test_inverse_batch_matches_single(self, simple_transformer):
        points = np.array([[10.0, 20.0], [13.0, 24.0]])
        batch_result = simple_transformer.inverse_transform_points("TRANSLATE", points)
        for i, (x_in, y_in) in enumerate(points):
            x_s, y_s = simple_transformer.inverse_transform("TRANSLATE", x_in, y_in)
            assert batch_result[i, 0] == pytest.approx(x_s)
            assert batch_result[i, 1] == pytest.approx(y_s)

    def test_bad_shape_raises(self, simple_transformer):
        with pytest.raises(ValueError, match="pairs"):
            simple_transformer.transform_points("IDENTITY", np.array([[1, 2, 3]]))


# ---------------------------------------------------------------------------
# Calibration validation
# ---------------------------------------------------------------------------

class TestValidation:
    def test_exact_calibration_has_zero_residual(self, simple_transformer):
        errors = simple_transformer.validate()
        for name, err in errors.items():
            assert err == pytest.approx(0.0, abs=1e-12), f"{name} has nonzero calibration error"

    def test_calibration_residuals_shape(self, simple_transformer):
        t = simple_transformer.get_transform("TRANSLATE")
        residuals = t.calibration_residuals()
        assert residuals.shape == (3, 2)

    def test_real_yaml_calibration_errors(self, real_yaml_path):
        t = CoordinateTransformer.from_yaml(real_yaml_path)
        errors = t.validate()
        for name, err in errors.items():
            assert err < 1e-10, f"{name} calibration error {err} exceeds tolerance"


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

class TestSummary:
    def test_summary_structure(self, simple_transformer):
        s = simple_transformer.summary()
        assert "canonical_system" in s
        assert "instruments" in s
        assert set(s["instruments"].keys()) == {"IDENTITY", "SCALE2X", "TRANSLATE"}
        for name, info in s["instruments"].items():
            assert "units" in info
            assert "matrix" in info
            assert "max_calibration_error" in info


# ---------------------------------------------------------------------------
# Real instrument data
# ---------------------------------------------------------------------------

class TestRealInstruments:
    """Tests against the actual calibration YAML shipped with the repo."""

    def test_maxima_calibration_point_roundtrip(self, real_yaml_path):
        t = CoordinateTransformer.from_yaml(real_yaml_path)
        # First calibration point: instrument (-14, -20) -> sample (0, 0)
        x, y = t.transform("MAXIMA", -14.0, -20.0)
        assert x == pytest.approx(0.0, abs=1e-10)
        assert y == pytest.approx(0.0, abs=1e-10)

    def test_sphinx_calibration_point_roundtrip(self, real_yaml_path):
        t = CoordinateTransformer.from_yaml(real_yaml_path)
        # Third calibration point: instrument (165.189, 107.350) -> sample (40, 40)
        x, y = t.transform("SPHINX", 165.189, 107.350)
        assert x == pytest.approx(40.0, abs=1e-10)
        assert y == pytest.approx(40.0, abs=1e-10)

    def test_helix_calibration_point_roundtrip(self, real_yaml_path):
        t = CoordinateTransformer.from_yaml(real_yaml_path)
        # First calibration point: instrument (8, 8) -> sample (32, 8)
        x, y = t.transform("HELIX", 8.0, 8.0)
        assert x == pytest.approx(32.0, abs=1e-10)
        assert y == pytest.approx(8.0, abs=1e-10)

    @pytest.mark.parametrize("instrument", ["MAXIMA", "HELIX", "SPHINX"])
    def test_real_roundtrip(self, real_yaml_path, instrument):
        t = CoordinateTransformer.from_yaml(real_yaml_path)
        x_orig, y_orig = 20.0, 20.0
        x_s, y_s = t.transform(instrument, x_orig, y_orig)
        x_back, y_back = t.inverse_transform(instrument, x_s, y_s)
        assert x_back == pytest.approx(x_orig, abs=1e-10)
        assert y_back == pytest.approx(y_orig, abs=1e-10)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

class TestCLI:
    def test_forward_transform_cli(self, real_yaml_path):
        result = subprocess_run(
            ["python", "-m", "coordinate_transformer", str(real_yaml_path), "MAXIMA", "-14", "-20"],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        assert "sample" in result.stdout
        assert "0.000000" in result.stdout

    def test_inverse_transform_cli(self, real_yaml_path):
        result = subprocess_run(
            ["python", "-m", "coordinate_transformer", str(real_yaml_path), "MAXIMA", "0", "0", "--inverse"],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        assert "MAXIMA" in result.stdout
        assert "-14.000000" in result.stdout

    def test_show_matrix_cli(self, real_yaml_path):
        result = subprocess_run(
            ["python", "-m", "coordinate_transformer", str(real_yaml_path), "HELIX", "8", "8", "--show-matrix"],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        assert "Affine matrix" in result.stdout
