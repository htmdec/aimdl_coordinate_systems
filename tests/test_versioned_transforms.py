"""Tests for versioned coordinate transforms."""

from __future__ import annotations

from datetime import datetime, timezone
from subprocess import run as subprocess_run

import pytest
import yaml

from coordinate_transformer import CoordinateTransformer

# -- Timestamps for testing --------------------------------------------------

BOUNDARY = datetime(2026, 6, 15, tzinfo=timezone.utc)
JUST_BEFORE = datetime(2026, 6, 14, 23, 59, 59, tzinfo=timezone.utc)
JUST_AFTER = datetime(2026, 6, 15, 0, 0, 1, tzinfo=timezone.utc)
WELL_BEFORE = datetime(2020, 1, 1, tzinfo=timezone.utc)
WELL_AFTER = datetime(2030, 1, 1, tzinfo=timezone.utc)

# -- Configs -----------------------------------------------------------------

VERSIONED_CONFIG = {
    "canonical_system": {"units": "mm"},
    "instruments": [
        {
            "name": "RECAL",
            "units": "mm",
            "versions": [
                {
                    "version": "v1",
                    "valid_from": None,
                    "valid_until": "2026-06-15T00:00:00+00:00",
                    "calibration_points": [
                        {"instrument": [0, 0], "sample": [10, 20]},
                        {"instrument": [1, 0], "sample": [11, 20]},
                        {"instrument": [0, 1], "sample": [10, 21]},
                    ],
                },
                {
                    "version": "v2",
                    "valid_from": "2026-06-15T00:00:00+00:00",
                    "valid_until": None,
                    "calibration_points": [
                        {"instrument": [0, 0], "sample": [20, 30]},
                        {"instrument": [1, 0], "sample": [21, 30]},
                        {"instrument": [0, 1], "sample": [20, 31]},
                    ],
                },
            ],
        },
        {
            "name": "STABLE",
            "units": "mm",
            "calibration_points": [
                {"instrument": [0, 0], "sample": [0, 0]},
                {"instrument": [1, 0], "sample": [1, 0]},
                {"instrument": [0, 1], "sample": [0, 1]},
            ],
        },
    ],
}

GAP_CONFIG = {
    "canonical_system": {"units": "mm"},
    "instruments": [
        {
            "name": "GAPPY",
            "units": "mm",
            "versions": [
                {
                    "version": "v1",
                    "valid_from": None,
                    "valid_until": "2026-06-01T00:00:00+00:00",
                    "calibration_points": [
                        {"instrument": [0, 0], "sample": [10, 20]},
                        {"instrument": [1, 0], "sample": [11, 20]},
                        {"instrument": [0, 1], "sample": [10, 21]},
                    ],
                },
                {
                    "version": "v2",
                    "valid_from": "2026-07-01T00:00:00+00:00",
                    "valid_until": None,
                    "calibration_points": [
                        {"instrument": [0, 0], "sample": [20, 30]},
                        {"instrument": [1, 0], "sample": [21, 30]},
                        {"instrument": [0, 1], "sample": [20, 31]},
                    ],
                },
            ],
        },
    ],
}

OVERLAP_CONFIG = {
    "canonical_system": {"units": "mm"},
    "instruments": [
        {
            "name": "OVERLAPPY",
            "units": "mm",
            "versions": [
                {
                    "version": "v1",
                    "valid_from": None,
                    "valid_until": "2026-07-01T00:00:00+00:00",
                    "calibration_points": [
                        {"instrument": [0, 0], "sample": [10, 20]},
                        {"instrument": [1, 0], "sample": [11, 20]},
                        {"instrument": [0, 1], "sample": [10, 21]},
                    ],
                },
                {
                    "version": "v2",
                    "valid_from": "2026-06-01T00:00:00+00:00",
                    "valid_until": None,
                    "calibration_points": [
                        {"instrument": [0, 0], "sample": [20, 30]},
                        {"instrument": [1, 0], "sample": [21, 30]},
                        {"instrument": [0, 1], "sample": [20, 31]},
                    ],
                },
            ],
        },
    ],
}

ALL_EXPIRED_CONFIG = {
    "canonical_system": {"units": "mm"},
    "instruments": [
        {
            "name": "RETIRED",
            "units": "mm",
            "versions": [
                {
                    "version": "v1",
                    "valid_from": None,
                    "valid_until": "2025-01-01T00:00:00+00:00",
                    "calibration_points": [
                        {"instrument": [0, 0], "sample": [10, 20]},
                        {"instrument": [1, 0], "sample": [11, 20]},
                        {"instrument": [0, 1], "sample": [10, 21]},
                    ],
                },
                {
                    "version": "v2",
                    "valid_from": "2025-01-01T00:00:00+00:00",
                    "valid_until": "2026-01-01T00:00:00+00:00",
                    "calibration_points": [
                        {"instrument": [0, 0], "sample": [20, 30]},
                        {"instrument": [1, 0], "sample": [21, 30]},
                        {"instrument": [0, 1], "sample": [20, 31]},
                    ],
                },
            ],
        },
    ],
}


MULTI_OPEN_CONFIG = {
    "canonical_system": {"units": "mm"},
    "instruments": [
        {
            "name": "DUPES",
            "units": "mm",
            "versions": [
                {
                    "version": "v1",
                    "valid_from": None,
                    "valid_until": None,
                    "calibration_points": [
                        {"instrument": [0, 0], "sample": [10, 20]},
                        {"instrument": [1, 0], "sample": [11, 20]},
                        {"instrument": [0, 1], "sample": [10, 21]},
                    ],
                },
                {
                    "version": "v2",
                    "valid_from": "2026-06-15T00:00:00+00:00",
                    "valid_until": None,
                    "calibration_points": [
                        {"instrument": [0, 0], "sample": [20, 30]},
                        {"instrument": [1, 0], "sample": [21, 30]},
                        {"instrument": [0, 1], "sample": [20, 31]},
                    ],
                },
            ],
        },
    ],
}


@pytest.fixture
def versioned_transformer():
    return CoordinateTransformer(VERSIONED_CONFIG)


# ---------------------------------------------------------------------------
# TestIsValidAt
# ---------------------------------------------------------------------------

class TestIsValidAt:
    def _get_versions(self, transformer, instrument_name):
        return transformer._transforms[instrument_name]

    def test_within_range(self, versioned_transformer):
        v1 = self._get_versions(versioned_transformer, "RECAL")[0]
        assert v1.is_valid_at(WELL_BEFORE)

    def test_before_range(self, versioned_transformer):
        v2 = self._get_versions(versioned_transformer, "RECAL")[1]
        assert not v2.is_valid_at(WELL_BEFORE)

    def test_after_range(self, versioned_transformer):
        v1 = self._get_versions(versioned_transformer, "RECAL")[0]
        assert not v1.is_valid_at(WELL_AFTER)

    def test_boundary_inclusive(self, versioned_transformer):
        v2 = self._get_versions(versioned_transformer, "RECAL")[1]
        assert v2.is_valid_at(BOUNDARY)

    def test_boundary_exclusive(self, versioned_transformer):
        v1 = self._get_versions(versioned_transformer, "RECAL")[0]
        assert not v1.is_valid_at(BOUNDARY)

    def test_unbounded_start(self, versioned_transformer):
        v1 = self._get_versions(versioned_transformer, "RECAL")[0]
        assert v1.valid_from is None
        assert v1.is_valid_at(WELL_BEFORE)

    def test_unbounded_end(self, versioned_transformer):
        v2 = self._get_versions(versioned_transformer, "RECAL")[1]
        assert v2.valid_until is None
        assert v2.is_valid_at(WELL_AFTER)

    def test_fully_unbounded(self, versioned_transformer):
        stable = self._get_versions(versioned_transformer, "STABLE")[0]
        assert stable.valid_from is None
        assert stable.valid_until is None
        assert stable.is_valid_at(WELL_BEFORE)
        assert stable.is_valid_at(WELL_AFTER)

    def test_naive_datetime_raises(self, versioned_transformer):
        v1 = self._get_versions(versioned_transformer, "RECAL")[0]
        with pytest.raises(ValueError, match="timezone-aware"):
            v1.is_valid_at(datetime(2026, 1, 1))


# ---------------------------------------------------------------------------
# TestVersionResolution
# ---------------------------------------------------------------------------

class TestVersionResolution:
    def test_no_timestamp_returns_current(self, versioned_transformer):
        t = versioned_transformer.get_transform("RECAL")
        x, y = t.transform_point(0, 0)
        assert x == pytest.approx(20.0)
        assert y == pytest.approx(30.0)

    def test_before_boundary_returns_v1(self, versioned_transformer):
        x, y = versioned_transformer.transform("RECAL", 0, 0, timestamp=JUST_BEFORE)
        assert x == pytest.approx(10.0)
        assert y == pytest.approx(20.0)

    def test_after_boundary_returns_v2(self, versioned_transformer):
        x, y = versioned_transformer.transform("RECAL", 0, 0, timestamp=JUST_AFTER)
        assert x == pytest.approx(20.0)
        assert y == pytest.approx(30.0)

    def test_at_exact_boundary_returns_v2(self, versioned_transformer):
        x, y = versioned_transformer.transform("RECAL", 0, 0, timestamp=BOUNDARY)
        assert x == pytest.approx(20.0)
        assert y == pytest.approx(30.0)

    def test_well_before_returns_v1(self, versioned_transformer):
        x, y = versioned_transformer.transform("RECAL", 0, 0, timestamp=WELL_BEFORE)
        assert x == pytest.approx(10.0)
        assert y == pytest.approx(20.0)

    def test_naive_timestamp_raises(self, versioned_transformer):
        with pytest.raises(ValueError):
            versioned_transformer.get_transform("RECAL", timestamp=datetime(2026, 1, 1))

    def test_unknown_instrument_raises(self, versioned_transformer):
        with pytest.raises(KeyError):
            versioned_transformer.get_transform("NONEXISTENT")


# ---------------------------------------------------------------------------
# TestVersionResolutionEdgeCases
# ---------------------------------------------------------------------------

class TestVersionResolutionEdgeCases:
    def test_gap_raises(self):
        t = CoordinateTransformer(GAP_CONFIG)
        with pytest.raises(ValueError, match="No valid"):
            t.get_transform("GAPPY", timestamp=datetime(2026, 6, 15, tzinfo=timezone.utc))

    def test_overlap_raises(self):
        t = CoordinateTransformer(OVERLAP_CONFIG)
        with pytest.raises(ValueError, match="Overlapping"):
            t.get_transform("OVERLAPPY", timestamp=datetime(2026, 6, 15, tzinfo=timezone.utc))

    def test_all_expired_no_timestamp(self):
        t = CoordinateTransformer(ALL_EXPIRED_CONFIG)
        transform = t.get_transform("RETIRED")
        x, y = transform.transform_point(0, 0)
        assert x == pytest.approx(20.0)
        assert y == pytest.approx(30.0)

    def test_all_expired_within_v1(self):
        t = CoordinateTransformer(ALL_EXPIRED_CONFIG)
        transform = t.get_transform("RETIRED", timestamp=datetime(2024, 6, 1, tzinfo=timezone.utc))
        x, y = transform.transform_point(0, 0)
        assert x == pytest.approx(10.0)
        assert y == pytest.approx(20.0)


# ---------------------------------------------------------------------------
# TestBackwardCompatibility
# ---------------------------------------------------------------------------

class TestBackwardCompatibility:
    def test_flat_format_loads(self, versioned_transformer):
        assert "STABLE" in versioned_transformer.instruments()

    def test_flat_format_no_timestamp(self, versioned_transformer):
        x, y = versioned_transformer.transform("STABLE", 5.0, 7.0)
        assert x == pytest.approx(5.0)
        assert y == pytest.approx(7.0)

    def test_flat_format_with_timestamp(self, versioned_transformer):
        x, y = versioned_transformer.transform("STABLE", 5.0, 7.0, timestamp=WELL_BEFORE)
        assert x == pytest.approx(5.0)
        assert y == pytest.approx(7.0)

    def test_instruments_list(self, versioned_transformer):
        assert set(versioned_transformer.instruments()) == {"RECAL", "STABLE"}

    def test_transform_backward_compatible_signature(self, versioned_transformer):
        x, y = versioned_transformer.transform("STABLE", 5.0, 7.0)
        assert x == pytest.approx(5.0)
        assert y == pytest.approx(7.0)


# ---------------------------------------------------------------------------
# TestTimestampPropagation
# ---------------------------------------------------------------------------

class TestTimestampPropagation:
    def test_transform_with_timestamp(self, versioned_transformer):
        x, y = versioned_transformer.transform("RECAL", 0, 0, timestamp=JUST_BEFORE)
        assert x == pytest.approx(10.0)
        assert y == pytest.approx(20.0)

    def test_inverse_transform_with_timestamp(self, versioned_transformer):
        x, y = versioned_transformer.inverse_transform("RECAL", 10, 20, timestamp=JUST_BEFORE)
        assert x == pytest.approx(0.0)
        assert y == pytest.approx(0.0)

    def test_transform_points_with_timestamp(self, versioned_transformer):
        import numpy as np
        points = np.array([[0.0, 0.0]])
        result = versioned_transformer.transform_points("RECAL", points, timestamp=JUST_BEFORE)
        assert result[0, 0] == pytest.approx(10.0)
        assert result[0, 1] == pytest.approx(20.0)

    def test_inverse_transform_points_with_timestamp(self, versioned_transformer):
        import numpy as np
        points = np.array([[10.0, 20.0]])
        result = versioned_transformer.inverse_transform_points("RECAL", points, timestamp=JUST_BEFORE)
        assert result[0, 0] == pytest.approx(0.0)
        assert result[0, 1] == pytest.approx(0.0)

    def test_roundtrip_with_timestamp(self, versioned_transformer):
        x_orig, y_orig = 3.0, 4.0
        x_s, y_s = versioned_transformer.transform("RECAL", x_orig, y_orig, timestamp=JUST_BEFORE)
        x_back, y_back = versioned_transformer.inverse_transform("RECAL", x_s, y_s, timestamp=JUST_BEFORE)
        assert x_back == pytest.approx(x_orig, abs=1e-10)
        assert y_back == pytest.approx(y_orig, abs=1e-10)


# ---------------------------------------------------------------------------
# TestValidateAndSummary
# ---------------------------------------------------------------------------

class TestValidateAndSummary:
    def test_validate_single_version(self, versioned_transformer):
        errors = versioned_transformer.validate()
        assert "STABLE" in errors

    def test_validate_multi_version(self, versioned_transformer):
        errors = versioned_transformer.validate()
        assert "RECAL/v1" in errors
        assert "RECAL/v2" in errors

    def test_summary_single_version_format(self, versioned_transformer):
        s = versioned_transformer.summary()
        stable = s["instruments"]["STABLE"]
        assert "units" in stable
        assert "matrix" in stable
        assert "max_calibration_error" in stable

    def test_summary_multi_version_format(self, versioned_transformer):
        s = versioned_transformer.summary()
        recal = s["instruments"]["RECAL"]
        assert "versions" in recal
        assert len(recal["versions"]) == 2


# ---------------------------------------------------------------------------
# TestYAMLTimezoneEnforcement
# ---------------------------------------------------------------------------

class TestYAMLTimezoneEnforcement:
    def test_naive_timestamp_in_yaml_raises(self):
        bad_config = {
            "canonical_system": {"units": "mm"},
            "instruments": [
                {
                    "name": "BAD",
                    "units": "mm",
                    "versions": [
                        {
                            "version": "v1",
                            "valid_from": "2026-06-15T00:00:00",
                            "valid_until": None,
                            "calibration_points": [
                                {"instrument": [0, 0], "sample": [0, 0]},
                                {"instrument": [1, 0], "sample": [1, 0]},
                                {"instrument": [0, 1], "sample": [0, 1]},
                            ],
                        },
                    ],
                },
            ],
        }
        with pytest.raises(ValueError, match="timezone"):
            CoordinateTransformer(bad_config)


# ---------------------------------------------------------------------------
# TestListVersions
# ---------------------------------------------------------------------------

class TestListVersions:
    def test_list_versions_multi(self, versioned_transformer):
        versions = versioned_transformer.list_versions("RECAL")
        assert len(versions) == 2
        labels = [v["version"] for v in versions]
        assert labels == ["v1", "v2"]
        assert versions[0]["valid_from"] is None
        assert versions[0]["valid_until"] == "2026-06-15T00:00:00+00:00"
        assert versions[1]["valid_from"] == "2026-06-15T00:00:00+00:00"
        assert versions[1]["valid_until"] is None

    def test_list_versions_single(self, versioned_transformer):
        versions = versioned_transformer.list_versions("STABLE")
        assert len(versions) == 1

    def test_list_versions_unknown_raises(self, versioned_transformer):
        with pytest.raises(KeyError):
            versioned_transformer.list_versions("NOPE")


# ---------------------------------------------------------------------------
# TestValidateVersionContinuity
# ---------------------------------------------------------------------------

class TestValidateVersionContinuity:
    def test_clean_continuity(self, versioned_transformer):
        warnings = versioned_transformer.validate_version_continuity("RECAL")
        assert warnings == []

    def test_gap_detected(self):
        t = CoordinateTransformer(GAP_CONFIG)
        warnings = t.validate_version_continuity("GAPPY")
        assert any("Gap" in w for w in warnings)

    def test_overlap_detected(self):
        t = CoordinateTransformer(OVERLAP_CONFIG)
        warnings = t.validate_version_continuity("OVERLAPPY")
        assert any("Overlap" in w for w in warnings)

    def test_multiple_open_ended_detected(self):
        t = CoordinateTransformer(MULTI_OPEN_CONFIG)
        warnings = t.validate_version_continuity("DUPES")
        assert any("Multiple open-ended" in w for w in warnings)

    def test_all_expired_warning(self):
        t = CoordinateTransformer(ALL_EXPIRED_CONFIG)
        warnings = t.validate_version_continuity("RETIRED")
        assert any("No current version" in w for w in warnings)

    def test_single_version_clean(self, versioned_transformer):
        warnings = versioned_transformer.validate_version_continuity("STABLE")
        assert warnings == []


# ---------------------------------------------------------------------------
# TestCLI
# ---------------------------------------------------------------------------

@pytest.fixture
def versioned_yaml_path(tmp_path):
    path = tmp_path / "versioned.yaml"
    with path.open("w", encoding="utf-8") as f:
        yaml.dump(VERSIONED_CONFIG, f)
    return path


class TestCLI:
    def test_cli_forward_default_version(self, versioned_yaml_path):
        result = subprocess_run(
            ["python", "-m", "coordinate_transformer",
             str(versioned_yaml_path), "RECAL", "0", "0"],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        assert "20.000000" in result.stdout

    def test_cli_with_timestamp_v1(self, versioned_yaml_path):
        result = subprocess_run(
            ["python", "-m", "coordinate_transformer",
             str(versioned_yaml_path), "RECAL", "0", "0",
             "--timestamp", "2026-01-01T00:00:00+00:00"],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        assert "10.000000" in result.stdout

    def test_cli_with_timestamp_v2(self, versioned_yaml_path):
        result = subprocess_run(
            ["python", "-m", "coordinate_transformer",
             str(versioned_yaml_path), "RECAL", "0", "0",
             "--timestamp", "2026-07-01T00:00:00+00:00"],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        assert "20.000000" in result.stdout

    def test_cli_list_versions(self, versioned_yaml_path):
        result = subprocess_run(
            ["python", "-m", "coordinate_transformer",
             str(versioned_yaml_path), "RECAL", "--list-versions"],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        assert "v1" in result.stdout
        assert "v2" in result.stdout
        assert "(current)" in result.stdout

    def test_cli_naive_timestamp_error(self, versioned_yaml_path):
        result = subprocess_run(
            ["python", "-m", "coordinate_transformer",
             str(versioned_yaml_path), "RECAL", "0", "0",
             "--timestamp", "2026-01-01T00:00:00"],
            capture_output=True,
            text=True,
        )
        assert result.returncode != 0
        combined = result.stdout + result.stderr
        assert "timezone" in combined
