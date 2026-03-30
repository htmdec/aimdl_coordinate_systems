# Coordinate Transformer

Config-driven affine coordinate transforms for lab instruments. (updated to v0.2.1 for PyPI)

This repository provides a small Python module that reads instrument calibration data from `instrument_coordinate_transforms.yaml`, fits a 2D affine transform for each instrument, and converts `(x, y)` points between an instrument coordinate system and a canonical sample coordinate system.

The current configuration includes these instruments:

- `MAXIMA`
- `HELIX`
- `SPHINX`

## Repository contents

- `src/coordinate_transformer/` is the package with the `CoordinateTransformer` class and a small CLI
- `instrument_coordinate_transforms.yaml` is the calibration data config based on the canonical sample coordinate system
- `coordinate_transform_example.ipynb` is a simple example notebook showing the transformer in action
- `tests/` is the test suite (40 tests covering transforms, validation, batch operations, and the CLI)
- `pyproject.toml` has project metadata, build configuration, and dependency groups
- `.circleci/` contains the CircleCI pipeline configuration for continuous integration
- `.github/workflows/` contains the GitHub Actions workflow for publishing to PyPI on tagged releases
- `LICENSE` is the MIT license

## How it works

For each instrument, the YAML file contains three calibration point pairs:

- `instrument: [x_inst, y_inst]`
- `sample: [x_sample, y_sample]`

The module fits a 2D affine transform from instrument coordinates to sample coordinates. With three non-collinear calibration pairs, the transform is determined exactly. If you later provide more than three calibration pairs, the code will fit the transform by least squares.

## Requirements

- Python 3.9+

Core dependencies (installed automatically):

- `numpy`
- `PyYAML`

Optional extras:

- `pip install coordinate-transformer[notebook]` adds `jupyter` and `ipykernel` for running the example notebook
- `pip install coordinate-transformer[test]` adds `pytest`, `pytest-cov`, and `ruff` for testing and linting

## Installation

### From PyPI

```bash
pip install coordinate-transformer
```

### Development installation

Clone the repository, create a virtual environment, and install in editable mode with test dependencies:

#### macOS / Linux

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -e ".[test]"
```

#### Windows PowerShell

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -e ".[test]"
```

## Running tests

```bash
pip install -e ".[test]"
pytest
```

Linting is available via:

```bash
ruff check .
```

CI runs automatically on CircleCI (for every push) and GitHub Actions (for tagged releases).

## YAML configuration

The transformer is driven by `instrument_coordinate_transforms.yaml`.

The top-level structure is:

```yaml
canonical_system:
  origin: bottom-left
  x_positive: right
  y_positive: up
  units: mm
  sample_width: 40.0
  sample_height: 40.0

instruments:
  - name: MAXIMA
    units: mm
    calibration_points:
      - instrument: [x1, y1]
        sample: [x1s, y1s]
      - instrument: [x2, y2]
        sample: [x2s, y2s]
      - instrument: [x3, y3]
        sample: [x3s, y3s]
```

To update an instrument, edit its calibration points in the YAML. To add another instrument, add another entry under `instruments` with at least three non-collinear calibration pairs.

## Using the class

Import the class and load the YAML file:

```python
from coordinate_transformer import CoordinateTransformer

transformer = CoordinateTransformer.from_yaml("instrument_coordinate_transforms.yaml")
```

### List available instruments

```python
transformer.instruments()
```

### Transform one point into sample coordinates

```python
x_sample, y_sample = transformer.transform("MAXIMA", -14.0, -20.0)
print(x_sample, y_sample)
```

### Inverse transform from sample coordinates back to an instrument frame

```python
x_inst, y_inst = transformer.inverse_transform("SPHINX", 40.0, 40.0)
print(x_inst, y_inst)
```

### Transform a batch of points

```python
import numpy as np

points = np.array([
    [125.319, 148.213],
    [124.924, 107.753],
    [165.189, 107.350],
])

sample_points = transformer.transform_points("SPHINX", points)
print(sample_points)
```

### Inspect the affine matrix for one instrument

```python
matrix = transformer.get_transform("HELIX").matrix
print(matrix)
```

### Validate calibration fit

```python
errors = transformer.validate()
print(errors)
```

This returns the maximum calibration-point residual for each instrument.

## API summary

### `CoordinateTransformer`

Main methods:

- `CoordinateTransformer.from_yaml(path)`
- `instruments()`
- `get_transform(instrument_name)`
- `transform(instrument_name, x, y)`
- `inverse_transform(instrument_name, x, y)`
- `transform_points(instrument_name, points)`
- `inverse_transform_points(instrument_name, points)`
- `validate()`
- `summary()`

### `InstrumentTransform`

Returned by `get_transform(...)`. Useful attributes and methods:

- `name`
- `units`
- `matrix`
- `inverse_matrix`
- `transform_point(x, y)`
- `inverse_transform_point(x, y)`
- `transform_points(points)`
- `inverse_transform_points(points)`
- `calibration_residuals()`
- `max_calibration_error()`

## Command-line usage

The module also includes a small CLI.

### Transform an instrument point into sample coordinates

```bash
python -m coordinate_transformer instrument_coordinate_transforms.yaml MAXIMA -14 -20
```

### Inverse transform from sample coordinates into instrument coordinates

```bash
python -m coordinate_transformer instrument_coordinate_transforms.yaml SPHINX 40 40 --inverse
```

### Show the fitted affine matrix

```bash
python -m coordinate_transformer instrument_coordinate_transforms.yaml HELIX 8 8 --show-matrix
```

### Using the console script

After installation, `coordinate-transformer` is also available as a standalone command:

```bash
coordinate-transformer instrument_coordinate_transforms.yaml MAXIMA -14 -20
```

## Running the example notebook

After installing the environment:

```bash
jupyter notebook coordinate_transform_example.ipynb
```

The notebook demonstrates how to:

- load the YAML configuration
- inspect the canonical coordinate system
- list instruments
- validate calibration errors
- transform a single point
- transform a batch of points
- perform an inverse transform
- inspect a fitted affine matrix

If you want to use the notebook with a dedicated kernel, you can register the environment explicitly:

```bash
python -m ipykernel install --user --name coordinate-transformer --display-name "Python (coordinate-transformer)"
```

Then choose that kernel in Jupyter.

## Versioning and releases

The version is derived automatically from git tags using [setuptools-scm](https://github.com/pypa/setuptools-scm). To release a new version:

1. Tag the commit with a `v` prefix (e.g., `git tag v0.2.0`)
2. Push the tag: `git push origin --tags`
3. The GitHub Actions workflow will publish to PyPI automatically

## Notes

- At least three non-collinear calibration points are required per instrument.
- If the calibration points are degenerate, the code raises an error.
- Units are recorded per instrument, but the transform assumes your calibration pairs already define the correct mapping into the canonical sample units.
- The current implementation uses a full affine model, so it can represent translation, rotation, reflection, scaling, shear, or combinations of those.

## Typical update workflow

1. Edit `instrument_coordinate_transforms.yaml`
2. Replace or add calibration points for the relevant instrument
3. Reload the transformer:

```python
transformer = CoordinateTransformer.from_yaml("instrument_coordinate_transforms.yaml")
```

4. Re-run `transformer.validate()`
5. Re-run the example notebook if needed
