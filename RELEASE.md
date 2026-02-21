# Release Guide

This project uses Semantic Versioning (`MAJOR.MINOR.PATCH`).

- `PATCH`: bug fixes (`1.0.1` -> `1.0.2`)
- `MINOR`: backward-compatible features (`1.0.0` -> `1.1.0`)
- `MAJOR`: breaking changes (`1.0.0` -> `2.0.0`)

## Important Rule

PyPI and TestPyPI do **not** allow overwriting an existing uploaded file/version.

- If `1.0.0` is already published, publish a new version (for example `1.0.1`).
- If `1.0.1` is already published, publish a new version (for example `1.0.2`).
- Use `--skip-existing` only to make re-runs idempotent; it does not replace files.

## Pre-release Checklist

1. Docs/API sync
1. Git hygiene
1. Build artifacts
1. TestPyPI validation
1. PyPI release
1. Post-release verification

## Commands

### 1. Version bump

Update version in:
- `pyproject.toml`
- `src/forestplotx/__init__.py`
- `README.md` (Current version line)
- `CHANGELOG.md` (move notes from `Unreleased` to new version section)

### 2. Run tests

```bash
pytest
python tests/run_visual_tests.py
```

### 3. Clean + build + artifact check

```bash
rm -rf dist/ build/ src/*.egg-info
python -m build
python -m twine check dist/*
```

If `python -m build` fails in an offline environment, use:

```bash
python -m build --no-isolation
```

### 4. Upload to TestPyPI

```bash
python -m twine upload --repository testpypi dist/*
```

Retry-safe:

```bash
python -m twine upload --repository testpypi --skip-existing dist/*
```

### 5. Fresh install smoke test from TestPyPI

```bash
python -m venv /tmp/fpx-testpypi-smoke
source /tmp/fpx-testpypi-smoke/bin/activate
python -m pip install --upgrade pip
python -m pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple forestplotx==<VERSION>
python -c "import forestplotx; print(forestplotx.__version__)"
deactivate
```

### 6. Upload to PyPI

```bash
python -m twine upload dist/*
```

### 7. Fresh install smoke test from PyPI

```bash
python -m venv /tmp/fpx-pypi-smoke
source /tmp/fpx-pypi-smoke/bin/activate
python -m pip install --upgrade pip
python -m pip install forestplotx==<VERSION>
python -c "import forestplotx; print(forestplotx.__version__)"
deactivate
```

### 8. Functional smoke check (`forest_plot(save=...)`)

```bash
python - <<'PY'
import numpy as np
import pandas as pd
from forestplotx.plot import forest_plot

df = pd.DataFrame({
    "predictor": ["x1", "x2"],
    "outcome": ["y1", "y1"],
    "Estimate": [0.0, 0.2],
    "CI_low": [-0.1, 0.1],
    "CI_high": [0.1, 0.3],
    "p_value": [0.2, 0.01],
})

fig, _ = forest_plot(
    df=df,
    model_type="linear",
    show=False,
    save="/tmp/fpx-release-smoke/nested/forest_plot.png",
)
print("saved")
PY
```

Expected result:
- File exists at `/tmp/fpx-release-smoke/nested/forest_plot.png`
- No save-path errors when parent directories do not already exist.
