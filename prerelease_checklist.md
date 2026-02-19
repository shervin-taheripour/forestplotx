# forestplotx v1.0.0 — Pre-Release Checklist

## Code completeness

- [ ] All visual test cases pass (run `tests/run_visual_tests.py`, review PNGs)
- [ ] All pytest tests pass (`pytest` from repo root)
- [ ] `block_spacing` either removed from function signature or prefixed as internal (`_block_spacing`) — currently still in signature but undocumented
- [ ] Figure height floor added: `fig_height = max(0.3 * n + 1.5, 5.0)`
- [ ] No stale files in repo (`plot_utils.py`, `refactoring_plan_v2.md`, `project_scope.md` — remove or move to `docs/`)

## Package metadata (`pyproject.toml`)

- [ ] `name` = `"forestplotx"`
- [ ] `version` = `"1.0.0"` (matches `__init__.py`)
- [ ] `description` — one-line summary present
- [ ] `readme` = `"README.md"`
- [ ] `license` = MIT
- [ ] `requires-python` = `">=3.10"`
- [ ] `dependencies` list complete (`matplotlib>=3.7`, `numpy>=1.24`, `pandas>=2.0`)
- [ ] `[project.optional-dependencies]` includes `dev = ["pytest>=7.0"]`
- [ ] `authors` field set with your name and email
- [ ] `classifiers` added (see below)
- [ ] `[project.urls]` includes `Homepage` and `Repository` pointing to GitHub

### Recommended classifiers

```toml
classifiers = [
    "Development Status :: 5 - Production/Stable",
    "Intended Audience :: Science/Research",
    "License :: OSI Approved :: MIT License",
    "Programming Language :: Python :: 3",
    "Programming Language :: Python :: 3.10",
    "Programming Language :: Python :: 3.11",
    "Programming Language :: Python :: 3.12",
    "Programming Language :: Python :: 3.13",
    "Topic :: Scientific/Engineering :: Visualization",
]
```

## Package structure

- [ ] `src/forestplotx/__init__.py` exports `forest_plot`, `normalize_model_output`, `__version__`
- [ ] `__all__` is defined
- [ ] No accidental imports of test or dev code
- [ ] `py.typed` marker file present in `src/forestplotx/` (PEP 561, optional but good practice)

## Files to include in distribution

- [ ] `README.md`
- [ ] `LICENSE`
- [ ] `pyproject.toml`
- [ ] `src/forestplotx/` (all `.py` files)
- [ ] Nothing else ships (no `tests/`, `data/`, `notebooks/`, `examples/`, `*.egg-info/`)

### Verify with:

```bash
python -m build
tar -tzf dist/forestplotx-1.0.0.tar.gz  # inspect sdist contents
unzip -l dist/forestplotx-1.0.0-py3-none-any.whl  # inspect wheel contents
```

## Files to exclude from distribution

- [ ] Add or verify `MANIFEST.in` if needed (setuptools with `src` layout usually handles this, but verify)
- [ ] `.gitignore` includes: `*.egg-info/`, `dist/`, `build/`, `__pycache__/`, `.pytest_cache/`, `tests/test_results/`
- [ ] `egg-info/` directory not committed to git

## README quality

- [ ] API signature matches actual code
- [ ] All documented parameters exist in code
- [ ] No undocumented public parameters (currently `block_spacing` is in code but not README — resolve)
- [ ] Quick Start example is copy-pasteable and works
- [ ] Test counts in README match actual test counts
- [ ] No broken links

## Testing

- [ ] `pytest` passes with zero failures
- [ ] Visual test matrix passes (no FAIL rows in CSV log)
- [ ] Tests don't require display server (`matplotlib.use("Agg")` in test files)
- [ ] Tests don't leave behind files (temp files cleaned up)

## Build and install verification

```bash
# Clean build
rm -rf dist/ build/ src/*.egg-info/
python -m build

# Install from wheel in fresh venv
python -m venv /tmp/fpx-test
source /tmp/fpx-test/bin/activate
pip install dist/forestplotx-1.0.0-py3-none-any.whl

# Verify
python -c "import forestplotx; print(forestplotx.__version__)"
python -c "from forestplotx import forest_plot, normalize_model_output"

# Run quick smoke test
python -c "
import pandas as pd
from forestplotx import forest_plot
df = pd.DataFrame({
    'predictor': ['A', 'B'],
    'outcome': ['Y', 'Y'],
    'Estimate': [0.5, -0.3],
    'CI_low': [0.1, -0.8],
    'CI_high': [0.9, 0.2],
    'p_value': [0.01, 0.2],
})
fig, axes = forest_plot(df, model_type='binom', show=False)
print('OK')
"
deactivate
```

## PyPI upload

```bash
# Test PyPI first
twine upload --repository testpypi dist/*
pip install --index-url https://test.pypi.org/simple/ forestplotx

# Production
twine upload dist/*
```

- [ ] `twine` installed (`pip install twine build`)
- [ ] PyPI account created, API token configured
- [ ] Package name `forestplotx` is available on PyPI
- [ ] Test upload to TestPyPI succeeds
- [ ] Install from TestPyPI succeeds
- [ ] Production upload to PyPI

## Git hygiene

- [ ] All changes committed
- [ ] Tag: `git tag -a v1.0.0 -m "forestplotx v1.0.0"`
- [ ] Push tag: `git push origin v1.0.0`
- [ ] GitHub release created from tag (optional but recommended)
- [ ] GitHub repo description and topics set (e.g., "forest-plot", "visualization", "statistics", "matplotlib")

## Post-release

- [ ] Verify `pip install forestplotx` works from PyPI
- [ ] Verify README renders correctly on PyPI project page
- [ ] Add GitHub repo URL to PyPI project page (via `[project.urls]` in pyproject.toml)
