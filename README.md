![CI](https://github.com/shervin-taheripour/forestplotx/actions/workflows/ci.yml/badge.svg)
[![PyPI version](https://img.shields.io/pypi/v/forestplotx.svg)](https://pypi.org/project/forestplotx/)
[![Python](https://img.shields.io/pypi/pyversions/forestplotx.svg)](https://pypi.org/project/forestplotx/)
[![License](https://img.shields.io/pypi/l/forestplotx.svg)](LICENSE)
# forestplotx

Publication-ready forest plots for regression model outputs in Python.

`forestplotx` takes DataFrame output from logistic, linear, ordinal, or gamma regression models and produces a combined table + forest plot figure ready for papers, reports, and presentations.

![Comparison After](https://raw.githubusercontent.com/shervin-taheripour/forestplotx/v1.0.2/examples/comparison_after.png)

## Features

- **Multiple model types** — binomial (logistic), linear, gamma, and ordinal (cumulative logit)
- **Automatic effect-scale handling** — exponentiation, log-scale axes, and reference lines driven by link function
- **Flexible column detection** — accepts `OR`, `Ratio`, `Estimate`, `beta`, `Coef`, or `effect` as input
- **Dual-outcome layout** — side-by-side comparison of up to two outcomes
- **Category grouping** — optional row grouping with bold category headers
- **Deterministic layout presets** — fixed internal geometry for 4 core display cases
- **Readable log-axis handling** — `decimal` and `power10` tick styles for ratio plots
- **Label overrides** — visible table-column and x-axis label overrides without changing normalization
- **Static matplotlib output** — high-resolution, saveable figures

## Layout Examples

- `examples/layout_case1_general_true_two_outcomes.png`
- `examples/layout_case2_general_true_one_outcome.png`
- `examples/layout_case3_general_false_two_outcomes.png`
- `examples/layout_case4_general_false_one_outcome.png`

## Installation

```bash
pip install forestplotx
```

Requires Python >= 3.10. Dependencies: `matplotlib>=3.7`, `numpy>=1.24`, `pandas>=2.0`.

### Development install (reproducible environment)

```bash
pip install -r requirements.txt
pip install -e ".[dev]"
```

## Quick Start

```python
import pandas as pd
import forestplotx as fpx

df = pd.DataFrame({
    "predictor": ["Age", "Sex", "BMI", "Smoking"],
    "outcome":   ["Mortality"] * 4,
    "Estimate":  [-0.12, 0.85, 0.30, 0.55],
    "CI_low":    [-0.35, 0.42, 0.05, 0.20],
    "CI_high":   [ 0.11, 1.28, 0.55, 0.90],
    "p_value":   [0.300, 0.001, 0.020, 0.003],
})

fig, axes = fpx.forest_plot(df, model_type="binom")
```

## Supported Model Types

| `model_type` | Example models | Link | Effect label (table) | X-axis label | Reference line |
|:-------------|:---------------|:-----|:---------------------|:-------------|:---------------|
| `"binom"` | Logistic regression | logit | OR | Odds Ratio (log scale) | 1.0 |
| `"gamma"` | Gamma GLM / GLMM | log | Ratio | Ratio (log scale) | 1.0 |
| `"linear"` | Linear regression | identity | β | β (coefficient) | 0.0 |
| `"ordinal"` | Ordinal regression | logit | OR | Odds Ratio (log scale) | 1.0 |

The `link` parameter can override the default; for example, `model_type="binom", link="identity"` skips exponentiation and plots on a linear scale.

## Input DataFrame

### Required columns

| Column | Description |
|:-------|:------------|
| `predictor` | Row labels (predictor names) |
| `outcome` | Outcome name (used for column headers and filtering) |
| Effect column | One of: `OR`, `Ratio`, `Estimate`, `beta`, `Coef`, `effect` |
| `CI_low` / `ci_low` | Lower bound of 95% CI |
| `CI_high` / `ci_high` | Upper bound of 95% CI |

### Optional columns

| Column | Description |
|:-------|:------------|
| `p_value` | P-value (bold formatting when < 0.05) |
| `category` | Group predictors under category headers |
| `n` | Event count |
| `N` | Total count |

**Note:** For `logit`/`log` links, `exponentiate=None` applies model-based exponentiation with a warning; set `exponentiate=False` if your data is already on effect scale.
Displayed CI values in the table use bracket notation: `[low,high]`.

## API Reference

### `forest_plot()`

```python
fig, axes = fpx.forest_plot(
    df,
    outcomes=None,
    save=None,
    model_type="binom",
    link=None,
    exponentiate=None,
    table_only=False,
    legend_labels=None,
    point_colors=None,
    column_labels=None,
    x_label_override=None,
    footer_text=None,
    tick_style="decimal",
    clip_outliers=False,
    clip_quantiles=(0.02, 0.98),
    base_decimals=2,
    show=True,
    show_general_stats=True,
    bold_override=None,
)
```

**Returns:** `(fig, axes)` — matplotlib Figure and axes tuple.

### Layout Behavior (v1.1)

`forest_plot()` uses fixed internal layout presets for:

1. `show_general_stats=True` + two outcomes
2. `show_general_stats=True` + one outcome
3. `show_general_stats=False` + two outcomes
4. `show_general_stats=False` + one outcome

Within each case, deterministic internal pressure tiers (`standard`, `expanded`, `max`) are applied from the final rendered strings.

Other layout guardrails:
- `base_decimals` is capped at 3 internally to protect dense layouts
- small row counts use a tighter figure-height heuristic
- long predictor labels are truncated with a warning based on layout-specific caps
- when general stats are shown, large `n`/`N` values compact only from `>= 1,000`
- compact units use a shared `k/M/B/T` scale within each row
- effect / CI values use the same compact unit family from `>= 1,000`
- missing outcome triplets are blanked; full-row gray is used only when all displayed outcomes are missing

### Title Handling

`forest_plot()` intentionally does not include a `title` parameter in v1.x.
Use the returned `fig` if you need a title for slides or reports.

### Exponentiation Safety

- Use `exponentiate=None` (default) for model/link-based automatic handling.
- Use `exponentiate=False` if your input is already on effect scale.
- Use `exponentiate=True` only when input is definitely on log scale.

### Axis Behavior

- Log-axis limits are data-driven after optional clipping; they are not forced symmetric around the reference.
- `clip_outliers=True` uses magnitude-based clipping on log axes, centered on the median CI bounds.
- `clip_quantiles` is retained for API compatibility and still applies to linear-axis clipping.
- `tick_style="decimal"` uses readable decimal ticks:
  - denser near-reference ticks for moderate spans
  - `1-2-5` progression for wider spans
  - compact labels for very large values
- `tick_style="power10"` uses readable power-of-ten labels for very wide ratio ranges.

### Label Overrides

```python
fig, axes = fpx.forest_plot(
    df,
    model_type="gamma",
    exponentiate=False,
    column_labels={
        "effect": "IRR",
        "ci": "95% CI",
        "p": "P",
        "n": "Cases",
        "N": "Total",
        "Freq": "Share",
    },
    x_label_override="IRR",
)
```

Supported `column_labels` keys:
- `effect`
- `ci`
- `p`
- `n`
- `N`
- `Freq`

### `normalize_model_output()`

```python
clean_df, config = fpx.normalize_model_output(
    df, model_type="binom", link=None, exponentiate=None
)
```

Standardizes columns, applies exponentiation policy, and returns axis metadata.

## Examples

### Category grouping

```python
df["category"] = ["Demographics", "Demographics", "Clinical", "Clinical"]
fig, axes = fpx.forest_plot(df, model_type="binom")
```

### Dual outcomes

```python
fig, axes = fpx.forest_plot(
    df_two_outcomes,
    model_type="binom",
    outcomes=["Mortality", "Readmission"],
    legend_labels=["30-day mortality", "90-day readmission"],
)
```

### Custom marker colors

```python
fig, axes = fpx.forest_plot(
    df_two_outcomes,
    model_type="binom",
    outcomes=["Mortality", "Readmission"],
    point_colors=["#2C5F8A", "#D4763A"],
)
```

### Linear model

```python
fig, axes = fpx.forest_plot(df_linear, model_type="linear")
```

### Save to file

```python
fig, axes = fpx.forest_plot(df, model_type="binom", save="forest_plot.png")
```

### Programmatic use (no display)

```python
fig, axes = fpx.forest_plot(df, model_type="binom", show=False)
fig.suptitle("My Forest Plot", fontsize=16)
fig.savefig("custom_plot.pdf", dpi=300)
```

## Testing

```bash
pytest
python tests/run_visual_tests.py
```

### Test files

| File | Module under test |
|:-----|:------------------|
| `tests/test_normalization.py` | `_normalize.py` |
| `tests/test_layout.py` | `_layout.py` |
| `tests/test_axes_config.py` | `_axes_config.py` |
| `tests/test_plot_smoke.py` | `plot.py` |

## Scope

`forestplotx` v1.x is focused on static, publication-quality forest plots for common regression model types.

**Not included:** interactive plots, Cox/Poisson models, theming engine, or GUI.

## Versioning

`forestplotx` follows semantic versioning (SemVer).

- `MAJOR` – breaking API changes
- `MINOR` – backward-compatible feature additions
- `PATCH` – bug fixes and internal improvements

Current version: **1.1.0**

## License

MIT
