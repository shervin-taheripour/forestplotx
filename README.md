# forestplotx

Publication-ready forest plots for regression model outputs in Python.

`forestplotx` takes DataFrame output from logistic, linear, ordinal, or gamma regression models and produces a combined table + forest plot figure — ready for papers, reports, and presentations.

## Features

- **Multiple model types** — binomial (logistic), linear, gamma, and ordinal (cumulative logit)
- **Automatic effect-scale handling** — exponentiation, log-scale axes, and reference lines driven by link function
- **Flexible column detection** — accepts `OR`, `Ratio`, `Estimate`, `beta`, `Coef`, or `effect` as input
- **Dual-outcome layout** — side-by-side comparison of up to two outcomes
- **Category grouping** — optional row grouping with shaded bands
- **Static matplotlib output** — high-resolution, saveable figures

## Installation

```bash
pip install forestplotx
```

Requires Python ≥ 3.10. Dependencies: `matplotlib`, `numpy`, `pandas`.

## Quick Start

```python
import pandas as pd
import forestplotx as fpx

# Example: logistic regression output
df = pd.DataFrame({
    "predictor": ["Age", "Sex", "BMI", "Smoking"],
    "outcome":   ["Mortality"] * 4,
    "OR":        [-0.12, 0.85, 0.30, 0.55],   # log-odds (pre-exponentiation)
    "CI_low":    [-0.35, 0.42, 0.05, 0.20],
    "CI_high":   [ 0.11, 1.28, 0.55, 0.90],
    "p_value":   [0.300, 0.001, 0.020, 0.003],
})

fig, axes = fpx.forest_plot(df, model_type="binom")
```

## Supported Model Types

| `model_type` | Link | Effect label | X-axis | Reference line |
|:-------------|:-----|:-------------|:-------|:---------------|
| `"binom"` | logit | OR | Odds Ratio (log scale) | 1.0 |
| `"gamma"` | log | Ratio | Ratio (log scale) | 1.0 |
| `"linear"` | identity | Coef | Effect Size | 0.0 |
| `"ordinal"` | logit | OR | Odds Ratio (log scale) | 1.0 |

The `link` parameter can override the default — for example, `model_type="binom", link="identity"` will skip exponentiation and plot on a linear scale.

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
| `p_value` | P-value (bold formatting applied when < 0.05) |
| `category` | Group predictors under category headers |
| `n` | Event count |
| `N` | Total count |

**Note:** For `logit` and `log` link functions, provide effects on the log scale — `forestplotx` exponentiates automatically.

## API Reference

### `forest_plot()`

```python
fig, axes = fpx.forest_plot(
    df,                              # DataFrame with model output
    model_type="binom",              # "binom" | "gamma" | "linear" | "ordinal"
    link=None,                       # Override default link function
    outcomes=None,                   # list[str], max 2; auto-detected if None
    palette=None,                    # dict mapping category → hex color
    legend_labels=None,              # list[str] override for legend entries
    footer_text=None,                # Italic footer below the plot
    show_general_stats=True,         # Show n / N / Freq columns
    bold_override=None,              # Manual bold control per predictor/outcome
    font_size=14,                    # Base font size
    block_spacing=6.0,               # Horizontal spacing between table blocks
    base_decimals=2,                 # Decimal places for effect / CI values
    table_only=False,                # Render table without forest panel
    save=None,                       # File path to save (e.g. "plot.png")
)
```

**Returns:** `(fig, axes)` — matplotlib Figure and axes tuple for further customization.

### `normalize_model_output()`

```python
clean_df, config = fpx.normalize_model_output(df, model_type="binom", link=None)
```

Standardizes column names, applies exponentiation, and returns a config dict with axis metadata. Useful if you want the normalized data without plotting.

## Examples

### Category grouping

```python
df["category"] = ["Demographics", "Demographics", "Clinical", "Clinical"]

fig, axes = fpx.forest_plot(
    df,
    model_type="binom",
    palette={"Demographics": "#E8F0FE", "Clinical": "#FEF3E8"},
)
```

### Dual outcomes

```python
# DataFrame with two outcomes per predictor
fig, axes = fpx.forest_plot(
    df_two_outcomes,
    model_type="binom",
    outcomes=["Mortality", "Readmission"],
    legend_labels=["30-day mortality", "90-day readmission"],
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

## Testing

The test suite lives in `tests/` and covers all internal modules with no image comparisons — structural and behavioral assertions only.

```bash
pytest
```

### Test files

| File | Module under test | Tests |
|:-----|:------------------|------:|
| `tests/test_normalization.py` | `_normalize.py` | 7 |
| `tests/test_layout.py` | `_layout.py` | 34 |
| `tests/test_axes_config.py` | `_axes_config.py` | 63 |

### Coverage summary

**`test_layout.py`** — `build_row_layout()`

- Flat layout (no `category` column): sequential y-positions, correct row count, all `is_cat=False`, `"Uncategorized"` labels, predictor order preserved, required columns present
- NaN predictor rows dropped; empty DataFrame raises `ValueError`
- Categorized layout: category header rows inserted, total = categories + predictors (parametrized), correct `is_cat` flags and per-predictor category labels, all-NaN category falls back to flat
- Dual-outcome DataFrames: `unique()` deduplication keeps one row per predictor regardless of outcome count

**`test_axes_config.py`** — `configure_forest_axis()` and helpers

- `_nice_linear_step`: 8 parametrized input→output pairs, zero, negative, tiny positive values
- `_decimals_from_ticks`: empty/single-tick → 2, step-inferred decimals (0/1/2), `max_decimals` cap
- Reference line: `axvline` placed at correct x for logit (1.0), log (1.0), identity (0.0); `#910C07` color; dashed style; threshold override
- X-scale: `"log"` for logit/log links, `"linear"` for identity; empty data and `thresholds=None` do not crash
- X-label: correct label per link (`"Odds Ratio"` / `"Ratio"` / `"Effect Size"`), threshold override, font size propagated
- Y-ticks cleared; y-limits applied from `thresholds["y_limits"]`
- Spine visibility: top/right/left hidden, bottom visible
- X-limits contain full data range for log and linear axes; negative reference raises `ValueError`; span=0 edge case handled
- End-to-end parametrized across all four model types: binom, gamma, linear, ordinal
- `show_general_stats=True/False` both produce consistent output (documents no-op behaviour on axis)
- Tick count heuristic: `num_ticks` in {3, 5, 7} for log and linear axes
- `tick_style="power10"` does not crash; single vs dual outcome `lo_all`/`hi_all` arrays both handled

## Scope

`forestplotx` v1.0 is intentionally focused. It produces static, publication-quality forest plots for common regression model types.

**Not included:** interactive plots, Cox/Poisson models, theming engine, or GUI.

## License

MIT
