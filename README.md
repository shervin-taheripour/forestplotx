# forestplotx

Publication-ready forest plots for regression model outputs in Python.

`forestplotx` takes DataFrame output from logistic, linear, ordinal, or gamma regression models and produces a combined table + forest plot figure — ready for papers, reports, and presentations.

![Two Outcomes with General Stats](examples/layout_case1_general_true_two_outcomes.png)

## Features

- **Multiple model types** — binomial (logistic), linear, gamma, and ordinal (cumulative logit)
- **Automatic effect-scale handling** — exponentiation, log-scale axes, and reference lines driven by link function
- **Flexible column detection** — accepts `OR`, `Ratio`, `Estimate`, `beta`, `Coef`, or `effect` as input
- **Dual-outcome layout** — side-by-side comparison of up to two outcomes
- **Category grouping** — optional row grouping with bold category headers
- **Deterministic layout presets** — fixed internal geometry for 4 core display cases
- **Adaptive small-table sizing** — compact height heuristic for low row counts
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

Requires Python ≥ 3.10. Dependencies: `matplotlib>=3.7`, `numpy>=1.24`, `pandas>=2.0`.

### Development install (reproducible environment)

```bash
pip install -r requirements.txt   # pin exact versions used during development
pip install -e ".[dev]"           # install forestplotx itself in editable mode
```

`requirements.txt` pins the full transitive closure of runtime + test dependencies. `pyproject.toml` declares the minimum-version constraints used when installing normally.

## Quick Start

```python
import pandas as pd
import forestplotx as fpx

# Example: logistic regression output
df = pd.DataFrame({
    "predictor": ["Age", "Sex", "BMI", "Smoking"],
    "outcome":   ["Mortality"] * 4,
    "Estimate":  [-0.12, 0.85, 0.30, 0.55],   # log-odds (pre-exponentiation)
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

**Note:** For `logit`/`log` links, `exponentiate=None` applies model-based exponentiation with a warning; set `exponentiate=False` if your data is already on effect scale.
Displayed CI values in the table use bracket notation: `[low,high]`.

## API Reference

### `forest_plot()`

```python
fig, axes = fpx.forest_plot(
    df,                              # DataFrame with model output
    model_type="binom",              # "binom" | "gamma" | "linear" | "ordinal"
    link=None,                       # Override default link function
    exponentiate=None,               # None=auto by link, True=force, False=disable
    outcomes=None,                   # list[str], max 2; auto-detected if None
    legend_labels=None,              # list[str] override for legend entries
    footer_text=None,                # Italic footer (wrapped/capped internally)
    show_general_stats=True,         # Show n / N / Freq columns
    bold_override=None,              # Manual bold control per predictor/outcome
    base_decimals=2,                 # Decimal places for effect / CI values
    tick_style="decimal",            # "decimal" or "power10" (readable log10 exponents)
    clip_outliers=False,             # Clip axis limits by quantiles (opt-in)
    clip_quantiles=(0.02, 0.98),     # Low/high quantiles used when clipping
    point_colors=None,               # list[str], up to 2 hex codes for outcome markers
    table_only=False,                # Render table without forest panel
    show=True,                       # Call plt.show(); set False for programmatic use
    save=None,                       # File path to save (e.g. "plot.png")
)
```

**Returns:** `(fig, axes)` — matplotlib Figure and axes tuple. When `show=False`, the figure is returned without displaying, allowing further customization before calling `plt.show()` manually.
When `exponentiate=None`, auto exponentiation for log/logit links emits a warning so users can verify input scale.

### Layout Behavior (v1)

`forest_plot()` uses fixed internal layout presets (including internal font size) for:

1. `show_general_stats=True` + two outcomes
2. `show_general_stats=True` + one outcome
3. `show_general_stats=False` + two outcomes
4. `show_general_stats=False` + one outcome

This is intentional to keep output stable and publication-ready across common use cases.
`base_decimals` is capped at 3 internally to prevent table collisions in dense layouts.
For small row counts, figure height uses a tighter internal heuristic to reduce excessive whitespace.
Long footer text is wrapped and capped to 3 lines with ellipsis for overflow protection.

### Exponentiation Safety

- Use `exponentiate=None` (default) for model/link-based automatic handling.
- Use `exponentiate=False` if your input is already on effect scale (e.g., OR/Ratio, not log-coefficients).
- Use `exponentiate=True` only when input is definitely on log scale and needs transformation.
- Read warnings: they include auto-exponentiation context and column mapping (effect column + `CI_low`/`CI_high` combined into `95% CI`).

### `normalize_model_output()`

```python
clean_df, config = fpx.normalize_model_output(
    df, model_type="binom", link=None, exponentiate=None
)
```

Standardizes columns, applies exponentiation policy, and returns axis metadata.
`config` includes `exponentiated` and `renamed_columns` for transparency.

## Examples

### Category grouping

```python
df["category"] = ["Demographics", "Demographics", "Clinical", "Clinical"]

fig, axes = fpx.forest_plot(df, model_type="binom")
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
# Further customization...
fig.suptitle("My Forest Plot", fontsize=16)
fig.savefig("custom_plot.pdf", dpi=300)
```

In notebooks, `show=False` prevents internal `plt.show()`, but Jupyter may still auto-render
the returned figure object. Use `plt.close(fig)` to suppress display.

## Testing

The test suite lives in `tests/` and covers all internal modules with no image comparisons — structural and behavioral assertions only.

Install dev dependencies first (see [Installation](#installation)), then:

```bash
pytest
```

### Test files

| File | Module under test | Tests |
|:-----|:------------------|------:|
| `tests/test_normalization.py` | `_normalize.py` | 8 |
| `tests/test_layout.py` | `_layout.py` | 33 |
| `tests/test_axes_config.py` | `_axes_config.py` | 64 |
| `tests/test_plot_smoke.py` | `plot.py` | 2 |

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
- `tick_style="power10"` uses readable rounded log10 exponents; single vs dual outcome `lo_all`/`hi_all` arrays both handled

## Scope

`forestplotx` v1.0 is intentionally focused. It produces static, publication-quality forest plots for common regression model types.

**Not included:** interactive plots, Cox/Poisson models, theming engine, or GUI.

## License

MIT
