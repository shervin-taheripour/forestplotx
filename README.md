# forestplotx

`forestplotx` creates publication-style forest plots that combine a clean text table with a forest panel, with deterministic formatting for common regression outputs.

## Features

- Publication-style table + forest composition
- Supports `binom`, `gamma`, `linear`, and `ordinal` model outputs
- One or two outcomes per plot
- Deterministic internal layout presets for stable output
- Readable log-axis handling in both `decimal` and `power10` styles
- Optional footer text for manuscript-style notes
- Visible column-header and x-axis label overrides

**Note:** For `logit`/`log` links, `exponentiate=None` applies model-based exponentiation with a warning; set `exponentiate=False` if your data is already on effect scale.
Displayed CI values in the table use bracket notation: `[low,high]`.

## API Reference

### `forest_plot()`

```python
fig, axes = fpx.forest_plot(
    df,                              # DataFrame with model output
    outcomes=None,                   # list[str], max 2; auto-detected if None
    save=None,                       # File path to save (e.g. "plot.png")
    model_type="binom",              # "binom" | "gamma" | "linear" | "ordinal"
    link=None,                       # Override default link function
    exponentiate=None,               # None=auto by link, True=force, False=disable
    table_only=False,                # Render table without forest panel
    legend_labels=None,              # list[str] override for legend entries
    point_colors=None,               # list[str], up to 2 hex codes for outcome markers
    column_labels=None,              # dict override for table column labels
    x_label_override=None,           # Override forest x-axis label
    footer_text=None,                # Italic footer (wrapped/capped internally)
    tick_style="decimal",            # "decimal" or "power10"
    clip_outliers=False,             # Opt-in clipping of extreme CI-driven axis outliers
    clip_quantiles=(0.02, 0.98),     # Retained for API compatibility
    base_decimals=2,                 # Decimal places for effect / CI values
    show=True,                       # Call plt.show(); set False for programmatic use
    show_general_stats=True,         # Show n / N / Freq columns
    bold_override=None,              # Manual bold control per predictor/outcome
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
Within each layout case, deterministic pressure tiers are applied internally (`standard`, `expanded`, `max`) based on the final rendered string widths.
Predictor labels are truncated (with warning) when they exceed layout-specific caps:
1. `show_general_stats=True` + two outcomes: 21 chars
2. `show_general_stats=True` + one outcome: 24 chars
3. `show_general_stats=False` + two outcomes: 26 chars
4. `show_general_stats=False` + one outcome: 25 chars
When general stats are shown, large `n`/`N` values are compacted (e.g., `9.9k`) to preserve column readability.
Compaction activates only when counts reach `>= 1,000` and uses a shared unit across both `n` and `N` (`k`, `M`, `B`, `T`) for consistent within-row formatting.
Very large values beyond display range are capped as `>999T` with a warning.
Effect / CI display uses the same compact unit family (`k`, `M`, `B`, `T`) once values reach `>= 1,000`, followed by deterministic decimal trimming to keep tables readable.
Rows are fully grayed only when all displayed outcomes are missing; if at least one outcome is valid, only the missing outcome triplet (`effect`, `95% CI`, `p`) is blanked and gray-marked.

### Title Handling

`forest_plot()` intentionally does not include a `title` parameter in v1.
This is by design for publication workflows where figure titles/captions are managed in the manuscript rather than embedded inside the plot image.
If needed for slides or reports, add a title externally on the returned matplotlib figure object.

### Exponentiation Safety

- Use `exponentiate=None` (default) for model/link-based automatic handling.
- Use `exponentiate=False` if your input is already on effect scale (e.g., OR/Ratio, not log-coefficients).
- Use `exponentiate=True` only when input is definitely on log scale and needs transformation.
- Read warnings: they include auto-exponentiation context and column mapping (effect column + `CI_low`/`CI_high` combined into `95% CI`).

### Axis Behavior

- Log-axis limits are data-driven after optional clipping; they are not forced symmetric around the reference value.
- `clip_outliers=True` uses magnitude-based clipping centered on the median CI bounds, which works much better for small samples with one extreme interval.
- `tick_style="decimal"` uses readable decimal ticks:
  - dense near-reference ticks for moderate spans
  - `1-2-5` progression for wider spans
  - compact notation for very large tick labels when needed
- `tick_style="power10"` keeps readable power-of-ten labels for very wide ratio ranges.

### Label Overrides

Use `column_labels` to override visible table headers without changing the underlying model type:

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
