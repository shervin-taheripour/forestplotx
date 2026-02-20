# forestplotx — Manual Visual Test Plan

## Dataset Summary

| Dataset | Effect col | Scale | Outcomes | Predictors | Categories | Has n/N |
|:--------|:-----------|:------|:---------|:-----------|:-----------|:--------|
| binom   | `OR`       | **Already exponentiated** (0.37–2.74) | 3 | 18 | 4 | No |
| gamma   | `Estimate` | Log-scale (−0.07 to 0.21) | 2 | 16 | 4 | No |
| linear  | `Estimate` | Raw coefficients (−2.96 to 3.16) | 2 | 10 | 4 | No |
| ordinal | `Estimate` | Log-scale (−1.94 to 1.44) | 2 | 10 | 4 | No |

**Critical note:** The binom dataset uses `OR` column with pre-exponentiated values.
Default `model_type="binom"` will attempt to exponentiate again → **double exponentiation**.
This dataset *requires* `exponentiate=False` for correct output.

---

## Test Matrix

### A. Core model types — baseline correctness (4 tests)

Each test: verify axis scale, reference line position, effect label, x-axis label, and that
point estimates + CIs look plausible (no absurd ranges, no clipping, no overlaps).

| ID | Dataset | Call | Verify |
|:---|:--------|:-----|:-------|
| A1 | binom | `forest_plot(df, model_type="binom", exponentiate=False, show=False)` | Log scale, ref=1.0, label="Odds Ratio", ORs centered ~1.0, CI bars reasonable |
| A2 | gamma | `forest_plot(df, model_type="gamma", show=False)` | Log scale, ref=1.0, label="Ratio", exponentiated values near 1.0 |
| A3 | linear | `forest_plot(df, model_type="linear", show=False)` | Linear scale, ref=0.0, label="Effect Size", symmetric range |
| A4 | ordinal | `forest_plot(df, model_type="ordinal", show=False)` | Log scale, ref=1.0, label="Odds Ratio", thresholds stripped if any |

**What to look for:** reference line visible and centered, all points + CIs within axis limits,
tick labels readable and not overlapping, category headers bold, predictor text aligned.

---

### B. Exponentiation control (5 tests)

This is the highest-priority visual validation — the double-exponentiation bug.

| ID | Dataset | Call | Expected result |
|:---|:--------|:-----|:----------------|
| B1 | binom | `model_type="binom"` (no exponentiate flag) | ⚠️ **Double-exponentiated** — ORs become exp(2.73)≈15.3 etc. Axis should be wildly stretched. Confirms the bug is visible. |
| B2 | binom | `model_type="binom", exponentiate=False` | ✅ Correct — ORs plotted as-is on log scale around 1.0 |
| B3 | binom | `model_type="binom", exponentiate=True` | Same as B1 — explicit True forces exponentiation |
| B4 | gamma | `model_type="gamma", exponentiate=False` | Raw log-scale estimates plotted on log axis — values near 0 on a log scale will look wrong (values < 0 can't be plotted on log). Confirms user gets what they asked for. |
| B5 | gamma | `model_type="gamma", exponentiate=None` (default) | ✅ Correct — auto-exponentiation, warning emitted |

**What to look for:** B1 vs B2 should be dramatically different. B2 should look like a
sensible clinical forest plot. B1 should look absurd (huge CIs, stretched axis).
Verify warning is emitted for B5 (check stderr/console).

---

### C. Outcome handling (4 tests)

| ID | Dataset | Call | Verify |
|:---|:--------|:-----|:-------|
| C1 | binom | `outcomes=["outcome1"]` | Single outcome column, no second legend entry, no second marker series |
| C2 | binom | `outcomes=["outcome1", "outcome2"]` | Two outcome columns side by side, two marker series in forest panel, legend shows both |
| C3 | binom | `outcomes=["outcome1", "outcome2", "outcome3"]` | Silently truncated to first 2 — same as C2 |
| C4 | linear | `outcomes=None` | Auto-detected from data (should pick both outcomes) |

**What to look for:** C1 has one set of columns in the table. C2 has two. C3 should be
identical to C2. Legend entries match outcome labels. Dual-outcome markers vertically
offset (not overlapping).

---

### D. Axis scaling stress tests (6 tests)

These require **modified datasets** — see Section G below for dataset edits.

| ID | Dataset | Modification | Verify |
|:---|:--------|:-------------|:-------|
| D1 | binom | Inject one OR=15.0, CI=[8.0, 28.0] (extreme outlier) | Axis stretches to accommodate, other points compressed but still visible, no tick label overlap |
| D2 | binom | Set all ORs to 1.0 ± 0.01 (very tight range) | Axis zooms in tightly around 1.0, fine-grained ticks visible |
| D3 | linear | Inject one Estimate=50.0 (extreme outlier) | Linear axis stretches, reference line at 0.0 still visible |
| D4 | linear | Set all Estimates to 0.0 ± 0.001 (near-zero span) | Axis handles span≈0 gracefully, non-degenerate range |
| D5 | gamma | All estimates = 0.0 exactly (span=0 after exp → all 1.0) | Axis doesn't collapse, reference line visible |
| D6 | binom | Mix of tiny (0.01) and large (10.0) ORs with `exponentiate=False` | Log axis covers full range, ticks readable |

**What to look for:** no axis collapse, no infinite limits, no overlapping tick labels,
reference line always visible, extreme points don't clip.

---

### E. Visual formatting parameters (8 tests)

| ID | Base dataset | Parameter under test | Call snippet | Verify |
|:---|:-------------|:---------------------|:-------------|:-------|
| E1 | linear | Layout preset (True, True) | (default, two outcomes) | Matches case 1 gamma reference |
| E2 | linear | Layout preset (True, False) | `show_general_stats=True`, one outcome | Matches case 2 gamma reference |
| E3 | gamma | `base_decimals=0` | `base_decimals=0` | Effect/CI values show integers only (e.g., "1" not "1.02") |
| E4 | gamma | `base_decimals=4` | `base_decimals=4` | Effect/CI values show 4 decimal places |
| E5 | binom | `tick_style="power10"` | `tick_style="power10", exponentiate=False` | Axis labels show 10^x notation, no crash, readable |
| E6 | binom | `tick_style="decimal"` | `tick_style="decimal", exponentiate=False` | Axis labels show plain decimal numbers (default) |
| E7 | linear | `point_colors=["#FF0000", "#0000FF"]` | `point_colors=[...]` | Outcome 1 markers red, outcome 2 blue |
| E8 | linear | `point_colors=["#2C5F8A"]` (single color, 2 outcomes) | `point_colors=["#2C5F8A"]` | First outcome uses provided color, second falls back to default |

---

### F. Layout and structural parameters (10 tests)

| ID | Base dataset | Parameter under test | Call snippet | Verify |
|:---|:-------------|:---------------------|:-------------|:-------|
| F1 | linear | `table_only=True` | `table_only=True` | No forest panel at all, only text table, no crash |
| F2 | linear | `show_general_stats=False` | `show_general_stats=False` | n/N/Freq columns hidden, outcome columns shift left |
| F3 | linear | `show_general_stats=True` | (default) + modified dataset with n/N | n/N/Freq headers visible, values populated, headers centered over values |
| F4 | binom | Short footer | `footer_text="Adjusted for age and sex"` | Italic gray text below plot frame, fully visible, not clipped |
| F5 | binom | Long footer | `footer_text="Adjusted for age, sex, BMI, smoking status, diabetes, hypertension, and prior cardiovascular events. Sensitivity analyses excluded patients with missing lab values (n=47)."` | Text wraps or truncates gracefully, no collision with frame |
| F6 | binom | Footer with special chars | `footer_text="†p<0.05 after Bonferroni correction; ‡excluding n=12 protocol deviations"` | Special characters render correctly |
| F7 | linear | `block_spacing=3.0` | `block_spacing=3.0` | Columns tighter together — check for overlap |
| F8 | linear | `block_spacing=10.0` | `block_spacing=10.0` | Columns spread wide — check for truncation |
| F9 | binom | `bold_override={"predictor2": {"outcome1": True}}` | `bold_override={...}, exponentiate=False` | predictor2/outcome1 effect value forced bold regardless of p |
| F10 | binom | `clip_outliers=True` on wide data | D6 dataset + `clip_outliers=True, exponentiate=False` | Arrow markers (< >) at clipped CI edges, axis range tighter than D6 |

---

### G. Suggested dataset modifications

Create these as one-off modified DataFrames in your test notebook. Don't modify the
original CSVs.

```python
import pandas as pd
import numpy as np

df_binom = pd.read_csv("data_modeltype_binom.csv")
df_gamma = pd.read_csv("data_modeltype_gamma.csv")
df_linear = pd.read_csv("data_modeltype_linear.csv")

# --- G1: Binom outlier (for D1) ---
df_binom_outlier = df_binom.copy()
mask = (df_binom_outlier["predictor"] == "predictor1") & (df_binom_outlier["outcome"] == "outcome1")
df_binom_outlier.loc[mask, "OR"] = 15.0
df_binom_outlier.loc[mask, "CI_low"] = 8.0
df_binom_outlier.loc[mask, "CI_high"] = 28.0

# --- G2: Binom tight range (for D2) ---
df_binom_tight = df_binom.copy()
df_binom_tight["OR"] = 1.0 + np.random.uniform(-0.01, 0.01, len(df_binom_tight))
df_binom_tight["CI_low"] = df_binom_tight["OR"] - 0.005
df_binom_tight["CI_high"] = df_binom_tight["OR"] + 0.005

# --- G3: Linear outlier (for D3) ---
df_linear_outlier = df_linear.copy()
mask = (df_linear_outlier["predictor"] == "predictor1") & (df_linear_outlier["outcome"] == "outcome1")
df_linear_outlier.loc[mask, "Estimate"] = 50.0
df_linear_outlier.loc[mask, "CI_low"] = 45.0
df_linear_outlier.loc[mask, "CI_high"] = 55.0

# --- G4: Linear near-zero span (for D4) ---
df_linear_zero = df_linear.copy()
df_linear_zero["Estimate"] = np.random.uniform(-0.001, 0.001, len(df_linear_zero))
df_linear_zero["CI_low"] = df_linear_zero["Estimate"] - 0.0005
df_linear_zero["CI_high"] = df_linear_zero["Estimate"] + 0.0005

# --- G5: Gamma all-zero (for D5) ---
df_gamma_zero = df_gamma.copy()
df_gamma_zero["Estimate"] = 0.0
df_gamma_zero["CI_low"] = -0.01
df_gamma_zero["CI_high"] = 0.01

# --- G6: Binom wide log range (for D6 and F10) ---
df_binom_wide = df_binom.copy()
mask1 = (df_binom_wide["predictor"] == "predictor1") & (df_binom_wide["outcome"] == "outcome1")
mask2 = (df_binom_wide["predictor"] == "predictor2") & (df_binom_wide["outcome"] == "outcome1")
df_binom_wide.loc[mask1, ["OR", "CI_low", "CI_high"]] = [0.01, 0.005, 0.02]
df_binom_wide.loc[mask2, ["OR", "CI_low", "CI_high"]] = [10.0, 5.0, 20.0]

# --- G7: Add n/N columns for show_general_stats testing ---
df_linear_with_counts = df_linear.copy()
df_linear_with_counts["n"] = np.random.randint(10, 100, len(df_linear_with_counts))
df_linear_with_counts["N"] = df_linear_with_counts["n"] + np.random.randint(50, 200, len(df_linear_with_counts))

# --- G8: NaN effect values (null handling) ---
df_binom_nulls = df_binom.copy()
mask = (df_binom_nulls["predictor"] == "predictor3") & (df_binom_nulls["outcome"] == "outcome1")
df_binom_nulls.loc[mask, ["OR", "CI_low", "CI_high"]] = np.nan

# --- G9: Linear with only 3 predictors ---
keep_preds = df_linear["predictor"].dropna().unique()[:3]
df_linear_3pred = df_linear[df_linear["predictor"].isin(keep_preds)].copy()

# --- G10: Linear expanded to 30 predictors ---
rows = []
base_preds = list(df_linear["predictor"].dropna().unique())
base_outcomes = list(df_linear["outcome"].dropna().unique())
for i in range(30):
    base_pred = base_preds[i % len(base_preds)]
    for outcome in base_outcomes[:2]:
        src = df_linear[(df_linear["predictor"] == base_pred) & (df_linear["outcome"] == outcome)]
        src = src.iloc[0].to_dict()
        src["predictor"] = f"predictor_{i+1:02d}"
        src["outcome"] = outcome
        rows.append(src)
df_linear_30pred = pd.DataFrame(rows)

# --- G11: Linear where only 2 predictors have values (8 predictors NaN) ---
df_linear_sparse = df_linear.copy()
keep_preds = set(df_linear_sparse["predictor"].dropna().unique()[:2])
mask_nan = ~df_linear_sparse["predictor"].isin(keep_preds)
df_linear_sparse.loc[mask_nan, ["Estimate", "CI_low", "CI_high"]] = np.nan

# --- G12: Partial missing in single-outcome view with general stats ---
df_linear_with_counts_partial_single = df_linear_with_counts.copy()
mask = (df_linear_with_counts_partial_single["predictor"] == "predictor3") & (df_linear_with_counts_partial_single["outcome"] == "outcome1")
df_linear_with_counts_partial_single.loc[mask, "p_value"] = np.nan

# --- G13: Partial missing in two-outcome view with general stats ---
df_linear_with_counts_partial_dual = df_linear_with_counts.copy()
mask = (df_linear_with_counts_partial_dual["predictor"] == "predictor3") & (df_linear_with_counts_partial_dual["outcome"] == "outcome1")
df_linear_with_counts_partial_dual.loc[mask, "CI_high"] = np.nan

# --- G14: Long predictor label for truncation tests ---
df_linear_long_pred = df_linear.copy()
first_pred = df_linear_long_pred["predictor"].dropna().unique()[0]
df_linear_long_pred.loc[
    df_linear_long_pred["predictor"] == first_pred,
    "predictor",
] = "Pneumonoultramicroscopicsilicovolcanoconiosis"

# --- G15: Long predictor label with n/N columns ---
df_linear_with_counts_long_pred = df_linear_with_counts.copy()
first_pred = df_linear_with_counts_long_pred["predictor"].dropna().unique()[0]
df_linear_with_counts_long_pred.loc[
    df_linear_with_counts_long_pred["predictor"] == first_pred,
    "predictor",
] = "Pneumonoultramicroscopicsilicovolcanoconiosis"
```

---

### H. Save / show behavior (3 tests)

| ID | Call | Verify |
|:---|:-----|:-------|
| H1 | `show=True, save=None` | Plot displayed, no file created |
| H2 | `show=False, save="test_output.png"` | No display, file saved at path, file is valid PNG |
| H3 | `show=False, save=True` | No display, file saved as `forestplot.png` (default name) |

---

### I. Edge cases and error paths (4 tests)

| ID | Scenario | Call | Expected |
|:---|:---------|:-----|:---------|
| I1 | Empty DataFrame | `forest_plot(pd.DataFrame(...empty...))` | `ValueError` raised |
| I2 | Missing effect column | DataFrame with no recognized effect col | `ValueError` raised |
| I3 | Invalid model_type | `model_type="poisson"` | `ValueError` raised |
| I4 | Category column all NaN | `df["category"] = np.nan` | Falls back to flat layout, no crash |

These are already covered by pytest but worth a quick visual confirmation that the error
messages are clear and helpful.

---

### J. Additional stress cases (7 tests)

| ID | Base dataset | Scenario | Call snippet | Verify |
|:---|:-------------|:---------|:-------------|:-------|
| J1 | binom | Extremely long footer | `footer_text="<very long text>"` | Footer remains visible, no frame collision, no axis overlap |
| J2 | binom | Multiline footer (3 lines) | `footer_text="line1\nline2\nline3"` | All lines rendered, readable, and contained in figure |
| J3 | linear | Only 3 predictors | Use modified dataset with 3 predictors | Height floor/layout remains readable; rows not stretched awkwardly |
| J4 | linear | 30 predictors | Use expanded dataset with 30 predictors | Dynamic height scales correctly; no clipping/crowding in table/forest |
| J5 | linear | 2 predictors with values + 8 NaN predictors | Use modified dataset with NaN effect/CI for 8 predictors | NaN rows remain listed and appear grayed out in table + forest markers |
| J6 | linear + counts | Single-outcome partial missing with general stats | Use `df_linear_with_counts_partial_single`, `outcomes=["outcome1"]` | Row remains readable (not full gray), missing outcome triplet blanked, forest marker for missing row shown in gray |
| J7 | linear + counts | Two-outcome partial missing in one outcome with general stats | Use `df_linear_with_counts_partial_dual`, `outcomes=["outcome1","outcome2"]` | Predictor/general stats remain black when one outcome is valid; only missing outcome triplet blanked and gray marker shown |

These tests are implemented in `tests/run_visual_tests.py` as `J1`–`J5`.

### K. Predictor truncation stress tests (4 tests)

| ID | Base dataset | Scenario | Call snippet | Verify |
|:---|:-------------|:---------|:-------------|:-------|
| K1 | linear + counts | Long predictor label, layout (True, True) | `show_general_stats=True, outcomes=["outcome1","outcome2"]` | Long label is truncated with ellipsis; no collision with n/N/Freq or outcome1 block |
| K2 | linear + counts | Long predictor label, layout (True, False) | `show_general_stats=True, outcomes=["outcome1"]` | Truncation preserves one-outcome layout readability |
| K3 | linear | Long predictor label, layout (False, True) | `show_general_stats=False, outcomes=["outcome1","outcome2"]` | Truncation preserves two-outcome table readability |
| K4 | linear | Long predictor label, layout (False, False) | `show_general_stats=False, outcomes=["outcome1"]` | Truncation preserves compact single-outcome table readability |

---

## Execution Notes

- Run all tests with `show=False` and `plt.close(fig)` to avoid memory leaks
- For visual inspection, use `show=False` then `fig.savefig(f"test_{ID}.png", dpi=150)` to generate a reviewable gallery
- Name output files by test ID (e.g., `test_A1.png`) for systematic comparison
- After each run, check: (1) no warnings except expected exponentiation warning, (2) no clipped content, (3) visual consistency across model types

## Suggested test notebook structure

```python
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import forestplotx as fpx
import os

OUT = "visual_tests/"
os.makedirs(OUT, exist_ok=True)

def run_test(test_id, df, description, **kwargs):
    """Run one visual test, save output, log result."""
    print(f"[{test_id}] {description}")
    try:
        fig, axes = fpx.forest_plot(df, show=False, **kwargs)
        fig.savefig(f"{OUT}/{test_id}.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  ✓ saved {test_id}.png")
    except Exception as e:
        print(f"  ✗ {type(e).__name__}: {e}")

# Then call:
# run_test("A1", df_binom, "Binom baseline", model_type="binom", exponentiate=False)
# run_test("A2", df_gamma, "Gamma baseline", model_type="gamma")
# ... etc.
```

---

## Priority order for testing

1. **B1–B5** (exponentiation) — highest risk of silent data corruption
2. **D1–D6** (axis scaling) — most likely to produce visual artifacts
3. **A1–A4** (baseline correctness) — foundational sanity
4. **C1–C4** (outcomes) — layout correctness
5. **E1–E8** (formatting) — visual polish
6. **F1–F10** (layout params) — structural edge cases including footer and clip_outliers
7. **H1–H3** (save/show) — output pipeline
8. **I1–I4** (error paths) — defensive behavior
