# forestplotx — Updated Refactoring Plan (v2)

## Current State After Initial Changes

You've already:
- Renamed `detailed_forest_plot` → `forest_plot` ✓
- Extracted `_normalize.py` as a clean standalone module ✓
- Renamed `exp` → `gamma` in `DEFAULT_LINK` ✓
- Cleaned up unused parameters (`binom_threshold`, `exp_threshold`, `linear_threshold`, `max_decimals`, `num_ticks`) ✓
- Standardized param naming (`FONT_SIZE` → `font_size`, `BLOCK_SPACING` → `block_spacing`) ✓
- Import now uses `from ._normalize import _normalize_model_output` (package-relative) ✓

## What's Left

### 1. `_normalize.py` — Done (minor suggestion)

The module is clean. One small note: the `_EFFECT_CANDIDATES` list is defined inside the function body. Since it's a constant, it could live at module level for clarity and testability — but this is cosmetic, not blocking.

### 2. `_layout.py` — Extract row assembly + positioning

**What to extract from `plot_utils.py`:**

```
Lines 29–37   → BLOCK_X computation
Lines 39–40   → GENERAL_OFFSETS, OUTCOME_OFFSETS constants
Lines 69–85   → Category/predictor row assembly logic
Lines 87–89   → Row count validation
Lines 190–193 → Text axis limits computation
```

**Proposed public surface of `_layout.py`:**

```python
def build_table_rows(df, categories=True):
    """Build ordered row list with category headers.
    
    Returns:
        table_rows: list[dict] with keys 'predictor', 'is_cat'
        row_cats: list[str] category assignment per row
    """

def compute_block_positions(block_spacing, show_general_stats, has_second_outcome):
    """Compute x-positions for text table columns.
    
    Returns:
        dict with keys: predictor, general, outcome1, outcome2
    """

# Module-level constants
GENERAL_OFFSETS = [0, 1.3, 2.6]
OUTCOME_OFFSETS = [0, 2.0, 4.1]
```

### 3. `_axes_config.py` — Extract axis formatting

**What to extract from `plot_utils.py`:**

```
Lines 235–276 → All axis scaling, tick formatting, xlim logic
```

This is the messiest part of the current code — nested `if model_type` / `if use_log` branches with inline `MaxNLocator` import.

**Proposed public surface of `_axes_config.py`:**

```python
def configure_forest_axis(ax, model_type, plot_config, lo_all, hi_all):
    """Configure x-axis scale, ticks, limits, and formatting.
    
    Handles:
    - Log vs linear scale selection
    - Model-type-aware tick placement (binom/gamma nice ticks, linear MaxNLocator)
    - Reference line value clamping
    - Minor tick suppression on log axes
    """
```

Single function, single responsibility. All the `if model_type in ("binom", "gamma")` branching lives here.

### 4. `plot.py` — Orchestration (renamed from `plot_utils.py`)

After extraction, `plot.py` retains:
- `forest_plot()` — the public entry point
- `format_effect_ci_p()` — promoted from closure to module-level helper (prefixed `_`)
- Figure creation (subplot setup, gridspec)
- Text table rendering loop (headers + data rows)
- Forest panel rendering loop (errorbars, markers, legend)
- Footer, frame, save/show logic

**Remaining concern:** The text rendering loop (lines 109–188) and forest rendering loop (lines 195–279) are still ~80 and ~85 lines respectively. For v1.0 this is acceptable — they're linear, sequential rendering code, not branching logic. Further splitting into `_render_table.py` / `_render_forest.py` is a v1.1 consideration if the file grows.

### 5. `__init__.py` — Public API

```python
from .plot import forest_plot
from ._normalize import _normalize_model_output as normalize_model_output

__all__ = ["forest_plot", "normalize_model_output"]
__version__ = "1.0.0"
```

---

## Outstanding Issues Still in `plot_utils.py`

| Issue | Location | Severity | Suggested Fix |
|-------|----------|----------|---------------|
| `plt.show()` called unconditionally | L292 | Medium | Add `show: bool = True` param; only call if True |
| No `fig` returned | end of func | Medium | `return fig, (ax_text, ax_forest)` |
| `format_effect_ci_p` is a closure | L60–67 | Low | Promote to `_format_effect_ci_p()` at module level, pass `base_decimals` |
| Unicode dash `–` appears garbled | L65 | Low | Confirm encoding; use `"\u2013"` explicitly |
| `save` then `show` then `close` ordering | L289–295 | Low | Save should happen after show or use `fig.savefig` before `plt.show()` (current order is fine for non-interactive, but `plt.close` after `plt.show` is a no-op in interactive) |
| `table_only` creates different axes shape | L91–100 | Low | Keep for now; document as internal |
| Inline `from matplotlib.ticker import MaxNLocator` | L241 | Low | Moves to `_axes_config.py` top-level import |

---

## Agreed Module Structure

```
forestplotx/
├── __init__.py        # Public API: forest_plot, normalize_model_output
├── plot.py            # Main orchestration + rendering
├── _normalize.py      # Data normalization + link logic       ← DONE
├── _layout.py         # Row assembly + grouping + positioning
├── _axes_config.py    # Axis scaling + ticks + reference line
```

## Refactoring Order

1. **`_layout.py`** — Extract `build_table_rows()` + `compute_block_positions()` + constants
2. **`_axes_config.py`** — Extract `configure_forest_axis()`
3. **`plot.py`** — Rename from `plot_utils.py`, wire imports, add `show` param + return `fig`
4. **`__init__.py`** — Wire public API
5. **Tests** — Smoke tests for each module
6. **`pyproject.toml`** + **`README.md`** + **`LICENSE`**
