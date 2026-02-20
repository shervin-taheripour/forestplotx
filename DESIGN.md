# forestplotx Design Notes (v1.0.1)

## Scope and Intent
`forestplotx` targets publication-ready forest plots for common regression outputs with minimal user-side formatting work.  
The design favors predictable defaults and explicit behavior over highly dynamic auto-layout.

## Technical Motivation
The primary pain point that motivated this project was the mismatch between what publication tables require and what default matplotlib composition provides for a combined table + forest plot.

- In dual-axis compositions, row alignment between the text table and forest markers is easy to drift and hard to keep stable across figure sizes and content variations.
- Standard table rendering (`matplotlib.table`) is not robust for dense scientific output (effect, CI, p-values, multi-outcome blocks) and becomes unreadable quickly.
- Wide-format output (especially two outcomes side-by-side) needs strict control over column spacing and precision formatting, which generic auto-layout does not reliably provide.

Design response in `forestplotx`:
- Build a "fake table" via explicit row assembly and deterministic text placement.
- Drive both table and forest from one shared y-position model (`_layout.py`) to enforce row-level alignment.
- Use fixed layout presets rather than free-form auto-spacing to preserve readability and reproducibility.

## Module Architecture
- `plot.py`: orchestration layer (table rendering, plotting, legend/footer, save/show behavior)
- `_normalize.py`: input normalization (column mapping, optional exponentiation, unified output schema)
- `_layout.py`: row assembly and y-position generation (no matplotlib state)
- `_axes_config.py`: axis scaling, ticks, labels, reference-line behavior, clipping logic

This split keeps data transformation, layout, and axis logic independently testable.

## Data Flow
1. User calls `forest_plot(df=..., model_type=...)`
2. `_normalize_model_output()` maps model-specific columns to canonical columns
3. `_layout.build_row_layout()` builds predictor/category rows and y positions
4. `plot.py` renders table + points/CIs
5. `_axes_config.configure_forest_axis()` sets x-scale, limits, ticks, and labels

## Core Design Decisions and Trade-offs
Each decision below records what was chosen, what was rejected, and why, so the engineering rationale remains auditable over time.

- **Fixed 4-case layout presets** (general stats on/off × one/two outcomes):
  - Chosen over generic auto-spacing because matplotlib text geometry is unstable across cases.
  - Trade-off: less user flexibility, much higher visual consistency.

- **Internalized layout controls**:
  - `font_size` and `block_spacing` removed from public API for v1 stability.
  - Trade-off: fewer knobs, fewer broken layouts.

- **Explicit exponentiation control with safe defaults**:
  - `exponentiate` supports override; when omitted, model/link defaults apply.
  - Warnings are emitted for potentially incorrect scale assumptions.
  - Trade-off: extra warnings, fewer silent mistakes.

- **Axis readability prioritized over strict data-bound scaling**:
  - Tight-range log handling ensures readable ticks around reference.
  - Minimum visible tick structure enforced for very narrow spans.
  - Trade-off: small synthetic padding may be added to improve interpretation.

- **Outlier clipping is opt-in**:
  - `clip_outliers=False` by default to avoid hiding information silently.
  - When enabled, quantile-based clipping improves readability in outlier-heavy plots.
  - Trade-off: defaults preserve raw scale; optional mode improves practical interpretability.

- **Precision and formatting guardrails**:
  - `base_decimals` capped at 3 to protect table layout.
  - CI rendered as bracketed intervals (`[low,high]`) to avoid sign ambiguity.
  - General stats counts switch to compact units only at `>= 10.000`, using one shared unit across `n` and `N` (`k/M/B/T`) for consistency.
  - Extremely large counts are display-capped (`>999T`) with warning rather than overflowing layout.
  - Missing-value rendering is strict per outcome triplet (`effect`, CI, `p`): partial-missing triplets are blanked; full-row gray is used only when all displayed outcomes are missing.
  - Long footers wrapped and capped to prevent layout breakage.

## Quality and Release Discipline
- Unit and behavior tests for each internal module (`pytest`).
- Visual regression matrix via scripted plot generation (`tests/run_visual_tests.py`).
- Packaging validation: build, metadata checks, TestPyPI smoke install, PyPI install verification.
- CI integration target: GitHub Actions on push/PR to run the same checks continuously (and back a README CI badge).

## Known Boundaries (v1)
- Layout is optimized for publication use cases, not arbitrary custom typography.
- Extreme invalid-input scenarios are warned about rather than fully auto-corrected.
- More adaptive layout and symmetric clipping refinements are deferred to future minor releases.
