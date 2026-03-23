# forestplotx Design Notes (v1.1.0)

## Scope and Intent
`forestplotx` targets publication-ready forest plots for common regression outputs with minimal user-side formatting work.
The design favors predictable defaults and explicit behavior over highly dynamic auto-layout.

## Technical Motivation
The primary pain point that motivated this project was the mismatch between what publication tables require and what default matplotlib composition provides for a combined table + forest plot.

- In dual-axis compositions, row alignment between the text table and forest markers is easy to drift and hard to keep stable across figure sizes and content variations.
- Standard table rendering (`matplotlib.table`) is not robust for dense scientific output and becomes unreadable quickly.
- Wide-format output needs strict control over column spacing and precision formatting, which generic auto-layout does not reliably provide.

Design response in `forestplotx`:
- Build a fake table via explicit row assembly and deterministic text placement.
- Drive both table and forest from one shared y-position model (`_layout.py`).
- Use fixed layout presets rather than free-form auto-spacing.

## Module Architecture
- `plot.py`: orchestration layer (table rendering, plotting, legend/footer, save/show behavior)
- `_normalize.py`: input normalization (column mapping, optional exponentiation, unified output schema)
- `_layout.py`: row assembly and y-position generation
- `_axes_config.py`: axis scaling, ticks, labels, reference-line behavior, clipping logic

## Data Flow
1. User calls `forest_plot(df=..., model_type=...)`
2. `_normalize_model_output()` maps model-specific columns to canonical columns
3. `_layout.build_row_layout()` builds predictor/category rows and y positions
4. `plot.py` renders table + points/CIs
5. `_axes_config.configure_forest_axis()` sets x-scale, limits, ticks, and labels

## Core Design Decisions and Trade-offs

- **Fixed 4-case layout presets**:
  - Chosen over generic auto-spacing because matplotlib text geometry is unstable across cases.
  - Trade-off: less user flexibility, much higher visual consistency.

- **Deterministic internal pressure tiers**:
  - `standard`, `expanded`, and `max` tiers respond to rendered string pressure inside each core layout case.
  - Trade-off: more internal logic, fewer user-facing knobs.

- **Explicit exponentiation control with safe defaults**:
  - `exponentiate` supports override; when omitted, model/link defaults apply.
  - Warnings are emitted for potentially incorrect scale assumptions.

- **Axis readability prioritized over formal symmetry**:
  - Log-axis limits are data-driven after optional clipping rather than forced symmetric around the reference.
  - Decimal log ticks use denser near-reference ticks for moderate spans and `1-2-5` progression for wider spans.
  - `power10` mode remains available for wide ratio ranges.

- **Outlier clipping is opt-in**:
  - `clip_outliers=False` by default to avoid hiding information silently.
  - On log axes, clipping uses a magnitude-based rule centered on median CI bounds.
  - On linear axes, quantile clipping remains available.

- **Precision and formatting guardrails**:
  - `base_decimals` capped at 3 to protect table layout.
  - CI rendered as bracketed intervals (`[low,high]`).
  - General stats counts switch to compact units only at `>= 1,000`, using one shared unit across `n` and `N`.
  - Effect and CI values use the same compact unit family from `>= 1,000`.
  - Missing-value rendering is strict per outcome triplet.
  - Long footers are wrapped and constrained to preserve layout.

- **Visible-label overrides are explicit**:
  - `column_labels` allows header overrides for `effect`, `ci`, `p`, `n`, `N`, and `Freq`.
  - `x_label_override` allows direct x-axis label overrides without changing normalization behavior.

## Quality and Release Discipline
- Unit and behavior tests for each internal module (`pytest`).
- Visual regression matrix via `tests/run_visual_tests.py`.
- Packaging validation: build, metadata checks, TestPyPI smoke install, PyPI install verification.
- CI target: GitHub Actions on push/PR.

## Known Boundaries (v1)
- Layout is optimized for publication use cases, not arbitrary custom typography.
- Extreme invalid-input scenarios are warned about rather than fully auto-corrected.
- Footer composition remains the most sensitive layout area relative to the rest of the system.

## Future Considerations

### 1) Title handling for publication workflows
Current policy:
- `forest_plot()` has no `title` parameter by design in v1.x.
- This keeps figure geometry deterministic and aligns with journal-style submissions where titles/captions are placed in the manuscript.

### 2) Footer/layout cleanup
Current policy:
- Main panel spacing and axis logic are deterministic and stable.
- Footer placement is bounded and improved, but still more sensitive than the main table/axis system.

Potential refinement:
- Further isolate footer geometry into a simpler physical-unit layout path if broader footer-heavy use cases demand it.
