from collections.abc import Mapping
import math
from typing import Any

import numpy as np
from matplotlib.axes import Axes
from matplotlib.ticker import FixedLocator, FuncFormatter, NullFormatter, NullLocator


def _nice_linear_step(raw_step: float) -> float:
    """Return a human-readable step size (1/2/5 x 10^k)."""
    if raw_step <= 0:
        return 1.0
    exponent = math.floor(math.log10(raw_step))
    fraction = raw_step / (10**exponent)
    if fraction <= 1:
        nice_fraction = 1
    elif fraction <= 2:
        nice_fraction = 2
    elif fraction <= 5:
        nice_fraction = 5
    else:
        nice_fraction = 10
    return nice_fraction * (10**exponent)


def _format_decimal(value: float, precision: int = 6) -> str:
    """Format decimals consistently without scientific notation."""
    return np.format_float_positional(value, precision=precision, trim="-")


def _decimals_from_ticks(ticks: np.ndarray, max_decimals: int = 3) -> int:
    """Infer a readable fixed decimal count from adjacent tick spacing."""
    if len(ticks) < 2:
        return 2
    diffs = np.diff(np.sort(np.asarray(ticks, dtype=float)))
    diffs = diffs[np.isfinite(diffs) & (diffs > 0)]
    if not len(diffs):
        return 2
    min_diff = float(np.min(diffs))
    decimals = int(max(0, -math.floor(math.log10(min_diff))))
    return max(0, min(max_decimals, decimals))


def _nice_log_step(raw_step: float) -> float:
    """Return a readable log10 step size."""
    candidates = [0.05, 0.1, 0.2, 0.25, 0.5, 1.0]
    for cand in candidates:
        if cand >= raw_step:
            return cand
    return raw_step


def configure_forest_axis(
    ax: Axes,
    model_type: str,
    link: str | None,
    thresholds: Mapping[str, Any] | None,
    num_ticks: int,
    font_size: int,
    show_general_stats: bool,
) -> Axes:
    """
    Configure forest-panel axis scaling, ticks, and visual styling.

    Parameters
    ----------
    ax : Axes
        Matplotlib axis for the forest panel.
    model_type : str
        Model family name (e.g., ``"binom"``, ``"gamma"``, ``"linear"``).
    link : str | None
        Link function name used by the model output normalization.
    thresholds : Mapping[str, Any] | None
        Explicit axis inputs. Supported keys include:
        ``reference_line``, ``x_label``, ``use_log``, ``lo_all``, ``hi_all``,
        and ``y_limits``.
    num_ticks : int
        Target number of major ticks for linear locators.
    font_size : int
        Axis label font size.
    show_general_stats : bool
        Included for API symmetry with plot orchestration.

    Returns
    -------
    Axes
        The configured axis.
    """
    _ = show_general_stats
    cfg = dict(thresholds or {})
    link_defaults = {
        "logit": {"reference_line": 1.0, "use_log": True, "x_label": "Odds Ratio"},
        "log": {"reference_line": 1.0, "use_log": True, "x_label": "Ratio"},
        "identity": {"reference_line": 0.0, "use_log": False, "x_label": "Effect Size"},
    }
    defaults = link_defaults.get(link or "identity", link_defaults["identity"])

    ref_val = float(cfg.get("reference_line", defaults["reference_line"]))
    use_log = bool(cfg.get("use_log", defaults["use_log"]))
    x_label = str(cfg.get("x_label", defaults["x_label"]))
    tick_style = str(cfg.get("tick_style", "decimal"))
    clip_outliers = bool(cfg.get("clip_outliers", False))
    clip_quantiles = cfg.get("clip_quantiles", (0.02, 0.98))
    lo_all = np.asarray(cfg.get("lo_all", []), dtype=float)
    hi_all = np.asarray(cfg.get("hi_all", []), dtype=float)
    y_limits = cfg.get("y_limits")

    ax.axvline(ref_val, color="#910C07", lw=1.2, ls="--")
    ax.set_yticks([])
    if y_limits is not None:
        ax.set_ylim(y_limits[0], y_limits[1])

    ax.set_xlabel(x_label, fontsize=font_size)
    if len(lo_all) and len(hi_all):
        finite_lo = lo_all[np.isfinite(lo_all)]
        finite_hi = hi_all[np.isfinite(hi_all)]
        if not len(finite_lo) or not len(finite_hi):
            return ax

        if clip_outliers:
            q_low, q_high = clip_quantiles
            q_low = float(q_low)
            q_high = float(q_high)
            if not (0.0 <= q_low < q_high <= 1.0):
                raise ValueError("clip_quantiles must satisfy 0 <= low < high <= 1.")
            data_min = float(np.quantile(finite_lo, q_low))
            data_max = float(np.quantile(finite_hi, q_high))
        else:
            data_min = float(np.min(finite_lo))
            data_max = float(np.max(finite_hi))

        ax.set_xscale("log" if use_log else "linear")

        if use_log:
            if ref_val <= 0:
                raise ValueError(
                    "Log-scaled forest axis requires a positive reference value."
                )
            positive_candidates = [v for v in (data_min, data_max, ref_val) if v > 0]
            if not positive_candidates:
                raise ValueError(
                    "Log-scaled forest axis requires positive effect/CI values."
                )

            pmin = min(positive_candidates)
            pmax = max(positive_candidates)
            target_ticks = max(int(num_ticks), 3)
            if target_ticks % 2 == 0:
                target_ticks -= 1
            n_side_target = max((target_ticks - 1) // 2, 1)

            span_decades = max(abs(math.log10(pmin / ref_val)), abs(math.log10(pmax / ref_val)))
            axis_span_decades = max(span_decades * 1.05, 0.06)
            axis_span_decades = min(axis_span_decades, span_decades + 0.08)
            raw_step = axis_span_decades / n_side_target
            step_decades = _nice_log_step(raw_step)
            n_side = max(1, int(axis_span_decades / step_decades))
            exponents = np.arange(-n_side, n_side + 1, dtype=float) * step_decades
            ticks = ref_val * np.power(10.0, exponents)
            axis_ratio = 10 ** axis_span_decades
            ax.set_xlim(ref_val / axis_ratio, ref_val * axis_ratio)
            ax.xaxis.set_major_locator(FixedLocator(ticks))

            if tick_style == "power10":

                def _power10_formatter(x: float, _pos: int) -> str:
                    exp = math.log10(x / ref_val)
                    rounded = round(exp, 2)
                    if math.isclose(rounded, 0.0, abs_tol=1e-9):
                        rounded = 0.0
                    exp_txt = f"{rounded:.2f}".rstrip("0").rstrip(".")
                    if math.isclose(ref_val, 1.0):
                        return rf"$10^{{{exp_txt}}}$"
                    return rf"${_format_decimal(ref_val)}\times10^{{{exp_txt}}}$"

                ax.xaxis.set_major_formatter(FuncFormatter(_power10_formatter))
            else:
                decimals = max(2, _decimals_from_ticks(ticks))
                ax.xaxis.set_major_formatter(
                    FuncFormatter(lambda x, _pos, d=decimals: f"{x:.{d}f}")
                )

            ax.xaxis.set_minor_locator(NullLocator())
            ax.xaxis.set_minor_formatter(NullFormatter())
        else:
            span = max(abs(data_min - ref_val), abs(data_max - ref_val))
            if span == 0:
                span = max(1e-3, abs(ref_val) * 0.1)
            target_ticks = max(int(num_ticks), 3)
            raw_step = (2 * span) / max(target_ticks - 1, 1)
            step = _nice_linear_step(raw_step)
            kmax = max(1, math.ceil(span / step))
            ticks = ref_val + np.arange(-kmax, kmax + 1, dtype=float) * step
            xmin = ref_val - kmax * step
            xmax = ref_val + kmax * step

            ax.set_xlim(xmin, xmax)
            ax.xaxis.set_major_locator(FixedLocator(ticks))
            decimals = _decimals_from_ticks(ticks)
            ax.xaxis.set_major_formatter(
                FuncFormatter(lambda x, _pos, d=decimals: f"{x:.{d}f}")
            )

    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)

    return ax
