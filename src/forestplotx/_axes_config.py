from collections.abc import Mapping
import math
import warnings
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
    candidates = [0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.25, 0.5, 1.0]
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
        "identity": {"reference_line": 0.0, "use_log": False, "x_label": "β (coefficient)"},
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

        ax.set_xscale("log" if use_log else "linear")

        if use_log:
            if ref_val <= 0:
                raise ValueError(
                    "Log-scaled forest axis requires a positive reference value."
                )
            finite_eff = np.asarray(cfg.get("eff_all", []), dtype=float)
            finite_eff = finite_eff[np.isfinite(finite_eff)]
            has_nonpositive = bool(
                np.any(finite_lo <= 0)
                or np.any(finite_hi <= 0)
                or np.any(finite_eff <= 0)
            )
            if has_nonpositive:
                warnings.warn(
                    "Log-scaled forest axis received nonpositive effect/CI values. "
                    "These values cannot be represented on a log axis and may be clipped. "
                    "Check whether your data is already exponentiated or set exponentiate=True "
                    "when input is on the link scale.",
                    UserWarning,
                    stacklevel=2,
                )
            positive_lo = finite_lo[finite_lo > 0]
            positive_hi = finite_hi[finite_hi > 0]
            positive_eff = finite_eff[finite_eff > 0]
            positive_values = np.concatenate(
                [positive_lo, positive_hi, positive_eff]
            )
            positive_candidates = [*positive_values.tolist(), ref_val]
            if not positive_candidates:
                raise ValueError(
                    "Log-scaled forest axis requires positive effect/CI values."
                )

            if clip_outliers and len(positive_values):
                clip_factor = 10.0

                if len(positive_lo):
                    lo_baseline = float(np.median(positive_lo))
                    lo_threshold = lo_baseline / clip_factor if lo_baseline > 0 else 0.0
                    lo_inliers = positive_lo[positive_lo >= lo_threshold]
                    clipped_pmin = float(np.min(lo_inliers)) if len(lo_inliers) else float(np.min(positive_lo))
                else:
                    clipped_pmin = float(np.min(positive_values))

                if len(positive_hi):
                    hi_baseline = float(np.median(positive_hi))
                    hi_threshold = hi_baseline * clip_factor
                    hi_inliers = positive_hi[positive_hi <= hi_threshold]
                    clipped_pmax = float(np.max(hi_inliers)) if len(hi_inliers) else float(np.max(positive_hi))
                else:
                    clipped_pmax = float(np.max(positive_values))

                pmin = min(clipped_pmin, ref_val)
                pmax = max(clipped_pmax, ref_val)
            else:
                pmin = min(positive_candidates)
                pmax = max(positive_candidates)
            target_ticks = max(int(num_ticks), 3)
            log_min = math.log10(pmin)
            log_max = math.log10(pmax)
            span_decades = max(log_max - log_min, 0.0)
            pad_decades = max(0.08, min(0.25, span_decades * 0.08))
            axis_log_min = log_min - pad_decades
            axis_log_max = log_max + pad_decades
            axis_span_decades = axis_log_max - axis_log_min

            raw_step = axis_span_decades / max(target_ticks - 1, 1)
            if span_decades > 3:
                step_decades = max(1.0, _nice_log_step(raw_step))
            else:
                step_decades = _nice_log_step(raw_step)

            tick_start = math.ceil(axis_log_min / step_decades) * step_decades
            tick_end = math.floor(axis_log_max / step_decades) * step_decades
            if tick_end < tick_start:
                tick_logs = np.array([axis_log_min, 0.0, axis_log_max], dtype=float)
            else:
                tick_logs = np.arange(
                    tick_start,
                    tick_end + 0.5 * step_decades,
                    step_decades,
                )
                if not np.any(np.isclose(tick_logs, 0.0, atol=1e-9)):
                    tick_logs = np.sort(np.append(tick_logs, 0.0))

            xmin = 10 ** axis_log_min
            xmax = 10 ** axis_log_max
            ax.set_xlim(xmin, xmax)
            ticks_in = np.power(10.0, tick_logs)
            ticks_in = ticks_in[(ticks_in >= xmin) & (ticks_in <= xmax)]
            ticks_in = np.unique(np.asarray(ticks_in, dtype=float))

            tick_data_min = max(pmin, np.nextafter(0.0, 1.0))
            tick_data_max = pmax
            moderate_decimal_span = (
                tick_style == "decimal"
                and pmin >= 0.2
                and pmax <= 10.0
                and span_decades <= 1.4
            )
            if moderate_decimal_span:
                readable_ticks = np.array(
                    [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.2, 1.5, 2.0, 3.0, 4.0, 5.0, 7.0, 10.0],
                    dtype=float,
                )
                ticks_in = readable_ticks[(readable_ticks >= tick_data_min) & (readable_ticks <= tick_data_max)]
                if len(ticks_in) > 8:
                    keep = []
                    for idx, tick in enumerate(ticks_in):
                        if idx % 2 == 0 or math.isclose(tick, 1.0, abs_tol=1e-9):
                            keep.append(tick)
                    ticks_in = np.array(sorted(set(keep)), dtype=float)
            elif tick_style == "decimal":
                decade_min = int(math.floor(axis_log_min))
                decade_max = int(math.ceil(axis_log_max))
                readable_ticks = []
                for decade in range(decade_min, decade_max + 1):
                    base = 10.0 ** decade
                    for mult in (1.0, 2.0, 5.0):
                        tick = mult * base
                        if tick_data_min <= tick <= tick_data_max:
                            readable_ticks.append(tick)
                if readable_ticks:
                    ticks_in = np.array(sorted(set(readable_ticks)), dtype=float)
                    if not np.any(np.isclose(ticks_in, ref_val, atol=1e-9)) and tick_data_min <= ref_val <= tick_data_max:
                        ticks_in = np.array(sorted(np.append(ticks_in, ref_val)), dtype=float)
                    if len(ticks_in) > 9:
                        min_log_gap = axis_span_decades / 7.0
                        keep = [float(ticks_in[0])]
                        for tick in ticks_in[1:-1]:
                            if math.isclose(tick, ref_val, abs_tol=1e-9):
                                keep.append(float(tick))
                                continue
                            if math.log10(float(tick)) - math.log10(float(keep[-1])) >= min_log_gap:
                                keep.append(float(tick))
                        keep.append(float(ticks_in[-1]))
                        if tick_data_min <= ref_val <= tick_data_max and not any(math.isclose(t, ref_val, abs_tol=1e-9) for t in keep):
                            keep.append(ref_val)
                        ticks_in = np.array(sorted(set(keep)), dtype=float)

            if len(ticks_in) < 3:
                ticks_in = np.array([xmin, ref_val, xmax], dtype=float)
            ax.xaxis.set_major_locator(FixedLocator(ticks_in))

            if tick_style == "power10":

                def _power10_formatter(x: float, _pos: int) -> str:
                    exp = round(math.log10(x), 6)
                    if math.isclose(exp, round(exp), abs_tol=1e-9):
                        exp_txt = str(int(round(exp)))
                    else:
                        exp_txt = f"{exp:.2f}".rstrip("0").rstrip(".")
                    return rf"$10^{{{exp_txt}}}$"

                ax.xaxis.set_major_formatter(FuncFormatter(_power10_formatter))
            else:
                decimals = _decimals_from_ticks(ticks_in)

                def _decimal_log_formatter(x: float, _pos: int, d: int = decimals) -> str:
                    abs_x = abs(float(x))
                    if abs_x >= 1e12:
                        return _format_decimal(x / 1e12, precision=1).rstrip("0").rstrip(".") + "T"
                    if abs_x >= 1e9:
                        return _format_decimal(x / 1e9, precision=1).rstrip("0").rstrip(".") + "B"
                    if abs_x >= 1e6:
                        return _format_decimal(x / 1e6, precision=1).rstrip("0").rstrip(".") + "M"
                    if abs_x >= 1e3:
                        return _format_decimal(x / 1e3, precision=1).rstrip("0").rstrip(".") + "k"
                    if abs_x >= 10:
                        return _format_decimal(x, precision=0)
                    return _format_decimal(x, precision=max(d + 1, 1))

                ax.xaxis.set_major_formatter(FuncFormatter(_decimal_log_formatter))

            ax.xaxis.set_minor_locator(NullLocator())
            ax.xaxis.set_minor_formatter(NullFormatter())
        else:
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

            if clip_outliers:
                q_high = float(clip_quantiles[1])
                # Linear outliers are visually dominant; keep clipping robust by capping
                # the effective upper quantile used for span control.
                q_high = min(q_high, 0.90)
                distances = np.concatenate(
                    [
                        np.abs(finite_lo - ref_val),
                        np.abs(finite_hi - ref_val),
                    ]
                )
                distances = distances[np.isfinite(distances)]
                if len(distances):
                    span = float(np.quantile(distances, q_high))
                else:
                    span = max(abs(data_min - ref_val), abs(data_max - ref_val))
            else:
                span = max(abs(data_min - ref_val), abs(data_max - ref_val))
                # Flag outlier-dominated ranges where one extreme compresses the majority.
                distances = np.concatenate(
                    [
                        np.abs(finite_lo - ref_val),
                        np.abs(finite_hi - ref_val),
                    ]
                )
                distances = distances[np.isfinite(distances)]
                if len(distances) >= 8:
                    q95 = float(np.quantile(distances, 0.95))
                    if q95 > 0 and span / q95 >= 5:
                        warnings.warn(
                            "Linear axis appears outlier-dominated. Consider clip_outliers=True "
                            "to improve readability while preserving raw table values.",
                            UserWarning,
                            stacklevel=2,
                        )
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
