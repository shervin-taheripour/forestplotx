import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from os import PathLike
from matplotlib.patches import Rectangle
import textwrap
import warnings

from ._axes_config import configure_forest_axis
from ._layout import build_row_layout
from ._normalize import _normalize_model_output


def forest_plot(
    df,
    outcomes=None,
    save=None,
    model_type="binom",
    link=None,
    exponentiate: bool | None = None,
    table_only=False,
    legend_labels=None,
    point_colors: list[str] | None = None,
    footer_text=None,
    tick_style: str = "decimal",
    clip_outliers: bool = False,
    clip_quantiles: tuple[float, float] = (0.02, 0.98),
    base_decimals=2,
    show: bool = True,
    show_general_stats: bool = True,
    bold_override: dict | None = None,
):
    """
    Render a publication-style forest plot with an aligned text table and optional footer.

    This function normalizes model-output columns (via `_normalize_model_output`), builds a
    row layout (via `_layout.build_row_layout`), and draws a two-panel figure:
    table on the left and forest axis on the right. Up to two outcomes are displayed.

    Key behavior:
    - Layout is controlled by fixed internal presets for 4 cases:
      (show_general_stats=True/False) x (one/two outcomes).
    - Internal font size and block spacing are fixed for layout stability.
    - `base_decimals` is capped at 3 to prevent dense-table collisions.
    - CI text is displayed in bracket notation: `[low,high]`.
    - For log-link models, exponentiation can be auto/resolved with `exponentiate`.
    - `clip_outliers` is opt-in and affects axis limits only (table values remain raw).

    Parameters
    ----------
    df : pandas.DataFrame
        Input model-output data. Must include `predictor`, `outcome`, one recognized
        effect column (`OR`, `Ratio`, `Estimate`, `beta`, `Coef`, or `effect`), and
        CI bounds (`CI_low`/`ci_low`, `CI_high`/`ci_high`).
    outcomes : list[str] | None, default None
        Outcomes to plot. If None, auto-detected from `df`. Maximum 2 outcomes are used.
    save : str | os.PathLike | bool | None, default None
        Save behavior:
        - `None` or `False`: do not save
        - `True`: save as `"forestplot.png"`
        - path-like/string: save to that path
    model_type : {"binom","gamma","linear","ordinal"}, default "binom"
        Model family used for normalization and axis defaults.
    link : str | None, default None
        Optional link override. If None, defaults from `model_type` are used.
    exponentiate : bool | None, default None
        Exponentiation policy:
        - `None`: automatic by link
        - `True`: force exponentiation
        - `False`: disable exponentiation
    table_only : bool, default False
        If True, render only the table panel (no forest axis).
    legend_labels : list[str] | None, default None
        Custom legend labels for plotted outcomes.
    point_colors : list[str] | None, default None
        Marker colors for outcomes (up to 2). Missing entries fall back to defaults.
    footer_text : str | None, default None
        Optional footer. Wrapped internally and capped to 3 lines with ellipsis for overflow.
    tick_style : {"decimal","power10"}, default "decimal"
        Tick label style for log-axis rendering.
    clip_outliers : bool, default False
        If True, use quantile-based clipping for axis limits to improve readability in
        outlier-heavy plots. Clipped CIs are marked at axis boundaries.
    clip_quantiles : tuple[float, float], default (0.02, 0.98)
        Quantiles used when `clip_outliers=True`; must satisfy `0 <= low < high <= 1`.
    base_decimals : int, default 2
        Decimal precision for effect/CI display (internally capped at 3).
    show : bool, default True
        If True, call `plt.show()`. In notebooks, returned figures may still auto-render
        even when `show=False`.
    show_general_stats : bool, default True
        Show/hide `n`, `N`, and `Freq` table columns.
    bold_override : dict | None, default None
        Optional manual bold override map by predictor/outcome.

    Returns
    -------
    tuple
        `(fig, (ax_text, ax_forest))`, where `ax_forest` is `None` when `table_only=True`.
    """
    if save is None or save is False:
        save_path = None
    elif save is True:
        save_path = "forestplot.png"
    elif isinstance(save, (str, PathLike)):
        save_path = save
    else:
        raise TypeError("save must be a path string/path-like, None, or bool.")

    bold_override = bold_override or {}
    if outcomes is None:
        outcomes = df["outcome"].unique().tolist()
    if len(outcomes) > 2:
        outcomes = outcomes[:2]
    has_second = len(outcomes) == 2

    # --- Normalize model output ----------------------------------------------
    df, plot_config = _normalize_model_output(
        df,
        model_type=model_type,
        link=link,
        exponentiate=exponentiate,
    )
    effect_label = plot_config["effect_label"]
    ref_val = plot_config["reference_line"]
    use_log = plot_config["use_log"]
    xlabel = plot_config["x_label"]
    ci_label = "95% CI"
    base_decimals = min(int(base_decimals), 3)
    _BLOCK_SPACING = 6.0

    def format_effect_ci_p(eff, lo, hi, p):
        d = base_decimals
        eff_s = f"{eff:.{d}f}" if pd.notnull(eff) else ""
        lo_s = f"{lo:.{d}f}" if pd.notnull(lo) else ""
        hi_s = f"{hi:.{d}f}" if pd.notnull(hi) else ""
        ci_s = f"[{lo_s},{hi_s}]" if lo_s and hi_s else ""
        p_s = "" if pd.isnull(p) else ("<0.001" if p < 0.001 else f"{p:.3f}")
        return eff_s, ci_s, p_s, d

    layout = build_row_layout(df)
    table_rows = layout["rows"].to_dict("records")
    y_positions = layout["y_positions"]
    n = layout["meta"]["n"]
    fig_height = max(0.3 * n + 1.5, 5.0)
    if n <= 8:
        fig_height = max(0.26 * n + 1.2, 3.4)

    # 4-case layout presets:
    # (show_general_stats, has_second_outcome) -> geometry config.
    # For now, all four cases intentionally share the same values.
    layout_presets = {
        (True, True): {
            "block_mult": {"general": 1.06, "outcome1": 1.74, "outcome2": 2.74},
            "general_offsets": [0.0, 1.45, 3.15],
            "outcome_offsets": [0.0, 2.0, 4.1],
            "fig_width": 16,
            "width_ratios": [1.9, 1.1],
        },
        (True, False): {
            "block_mult": {"general": 1.2, "outcome1": 1.90, "outcome2": 2.9},
            "general_offsets": [0.0, 1.3, 2.8],
            "outcome_offsets": [0.0, 2.0, 4.1],
            "fig_width": 13,
            "width_ratios": [1.9, 1.3],
        },
        (False, True): {
            "block_mult": {"general": 1.2, "outcome1": 1.30, "outcome2": 2.30},
            "general_offsets": [0.0, 1.3, 2.6],
            "outcome_offsets": [0.0, 2.1, 4.3],
            "fig_width": 14,
            "width_ratios": [1.9, 1.2],
        },
        (False, False): {
            "block_mult": {"general": 1.2, "outcome1": 1.2, "outcome2": 2.9},
            "general_offsets": [0.0, 1.3, 2.6],
            "outcome_offsets": [0.0, 2.0, 4.1],
            "fig_width": 9.5,
            "width_ratios": [1.9, 1.2],
        },
    }
    layout_cfg = layout_presets[(show_general_stats, has_second)]
    predictor_label_caps = {
        (True, True): 21,
        (True, False): 24,
        (False, True): 26,
        (False, False): 25,
    }
    predictor_label_cap = predictor_label_caps[(show_general_stats, has_second)]
    render_font_size = 10
    BLOCK_X = {"predictor": 0.0}
    if show_general_stats:
        BLOCK_X["general"] = _BLOCK_SPACING * layout_cfg["block_mult"]["general"]
    BLOCK_X["outcome1"] = _BLOCK_SPACING * layout_cfg["block_mult"]["outcome1"]
    BLOCK_X["outcome2"] = _BLOCK_SPACING * layout_cfg["block_mult"]["outcome2"]
    GENERAL_OFFSETS = layout_cfg["general_offsets"]
    OUTCOME_OFFSETS = layout_cfg["outcome_offsets"]

    n_vals_all = (
        pd.to_numeric(df["n"], errors="coerce")
        if "n" in df.columns
        else pd.Series(dtype=float)
    )
    N_vals_all = (
        pd.to_numeric(df["N"], errors="coerce")
        if "N" in df.columns
        else pd.Series(dtype=float)
    )

    def _pick_compact_unit(vals: pd.Series) -> tuple[float, str]:
        if vals.empty or not vals.notna().any():
            return 1.0, ""
        vmax = float(vals.max())
        if vmax < 10_000:
            return 1.0, ""
        if vmax >= 1_000_000_000_000:
            return 1_000_000_000_000.0, "T"
        if vmax >= 1_000_000_000:
            return 1_000_000_000.0, "B"
        if vmax >= 1_000_000:
            return 1_000_000.0, "M"
        if vmax >= 10_000:
            return 1_000.0, "k"
        return 1.0, ""

    if show_general_stats:
        combined_counts = pd.concat([n_vals_all, N_vals_all], axis=0)
        shared_scale, shared_suffix = _pick_compact_unit(combined_counts)
    else:
        shared_scale, shared_suffix = 1.0, ""
    n_scale, n_suffix = shared_scale, shared_suffix
    N_scale, N_suffix = shared_scale, shared_suffix
    count_overflow_cap = 999.95 * 1_000_000_000_000.0
    n_overflow = bool(show_general_stats and not n_vals_all.empty and n_vals_all.notna().any() and float(n_vals_all.max()) > count_overflow_cap)
    N_overflow = bool(show_general_stats and not N_vals_all.empty and N_vals_all.notna().any() and float(N_vals_all.max()) > count_overflow_cap)
    if n_overflow or N_overflow:
        cols = []
        if n_overflow:
            cols.append("n")
        if N_overflow:
            cols.append("N")
        warnings.warn(
            f"Very large count values detected in {', '.join(cols)}; display is capped at >999T for readability.",
            UserWarning,
            stacklevel=2,
        )

    def _format_count(val: float, scale: float, suffix: str) -> str:
        if pd.isna(val):
            return ""
        v = float(val)
        if scale >= 1_000_000_000_000.0 and v > count_overflow_cap:
            return ">999T"
        if scale == 1.0:
            return f"{int(v):,}".replace(",", ".")
        compact = v / scale
        txt = f"{compact:.1f}".replace(".", ",")
        if txt.endswith(",0"):
            txt = txt[:-2]
        return f"{txt}{suffix}"

    predictor_display_map: dict[str, str] = {}
    truncated_predictors: list[str] = []
    for row in table_rows:
        if row["is_cat"]:
            continue
        pred = str(row["predictor"])
        if pred in predictor_display_map:
            continue
        if len(pred) > predictor_label_cap:
            predictor_display_map[pred] = pred[: predictor_label_cap - 3] + "..."
            truncated_predictors.append(pred)
        else:
            predictor_display_map[pred] = pred

    if truncated_predictors:
        shown = ", ".join(f"'{p}'" for p in truncated_predictors[:3])
        more = f" (+{len(truncated_predictors) - 3} more)" if len(truncated_predictors) > 3 else ""
        warnings.warn(
            f"Predictor label length exceeded cap ({predictor_label_cap}) for layout "
            f"(show_general_stats={show_general_stats}, two_outcomes={has_second}). "
            f"Labels were truncated for display: {shown}{more}.",
            UserWarning,
            stacklevel=2,
        )
    suppressed_null_triplets = 0

    if table_only:
        fig, ax_text = plt.subplots(1, 1, figsize=(15, fig_height))
        ax_forest = None
    else:
        fig, (ax_text, ax_forest) = plt.subplots(
            1,
            2,
            figsize=(layout_cfg["fig_width"], fig_height),
            gridspec_kw={"width_ratios": layout_cfg["width_ratios"]},
        )
        plt.subplots_adjust(wspace=0.02)
        plt.subplots_adjust(bottom=0.12)

    header_row_1, header_row_2 = -2.2, -1.0
    general_header_artists = [None, None, None]
    general_value_artists = [[], [], []]
    ax_text.text(
        BLOCK_X["predictor"],
        header_row_2,
        "Predictor",
        ha="left",
        va="center",
        fontweight="bold",
        fontsize=render_font_size,
    )
    if show_general_stats:
        for i, label in enumerate(["n", "N", "Freq"]):
            col_x = BLOCK_X["general"] + GENERAL_OFFSETS[i]
            general_header_artists[i] = ax_text.text(
                col_x,
                header_row_2,
                label,
                ha="center",
                va="center",
                fontweight="bold",
                fontsize=render_font_size,
            )
    ax_text.text(
        BLOCK_X["outcome1"] + OUTCOME_OFFSETS[1],
        header_row_1,
        outcomes[0],
        ha="center",
        va="center",
        fontweight="bold",
        fontsize=render_font_size,
    )
    for i, label in enumerate([effect_label, ci_label, "p"]):
        ax_text.text(
            BLOCK_X["outcome1"] + OUTCOME_OFFSETS[i],
            header_row_2,
            label,
            ha="center",
            va="center",
            fontweight="bold",
            fontsize=render_font_size,
        )
    if has_second:
        ax_text.text(
            BLOCK_X["outcome2"] + OUTCOME_OFFSETS[1],
            header_row_1,
            outcomes[1],
            ha="center",
            va="center",
            fontweight="bold",
            fontsize=render_font_size,
        )
        for i, label in enumerate([effect_label, ci_label, "p"]):
            ax_text.text(
                BLOCK_X["outcome2"] + OUTCOME_OFFSETS[i],
                header_row_2,
                label,
                ha="center",
                va="center",
                fontweight="bold",
                fontsize=render_font_size,
            )

    for y, row in zip(y_positions, table_rows):
        is_cat, pred = row["is_cat"], row["predictor"]
        if is_cat:
            ax_text.text(
                BLOCK_X["predictor"],
                y,
                pred,
                ha="left",
                va="center",
                fontsize=render_font_size,
                fontweight="bold",
            )
            continue

        dfrow_pred = df[df["predictor"] == pred]
        if dfrow_pred.empty:
            continue
        dfrow = dfrow_pred.iloc[0]
        style = dict(fontsize=render_font_size, fontweight="normal")

        any_valid_outcome = False
        for outcome in outcomes:
            dfo = df[(df["predictor"] == pred) & (df["outcome"] == outcome)]
            if dfo.empty:
                continue
            dfo = dfo.iloc[0]
            is_missing_triplet = (
                pd.isna(dfo.get("effect"))
                or pd.isna(dfo.get("ci_low"))
                or pd.isna(dfo.get("ci_high"))
                or pd.isna(dfo.get("p_value"))
            )
            if not is_missing_triplet:
                any_valid_outcome = True
                break
        is_null = not any_valid_outcome
        text_color = "#A0A0A0" if is_null else "black"

        ax_text.text(
            BLOCK_X["predictor"],
            y,
            predictor_display_map.get(str(pred), str(pred)),
            ha="left",
            va="center",
            color=text_color,
            **style,
        )

        if show_general_stats:
            n_val = ""
            N_val = ""
            if "n" in dfrow and pd.notnull(dfrow["n"]):
                n_val = _format_count(dfrow["n"], n_scale, n_suffix)
            if "N" in dfrow and pd.notnull(dfrow["N"]):
                N_val = _format_count(dfrow["N"], N_scale, N_suffix)
            freq_val = ""
            if n_val and N_val:
                try:
                    freq_val = f"{(int(dfrow['n']) / int(dfrow['N'])):.1%}"
                except Exception:
                    pass
            for i, val in enumerate([n_val, N_val, freq_val]):
                col_x = BLOCK_X["general"] + GENERAL_OFFSETS[i]
                txt = ax_text.text(
                    col_x,
                    y,
                    val,
                    ha="right",
                    va="center",
                    fontsize=render_font_size,
                    color=text_color,
                )
                general_value_artists[i].append(txt)

        for k, outcome in enumerate(outcomes):
            out_block = "outcome1" if k == 0 else "outcome2"
            dfrow = df[
                (df["predictor"] == pred) & (df["outcome"] == outcome)
            ]
            if dfrow.empty:
                continue
            dfrow = dfrow.iloc[0]
            eff, lo, hi, p = (
                dfrow.get("effect", np.nan),
                dfrow.get("ci_low", np.nan),
                dfrow.get("ci_high", np.nan),
                dfrow.get("p_value", np.nan),
            )
            if pd.isna(eff) or pd.isna(lo) or pd.isna(hi) or pd.isna(p):
                effect_val, ci, pval = "", "", ""
                suppressed_null_triplets += 1
            else:
                effect_val, ci, pval, _ = format_effect_ci_p(eff, lo, hi, p)
            p_bold = pd.notnull(p) and p < 0.05 and not (effect_val == "" and ci == "" and pval == "")
            manual_pred = bold_override.get(pred, {})
            manual_outcome = manual_pred.get(outcome, None)

            vals = [effect_val, ci, pval]
            for i, val in enumerate(vals):
                if i == 0:
                    auto_bold = p_bold
                    final_bold = manual_outcome if manual_outcome is not None else auto_bold
                elif i == 2:
                    final_bold = p_bold
                else:
                    final_bold = False

                ax_text.text(
                    BLOCK_X[out_block] + OUTCOME_OFFSETS[i],
                    y,
                    val,
                    ha="center",
                    va="center",
                    fontsize=render_font_size,
                    fontweight="bold" if final_bold else "normal",
                    color=text_color,
                )

    rightmost = (
        (BLOCK_X["outcome2"] + OUTCOME_OFFSETS[2] + 2.0)
        if has_second
        else (BLOCK_X["outcome1"] + OUTCOME_OFFSETS[2] + 2.0)
    )
    ax_text.set_xlim(-0.5, rightmost)
    ax_text.set_ylim(n - 0.5, -2.8)

    # Center n/N/Freq headers over the actual rendered value columns.
    if show_general_stats:
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        inv = ax_text.transData.inverted()
        for i in range(3):
            if general_header_artists[i] is None or not general_value_artists[i]:
                continue
            bboxes = [t.get_window_extent(renderer=renderer) for t in general_value_artists[i]]
            left = min(b.x0 for b in bboxes)
            right = max(b.x1 for b in bboxes)
            center_disp_x = 0.5 * (left + right)
            center_data_x = inv.transform((center_disp_x, 0.0))[0]
            general_header_artists[i].set_x(center_data_x)

    ax_text.axis("off")

    _DEFAULT_COLORS = ["#212427", "#D8CBBB"]
    if point_colors:
        colors = list(point_colors[:2])
        if len(colors) < 2:
            colors.extend(_DEFAULT_COLORS[len(colors) : 2])
    else:
        colors = _DEFAULT_COLORS.copy()

    if ax_forest is not None:
        markers = ["o", "s"]
        handles, labels, Y_OFFSET = [], [], 0.10
        eff_all, lo_all, hi_all = [], [], []
        for j, outcome in enumerate(outcomes):
            y_points, eff_vals, cil, cih = [], [], [], []
            for y, row in zip(y_positions, table_rows):
                if row["is_cat"]:
                    continue
                pred = row["predictor"]
                dfrow = df[
                    (df["predictor"] == pred) & (df["outcome"] == outcome)
                ]
                if dfrow.empty:
                    continue
                dfrow = dfrow.iloc[0]
                eff, lo, hi = (
                    dfrow.get("effect", np.nan),
                    dfrow.get("ci_low", np.nan),
                    dfrow.get("ci_high", np.nan),
                )
                is_null = pd.isna(eff) or pd.isna(lo) or pd.isna(hi) or pd.isna(dfrow.get("p_value"))
                color = "#C7C7C7" if is_null else colors[j]

                if not is_null:
                    eff_vals.append(eff)
                    cil.append(eff - lo)
                    cih.append(hi - eff)
                    y_points.append(y)
                    eff_all += [eff]
                    lo_all += [lo]
                    hi_all += [hi]
                else:
                    ax_forest.plot(
                        [ref_val],
                        [y],
                        marker=markers[j],
                        color=color,
                        alpha=0.6,
                        markersize=5,
                        zorder=2,
                    )

            if eff_vals:
                offset_sign = -1 if j == 1 else 1
                y_points_offset = [y + Y_OFFSET * offset_sign for y in y_points]
                h = ax_forest.errorbar(
                    eff_vals,
                    y_points_offset,
                    xerr=[cil, cih],
                    fmt=markers[j],
                    color=colors[j],
                    ecolor=colors[j],
                    capsize=3,
                    markersize=5,
                    lw=1.7,
                    label=(
                        legend_labels[j]
                        if legend_labels and j < len(legend_labels)
                        else outcome
                    ),
                )
                handles.append(h.lines[0])
                labels.append(outcome)
        configure_forest_axis(
            ax=ax_forest,
            model_type=model_type,
            link=plot_config["link"],
            thresholds={
                "reference_line": ref_val,
                "x_label": xlabel,
                "use_log": use_log,
                "lo_all": lo_all,
                "hi_all": hi_all,
                "y_limits": (n - 0.5, -2.8),
                "tick_style": tick_style,
                "clip_outliers": clip_outliers,
                "clip_quantiles": clip_quantiles,
            },
            num_ticks=6,
            font_size=render_font_size,
            show_general_stats=show_general_stats,
        )

        # Mark clipped confidence intervals at panel edges for transparency.
        xmin, xmax = ax_forest.get_xlim()
        for j, outcome in enumerate(outcomes):
            offset_sign = -1 if j == 1 else 1
            y_offset = Y_OFFSET * offset_sign
            for y, row in zip(y_positions, table_rows):
                if row["is_cat"]:
                    continue
                pred = row["predictor"]
                dfrow = df[
                    (df["predictor"] == pred) & (df["outcome"] == outcome)
                ]
                if dfrow.empty:
                    continue
                dfrow = dfrow.iloc[0]
                eff, lo, hi = (
                    dfrow.get("effect", np.nan),
                    dfrow.get("ci_low", np.nan),
                    dfrow.get("ci_high", np.nan),
                )
                if pd.isna(eff) or pd.isna(lo) or pd.isna(hi):
                    continue
                yy = y + y_offset
                if lo < xmin:
                    ax_forest.plot(
                        [xmin],
                        [yy],
                        marker="<",
                        color=colors[j],
                        markersize=6,
                        zorder=4,
                    )
                if hi > xmax:
                    ax_forest.plot(
                        [xmax],
                        [yy],
                        marker=">",
                        color=colors[j],
                        markersize=6,
                        zorder=4,
                    )
        ax_forest.legend(handles, labels, loc="upper left", fontsize=render_font_size)
    if suppressed_null_triplets > 0:
        warnings.warn(
            f"Suppressed effect/CI/p display for {suppressed_null_triplets} predictor-outcome rows due to missing values.",
            UserWarning,
            stacklevel=2,
        )

    if footer_text:
        wrapped = textwrap.wrap(str(footer_text), width=150)
        max_lines = 3
        if len(wrapped) > max_lines:
            wrapped = wrapped[:max_lines]
            if wrapped[-1]:
                wrapped[-1] = wrapped[-1].rstrip(". ") + "..."
        footer_display = "\n".join(wrapped)
        fig.text(
            0.5,
            0.01,
            footer_display,
            ha="center",
            va="bottom",
            fontsize=render_font_size * 0.9,
            color="dimgray",
            style="italic",
        )

    frame = Rectangle(
        (0.12, 0.005),
        0.8,
        0.9,
        transform=fig.transFigure,
        color="black",
        lw=1.2,
        fill=False,
    )
    fig.patches.append(frame)

    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    if show:
        plt.show()

    return fig, (ax_text, ax_forest)
