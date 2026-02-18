import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from os import PathLike
from matplotlib.patches import Rectangle

from ._axes_config import configure_forest_axis
from ._layout import build_row_layout
from ._normalize import _normalize_model_output


def forest_plot(
    df_final,
    outcomes=None,
    save=None,
    model_type="binom",
    link=None,
    exponentiate: bool | None = None,
    table_only=False,
    legend_labels=None,
    point_colors: list[str] | None = None,
    footer_text=None,
    block_spacing: float = 6.0,
    tick_style: str = "decimal",
    clip_outliers: bool = False,
    clip_quantiles: tuple[float, float] = (0.02, 0.98),
    base_decimals=2,
    show: bool = True,
    show_general_stats: bool = True,
    bold_override: dict | None = None,
):
    """
    Create a detailed forest plot with table and optional footer + frame.
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
        outcomes = df_final["outcome"].unique().tolist()
    if len(outcomes) > 2:
        outcomes = outcomes[:2]
    has_second = len(outcomes) == 2

    # --- Normalize model output ----------------------------------------------
    df_final, plot_config = _normalize_model_output(
        df_final,
        model_type=model_type,
        link=link,
        exponentiate=exponentiate,
    )
    effect_label = plot_config["effect_label"]
    ref_val = plot_config["reference_line"]
    use_log = plot_config["use_log"]
    xlabel = plot_config["x_label"]
    ci_label = "95% CI"

    def format_effect_ci_p(eff, lo, hi, p):
        d = base_decimals
        eff_s = f"{eff:.{d}f}" if pd.notnull(eff) else ""
        lo_s = f"{lo:.{d}f}" if pd.notnull(lo) else ""
        hi_s = f"{hi:.{d}f}" if pd.notnull(hi) else ""
        ci_s = f"{lo_s}\u2013{hi_s}" if lo_s and hi_s else ""
        p_s = "" if pd.isnull(p) else ("<0.001" if p < 0.001 else f"{p:.3f}")
        return eff_s, ci_s, p_s, d

    layout = build_row_layout(df_final)
    table_rows = layout["rows"].to_dict("records")
    y_positions = layout["y_positions"]
    n = layout["meta"]["n"]

    # 4-case layout presets:
    # (show_general_stats, has_second_outcome) -> geometry config.
    # For now, all four cases intentionally share the same values.
    layout_presets = {
        (True, True): {
            "block_mult": {"general": 1.09, "outcome1": 1.74, "outcome2": 2.74},
            "general_offsets": [0.0, 1.2, 2.7],
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
            "fig_width": 13,
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
    render_font_size = 10
    BLOCK_X = {"predictor": 0.0}
    if show_general_stats:
        BLOCK_X["general"] = block_spacing * layout_cfg["block_mult"]["general"]
    BLOCK_X["outcome1"] = block_spacing * layout_cfg["block_mult"]["outcome1"]
    BLOCK_X["outcome2"] = block_spacing * layout_cfg["block_mult"]["outcome2"]
    GENERAL_OFFSETS = layout_cfg["general_offsets"]
    OUTCOME_OFFSETS = layout_cfg["outcome_offsets"]

    if table_only:
        fig, ax_text = plt.subplots(1, 1, figsize=(15, 0.3 * n + 1.5))
        ax_forest = None
    else:
        fig, (ax_text, ax_forest) = plt.subplots(
            1,
            2,
            figsize=(layout_cfg["fig_width"], 0.3 * n + 1.5),
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

        dfrow_pred = df_final[df_final["predictor"] == pred]
        if dfrow_pred.empty:
            continue
        dfrow = dfrow_pred.iloc[0]
        style = dict(fontsize=render_font_size, fontweight="normal")

        is_null = (
            pd.isna(dfrow.get("effect"))
            or pd.isna(dfrow.get("ci_low"))
            or pd.isna(dfrow.get("ci_high"))
        )
        text_color = "#A0A0A0" if is_null else "black"

        ax_text.text(
            BLOCK_X["predictor"],
            y,
            pred,
            ha="left",
            va="center",
            color=text_color,
            **style,
        )

        if show_general_stats:
            n_val = f"{int(dfrow['n'])}" if "n" in dfrow and pd.notnull(dfrow["n"]) else ""
            N_val = f"{int(dfrow['N'])}" if "N" in dfrow and pd.notnull(dfrow["N"]) else ""
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
            dfrow = df_final[
                (df_final["predictor"] == pred) & (df_final["outcome"] == outcome)
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
            effect_val, ci, pval, _ = format_effect_ci_p(eff, lo, hi, p)
            p_bold = pd.notnull(p) and p < 0.05
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
                dfrow = df_final[
                    (df_final["predictor"] == pred) & (df_final["outcome"] == outcome)
                ]
                if dfrow.empty:
                    continue
                dfrow = dfrow.iloc[0]
                eff, lo, hi = (
                    dfrow.get("effect", np.nan),
                    dfrow.get("ci_low", np.nan),
                    dfrow.get("ci_high", np.nan),
                )
                is_null = pd.isna(eff) or pd.isna(lo) or pd.isna(hi)
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
                dfrow = df_final[
                    (df_final["predictor"] == pred) & (df_final["outcome"] == outcome)
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

    if footer_text:
        fig.text(
            0.5,
            0.01,
            footer_text,
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
