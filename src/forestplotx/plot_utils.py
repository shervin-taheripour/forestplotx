import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FuncFormatter, NullLocator, NullFormatter
from matplotlib.patches import Rectangle


def _normalize_model_output(df, model_type):
    """
    Normalize model output to standardized columns and apply
    model-appropriate transformations.

    Parameters
    ----------
    df : pd.DataFrame
        Model output containing an effect column (one of "OR", "Ratio",
        "Estimate", "beta", "Coef", "effect"), plus "CI_low"/"ci_low",
        "CI_high"/"ci_high", and "p_value".
    model_type : {"binom", "exp", "linear", "ordinal"}

    Returns
    -------
    clean_df : pd.DataFrame
        Copy with columns renamed to ``effect``, ``ci_low``, ``ci_high``,
        ``p_value``.  For binom/exp/ordinal the effect and CI columns are
        exponentiated.  For ordinal, threshold/cutpoint/intercept rows are
        removed.
    config : dict
        Keys: ``x_label``, ``reference_line``, ``use_log``, ``effect_label``.
    """
    _EFFECT_CANDIDATES = ["OR", "Ratio", "Estimate", "beta", "Coef", "effect"]

    _CONFIG = {
        "binom": {
            "x_label": "Odds Ratio (log scale)",
            "reference_line": 1.0,
            "use_log": True,
            "effect_label": "OR",
        },
        "exp": {
            "x_label": "Relative Mean",
            "reference_line": 1.0,
            "use_log": False,
            "effect_label": "Ratio",
        },
        "linear": {
            "x_label": "Effect Size (linear scale)",
            "reference_line": 0.0,
            "use_log": False,
            "effect_label": "Coef",
        },
        "ordinal": {
            "x_label": "Odds Ratio (ordinal logit)",
            "reference_line": 1.0,
            "use_log": True,
            "effect_label": "OR",
        },
    }

    if model_type not in _CONFIG:
        raise ValueError(
            f"Unknown model_type '{model_type}'. "
            f"Use one of: {list(_CONFIG.keys())}"
        )

    config = _CONFIG[model_type]
    df = df.copy()

    # --- Detect effect column ------------------------------------------------
    effect_col = None
    for candidate in _EFFECT_CANDIDATES:
        if candidate in df.columns:
            effect_col = candidate
            break

    if effect_col is None:
        raise ValueError(
            f"No effect column found. Expected one of: {_EFFECT_CANDIDATES}"
        )

    # --- Rename to standard names --------------------------------------------
    rename = {}
    if effect_col != "effect":
        rename[effect_col] = "effect"
    if "CI_low" in df.columns:
        rename["CI_low"] = "ci_low"
    if "CI_high" in df.columns:
        rename["CI_high"] = "ci_high"
    if rename:
        df = df.rename(columns=rename)

    # --- Ordinal: drop threshold / cutpoint / intercept rows -----------------
    if model_type == "ordinal":
        if "predictor" not in df.columns:
            raise ValueError("Ordinal model requires a 'predictor' column.")
        mask = df["predictor"].str.contains(
            r"(?i)^(?:threshold|cutpoint|intercept)", na=False, regex=True
        )
        df = df[~mask]

    # --- Exponentiate for non-linear link functions --------------------------
    if (
        model_type in ("binom", "ordinal", "exp")
        and effect_col in ("Estimate", "beta", "Coef")
    ):
        for col in ("effect", "ci_low", "ci_high"):
            if col in df.columns:
                df[col] = np.exp(df[col])

    return df, config


def detailed_forest_plot(
    df_final, 
    cat_palette=None, 
    outcomes=None, 
    save=False, 
    filename="forestplot.png",   
    model_type="binom",   
    table_only=False,
    legend_labels=None,
    footer_text=None,
    FONT_SIZE: int = 14,
    BLOCK_SPACING: float = 6.0,
    binom_threshold=(0.5, 2.0),
    exp_threshold=(0.8, 1.2),
    linear_threshold=0.5,
    base_decimals=2,
    max_decimals=4,
    show_general_stats: bool = True,
    bold_override: dict | None = None,
    num_ticks: int = 5,
):
    """
    Create a detailed forest plot with table and optional footer + frame.
    """
    bold_override = bold_override or {}
    BLOCK_X = {
        "predictor": 0,
        "general": BLOCK_SPACING * 1.2,
        "outcome1": BLOCK_SPACING * 1.9,
        "outcome2": BLOCK_SPACING * 2.9,
    }
    if not show_general_stats:
        BLOCK_X["outcome1"] = BLOCK_SPACING * 1.2
        BLOCK_X["outcome2"] = BLOCK_SPACING * 2.2

    GENERAL_OFFSETS = [0, 1.3, 2.6]
    OUTCOME_OFFSETS = [0, 2.0, 4.1]

    if outcomes is None:
        outcomes = df_final['outcome'].unique().tolist()
    if len(outcomes) > 2:
        outcomes = outcomes[:2]
    has_second = len(outcomes) == 2

    # --- Normalize model output ----------------------------------------------
    df_final, plot_config = _normalize_model_output(df_final, model_type)
    effect_label = plot_config["effect_label"]
    ref_val      = plot_config["reference_line"]
    use_log      = plot_config["use_log"]
    xlabel       = plot_config["x_label"]
    ci_label     = "95% CI"

    def format_effect_ci_p(eff, lo, hi, p):
        d = base_decimals
        eff_s = f"{eff:.{d}f}" if pd.notnull(eff) else ""
        lo_s  = f"{lo:.{d}f}"  if pd.notnull(lo)  else ""
        hi_s  = f"{hi:.{d}f}"  if pd.notnull(hi)  else ""
        ci_s  = f"{lo_s}–{hi_s}" if lo_s and hi_s else ""
        p_s   = "" if pd.isnull(p) else ("<0.001" if p < 0.001 else f"{p:.3f}")
        return eff_s, ci_s, p_s, d

    if 'category' in df_final.columns and df_final['category'].notna().any():
        cat_order = list(df_final['category'].dropna().unique())
        table_rows, row_is_cat, row_cats = [], [], []
        for cat in cat_order:
            table_rows.append({'predictor': cat, 'is_cat': True})
            row_is_cat.append(True)
            row_cats.append(cat)
            preds = df_final.loc[df_final['category'] == cat, 'predictor'].unique()
            for pred in preds:
                table_rows.append({'predictor': pred, 'is_cat': False})
                row_is_cat.append(False)
                row_cats.append(cat)
    else:
        preds = df_final['predictor'].dropna().unique()
        table_rows = [{'predictor': pred, 'is_cat': False} for pred in preds]
        row_is_cat = [False] * len(preds)
        row_cats = ['Uncategorized'] * len(preds)

    n = len(table_rows)
    if n == 0:
        raise ValueError("No rows to plot! Check DataFrame structure.")

    if table_only:
        fig, ax_text = plt.subplots(1, 1, figsize=(15, 0.3 * n + 1.5))
        ax_forest = None
    else:
        fig, (ax_text, ax_forest) = plt.subplots(
            1, 2, figsize=(16, 0.3 * n + 1.5),
            gridspec_kw={"width_ratios": [1.9, 1.1]}
        )
        plt.subplots_adjust(wspace=0.02)
        plt.subplots_adjust(bottom=0.12)


    if cat_palette:
        for y, (cat, is_cat) in enumerate(zip(row_cats, row_is_cat)):
            color = cat_palette.get(cat, "#f5f5f5")
            for ax in [ax_text, ax_forest] if ax_forest else [ax_text]:
                ax.axhspan(y - 0.5, y + 0.5, color=color, alpha=0.13, zorder=-2)

    header_row_1, header_row_2 = -2.2, -1.0
    ax_text.text(BLOCK_X["predictor"], header_row_2, "Predictor",
                 ha='left', va='center', fontweight='bold', fontsize=FONT_SIZE)
    if show_general_stats:
        for i, label in enumerate(["n", "N", "Freq"]):
            ax_text.text(BLOCK_X["general"] + GENERAL_OFFSETS[i], header_row_2, label,
                         ha='center', va='center', fontweight='bold', fontsize=FONT_SIZE)
    ax_text.text(BLOCK_X["outcome1"] + OUTCOME_OFFSETS[1], header_row_1, outcomes[0],
                 ha='center', va='center', fontweight='bold', fontsize=FONT_SIZE)
    for i, label in enumerate([effect_label, ci_label, "p"]):
        ax_text.text(BLOCK_X["outcome1"] + OUTCOME_OFFSETS[i], header_row_2, label,
                     ha='center', va='center', fontweight='bold', fontsize=FONT_SIZE)
    if has_second:
        ax_text.text(BLOCK_X["outcome2"] + OUTCOME_OFFSETS[1], header_row_1, outcomes[1],
                     ha='center', va='center', fontweight='bold', fontsize=FONT_SIZE)
        for i, label in enumerate([effect_label, ci_label, "p"]):
            ax_text.text(BLOCK_X["outcome2"] + OUTCOME_OFFSETS[i], header_row_2, label,
                         ha='center', va='center', fontweight='bold', fontsize=FONT_SIZE)

    for y, row in enumerate(table_rows):
        is_cat, pred = row['is_cat'], row['predictor']
        if is_cat:
            ax_text.text(BLOCK_X["predictor"], y, pred, ha='left', va='center',
                         fontsize=FONT_SIZE, fontweight='bold')
            continue

        dfrow_pred = df_final[df_final['predictor'] == pred]
        if dfrow_pred.empty:
            continue
        dfrow = dfrow_pred.iloc[0]
        style = dict(fontsize=FONT_SIZE, fontweight='normal')

        is_null = pd.isna(dfrow.get("effect")) or pd.isna(dfrow.get("ci_low")) or pd.isna(dfrow.get("ci_high"))
        text_color = "#A0A0A0" if is_null else "black"

        ax_text.text(BLOCK_X["predictor"], y, pred, ha='left', va='center', color=text_color, **style)

        if show_general_stats:
            n_val = f"{int(dfrow['n'])}" if 'n' in dfrow and pd.notnull(dfrow['n']) else ""
            N_val = f"{int(dfrow['N'])}" if 'N' in dfrow and pd.notnull(dfrow['N']) else ""
            freq_val = ""
            if n_val and N_val:
                try:
                    freq_val = f"{(int(dfrow['n']) / int(dfrow['N'])):.1%}"
                except Exception:
                    pass
            for i, val in enumerate([n_val, N_val, freq_val]):
                ax_text.text(BLOCK_X["general"] + GENERAL_OFFSETS[i], y, val,
                             ha='center', va='center', fontsize=FONT_SIZE, color=text_color)

        for k, outcome in enumerate(outcomes):
            out_block = "outcome1" if k == 0 else "outcome2"
            dfrow = df_final[(df_final['predictor'] == pred) & (df_final['outcome'] == outcome)]
            if dfrow.empty:
                continue
            dfrow = dfrow.iloc[0]
            eff, lo, hi, p = dfrow.get("effect", np.nan), dfrow.get("ci_low", np.nan), dfrow.get("ci_high", np.nan), dfrow.get("p_value", np.nan)
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
                    y, val,
                    ha='center', va='center',
                    fontsize=FONT_SIZE,
                    fontweight='bold' if final_bold else 'normal',
                    color=text_color
                )

    rightmost = (BLOCK_X["outcome2"] + OUTCOME_OFFSETS[2] + 2.0) if has_second else (BLOCK_X["outcome1"] + OUTCOME_OFFSETS[2] + 2.0)
    ax_text.set_xlim(-0.5, rightmost)
    ax_text.set_ylim(n - 0.5, -2.8)
    ax_text.axis('off')

    if ax_forest is not None:
        colors, markers = ['#212427', '#D8CBBB'], ['o', 's']
        handles, labels, Y_OFFSET = [], [], 0.10
        eff_all, lo_all, hi_all = [], [], []
        for j, outcome in enumerate(outcomes):
            y_points, eff_vals, cil, cih = [], [], [], []
            for y, row in enumerate(table_rows):
                if row['is_cat']:
                    continue
                pred = row['predictor']
                dfrow = df_final[(df_final['predictor'] == pred) & (df_final['outcome'] == outcome)]
                if dfrow.empty:
                    continue
                dfrow = dfrow.iloc[0]
                eff, lo, hi = dfrow.get("effect", np.nan), dfrow.get("ci_low", np.nan), dfrow.get("ci_high", np.nan)
                is_null = pd.isna(eff) or pd.isna(lo) or pd.isna(hi)
                color = "#C7C7C7" if is_null else colors[j]

                if not is_null:
                    eff_vals.append(eff)
                    cil.append(eff - lo)
                    cih.append(hi - eff)
                    y_points.append(y)
                    eff_all += [eff]; lo_all += [lo]; hi_all += [hi]
                else:
                    ax_forest.plot([ref_val], [y], marker=markers[j], color=color, alpha=0.6, markersize=5, zorder=2)

            if eff_vals:
                offset_sign = -1 if j == 1 else 1
                y_points_offset = [y + Y_OFFSET * offset_sign for y in y_points]
                h = ax_forest.errorbar(
                    eff_vals, y_points_offset, xerr=[cil, cih],
                    fmt=markers[j], color=colors[j], ecolor=colors[j],
                    capsize=3, markersize=5, lw=1.7,
                    label=(legend_labels[j] if legend_labels and j < len(legend_labels) else outcome)
                )
                handles.append(h.lines[0]); labels.append(outcome)
        ax_forest.axvline(ref_val, color='#910C07', lw=1.2, ls='--')
        ax_forest.set_yticks([]); ax_forest.set_ylim(n - 0.5, -2.8)
        ax_forest.set_xlabel(xlabel, fontsize=FONT_SIZE)
        if len(lo_all) and len(hi_all):
            xmin, xmax = np.min(lo_all), np.max(hi_all)
            xmin = max(np.min(lo_all), 0.2)
            xmax = max(np.max(hi_all), xmin * 2.5)
            ax_forest.set_xscale("log" if use_log else "linear")
            ax_forest.set_xlim(xmin * 0.85, xmax * 1.15)
            from matplotlib.ticker import MaxNLocator

            if model_type in ("binom", "exp"):
                if use_log:
                    # Step 1: Get your axis min/max
                    xlim = ax_forest.get_xlim()
                    xmin, xmax = xlim
                    # Step 2: Predefined "nice" tick values (edit as needed!)
                    nice_ticks = np.array([0.2, 0.4, 0.6, 0.8, 1, 1.2, 1.5, 2, 3, 5])
                    # Only use those within the limits
                    ticks = nice_ticks[(nice_ticks >= xmin) & (nice_ticks <= xmax)]
                    # Step 3: Always add reference value if not included
                    if ref_val not in ticks and xmin < ref_val < xmax:
                        ticks = np.sort(np.append(ticks, ref_val))
                    # Step 4: Apply
                    ax_forest.set_xticks(ticks)
                    ax_forest.set_xticklabels([f"{t:.2f}".rstrip("0").rstrip(".") for t in ticks])
                    ax_forest.xaxis.set_minor_locator(NullLocator())
                    ax_forest.xaxis.set_minor_formatter(NullFormatter())
                else:
                    # Force nice linear ticks in tight range
                    ticks = [0.8, 0.9, 1.0, 1.1, 1.2]
                    ax_forest.set_xticks(ticks)
                    ax_forest.set_xlim(0.75, 1.25)  # Expand limits just a touch
                    ax_forest.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:.2f}".rstrip("0").rstrip(".")))
            else:
                rng = xmax - xmin
                if rng == 0:
                    rng = max(1.0, abs(xmax) * 0.2)
                ax_forest.set_xlim(xmin - 0.1 * rng, xmax + 0.1 * rng)
                ax_forest.xaxis.set_major_locator(plt.MaxNLocator(6))
        # Only set formatter for linear models (no-op for log since above replaces it)
        if not use_log:
            ax_forest.xaxis.set_major_formatter(FuncFormatter(
                lambda x, pos: f"{x:.2f}".rstrip("0").rstrip(".")
            ))
        for spine in ["top", "right", "left"]:
            ax_forest.spines[spine].set_visible(False)
        ax_forest.legend(handles, labels, loc="upper left", fontsize=FONT_SIZE)

    if footer_text:
        fig.text(0.5, 0.01, footer_text, ha='center', va='bottom', fontsize=FONT_SIZE * 0.9,
                 color='dimgray', style='italic')

    frame = Rectangle((0.12, 0.005), 0.8, 0.9,
                      transform=fig.transFigure, color="black", lw=1.2, fill=False)
    fig.patches.append(frame)

    if save:
        plt.savefig(filename, dpi=300, bbox_inches="tight")

    plt.show()

    if save:
        plt.close(fig)
