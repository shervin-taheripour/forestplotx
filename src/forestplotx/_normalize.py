import numpy as np

DEFAULT_LINK = {
    "binom": "logit",
    "ordinal": "logit",
    "gamma": "log",
    "linear": "identity",
}


def _normalize_model_output(df, model_type, link=None):
    """
    Normalize model output to standardized columns and apply
    link-driven transformations.
    """

    _EFFECT_CANDIDATES = ["OR", "Ratio", "Estimate", "beta", "Coef", "effect"]

    if model_type not in DEFAULT_LINK:
        raise ValueError(
            f"Unknown model_type '{model_type}'. "
            f"Use one of: {list(DEFAULT_LINK.keys())}"
        )

    # ---- Resolve link -------------------------------------------------------
    resolved_link = link or DEFAULT_LINK[model_type]

    # ---- Config derived from link ------------------------------------------
    if resolved_link in ("log", "logit"):
        reference_line = 1.0
        use_log = True
        exponentiate = True
    elif resolved_link == "identity":
        reference_line = 0.0
        use_log = False
        exponentiate = False
    else:
        raise ValueError(f"Unsupported link '{resolved_link}'")

    df = df.copy()

    # ---- Detect effect column ----------------------------------------------
    effect_col = None
    for candidate in _EFFECT_CANDIDATES:
        if candidate in df.columns:
            effect_col = candidate
            break

    if effect_col is None:
        raise ValueError(
            f"No effect column found. Expected one of: {_EFFECT_CANDIDATES}"
        )

    # ---- Standardize column names ------------------------------------------
    rename = {}
    if effect_col != "effect":
        rename[effect_col] = "effect"
    if "CI_low" in df.columns:
        rename["CI_low"] = "ci_low"
    if "CI_high" in df.columns:
        rename["CI_high"] = "ci_high"
    if rename:
        df = df.rename(columns=rename)

    # ---- Ordinal: remove threshold rows ------------------------------------
    if model_type == "ordinal":
        if "predictor" not in df.columns:
            raise ValueError("Ordinal model requires a 'predictor' column.")
        mask = df["predictor"].str.contains(
            r"(?i)^(?:threshold|cutpoint|intercept)", na=False, regex=True
        )
        df = df[~mask]

    # ---- Apply exponentiation based on link --------------------------------
    if exponentiate:
        for col in ("effect", "ci_low", "ci_high"):
            if col in df.columns:
                df[col] = np.exp(df[col])

    config = {
        "x_label": {
            "logit": "Odds Ratio",
            "log": "Ratio",
            "identity": "Effect Size",
        }[resolved_link],
        "reference_line": reference_line,
        "use_log": use_log,
        "link": resolved_link,
        "effect_label": {
            "logit": "OR",
            "log": "Ratio",
            "identity": "Coef",
        }[resolved_link],
    }

    return df, config
