import numpy as np
import pandas as pd
import pytest

from forestplotx.plot_utils import _normalize_model_output


# -----------------------------------------------------------
# Helper
# -----------------------------------------------------------

def make_base_df(effect_col="Estimate", effect_val=0.5):
    return pd.DataFrame({
        effect_col: [effect_val],
        "CI_low": [effect_val - 0.1],
        "CI_high": [effect_val + 0.1],
        "p_value": [0.01],
        "predictor": ["x1"],
        "outcome": ["y1"]
    })


# -----------------------------------------------------------
# Linear: no exponentiation
# -----------------------------------------------------------

def test_linear_no_exponentiation():
    df = make_base_df("Estimate", 0.5)
    clean_df, config = _normalize_model_output(df, "linear")

    assert np.isclose(clean_df["effect"].iloc[0], 0.5)
    assert config["reference_line"] == 0.0
    assert config["use_log"] is False


# -----------------------------------------------------------
# Binomial: exponentiation applied
# -----------------------------------------------------------

def test_binom_exponentiation():
    df = make_base_df("Estimate", 0.0)  # exp(0) = 1
    clean_df, config = _normalize_model_output(df, "binom")

    assert np.isclose(clean_df["effect"].iloc[0], 1.0)
    assert config["reference_line"] == 1.0
    assert config["use_log"] is True


# -----------------------------------------------------------
# Exp model: exponentiation applied
# -----------------------------------------------------------

def test_exp_exponentiation():
    df = make_base_df("Estimate", 0.0)
    clean_df, _ = _normalize_model_output(df, "exp")

    assert np.isclose(clean_df["effect"].iloc[0], 1.0)


# -----------------------------------------------------------
# Ordinal: exponentiation + threshold removal
# -----------------------------------------------------------

def test_ordinal_drops_threshold_rows():
    df = pd.DataFrame({
        "Estimate": [0.0, 0.5],
        "CI_low": [-0.1, 0.4],
        "CI_high": [0.1, 0.6],
        "p_value": [0.5, 0.01],
        "predictor": ["threshold1", "x1"],
        "outcome": ["y1", "y1"]
    })

    clean_df, _ = _normalize_model_output(df, "ordinal")

    assert len(clean_df) == 1
    assert clean_df["predictor"].iloc[0] == "x1"


# -----------------------------------------------------------
# Exponentiation guard
# -----------------------------------------------------------

def test_log_link_exponentiates():
    df = make_base_df("Estimate", 0.0)
    clean_df, config = _normalize_model_output(df, "gamma", link="log")

    assert clean_df["effect"].iloc[0] == 1.0
    assert config["reference_line"] == 1.0
    assert config["use_log"] is True

def test_identity_link_no_exponentiation():
    df = make_base_df("Estimate", 0.5)
    clean_df, config = _normalize_model_output(df, "linear", link="identity")

    assert clean_df["effect"].iloc[0] == 0.5
    assert config["reference_line"] == 0.0
    assert config["use_log"] is False


# -----------------------------------------------------------
# Missing effect column
# -----------------------------------------------------------

def test_missing_effect_column_raises():
    df = pd.DataFrame({
        "CI_low": [0.1],
        "CI_high": [0.2],
        "predictor": ["x1"],
        "outcome": ["y1"]
    })

    with pytest.raises(ValueError):
        _normalize_model_output(df, "linear")


# -----------------------------------------------------------
# Invalid model_type
# -----------------------------------------------------------

def test_invalid_model_type_raises():
    df = make_base_df()

    with pytest.raises(ValueError):
        _normalize_model_output(df, "invalid_model")
