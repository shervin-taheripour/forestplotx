import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import pytest

from forestplotx import forest_plot


def test_forest_plot_smoke_returns_figure_and_axes():
    df = pd.DataFrame(
        {
            "predictor": ["x1", "x2", "x1", "x2"],
            "outcome": ["y1", "y1", "y2", "y2"],
            "Estimate": [0.0, 0.2, 0.1, -0.1],
            "CI_low": [-0.1, 0.1, 0.0, -0.2],
            "CI_high": [0.1, 0.3, 0.2, 0.0],
            "p_value": [0.2, 0.01, 0.05, 0.3],
        }
    )

    with pytest.warns(UserWarning):
        fig, axes = forest_plot(
            df=df,
            outcomes=["y1", "y2"],
            model_type="binom",
            show=False,
        )

    ax_text, ax_forest = axes
    assert fig is not None
    assert ax_text is not None
    assert ax_forest is not None

    plt.close(fig)


def test_forest_plot_single_point_color_with_two_outcomes():
    df = pd.DataFrame(
        {
            "predictor": ["x1", "x2", "x1", "x2"],
            "outcome": ["y1", "y1", "y2", "y2"],
            "Estimate": [0.0, 0.2, 0.1, -0.1],
            "CI_low": [-0.1, 0.1, 0.0, -0.2],
            "CI_high": [0.1, 0.3, 0.2, 0.0],
            "p_value": [0.2, 0.01, 0.05, 0.3],
        }
    )

    with pytest.warns(UserWarning):
        fig, axes = forest_plot(
            df=df,
            outcomes=["y1", "y2"],
            model_type="binom",
            point_colors=["#2C5F8A"],
            show=False,
        )

    ax_text, ax_forest = axes
    assert fig is not None
    assert ax_text is not None
    assert ax_forest is not None

    plt.close(fig)


def test_forest_plot_truncates_long_predictor_label_with_warning():
    long_name = "Pneumonoultramicroscopicsilicovolcanoconiosis"
    df = pd.DataFrame(
        {
            "predictor": [long_name, long_name],
            "outcome": ["y1", "y2"],
            "Estimate": [0.1, -0.1],
            "CI_low": [0.0, -0.2],
            "CI_high": [0.2, 0.0],
            "p_value": [0.2, 0.3],
        }
    )

    with pytest.warns(UserWarning, match="Predictor label length exceeded cap"):
        fig, axes = forest_plot(
            df=df,
            outcomes=["y1", "y2"],
            model_type="linear",
            show_general_stats=False,
            show=False,
        )

    ax_text, _ = axes
    texts = [t.get_text() for t in ax_text.texts]
    assert any(txt.endswith("...") for txt in texts)
    assert long_name not in texts
    plt.close(fig)


def test_forest_plot_compacts_large_counts_with_general_stats():
    df = pd.DataFrame(
        {
            "predictor": ["x1", "x2", "x1", "x2"],
            "outcome": ["y1", "y1", "y2", "y2"],
            "Estimate": [0.0, 0.2, 0.1, -0.1],
            "CI_low": [-0.1, 0.1, 0.0, -0.2],
            "CI_high": [0.1, 0.3, 0.2, 0.0],
            "p_value": [0.2, 0.01, 0.05, 0.3],
            "n": [2400, 8600, 2400, 8600],
            "N": [78600, 50700, 78600, 50700],
        }
    )

    fig, axes = forest_plot(
        df=df,
        outcomes=["y1", "y2"],
        model_type="linear",
        show_general_stats=True,
        show=False,
    )

    ax_text, _ = axes
    texts = [t.get_text() for t in ax_text.texts]
    assert any("k" in txt for txt in texts)
    plt.close(fig)
