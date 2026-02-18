"""
Tests for forestplotx._axes_config:
  - configure_forest_axis
  - _nice_linear_step   (pure function)
  - _decimals_from_ticks (pure function)

Covers: all model types, log/linear scale, reference line, ticks,
        x/y limits, spine visibility, edge cases, show_general_stats.
No image comparison — structural and behavioral assertions only.
"""
import math

import matplotlib
matplotlib.use("Agg")  # non-interactive backend; must be set before pyplot import
import matplotlib.pyplot as plt
import numpy as np
import pytest

from forestplotx._axes_config import (
    _decimals_from_ticks,
    _nice_linear_step,
    configure_forest_axis,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def ax():
    fig, a = plt.subplots()
    yield a
    plt.close(fig)


# ── Threshold builder ─────────────────────────────────────────────────────────

_LINK_DEFAULTS = {
    "logit":    {"reference_line": 1.0, "use_log": True,  "x_label": "Odds Ratio"},
    "log":      {"reference_line": 1.0, "use_log": True,  "x_label": "Ratio"},
    "identity": {"reference_line": 0.0, "use_log": False, "x_label": "Effect Size"},
}


def thresholds_for(link, lo=None, hi=None, ref=None, y_limits=None, tick_style="decimal"):
    d = dict(_LINK_DEFAULTS[link])
    if ref is not None:
        d["reference_line"] = ref
    d["lo_all"] = lo or []
    d["hi_all"] = hi or []
    if y_limits is not None:
        d["y_limits"] = y_limits
    d["tick_style"] = tick_style
    return d


def _refline_xvalues(a):
    """Collect all x-values from Line2D objects on the axis."""
    xs = []
    for line in a.lines:
        xs.extend(line.get_xdata())
    return xs


# ═══════════════════════════════════════════════════════════════════════════════
# Pure-function unit tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestNiceLinearStep:

    @pytest.mark.parametrize(
        "raw, expected",
        [
            (0.05,  0.05),   # fraction=5 → nice=5 → 5*0.01
            (0.3,   0.5),    # fraction=3 → nice=5 → 5*0.1
            (0.8,   1.0),    # fraction=8 → nice=10 → 10*0.1
            (1.5,   2.0),    # fraction=1.5 → nice=2 → 2*1
            (3.0,   5.0),    # fraction=3 → nice=5 → 5*1
            (7.0,   10.0),   # fraction=7 → nice=10 → 10*1
            (15.0,  20.0),   # fraction=1.5 → nice=2 → 2*10
            (55.0, 100.0),   # fraction=5.5 → nice=10 → 10*10
        ],
    )
    def test_returns_nice_step(self, raw, expected):
        assert _nice_linear_step(raw) == pytest.approx(expected)

    def test_zero_input_returns_one(self):
        assert _nice_linear_step(0) == 1.0

    def test_negative_input_returns_one(self):
        assert _nice_linear_step(-5.0) == 1.0

    def test_very_small_positive_value(self):
        result = _nice_linear_step(1e-6)
        assert result > 0
        assert math.isfinite(result)


class TestDecimalsFromTicks:

    def test_single_tick_returns_two(self):
        assert _decimals_from_ticks(np.array([1.0])) == 2

    def test_empty_array_returns_two(self):
        assert _decimals_from_ticks(np.array([])) == 2

    @pytest.mark.parametrize(
        "ticks, expected",
        [
            ([1, 2, 3],          0),   # step=1  → 0 decimals
            ([0.5, 1.0, 1.5],    1),   # step=0.5 → 1 decimal
            ([0.1, 0.2, 0.3],    1),   # step=0.1 → 1 decimal
            ([0.01, 0.02, 0.03], 2),   # step=0.01 → 2 decimals
        ],
    )
    def test_decimals_inferred_from_spacing(self, ticks, expected):
        result = _decimals_from_ticks(np.array(ticks, dtype=float))
        assert result == expected

    def test_max_decimals_cap_is_respected(self):
        # step=0.0001 → raw=4, should be capped by max_decimals=3
        ticks = np.array([0.0001, 0.0002, 0.0003])
        result = _decimals_from_ticks(ticks, max_decimals=3)
        assert result <= 3

    def test_result_is_non_negative(self):
        ticks = np.array([100.0, 200.0, 300.0])
        result = _decimals_from_ticks(ticks)
        assert result >= 0


# ═══════════════════════════════════════════════════════════════════════════════
# configure_forest_axis — reference line
# ═══════════════════════════════════════════════════════════════════════════════

class TestReferenceLine:

    @pytest.mark.parametrize(
        "link, expected_ref",
        [
            ("logit",    1.0),
            ("log",      1.0),
            ("identity", 0.0),
        ],
    )
    def test_axvline_placed_at_reference_value(self, ax, link, expected_ref):
        configure_forest_axis(
            ax=ax,
            model_type="binom",
            link=link,
            thresholds=thresholds_for(link),
            num_ticks=5,
            font_size=12,
            show_general_stats=True,
        )
        xs = _refline_xvalues(ax)
        assert any(math.isclose(x, expected_ref, rel_tol=1e-9) for x in xs), (
            f"No line at x={expected_ref}; found x-values: {xs}"
        )

    def test_reference_line_color_is_red(self, ax):
        configure_forest_axis(
            ax=ax,
            model_type="binom",
            link="logit",
            thresholds=thresholds_for("logit"),
            num_ticks=5,
            font_size=12,
            show_general_stats=True,
        )
        # axvline is added first → first Line2D in ax.lines
        assert ax.lines[0].get_color() == "#910C07"

    def test_reference_line_is_dashed(self, ax):
        configure_forest_axis(
            ax=ax,
            model_type="binom",
            link="logit",
            thresholds=thresholds_for("logit"),
            num_ticks=5,
            font_size=12,
            show_general_stats=True,
        )
        assert ax.lines[0].get_linestyle() == "--"

    def test_custom_reference_via_thresholds(self, ax):
        t = thresholds_for("logit", lo=[1.2], hi=[2.5], ref=1.5)
        configure_forest_axis(
            ax=ax,
            model_type="binom",
            link="logit",
            thresholds=t,
            num_ticks=5,
            font_size=12,
            show_general_stats=True,
        )
        xs = _refline_xvalues(ax)
        assert any(math.isclose(x, 1.5, rel_tol=1e-9) for x in xs)


# ═══════════════════════════════════════════════════════════════════════════════
# configure_forest_axis — x-scale
# ═══════════════════════════════════════════════════════════════════════════════

class TestXScale:

    @pytest.mark.parametrize(
        "link, expected_scale",
        [
            ("logit",    "log"),
            ("log",      "log"),
            ("identity", "linear"),
        ],
    )
    def test_xscale_correct_for_link(self, ax, link, expected_scale):
        configure_forest_axis(
            ax=ax,
            model_type="binom",
            link=link,
            thresholds=thresholds_for(link, lo=[0.5, 0.7], hi=[1.5, 2.0]),
            num_ticks=5,
            font_size=12,
            show_general_stats=True,
        )
        assert ax.get_xscale() == expected_scale

    def test_empty_lo_hi_does_not_crash(self, ax):
        configure_forest_axis(
            ax=ax,
            model_type="binom",
            link="logit",
            thresholds=thresholds_for("logit"),  # empty lo/hi
            num_ticks=5,
            font_size=12,
            show_general_stats=True,
        )
        assert ax.get_xscale() in ("linear", "log")

    def test_none_thresholds_does_not_crash(self, ax):
        configure_forest_axis(
            ax=ax,
            model_type="linear",
            link="identity",
            thresholds=None,
            num_ticks=5,
            font_size=12,
            show_general_stats=False,
        )


# ═══════════════════════════════════════════════════════════════════════════════
# configure_forest_axis — x-label
# ═══════════════════════════════════════════════════════════════════════════════

class TestXLabel:

    @pytest.mark.parametrize(
        "link, expected_label",
        [
            ("logit",    "Odds Ratio"),
            ("log",      "Ratio"),
            ("identity", "Effect Size"),
        ],
    )
    def test_xlabel_derived_from_link(self, ax, link, expected_label):
        configure_forest_axis(
            ax=ax,
            model_type="binom",
            link=link,
            thresholds=thresholds_for(link),
            num_ticks=5,
            font_size=12,
            show_general_stats=True,
        )
        assert ax.get_xlabel() == expected_label

    def test_xlabel_overridden_by_thresholds(self, ax):
        t = thresholds_for("logit")
        t["x_label"] = "Custom Label"
        configure_forest_axis(
            ax=ax,
            model_type="binom",
            link="logit",
            thresholds=t,
            num_ticks=5,
            font_size=12,
            show_general_stats=True,
        )
        assert ax.get_xlabel() == "Custom Label"

    def test_xlabel_uses_font_size(self, ax):
        configure_forest_axis(
            ax=ax,
            model_type="binom",
            link="logit",
            thresholds=thresholds_for("logit"),
            num_ticks=5,
            font_size=16,
            show_general_stats=True,
        )
        assert ax.xaxis.label.get_fontsize() == 16


# ═══════════════════════════════════════════════════════════════════════════════
# configure_forest_axis — y-ticks
# ═══════════════════════════════════════════════════════════════════════════════

class TestYTicks:

    @pytest.mark.parametrize("link", ["logit", "log", "identity"])
    def test_yticks_cleared_for_all_links(self, ax, link):
        configure_forest_axis(
            ax=ax,
            model_type="binom",
            link=link,
            thresholds=thresholds_for(link),
            num_ticks=5,
            font_size=12,
            show_general_stats=True,
        )
        assert list(ax.get_yticks()) == []


# ═══════════════════════════════════════════════════════════════════════════════
# configure_forest_axis — y-limits
# ═══════════════════════════════════════════════════════════════════════════════

class TestYLimits:

    def test_ylimits_applied_from_thresholds(self, ax):
        t = thresholds_for("logit", y_limits=(9.5, -2.8))
        configure_forest_axis(
            ax=ax,
            model_type="binom",
            link="logit",
            thresholds=t,
            num_ticks=5,
            font_size=12,
            show_general_stats=True,
        )
        lo, hi = ax.get_ylim()
        assert math.isclose(lo, 9.5, rel_tol=1e-9)
        assert math.isclose(hi, -2.8, rel_tol=1e-9)

    def test_missing_y_limits_key_does_not_crash(self, ax):
        t = thresholds_for("logit")
        assert "y_limits" not in t
        configure_forest_axis(
            ax=ax,
            model_type="binom",
            link="logit",
            thresholds=t,
            num_ticks=5,
            font_size=12,
            show_general_stats=True,
        )  # should not raise


# ═══════════════════════════════════════════════════════════════════════════════
# configure_forest_axis — spine visibility
# ═══════════════════════════════════════════════════════════════════════════════

class TestSpineVisibility:

    @pytest.mark.parametrize("spine", ["top", "right", "left"])
    def test_non_bottom_spines_hidden(self, ax, spine):
        configure_forest_axis(
            ax=ax,
            model_type="binom",
            link="logit",
            thresholds=thresholds_for("logit"),
            num_ticks=5,
            font_size=12,
            show_general_stats=True,
        )
        assert not ax.spines[spine].get_visible(), f"spine '{spine}' should be hidden"

    def test_bottom_spine_remains_visible(self, ax):
        configure_forest_axis(
            ax=ax,
            model_type="binom",
            link="logit",
            thresholds=thresholds_for("logit"),
            num_ticks=5,
            font_size=12,
            show_general_stats=True,
        )
        assert ax.spines["bottom"].get_visible()


# ═══════════════════════════════════════════════════════════════════════════════
# configure_forest_axis — x-limits
# ═══════════════════════════════════════════════════════════════════════════════

class TestXLimits:

    def test_log_xlim_contains_full_data_range(self, ax):
        lo, hi = [0.5, 0.6], [1.5, 2.0]
        configure_forest_axis(
            ax=ax,
            model_type="binom",
            link="logit",
            thresholds=thresholds_for("logit", lo=lo, hi=hi),
            num_ticks=5,
            font_size=12,
            show_general_stats=True,
        )
        xmin, xmax = ax.get_xlim()
        assert xmin <= min(lo)
        assert xmax >= max(hi)

    def test_linear_xlim_contains_full_data_range(self, ax):
        lo, hi = [-0.5, -0.3], [0.5, 0.8]
        configure_forest_axis(
            ax=ax,
            model_type="linear",
            link="identity",
            thresholds=thresholds_for("identity", lo=lo, hi=hi),
            num_ticks=5,
            font_size=12,
            show_general_stats=True,
        )
        xmin, xmax = ax.get_xlim()
        assert xmin <= min(lo)
        assert xmax >= max(hi)

    def test_log_axis_raises_for_nonpositive_ref(self, ax):
        t = thresholds_for("logit", lo=[0.5], hi=[1.5], ref=-1.0)
        with pytest.raises(ValueError, match="positive reference"):
            configure_forest_axis(
                ax=ax,
                model_type="binom",
                link="logit",
                thresholds=t,
                num_ticks=5,
                font_size=12,
                show_general_stats=True,
            )

    def test_linear_single_point_span_zero_does_not_crash(self, ax):
        # Both lo and hi are at the reference value → span=0 edge case
        configure_forest_axis(
            ax=ax,
            model_type="linear",
            link="identity",
            thresholds=thresholds_for("identity", lo=[0.0], hi=[0.0]),
            num_ticks=5,
            font_size=12,
            show_general_stats=True,
        )
        xmin, xmax = ax.get_xlim()
        assert xmin < xmax  # still produces a valid (non-degenerate) range


# ═══════════════════════════════════════════════════════════════════════════════
# configure_forest_axis — return value
# ═══════════════════════════════════════════════════════════════════════════════

class TestReturnValue:

    def test_returns_the_same_axes_object(self, ax):
        result = configure_forest_axis(
            ax=ax,
            model_type="binom",
            link="logit",
            thresholds=thresholds_for("logit"),
            num_ticks=5,
            font_size=12,
            show_general_stats=True,
        )
        assert result is ax


# ═══════════════════════════════════════════════════════════════════════════════
# Parametrized cross-model tests
# ═══════════════════════════════════════════════════════════════════════════════

@pytest.mark.parametrize(
    "model_type, link, lo, hi, expected_scale, expected_ref",
    [
        ("binom",   "logit",    [0.5, 0.8], [1.2, 2.0],  "log",    1.0),
        ("gamma",   "log",      [0.7, 0.9], [1.1, 1.5],  "log",    1.0),
        ("linear",  "identity", [-0.5, 0.1], [0.5, 0.8], "linear", 0.0),
        ("ordinal", "logit",    [0.4, 0.6], [1.3, 1.8],  "log",    1.0),
    ],
)
def test_model_type_end_to_end(
    model_type, link, lo, hi, expected_scale, expected_ref, ax
):
    configure_forest_axis(
        ax=ax,
        model_type=model_type,
        link=link,
        thresholds=thresholds_for(link, lo=lo, hi=hi),
        num_ticks=5,
        font_size=12,
        show_general_stats=True,
    )
    assert ax.get_xscale() == expected_scale
    xs = _refline_xvalues(ax)
    assert any(math.isclose(x, expected_ref, rel_tol=1e-9) for x in xs)
    assert list(ax.get_yticks()) == []
    for spine in ("top", "right", "left"):
        assert not ax.spines[spine].get_visible()


# ═══════════════════════════════════════════════════════════════════════════════
# show_general_stats (API symmetry — parameter is a no-op on the axis)
# ═══════════════════════════════════════════════════════════════════════════════

@pytest.mark.parametrize("show_general_stats", [True, False])
def test_show_general_stats_both_values_produce_identical_xlabel(
    ax, show_general_stats
):
    configure_forest_axis(
        ax=ax,
        model_type="binom",
        link="logit",
        thresholds=thresholds_for("logit", lo=[0.5], hi=[1.5]),
        num_ticks=5,
        font_size=12,
        show_general_stats=show_general_stats,
    )
    assert ax.get_xlabel() == "Odds Ratio"


# ═══════════════════════════════════════════════════════════════════════════════
# Tick count heuristic
# ═══════════════════════════════════════════════════════════════════════════════

@pytest.mark.parametrize("num_ticks", [3, 5, 7])
def test_log_axis_tick_count_is_reasonable(ax, num_ticks):
    configure_forest_axis(
        ax=ax,
        model_type="binom",
        link="logit",
        thresholds=thresholds_for("logit", lo=[0.5, 0.6], hi=[1.5, 2.0]),
        num_ticks=num_ticks,
        font_size=12,
        show_general_stats=True,
    )
    ticks = ax.get_xticks()
    assert len(ticks) >= 2
    # Axis should not produce more than a liberal upper bound of ticks
    assert len(ticks) <= num_ticks + 4


@pytest.mark.parametrize("num_ticks", [3, 5, 7])
def test_linear_axis_tick_count_is_reasonable(ax, num_ticks):
    configure_forest_axis(
        ax=ax,
        model_type="linear",
        link="identity",
        thresholds=thresholds_for("identity", lo=[-1.0, -0.5], hi=[0.5, 1.0]),
        num_ticks=num_ticks,
        font_size=12,
        show_general_stats=True,
    )
    ticks = ax.get_xticks()
    assert len(ticks) >= 2
    assert len(ticks) <= num_ticks + 4


# ═══════════════════════════════════════════════════════════════════════════════
# tick_style="power10"
# ═══════════════════════════════════════════════════════════════════════════════

def test_tick_style_power10_does_not_crash(ax):
    configure_forest_axis(
        ax=ax,
        model_type="binom",
        link="logit",
        thresholds=thresholds_for("logit", lo=[0.5], hi=[2.0], tick_style="power10"),
        num_ticks=5,
        font_size=12,
        show_general_stats=True,
    )
    # Ticks should still be present
    assert len(ax.get_xticks()) >= 2


# ═══════════════════════════════════════════════════════════════════════════════
# Single vs dual outcome: lo_all / hi_all array sizes
# ═══════════════════════════════════════════════════════════════════════════════

@pytest.mark.parametrize(
    "lo, hi, label",
    [
        ([0.6], [1.4], "single outcome"),
        ([0.5, 0.7], [1.3, 1.8], "dual outcome"),
    ],
)
def test_single_and_dual_outcome_data_arrays(ax, lo, hi, label):
    configure_forest_axis(
        ax=ax,
        model_type="binom",
        link="logit",
        thresholds=thresholds_for("logit", lo=lo, hi=hi),
        num_ticks=5,
        font_size=12,
        show_general_stats=True,
    )
    xmin, xmax = ax.get_xlim()
    assert xmin <= min(lo), f"xlim too narrow on left for {label}"
    assert xmax >= max(hi), f"xlim too narrow on right for {label}"
