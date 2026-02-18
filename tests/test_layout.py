"""
Tests for forestplotx._layout.build_row_layout
Covers: flat layout, categorized layout, dual-outcome DFs, edge cases.
"""
import numpy as np
import pandas as pd
import pytest

from forestplotx._layout import build_row_layout


# ── Helpers ───────────────────────────────────────────────────────────────────

def flat_df(predictors, outcome="y1"):
    """DataFrame with no category column."""
    return pd.DataFrame(
        {
            "predictor": predictors,
            "outcome": [outcome] * len(predictors),
            "effect": [0.5] * len(predictors),
            "ci_low": [0.3] * len(predictors),
            "ci_high": [0.7] * len(predictors),
        }
    )


def categorized_df(cat_preds: dict, outcome="y1"):
    """Build a DF where keys=categories, values=list[predictor]."""
    rows = []
    for cat, preds in cat_preds.items():
        for pred in preds:
            rows.append(
                {
                    "predictor": pred,
                    "category": cat,
                    "outcome": outcome,
                    "effect": 0.5,
                    "ci_low": 0.3,
                    "ci_high": 0.7,
                }
            )
    return pd.DataFrame(rows)


# ── Flat layout ───────────────────────────────────────────────────────────────

class TestFlatLayout:
    """No 'category' column → all rows are predictor rows."""

    @pytest.mark.parametrize(
        "predictors",
        [
            ["age"],
            ["age", "sex", "bmi"],
            ["a", "b", "c", "d", "e"],
        ],
    )
    def test_y_positions_are_sequential_integers(self, predictors):
        result = build_row_layout(flat_df(predictors))
        assert result["y_positions"] == list(range(len(predictors)))

    @pytest.mark.parametrize(
        "predictors",
        [
            ["age"],
            ["age", "sex", "bmi"],
        ],
    )
    def test_row_count_matches_predictor_count(self, predictors):
        result = build_row_layout(flat_df(predictors))
        assert result["meta"]["n"] == len(predictors)
        assert len(result["rows"]) == len(predictors)

    def test_all_rows_not_is_cat(self):
        result = build_row_layout(flat_df(["age", "sex"]))
        assert all(not v for v in result["meta"]["row_is_cat"])

    def test_all_row_cats_are_uncategorized(self):
        result = build_row_layout(flat_df(["age", "sex"]))
        assert all(c == "Uncategorized" for c in result["meta"]["row_cats"])

    def test_predictor_order_preserved(self):
        preds = ["bmi", "age", "sex"]
        result = build_row_layout(flat_df(preds))
        assert list(result["rows"]["predictor"]) == preds

    def test_rows_dataframe_has_required_columns(self):
        result = build_row_layout(flat_df(["age"]))
        for col in ("predictor", "is_cat", "category"):
            assert col in result["rows"].columns, f"missing column: {col}"

    def test_nan_predictors_are_dropped(self):
        df = flat_df(["age", "sex"])
        # append a row with a NaN predictor
        extra = pd.DataFrame(
            [{"predictor": None, "outcome": "y1", "effect": 0.5, "ci_low": 0.3, "ci_high": 0.7}]
        )
        df = pd.concat([df, extra], ignore_index=True)
        result = build_row_layout(df)
        assert result["meta"]["n"] == 2  # NaN predictor dropped

    def test_empty_predictor_raises(self):
        df = pd.DataFrame({"predictor": pd.Series([], dtype=str), "outcome": []})
        with pytest.raises(ValueError, match="No rows"):
            build_row_layout(df)

    def test_single_predictor(self):
        result = build_row_layout(flat_df(["only_var"]))
        assert result["meta"]["n"] == 1
        assert result["y_positions"] == [0]
        assert result["rows"].iloc[0]["predictor"] == "only_var"
        assert result["rows"].iloc[0]["is_cat"] == False

    def test_meta_n_equals_len_y_positions(self):
        result = build_row_layout(flat_df(["a", "b", "c"]))
        assert result["meta"]["n"] == len(result["y_positions"])

    @pytest.mark.parametrize(
        "model_predictors",
        [
            # simulate different model type datasets (layout is model-type agnostic)
            ["age", "sex"],            # binom-style binary predictors
            ["bmi_cont", "crp_log"],   # gamma-style continuous
            ["latent_factor"],         # linear-style single predictor
            ["x1", "x2", "threshold1_kept"],  # ordinal-style (threshold already stripped)
        ],
    )
    def test_flat_layout_is_model_type_agnostic(self, model_predictors):
        """build_row_layout output shape doesn't depend on what the predictors represent."""
        result = build_row_layout(flat_df(model_predictors))
        assert result["meta"]["n"] == len(model_predictors)


# ── Categorized layout ────────────────────────────────────────────────────────

class TestCategorizedLayout:
    """'category' column present and non-null → category header rows inserted."""

    def test_category_headers_inserted_as_first_entries(self):
        result = build_row_layout(categorized_df({"Demographics": ["age", "sex"]}))
        is_cat = result["meta"]["row_is_cat"]
        # first row must be the category header
        assert is_cat[0] is True
        # remaining rows are data rows
        assert all(not f for f in is_cat[1:])

    @pytest.mark.parametrize(
        "cat_preds, expected_n",
        [
            ({"A": ["a1"]}, 2),                           # 1 header + 1 pred
            ({"A": ["a1", "a2"], "B": ["b1"]}, 5),        # 2 headers + 3 preds
            ({"A": ["a1"], "B": ["b1"], "C": ["c1"]}, 6), # 3 headers + 3 preds
        ],
    )
    def test_total_row_count_is_categories_plus_predictors(self, cat_preds, expected_n):
        result = build_row_layout(categorized_df(cat_preds))
        assert result["meta"]["n"] == expected_n

    def test_y_positions_are_sequential(self):
        result = build_row_layout(
            categorized_df({"A": ["a1", "a2"], "B": ["b1"]})
        )
        n = result["meta"]["n"]
        assert result["y_positions"] == list(range(n))

    def test_category_header_rows_have_is_cat_true(self):
        result = build_row_layout(
            categorized_df({"Demographics": ["age", "sex"], "Labs": ["crp"]})
        )
        rows = result["rows"]
        cat_rows = rows[rows["is_cat"]]
        assert set(cat_rows["predictor"]) == {"Demographics", "Labs"}

    def test_predictor_rows_carry_correct_category(self):
        result = build_row_layout(
            categorized_df({"Demo": ["age"], "Labs": ["crp"]})
        )
        rows = result["rows"]
        pred_rows = rows[~rows["is_cat"]]
        assert (
            pred_rows.loc[pred_rows["predictor"] == "age", "category"].iloc[0] == "Demo"
        )
        assert (
            pred_rows.loc[pred_rows["predictor"] == "crp", "category"].iloc[0] == "Labs"
        )

    def test_category_all_nan_falls_back_to_flat(self):
        df = flat_df(["age", "sex"])
        df["category"] = np.nan
        result = build_row_layout(df)
        # all-NaN category → flat path, no is_cat rows
        assert all(not f for f in result["meta"]["row_is_cat"])

    def test_single_predictor_single_category(self):
        result = build_row_layout(categorized_df({"Group": ["x1"]}))
        assert result["meta"]["n"] == 2  # 1 header + 1 predictor
        assert result["y_positions"] == [0, 1]
        rows = result["rows"]
        assert rows.iloc[0]["is_cat"] == True
        assert rows.iloc[0]["predictor"] == "Group"
        assert rows.iloc[1]["is_cat"] == False
        assert rows.iloc[1]["predictor"] == "x1"

    def test_meta_n_equals_len_y_positions_categorized(self):
        result = build_row_layout(
            categorized_df({"A": ["a1", "a2"], "B": ["b1", "b2"]})
        )
        assert result["meta"]["n"] == len(result["y_positions"])

    def test_row_cats_list_length_equals_n(self):
        result = build_row_layout(
            categorized_df({"Demo": ["age", "sex"], "Labs": ["crp"]})
        )
        assert len(result["meta"]["row_cats"]) == result["meta"]["n"]

    def test_row_is_cat_length_equals_n(self):
        result = build_row_layout(
            categorized_df({"Demo": ["age", "sex"], "Labs": ["crp"]})
        )
        assert len(result["meta"]["row_is_cat"]) == result["meta"]["n"]


# ── Dual-outcome DataFrames (layout is outcome-agnostic) ──────────────────────

class TestDualOutcomeLayout:
    """
    build_row_layout sees only predictor/category structure; it uses
    unique() on predictor so multiple outcome rows per predictor don't
    inflate the row count.
    """

    @pytest.mark.parametrize(
        "outcomes",
        [
            ["y1"],           # single outcome
            ["y1", "y2"],     # dual outcome
        ],
    )
    def test_predictor_rows_not_duplicated_per_outcome(self, outcomes):
        predictors = ["age", "sex", "bmi"]
        rows = [
            {"predictor": p, "outcome": o, "effect": 0.5, "ci_low": 0.3, "ci_high": 0.7}
            for p in predictors
            for o in outcomes
        ]
        df = pd.DataFrame(rows)
        result = build_row_layout(df)
        # regardless of outcome count, row count should equal number of unique predictors
        assert result["meta"]["n"] == len(predictors)

    def test_dual_outcome_y_positions_still_sequential(self):
        df = pd.DataFrame(
            {
                "predictor": ["age", "age", "sex", "sex"],
                "outcome": ["y1", "y2", "y1", "y2"],
                "effect": [0.5] * 4,
                "ci_low": [0.3] * 4,
                "ci_high": [0.7] * 4,
            }
        )
        result = build_row_layout(df)
        assert result["y_positions"] == [0, 1]

    def test_dual_outcome_with_categories(self):
        rows = []
        for cat, preds in {"Demo": ["age", "sex"], "Labs": ["crp"]}.items():
            for pred in preds:
                for outcome in ["y1", "y2"]:
                    rows.append(
                        {
                            "predictor": pred,
                            "category": cat,
                            "outcome": outcome,
                            "effect": 0.5,
                            "ci_low": 0.3,
                            "ci_high": 0.7,
                        }
                    )
        df = pd.DataFrame(rows)
        result = build_row_layout(df)
        # 2 cat headers + 3 unique predictors = 5
        assert result["meta"]["n"] == 5
