from typing import Any, TypedDict

import pandas as pd


class LayoutResult(TypedDict):
    """Structured row layout used by the forest plot renderer."""

    rows: pd.DataFrame
    y_positions: list[int]
    meta: dict[str, Any]


def build_row_layout(df_final: pd.DataFrame) -> LayoutResult:
    """
    Assemble row ordering and y-positions for forest plot table/points.

    Parameters
    ----------
    df_final : pd.DataFrame
        Normalized plotting dataframe expected to include a ``predictor``
        column and optionally a ``category`` column.

    Returns
    -------
    LayoutResult
        Dict with:
        - ``rows``: DataFrame with ``predictor``, ``is_cat``, ``category``.
        - ``y_positions``: Integer y-positions aligned with ``rows`` order.
        - ``meta``: Extra layout fields (`n`, `row_is_cat`, `row_cats`).
    """
    if "category" in df_final.columns and df_final["category"].notna().any():
        cat_order = list(df_final["category"].dropna().unique())
        table_rows: list[dict[str, Any]] = []
        row_is_cat: list[bool] = []
        row_cats: list[str] = []

        for cat in cat_order:
            table_rows.append({"predictor": cat, "is_cat": True, "category": cat})
            row_is_cat.append(True)
            row_cats.append(cat)

            preds = df_final.loc[df_final["category"] == cat, "predictor"].unique()
            for pred in preds:
                table_rows.append(
                    {"predictor": pred, "is_cat": False, "category": cat}
                )
                row_is_cat.append(False)
                row_cats.append(cat)
    else:
        preds = df_final["predictor"].dropna().unique()
        table_rows = [
            {"predictor": pred, "is_cat": False, "category": "Uncategorized"}
            for pred in preds
        ]
        row_is_cat = [False] * len(preds)
        row_cats = ["Uncategorized"] * len(preds)

    n = len(table_rows)
    if n == 0:
        raise ValueError("No rows to plot! Check DataFrame structure.")

    rows_df = pd.DataFrame(table_rows)
    y_positions = list(range(n))

    return {
        "rows": rows_df,
        "y_positions": y_positions,
        "meta": {"n": n, "row_is_cat": row_is_cat, "row_cats": row_cats},
    }
