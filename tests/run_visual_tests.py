#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import os
import sys
import traceback
import warnings
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import forestplotx as fpx


@dataclass
class Case:
    test_id: str
    description: str
    df_key: str
    kwargs: dict[str, Any]
    expect_error: str | None = None


def load_base_datasets() -> dict[str, pd.DataFrame]:
    data_dir = ROOT / "data"
    return {
        "binom": pd.read_csv(data_dir / "data_modeltype_binom.csv"),
        "gamma": pd.read_csv(data_dir / "data_modeltype_gamma.csv"),
        "linear": pd.read_csv(data_dir / "data_modeltype_linear.csv"),
        "ordinal": pd.read_csv(data_dir / "data_modeltype_ordinal.csv"),
    }


def build_modified_datasets(ds: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    np.random.seed(42)
    out = dict(ds)
    long_pred = "Pneumonoultramicroscopicsilicovolcanoconiosis"

    # G1
    d = ds["binom"].copy()
    m = (d["predictor"] == "predictor1") & (d["outcome"] == "outcome1")
    d.loc[m, ["OR", "CI_low", "CI_high"]] = [15.0, 8.0, 28.0]
    out["binom_outlier"] = d

    # G2
    d = ds["binom"].copy()
    d["OR"] = 1.0 + np.random.uniform(-0.01, 0.01, len(d))
    d["CI_low"] = d["OR"] - 0.005
    d["CI_high"] = d["OR"] + 0.005
    out["binom_tight"] = d

    # G3
    d = ds["linear"].copy()
    m = (d["predictor"] == "predictor1") & (d["outcome"] == "outcome1")
    d.loc[m, ["Estimate", "CI_low", "CI_high"]] = [50.0, 45.0, 55.0]
    out["linear_outlier"] = d

    # G4
    d = ds["linear"].copy()
    d["Estimate"] = np.random.uniform(-0.001, 0.001, len(d))
    d["CI_low"] = d["Estimate"] - 0.0005
    d["CI_high"] = d["Estimate"] + 0.0005
    out["linear_zero"] = d

    # G5
    d = ds["gamma"].copy()
    d["Estimate"] = 0.0
    d["CI_low"] = -0.01
    d["CI_high"] = 0.01
    out["gamma_zero"] = d

    # G6
    d = ds["binom"].copy()
    m1 = (d["predictor"] == "predictor1") & (d["outcome"] == "outcome1")
    m2 = (d["predictor"] == "predictor2") & (d["outcome"] == "outcome1")
    d.loc[m1, ["OR", "CI_low", "CI_high"]] = [0.01, 0.005, 0.02]
    d.loc[m2, ["OR", "CI_low", "CI_high"]] = [10.0, 5.0, 20.0]
    out["binom_wide"] = d

    # G7
    d = ds["linear"].copy()
    d["n"] = np.random.randint(10, 100, len(d))
    d["N"] = d["n"] + np.random.randint(50, 200, len(d))
    out["linear_with_counts"] = d

    # G8
    d = ds["binom"].copy()
    m = (d["predictor"] == "predictor3") & (d["outcome"] == "outcome1")
    d.loc[m, ["OR", "CI_low", "CI_high"]] = np.nan
    out["binom_nulls"] = d

    # G9: Linear subset with only 3 predictors (all outcomes retained)
    d = ds["linear"].copy()
    keep_preds = list(d["predictor"].dropna().unique()[:3])
    out["linear_3pred"] = d[d["predictor"].isin(keep_preds)].copy()

    # G10: Linear expanded to 30 predictors (2 outcomes each)
    d = ds["linear"].copy()
    base_preds = list(d["predictor"].dropna().unique())
    base_outcomes = list(d["outcome"].dropna().unique())
    rows: list[dict[str, Any]] = []
    for i in range(30):
        base_pred = base_preds[i % len(base_preds)]
        for outcome in base_outcomes[:2]:
            src = d[(d["predictor"] == base_pred) & (d["outcome"] == outcome)]
            if src.empty:
                src = d[d["predictor"] == base_pred].head(1)
            if src.empty:
                continue
            row = src.iloc[0].to_dict()
            row["predictor"] = f"predictor_{i+1:02d}"
            row["outcome"] = outcome
            # Small deterministic jitter to avoid complete duplication.
            delta = ((i % 5) - 2) * 0.02
            if "Estimate" in row and pd.notnull(row["Estimate"]):
                row["Estimate"] = float(row["Estimate"]) + delta
            if "CI_low" in row and pd.notnull(row["CI_low"]):
                row["CI_low"] = float(row["CI_low"]) + delta
            if "CI_high" in row and pd.notnull(row["CI_high"]):
                row["CI_high"] = float(row["CI_high"]) + delta
            rows.append(row)
    out["linear_30pred"] = pd.DataFrame(rows)

    # G11: Linear data where only 2/10 predictors have effect values; others are NaN.
    d = ds["linear"].copy()
    keep_preds = set(d["predictor"].dropna().unique()[:2])
    mask_nan = ~d["predictor"].isin(keep_preds)
    for col in ("Estimate", "CI_low", "CI_high"):
        if col in d.columns:
            d.loc[mask_nan, col] = np.nan
    out["linear_2_with_values_8_nan"] = d

    # G12: Partial missing in single-outcome view with general stats (missing p only).
    d = out["linear_with_counts"].copy()
    m = (d["predictor"] == "predictor3") & (d["outcome"] == "outcome1")
    d.loc[m, "p_value"] = np.nan
    out["linear_with_counts_partial_single"] = d

    # G13: Partial missing in two-outcome view with general stats (one outcome only).
    d = out["linear_with_counts"].copy()
    m = (d["predictor"] == "predictor3") & (d["outcome"] == "outcome1")
    d.loc[m, "CI_high"] = np.nan
    out["linear_with_counts_partial_dual"] = d

    # G14: Long predictor label for truncation tests.
    d = ds["linear"].copy()
    first_pred = d["predictor"].dropna().unique()[0]
    d.loc[d["predictor"] == first_pred, "predictor"] = long_pred
    out["linear_long_pred"] = d

    # G15: Long predictor label with n/N columns.
    d = out["linear_with_counts"].copy()
    first_pred = d["predictor"].dropna().unique()[0]
    d.loc[d["predictor"] == first_pred, "predictor"] = long_pred
    out["linear_with_counts_long_pred"] = d

    # Error-path helpers
    out["empty"] = pd.DataFrame(
        columns=["predictor", "outcome", "Estimate", "CI_low", "CI_high"]
    )
    out["missing_effect"] = pd.DataFrame(
        {"predictor": ["x1"], "outcome": ["y1"], "CI_low": [0.1], "CI_high": [0.2]}
    )
    out["binom_cat_nan"] = ds["binom"].assign(category=np.nan)
    return out


def build_cases() -> list[Case]:
    return [
        # A
        Case("A1", "binom baseline", "binom", dict(model_type="binom", exponentiate=False)),
        Case("A2", "gamma baseline", "gamma", dict(model_type="gamma")),
        Case("A3", "linear baseline", "linear", dict(model_type="linear")),
        Case("A4", "ordinal baseline", "ordinal", dict(model_type="ordinal")),
        # B
        Case("B1", "binom default exponentiation (expected bad)", "binom", dict(model_type="binom")),
        Case("B2", "binom exponentiate False", "binom", dict(model_type="binom", exponentiate=False)),
        Case("B3", "binom exponentiate True", "binom", dict(model_type="binom", exponentiate=True)),
        Case("B4", "gamma exponentiate False", "gamma", dict(model_type="gamma", exponentiate=False)),
        Case("B5", "gamma exponentiate default None", "gamma", dict(model_type="gamma", exponentiate=None)),
        # C
        Case("C1", "binom one outcome", "binom", dict(model_type="binom", exponentiate=False, outcomes=["outcome1"])),
        Case("C2", "binom two outcomes", "binom", dict(model_type="binom", exponentiate=False, outcomes=["outcome1", "outcome2"])),
        Case("C3", "binom three outcomes truncation", "binom", dict(model_type="binom", exponentiate=False, outcomes=["outcome1", "outcome2", "outcome3"])),
        Case("C4", "linear outcomes autodetect", "linear", dict(model_type="linear", outcomes=None)),
        # D
        Case("D1", "binom outlier", "binom_outlier", dict(model_type="binom", exponentiate=False)),
        Case("D2", "binom tight range", "binom_tight", dict(model_type="binom", exponentiate=False)),
        Case("D3", "linear outlier", "linear_outlier", dict(model_type="linear")),
        Case("D3b", "linear outlier clipped", "linear_outlier", dict(model_type="linear", clip_outliers=True)),
        Case("D4", "linear near-zero span", "linear_zero", dict(model_type="linear")),
        Case("D5", "gamma span==0 after exp", "gamma_zero", dict(model_type="gamma")),
        Case("D6", "binom wide range", "binom_wide", dict(model_type="binom", exponentiate=False)),
        # E
        Case("E1", "layout preset (True, True)", "linear", dict(model_type="linear", outcomes=["outcome1", "outcome2"], show_general_stats=True)),
        Case("E2", "layout preset (True, False)", "linear", dict(model_type="linear", outcomes=["outcome1"], show_general_stats=True)),
        Case("E3", "base_decimals 0", "gamma", dict(model_type="gamma", base_decimals=0)),
        Case("E4", "base_decimals 4", "gamma", dict(model_type="gamma", base_decimals=4)),
        Case("E5", "power10 ticks", "binom", dict(model_type="binom", exponentiate=False, tick_style="power10")),
        Case("E6", "decimal ticks", "binom", dict(model_type="binom", exponentiate=False, tick_style="decimal")),
        Case("E7", "custom point colors two", "linear", dict(model_type="linear", point_colors=["#FF0000", "#0000FF"])),
        Case("E8", "custom point colors one", "linear", dict(model_type="linear", point_colors=["#2C5F8A"])),
        # F
        Case("F1", "table only", "linear", dict(model_type="linear", table_only=True)),
        Case("F2", "hide general stats", "linear", dict(model_type="linear", show_general_stats=False)),
        Case("F3", "show general stats", "linear_with_counts", dict(model_type="linear", show_general_stats=True)),
        Case("F4", "footer short", "binom", dict(model_type="binom", exponentiate=False, footer_text="Adjusted for age and sex")),
        Case("F5", "footer long", "binom", dict(model_type="binom", exponentiate=False, footer_text="Adjusted for age, sex, BMI, smoking status, diabetes, hypertension, and prior cardiovascular events. Sensitivity analyses excluded patients with missing lab values (n=47).")),
        Case("F6", "footer special chars", "binom", dict(model_type="binom", exponentiate=False, footer_text="†p<0.05 after Bonferroni correction; ‡excluding n=12 protocol deviations")),
        Case("F7", "layout no general + two outcomes", "linear", dict(model_type="linear", show_general_stats=False, outcomes=["outcome1", "outcome2"])),
        Case("F8", "layout no general + one outcome", "linear", dict(model_type="linear", show_general_stats=False, outcomes=["outcome1"])),
        Case("F9", "bold override", "binom", dict(model_type="binom", exponentiate=False, bold_override={"predictor2": {"outcome1": True}})),
        Case("F10", "clip outliers on wide data", "binom_wide", dict(model_type="binom", exponentiate=False, clip_outliers=True)),
        # H
        Case("H1", "show true save none", "linear", dict(model_type="linear", show=True, save=None)),
        Case("H2", "show false save path", "linear", dict(model_type="linear", show=False, save="H2_saved.png")),
        Case("H3", "show false save true(default name)", "linear", dict(model_type="linear", show=False, save=True)),
        Case("H4", "show false save nested path", "linear", dict(model_type="linear", show=False, save="nested/H4_saved.png")),
        # I
        Case("I1", "empty df", "empty", dict(model_type="linear"), expect_error="ValueError"),
        Case("I2", "missing effect", "missing_effect", dict(model_type="linear"), expect_error="ValueError"),
        Case("I3", "invalid model_type", "linear", dict(model_type="poisson"), expect_error="ValueError"),
        Case("I4", "category all NaN", "binom_cat_nan", dict(model_type="binom", exponentiate=False)),
        # J
        Case("J1", "footer extremely long", "binom", dict(model_type="binom", exponentiate=False, footer_text="Adjusted for age, sex, BMI, smoking status, diabetes, hypertension, prior cardiovascular events, renal function stage, liver function profile, frailty index, baseline medication burden, and center-level random effects; sensitivity analyses excluded protocol deviations, missing baseline laboratory panels, and incomplete follow-up observations across all treatment strata.")),
        Case("J2", "footer multiline (3 lines)", "binom", dict(model_type="binom", exponentiate=False, footer_text="Line 1: adjusted for age/sex\nLine 2: robust SE + center effects\nLine 3: sensitivity excludes missing labs")),
        Case("J3", "only 3 predictors", "linear_3pred", dict(model_type="linear")),
        Case("J4", "30 predictors", "linear_30pred", dict(model_type="linear")),
        Case("J5", "2 predictors with values + 8 NaN (gray rows)", "linear_2_with_values_8_nan", dict(model_type="linear")),
        Case("J6", "single-outcome partial missing with general stats", "linear_with_counts_partial_single", dict(model_type="linear", show_general_stats=True, outcomes=["outcome1"])),
        Case("J7", "two-outcome partial missing one outcome with general stats", "linear_with_counts_partial_dual", dict(model_type="linear", show_general_stats=True, outcomes=["outcome1", "outcome2"])),
        # K: long predictor truncation behavior
        Case("K1", "long predictor label (True, True)", "linear_with_counts_long_pred", dict(model_type="linear", show_general_stats=True, outcomes=["outcome1", "outcome2"])),
        Case("K2", "long predictor label (True, False)", "linear_with_counts_long_pred", dict(model_type="linear", show_general_stats=True, outcomes=["outcome1"])),
        Case("K3", "long predictor label (False, True)", "linear_long_pred", dict(model_type="linear", show_general_stats=False, outcomes=["outcome1", "outcome2"])),
        Case("K4", "long predictor label (False, False)", "linear_long_pred", dict(model_type="linear", show_general_stats=False, outcomes=["outcome1"])),
    ]


def run_case(case: Case, datasets: dict[str, pd.DataFrame], out_plots: Path) -> dict[str, str]:
    row = {
        "test_id": case.test_id,
        "description": case.description,
        "status": "PASS",
        "error_type": "",
        "error_message": "",
        "warnings": "",
        "output_file": "",
    }
    df = datasets[case.df_key]
    kwargs = dict(case.kwargs)
    kwargs.setdefault("show", False)

    # Re-root explicit relative save paths into run directory.
    if isinstance(kwargs.get("save"), str):
        kwargs["save"] = str(out_plots / kwargs["save"])

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        fig = None
        try:
            fig, _axes = fpx.forest_plot(df=df, **kwargs)
            if case.expect_error:
                row["status"] = "FAIL"
                row["error_message"] = f"Expected {case.expect_error} but no exception"
            else:
                out_file = out_plots / f"{case.test_id}.png"
                fig.savefig(out_file, dpi=150, bbox_inches="tight")
                row["output_file"] = str(out_file.relative_to(ROOT))
        except Exception as exc:  # noqa: BLE001
            if case.expect_error and type(exc).__name__ == case.expect_error:
                row["status"] = "PASS_EXPECTED_ERROR"
            else:
                row["status"] = "FAIL"
                row["error_type"] = type(exc).__name__
                row["error_message"] = str(exc)
        finally:
            if fig is not None:
                plt.close(fig)

        row["warnings"] = " | ".join(str(x.message) for x in w)

    return row


def main() -> int:
    parser = argparse.ArgumentParser(description="Run forestplotx visual test matrix.")
    parser.add_argument(
        "--out",
        default="tests/test_results",
        help="Output root directory (default: tests/test_results)",
    )
    args = parser.parse_args()

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_root = ROOT / args.out / run_id
    out_plots = out_root / "plots"
    out_logs = out_root / "logs"
    out_plots.mkdir(parents=True, exist_ok=True)
    out_logs.mkdir(parents=True, exist_ok=True)

    datasets = build_modified_datasets(load_base_datasets())
    cases = build_cases()

    rows: list[dict[str, str]] = []
    print(f"Writing visual outputs to: {out_root}")
    for case in cases:
        print(f"[{case.test_id}] {case.description}")
        row = run_case(case, datasets, out_plots)
        rows.append(row)
        print(f"  -> {row['status']}")
        if row["error_message"]:
            print(f"     {row['error_type']}: {row['error_message']}")

    csv_path = out_logs / "visual_results.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "test_id",
                "description",
                "status",
                "error_type",
                "error_message",
                "warnings",
                "output_file",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    md_path = out_logs / "summary.md"
    passed = sum(1 for r in rows if r["status"].startswith("PASS"))
    failed = sum(1 for r in rows if r["status"] == "FAIL")
    md_path.write_text(
        "\n".join(
            [
                "# Visual Test Summary",
                "",
                f"- Total: {len(rows)}",
                f"- Passed: {passed}",
                f"- Failed: {failed}",
                "",
                f"CSV log: `{csv_path.relative_to(ROOT)}`",
                f"Plots: `{out_plots.relative_to(ROOT)}`",
            ]
        ),
        encoding="utf-8",
    )

    if failed:
        fail_log = out_logs / "failures.txt"
        fail_log.write_text(
            "\n\n".join(
                f"[{r['test_id']}] {r['description']}\n{r['error_type']}: {r['error_message']}\n{r['warnings']}"
                for r in rows
                if r["status"] == "FAIL"
            ),
            encoding="utf-8",
        )
        return 1
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception:  # noqa: BLE001
        traceback.print_exc()
        raise
