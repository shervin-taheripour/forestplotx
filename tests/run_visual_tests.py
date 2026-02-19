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
        # I
        Case("I1", "empty df", "empty", dict(model_type="linear"), expect_error="ValueError"),
        Case("I2", "missing effect", "missing_effect", dict(model_type="linear"), expect_error="ValueError"),
        Case("I3", "invalid model_type", "linear", dict(model_type="poisson"), expect_error="ValueError"),
        Case("I4", "category all NaN", "binom_cat_nan", dict(model_type="binom", exponentiate=False)),
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
            fig, _axes = fpx.forest_plot(df_final=df, **kwargs)
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
