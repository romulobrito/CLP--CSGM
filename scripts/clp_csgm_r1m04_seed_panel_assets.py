#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Aggregate R1-M04 10-seed CLP vs AE runs into paper appendix tables.

Reads:
  outputs/reviewer_revision/r1m04_seed_panel/runs/stepXX/tables/summary_by_seed.csv
Writes:
  outputs/reviewer_revision/r1m04_seed_panel/aggregate/tables/
  paper_clp_csgm/tables/robustness_crosswell_seed_panel.csv|.tex

Does not modify results_crosswell_clp_vs_ae.* or paired_delta_crosswell_*.
ASCII-only.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
from typing import Dict, List, Tuple

import pandas as pd

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RUNS_DIR = os.path.join(
    REPO_ROOT, "outputs", "reviewer_revision", "r1m04_seed_panel", "runs"
)
AGG_DIR = os.path.join(
    REPO_ROOT, "outputs", "reviewer_revision", "r1m04_seed_panel", "aggregate", "tables"
)
PAPER_TAB = os.path.join(REPO_ROOT, "paper_clp_csgm", "tables")
CLP_METHOD = "ridge_prior_csgm"
AE_METHOD = "ae_regression_ub"
FOCUS_METHODS = (CLP_METHOD, AE_METHOD)
STEP_RE = re.compile(r"^step(\d+)$")
# Student-t 0.975 quantile, df = 9 (n_seeds = 10). Fallback if scipy is absent.
T_CRIT_DF9 = 2.2621571627409915


def _t_crit(df: int) -> float:
    """Two-sided 95 percent Student-t half-width critical value."""
    if int(df) == 9:
        try:
            from scipy.stats import t as student_t

            return float(student_t.ppf(0.975, 9))
        except Exception:
            return float(T_CRIT_DF9)
    if int(df) <= 0:
        raise ValueError("t critical value needs df >= 1")
    try:
        from scipy.stats import t as student_t

        return float(student_t.ppf(0.975, int(df)))
    except Exception as exc:
        raise RuntimeError("scipy is required for df != 9") from exc


def _step_from_name(name: str) -> int:
    """Parse step08 -> 8."""
    matched = STEP_RE.match(name.strip())
    if matched is None:
        raise ValueError("run folder must match stepNN, got {}".format(name))
    return int(matched.group(1))


def _read_step_summary(run_dir: str) -> Tuple[int, pd.DataFrame, Dict[str, object]]:
    """Load one step folder seed-level summary and config."""
    name = os.path.basename(run_dir.rstrip(os.sep))
    step = _step_from_name(name)
    summary_path = os.path.join(run_dir, "tables", "summary_by_seed.csv")
    if not os.path.isfile(summary_path):
        raise FileNotFoundError(summary_path)
    frame = pd.read_csv(summary_path)
    required = {"seed", "measurement_ratio", "method", "rmse_mean"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError("missing columns in {}: {}".format(summary_path, sorted(missing)))
    cfg_path = os.path.join(run_dir, "config.json")
    meta: Dict[str, object] = {}
    if os.path.isfile(cfg_path):
        with open(cfg_path, "r", encoding="utf-8") as handle:
            dumped = json.load(handle)
        cw = dumped.get("cross_well_vc", {})
        if isinstance(cw, dict):
            meta = cw
            if "step" in cw:
                step = int(cw["step"])
    frame = frame[frame["method"].isin(FOCUS_METHODS)].copy()
    frame.insert(0, "step", step)
    return step, frame, meta


def _paired_rows(long_df: pd.DataFrame) -> pd.DataFrame:
    """One row per step x seed x rho with CLP and AE RMSE."""
    clp = long_df[long_df["method"] == CLP_METHOD][
        ["step", "seed", "measurement_ratio", "rmse_mean"]
    ].rename(columns={"rmse_mean": "clp_rmse"})
    ae = long_df[long_df["method"] == AE_METHOD][
        ["step", "seed", "measurement_ratio", "rmse_mean"]
    ].rename(columns={"rmse_mean": "ae_rmse"})
    merged = clp.merge(ae, on=["step", "seed", "measurement_ratio"], how="inner")
    merged["delta_rmse"] = merged["ae_rmse"] - merged["clp_rmse"]
    merged["clp_wins"] = merged["clp_rmse"] < merged["ae_rmse"]
    return merged.sort_values(["step", "measurement_ratio", "seed"]).reset_index(drop=True)


def _summary_by_cell(paired: pd.DataFrame) -> pd.DataFrame:
    """Mean paired DeltaRMSE and Student-t interval per (step, rho)."""
    rows: List[Dict[str, object]] = []
    grouped = paired.groupby(["step", "measurement_ratio"], sort=True)
    for (step, rho), grp in grouped:
        n_seeds = int(grp.shape[0])
        if n_seeds < 2:
            raise ValueError("need at least 2 seeds for cell step={} rho={}".format(step, rho))
        delta_mean = float(grp["delta_rmse"].mean())
        delta_std = float(grp["delta_rmse"].std(ddof=1))
        delta_se = delta_std / math.sqrt(n_seeds)
        half = float(_t_crit(n_seeds - 1)) * delta_se
        clp_mean = float(grp["clp_rmse"].mean())
        ae_mean = float(grp["ae_rmse"].mean())
        clp_wins = int(grp["clp_wins"].sum())
        ae_wins = int(n_seeds - clp_wins)
        includes_zero = bool(abs(delta_mean) <= half)
        rows.append(
            {
                "step": int(step),
                "rho": float(rho),
                "n_seeds": n_seeds,
                "clp_rmse_mean": clp_mean,
                "ae_rmse_mean": ae_mean,
                "mean_delta_rmse": delta_mean,
                "delta_se": delta_se,
                "delta_t95_half": half,
                "clp_wins": clp_wins,
                "ae_wins": ae_wins,
                "ci_includes_zero": includes_zero,
            }
        )
    return pd.DataFrame(rows)


def _write_latex(summary: pd.DataFrame, path: str) -> None:
    """Write a compact booktabs fragment for the appendix."""
    lines = [
        r"\begin{tabular}{rrrrrrrrr}",
        r"\toprule",
        r"step & $\rho$ & $n$ & mean $\Delta$RMSE & SE & "
        r"$t$ 95\% half-width & CLP wins & AE wins & CI includes 0 \\",
        r"\midrule",
    ]
    for _, row in summary.iterrows():
        includes = "yes" if bool(row["ci_includes_zero"]) else "no"
        lines.append(
            "{:d} & {:.2f} & {:d} & {:.5f} & {:.5f} & {:.5f} & {:d} & {:d} & {} \\\\".format(
                int(row["step"]),
                float(row["rho"]),
                int(row["n_seeds"]),
                float(row["mean_delta_rmse"]),
                float(row["delta_se"]),
                float(row["delta_t95_half"]),
                int(row["clp_wins"]),
                int(row["ae_wins"]),
                includes,
            )
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", ""])
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate R1-M04 seed-panel runs.")
    parser.add_argument("--runs-dir", type=str, default=RUNS_DIR)
    parser.add_argument("--agg-dir", type=str, default=AGG_DIR)
    parser.add_argument("--paper-tab", type=str, default=PAPER_TAB)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    runs_dir = os.path.abspath(str(args.runs_dir))
    if not os.path.isdir(runs_dir):
        raise SystemExit("missing runs dir: {}".format(runs_dir))
    names = sorted(
        name for name in os.listdir(runs_dir) if STEP_RE.match(name) is not None
    )
    if not names:
        raise SystemExit("no stepNN folders under {}".format(runs_dir))
    chunks: List[pd.DataFrame] = []
    for name in names:
        run_dir = os.path.join(runs_dir, name)
        _step, frame, _meta = _read_step_summary(run_dir)
        chunks.append(frame)
    long_df = pd.concat(chunks, ignore_index=True)
    paired = _paired_rows(long_df)
    summary = _summary_by_cell(paired)
    agg_dir = os.path.abspath(str(args.agg_dir))
    os.makedirs(agg_dir, exist_ok=True)
    long_path = os.path.join(agg_dir, "per_seed_focus.csv")
    paired_path = os.path.join(agg_dir, "paired_delta_clp_vs_ae.csv")
    summary_path = os.path.join(agg_dir, "summary_by_cell.csv")
    long_df.to_csv(long_path, index=False)
    paired.to_csv(paired_path, index=False)
    summary.to_csv(summary_path, index=False)
    paper_tab = os.path.abspath(str(args.paper_tab))
    os.makedirs(paper_tab, exist_ok=True)
    paper_csv = os.path.join(paper_tab, "robustness_crosswell_seed_panel.csv")
    paper_tex = os.path.join(paper_tab, "robustness_crosswell_seed_panel.tex")
    summary.to_csv(paper_csv, index=False)
    _write_latex(summary, paper_tex)
    print("wrote", long_path)
    print("wrote", paired_path)
    print("wrote", summary_path)
    print("wrote", paper_csv)
    print("wrote", paper_tex)


if __name__ == "__main__":
    main()
