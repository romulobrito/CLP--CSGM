#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Aggregate the R1-M08 explicit-mask AE supplement.

Reads summary_by_seed.csv from
  outputs/reviewer_revision/r1m08_explicit_mask/runs/stepXX/
Writes the appendix CSV and TeX. Does not edit the main 9-cell tables.
ASCII-only.
"""

from __future__ import annotations

import argparse
import math
import os
import re
from typing import Dict, List

import pandas as pd

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RUNS_DIR = os.path.join(
    REPO_ROOT, "outputs", "reviewer_revision", "r1m08_explicit_mask", "runs"
)
AGG_DIR = os.path.join(
    REPO_ROOT, "outputs", "reviewer_revision", "r1m08_explicit_mask", "aggregate", "tables"
)
PAPER_TAB = os.path.join(REPO_ROOT, "paper_clp_csgm", "tables")
CLP_METHOD = "ridge_prior_csgm"
AE_METHOD = "ae_regression_ub"
MASK_METHOD = "ae_regression_ub_mask"
STEP_RE = re.compile(r"^step(\d+)$")
T_CRIT_DF2 = 4.302652729911275


def _t_crit(df: int) -> float:
    """Two-sided 95 percent Student-t critical value."""
    if int(df) == 2:
        return float(T_CRIT_DF2)
    try:
        from scipy.stats import t as student_t
    except Exception as exc:
        raise RuntimeError("scipy is required for df other than 2") from exc
    return float(student_t.ppf(0.975, int(df)))


def _read_step(run_dir: str) -> pd.DataFrame:
    """Load one step folder and keep the three compared methods."""
    summary_path = os.path.join(run_dir, "tables", "summary_by_seed.csv")
    if not os.path.isfile(summary_path):
        raise FileNotFoundError(summary_path)
    frame = pd.read_csv(summary_path)
    required = {"seed", "measurement_ratio", "method", "rmse_mean"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError("missing columns in {}: {}".format(summary_path, sorted(missing)))
    name = os.path.basename(run_dir.rstrip(os.sep))
    matched = STEP_RE.match(name)
    if matched is None:
        raise ValueError("run folder must match stepNN, got {}".format(name))
    kept = frame[frame["method"].isin([CLP_METHOD, AE_METHOD, MASK_METHOD])].copy()
    kept.insert(0, "step", int(matched.group(1)))
    return kept


def _paired(long_df: pd.DataFrame) -> pd.DataFrame:
    """One row per step, seed, and rho."""
    keys = ["step", "seed", "measurement_ratio"]
    clp = long_df[long_df["method"] == CLP_METHOD][keys + ["rmse_mean"]].rename(
        columns={"rmse_mean": "clp_rmse"}
    )
    ae = long_df[long_df["method"] == AE_METHOD][keys + ["rmse_mean"]].rename(
        columns={"rmse_mean": "ae_rmse"}
    )
    mask = long_df[long_df["method"] == MASK_METHOD][keys + ["rmse_mean"]].rename(
        columns={"rmse_mean": "ae_mask_rmse"}
    )
    merged = clp.merge(ae, on=keys, how="inner").merge(mask, on=keys, how="inner")
    if merged.empty:
        raise ValueError("no paired rows; the mask method is missing from the runs.")
    merged["delta_ae_minus_clp"] = merged["ae_rmse"] - merged["clp_rmse"]
    merged["delta_mask_minus_clp"] = merged["ae_mask_rmse"] - merged["clp_rmse"]
    merged["clp_beats_ae"] = merged["clp_rmse"] < merged["ae_rmse"]
    merged["clp_beats_mask"] = merged["clp_rmse"] < merged["ae_mask_rmse"]
    merged["mask_beats_ae"] = merged["ae_mask_rmse"] < merged["ae_rmse"]
    return merged.sort_values(keys).reset_index(drop=True)


def _summary(paired: pd.DataFrame) -> pd.DataFrame:
    """Mean RMSE and paired gaps per cell."""
    rows: List[Dict[str, object]] = []
    grouped = paired.groupby(["step", "measurement_ratio"], sort=True)
    for (step, rho), grp in grouped:
        n_seeds = int(grp.shape[0])
        if n_seeds < 2:
            raise ValueError("need at least 2 seeds for step={} rho={}".format(step, rho))
        half_scale = _t_crit(n_seeds - 1) / math.sqrt(n_seeds)
        delta_ae = float(grp["delta_ae_minus_clp"].mean())
        delta_mask = float(grp["delta_mask_minus_clp"].mean())
        mask_std = float(grp["delta_mask_minus_clp"].std(ddof=1))
        mask_half = half_scale * mask_std
        rows.append(
            {
                "step": int(step),
                "rho": float(rho),
                "n_seeds": n_seeds,
                "clp_rmse_mean": float(grp["clp_rmse"].mean()),
                "ae_rmse_mean": float(grp["ae_rmse"].mean()),
                "ae_mask_rmse_mean": float(grp["ae_mask_rmse"].mean()),
                "delta_ae_minus_clp": delta_ae,
                "delta_mask_minus_clp": delta_mask,
                "delta_mask_se": mask_std / math.sqrt(n_seeds),
                "delta_mask_t95_half": mask_half,
                "clp_wins_vs_ae": int(grp["clp_beats_ae"].sum()),
                "clp_wins_vs_mask": int(grp["clp_beats_mask"].sum()),
                "mask_wins_vs_ae": int(grp["mask_beats_ae"].sum()),
                "mask_gap_includes_zero": bool(abs(delta_mask) <= mask_half),
            }
        )
    return pd.DataFrame(rows)


def _write_latex(summary: pd.DataFrame, path: str) -> None:
    """Write the appendix booktabs fragment."""
    lines = [
        r"\begin{tabular}{rrrrrrrrr}",
        r"\toprule",
        r"step & $\rho$ & CLP & AE $[u,b]$ & AE mask & "
        r"$\Delta$ AE & $\Delta$ mask & CLP$>$AE & CLP$>$mask \\",
        r"\midrule",
    ]
    for _, row in summary.iterrows():
        lines.append(
            "{:d} & {:.2f} & {:.4f} & {:.4f} & {:.4f} & {:.4f} & {:.4f} & {:d}/{:d} & {:d}/{:d} \\\\".format(
                int(row["step"]),
                float(row["rho"]),
                float(row["clp_rmse_mean"]),
                float(row["ae_rmse_mean"]),
                float(row["ae_mask_rmse_mean"]),
                float(row["delta_ae_minus_clp"]),
                float(row["delta_mask_minus_clp"]),
                int(row["clp_wins_vs_ae"]),
                int(row["n_seeds"]),
                int(row["clp_wins_vs_mask"]),
                int(row["n_seeds"]),
            )
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", ""])
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate the R1-M08 mask supplement.")
    parser.add_argument("--runs-dir", type=str, default=RUNS_DIR)
    parser.add_argument("--agg-dir", type=str, default=AGG_DIR)
    parser.add_argument("--paper-tab", type=str, default=PAPER_TAB)
    args = parser.parse_args()
    runs_dir = os.path.abspath(str(args.runs_dir))
    if not os.path.isdir(runs_dir):
        raise SystemExit("missing runs dir: {}".format(runs_dir))
    names = sorted(name for name in os.listdir(runs_dir) if STEP_RE.match(name) is not None)
    if not names:
        raise SystemExit("no stepNN folders under {}".format(runs_dir))
    long_df = pd.concat([_read_step(os.path.join(runs_dir, name)) for name in names], ignore_index=True)
    paired = _paired(long_df)
    summary = _summary(paired)
    agg_dir = os.path.abspath(str(args.agg_dir))
    os.makedirs(agg_dir, exist_ok=True)
    paired_path = os.path.join(agg_dir, "paired_delta_clp_ae_mask.csv")
    summary_path = os.path.join(agg_dir, "summary_by_cell.csv")
    paired.to_csv(paired_path, index=False)
    summary.to_csv(summary_path, index=False)
    paper_tab = os.path.abspath(str(args.paper_tab))
    os.makedirs(paper_tab, exist_ok=True)
    paper_csv = os.path.join(paper_tab, "robustness_crosswell_explicit_mask.csv")
    paper_tex = os.path.join(paper_tab, "robustness_crosswell_explicit_mask.tex")
    summary.to_csv(paper_csv, index=False)
    _write_latex(summary, paper_tex)
    print("wrote", paired_path)
    print("wrote", summary_path)
    print("wrote", paper_csv)
    print("wrote", paper_tex)


if __name__ == "__main__":
    main()
