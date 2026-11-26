#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Aggregate R1-M03 step-32 development-split runs into paper appendix tables.

Reads:
  outputs/reviewer_revision/r1m03_step32_dev_splits/runs/splitXX/tables/summary.csv
Writes:
  outputs/reviewer_revision/r1m03_step32_dev_splits/aggregate/tables/
  paper_clp_csgm/tables/robustness_crosswell_step32_dev_splits.csv|.tex

Does not modify results_crosswell_clp_vs_ae.*.
ASCII-only.
"""

from __future__ import annotations

import argparse
import json
import os
import re
from typing import Dict, List, Tuple

import pandas as pd

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RUNS_DIR = os.path.join(
    REPO_ROOT, "outputs", "reviewer_revision", "r1m03_step32_dev_splits", "runs"
)
AGG_DIR = os.path.join(
    REPO_ROOT, "outputs", "reviewer_revision", "r1m03_step32_dev_splits", "aggregate", "tables"
)
PAPER_TAB = os.path.join(REPO_ROOT, "paper_clp_csgm", "tables")
CLP_METHOD = "ridge_prior_csgm"
AE_METHOD = "ae_regression_ub"
FOCUS_METHODS = (CLP_METHOD, AE_METHOD)
SPLIT_RE = re.compile(r"^split(\d+)$")


def _split_id_from_name(name: str) -> int:
    """Parse split00 -> 0."""
    matched = SPLIT_RE.match(name.strip())
    if matched is None:
        raise ValueError("run folder must match splitNN, got {}".format(name))
    return int(matched.group(1))


def _read_split_summary(run_dir: str) -> Tuple[int, pd.DataFrame, Dict[str, object]]:
    """Load one split folder summary and config."""
    name = os.path.basename(run_dir.rstrip(os.sep))
    split_id = _split_id_from_name(name)
    summary_path = os.path.join(run_dir, "tables", "summary.csv")
    if not os.path.isfile(summary_path):
        raise FileNotFoundError(summary_path)
    frame = pd.read_csv(summary_path)
    required = {"measurement_ratio", "method", "rmse_mean"}
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
    frame = frame[frame["method"].isin(FOCUS_METHODS)].copy()
    frame.insert(0, "split_id", split_id)
    return split_id, frame, meta


def _paired_rows(long_df: pd.DataFrame) -> pd.DataFrame:
    """One row per split x rho with CLP and AE RMSE."""
    clp = long_df[long_df["method"] == CLP_METHOD][
        ["split_id", "measurement_ratio", "rmse_mean"]
    ].rename(columns={"rmse_mean": "clp_rmse"})
    ae = long_df[long_df["method"] == AE_METHOD][
        ["split_id", "measurement_ratio", "rmse_mean"]
    ].rename(columns={"rmse_mean": "ae_rmse"})
    merged = clp.merge(ae, on=["split_id", "measurement_ratio"], how="inner")
    merged["delta_rmse"] = merged["ae_rmse"] - merged["clp_rmse"]
    merged["clp_wins"] = merged["clp_rmse"] < merged["ae_rmse"]
    return merged.sort_values(["measurement_ratio", "split_id"]).reset_index(drop=True)


def _summary_by_rho(paired: pd.DataFrame, n_train: int, n_val: int) -> pd.DataFrame:
    """Mean/std across splits for the appendix table."""
    rows: List[Dict[str, object]] = []
    for rho, grp in paired.groupby("measurement_ratio", sort=True):
        n_splits = int(grp.shape[0])
        clp_mean = float(grp["clp_rmse"].mean())
        clp_std = float(grp["clp_rmse"].std(ddof=1)) if n_splits > 1 else 0.0
        ae_mean = float(grp["ae_rmse"].mean())
        ae_std = float(grp["ae_rmse"].std(ddof=1)) if n_splits > 1 else 0.0
        delta_mean = float(grp["delta_rmse"].mean())
        win_frac = float(grp["clp_wins"].mean())
        rows.append(
            {
                "rho": float(rho),
                "n_splits": n_splits,
                "n_train": int(n_train),
                "n_val": int(n_val),
                "clp_rmse_mean": clp_mean,
                "clp_rmse_std": clp_std,
                "ae_rmse_mean": ae_mean,
                "ae_rmse_std": ae_std,
                "delta_rmse_mean": delta_mean,
                "clp_win_frac": win_frac,
            }
        )
    return pd.DataFrame(rows)


def _write_latex(summary: pd.DataFrame, path: str) -> None:
    """Write a compact booktabs fragment for the appendix."""
    lines = [
        r"\begin{tabular}{rrrrrrrr}",
        r"\toprule",
        r"$\rho$ & $n_{\mathrm{splits}}$ & $n_{\mathrm{train}}$ & $n_{\mathrm{val}}$ & "
        r"CLP RMSE & AE RMSE & mean $\Delta$RMSE & CLP win frac \\",
        r"\midrule",
    ]
    for _, row in summary.iterrows():
        lines.append(
            "{:.2f} & {:d} & {:d} & {:d} & {:.4f} $\\pm$ {:.4f} & {:.4f} $\\pm$ {:.4f} & "
            "{:.4f} & {:.2f} \\\\".format(
                float(row["rho"]),
                int(row["n_splits"]),
                int(row["n_train"]),
                int(row["n_val"]),
                float(row["clp_rmse_mean"]),
                float(row["clp_rmse_std"]),
                float(row["ae_rmse_mean"]),
                float(row["ae_rmse_std"]),
                float(row["delta_rmse_mean"]),
                float(row["clp_win_frac"]),
            )
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", ""])
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate R1-M03 step-32 split runs.")
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
        name for name in os.listdir(runs_dir) if SPLIT_RE.match(name) is not None
    )
    if not names:
        raise SystemExit("no splitNN folders under {}".format(runs_dir))
    chunks: List[pd.DataFrame] = []
    n_train = 0
    n_val = 0
    for name in names:
        run_dir = os.path.join(runs_dir, name)
        _sid, frame, meta = _read_split_summary(run_dir)
        chunks.append(frame)
        if "n_train" in meta:
            n_train = int(meta["n_train"])
        if "n_val" in meta:
            n_val = int(meta["n_val"])
    long_df = pd.concat(chunks, ignore_index=True)
    paired = _paired_rows(long_df)
    summary = _summary_by_rho(paired, n_train=n_train, n_val=n_val)
    agg_dir = os.path.abspath(str(args.agg_dir))
    os.makedirs(agg_dir, exist_ok=True)
    long_path = os.path.join(agg_dir, "per_split_focus.csv")
    paired_path = os.path.join(agg_dir, "paired_delta_clp_vs_ae.csv")
    summary_path = os.path.join(agg_dir, "summary_by_rho.csv")
    long_df.to_csv(long_path, index=False)
    paired.to_csv(paired_path, index=False)
    summary.to_csv(summary_path, index=False)
    paper_tab = os.path.abspath(str(args.paper_tab))
    os.makedirs(paper_tab, exist_ok=True)
    paper_csv = os.path.join(paper_tab, "robustness_crosswell_step32_dev_splits.csv")
    paper_tex = os.path.join(paper_tab, "robustness_crosswell_step32_dev_splits.tex")
    summary.to_csv(paper_csv, index=False)
    _write_latex(summary, paper_tex)
    print("wrote", long_path)
    print("wrote", paired_path)
    print("wrote", summary_path)
    print("wrote", paper_csv)
    print("wrote", paper_tex)


if __name__ == "__main__":
    main()
