#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Aggregate the R2-04 Omega-placement runs. ASCII only."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

ROOT = Path("outputs/reviewer_revision/r2m04_omega_placement")
OFFSETS = (0, 10007, 20011, 30013, 40009)
STEPS = (8, 16, 32)
FOCUS = "summary_by_seed_focus_clp_csgm_vs_ub.csv"
CLP = "ridge_prior_csgm"
AE = "ae_regression_ub"


def _run_dir(offset: int, step: int) -> Path:
    name = "offset{0:05d}_step{1:02d}".format(offset, step)
    return ROOT / "runs" / name


def _load_focus(offset: int, step: int) -> pd.DataFrame:
    path = _run_dir(offset, step) / "tables" / FOCUS
    if not path.is_file():
        raise FileNotFoundError(str(path))
    frame = pd.read_csv(path)
    keep = frame[frame["method"].isin([CLP, AE])].copy()
    keep["measurement_seed_offset"] = int(offset)
    keep["step"] = int(step)
    keep["draw"] = list(OFFSETS).index(int(offset))
    return keep


def _load_omega(offset: int, step: int) -> pd.DataFrame:
    path = _run_dir(offset, step) / "tables" / "omega_sets.csv"
    if not path.is_file():
        raise FileNotFoundError(str(path))
    frame = pd.read_csv(path)
    frame["step"] = int(step)
    frame["draw"] = list(OFFSETS).index(int(offset))
    return frame


def _paired(focus: pd.DataFrame) -> pd.DataFrame:
    keys = ["step", "draw", "measurement_seed_offset", "seed", "measurement_ratio"]
    clp = focus[focus["method"] == CLP][keys + ["rmse_mean"]].rename(
        columns={"rmse_mean": "clp_rmse"}
    )
    ae = focus[focus["method"] == AE][keys + ["rmse_mean"]].rename(
        columns={"rmse_mean": "ae_rmse"}
    )
    paired = clp.merge(ae, on=keys, how="inner")
    paired["delta_ae_minus_clp"] = paired["ae_rmse"] - paired["clp_rmse"]
    paired["clp_lower"] = paired["delta_ae_minus_clp"] > 0.0
    return paired.sort_values(keys).reset_index(drop=True)


def _cell_summary(paired: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (step, rho), group in paired.groupby(["step", "measurement_ratio"], sort=True):
        published = group[group["draw"] == 0]
        published_clp = int(published["clp_lower"].sum())
        published_n = int(len(published))
        published_winner = "CLP" if published_clp * 2 >= published_n else "AE"
        clp_n = int(group["clp_lower"].sum())
        n = int(len(group))
        majority = "CLP" if clp_n * 2 >= n else "AE"
        rows.append(
            {
                "step": int(step),
                "rho": float(rho),
                "n": n,
                "clp_rmse_mean": float(group["clp_rmse"].mean()),
                "ae_rmse_mean": float(group["ae_rmse"].mean()),
                "delta_mean": float(group["delta_ae_minus_clp"].mean()),
                "clp_wins": clp_n,
                "ae_wins": n - clp_n,
                "published_winner": published_winner,
                "majority_winner": majority,
                "majority_matches_published": majority == published_winner,
            }
        )
    return pd.DataFrame(rows)


def _check_omega(omega: pd.DataFrame) -> None:
    keys = ["step", "model_seed", "measurement_ratio", "omega_order"]
    counts = omega.groupby(keys, dropna=False).size()
    repeated = counts[counts > 1]
    if len(repeated):
        raise RuntimeError("Repeated Omega inside one draw: {0}".format(repeated.head(5)))
    identity = ["step", "model_seed", "measurement_ratio"]
    nunique = omega.groupby(identity)["omega_order"].nunique()
    short = nunique[nunique < len(OFFSETS)]
    if len(short):
        raise RuntimeError("A seed-rho cell has fewer than 5 distinct Omegas.")


def main() -> None:
    """Write placement detail, cell summary, and the joined Omega table."""
    focus = pd.concat([_load_focus(offset, step) for offset in OFFSETS for step in STEPS], ignore_index=True)
    omega = pd.concat([_load_omega(offset, step) for offset in OFFSETS for step in STEPS], ignore_index=True)
    _check_omega(omega)
    paired = _paired(focus)
    summary = _cell_summary(paired)
    out = ROOT / "aggregate" / "tables"
    out.mkdir(parents=True, exist_ok=True)
    paired.to_csv(out / "placement_detail.csv", index=False)
    summary.to_csv(out / "summary_by_cell.csv", index=False)
    omega.to_csv(out / "omega_sets.csv", index=False)
    payload = {
        "offsets": list(OFFSETS),
        "steps": list(STEPS),
        "seeds": [7, 23, 41],
        "rhos": [0.05, 0.10, 0.20],
        "cells_matching_published_winner": int(summary["majority_matches_published"].sum()),
        "n_cells": int(len(summary)),
    }
    with (ROOT / "aggregate" / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    print(summary.to_string(index=False))
    print("cells matching published winner:", payload["cells_matching_published_winner"], "/", payload["n_cells"])


if __name__ == "__main__":
    main()
