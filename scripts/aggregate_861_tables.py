#!/usr/bin/env python3
"""Aggregate well-861 Tables 23 and 24 from complete local fold/seed results."""
from __future__ import annotations
import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd

import run_r1m11_861_phi_lab as protocol

METHODS = ("clp_csgm_ridge", "ae_regression_ub", "prior_only")
KEYS = ["fold_id", "seed", "rho", "method"]


def aggregate(detail):
    expected = set(itertools.product(range(3), protocol.SEEDS, protocol.RHOS, METHODS))
    actual = set(detail[KEYS].itertuples(index=False, name=None))
    if len(detail) != len(expected) or actual != expected:
        raise ValueError("Expected exactly 54 distinct fold/seed/rho/method rows.")
    if not np.isfinite(detail.rmse_holdout).all() or (detail.rmse_holdout < 0).any():
        raise ValueError("Invalid RMSE.")
    paired = detail.pivot(index=["fold_id", "seed", "rho"], columns="method", values="rmse_holdout").reset_index()
    paired["delta_ae_minus_clp"] = paired.ae_regression_ub - paired.clp_csgm_ridge
    paired["clp_wins"] = paired.delta_ae_minus_clp > 0
    pooled = paired.groupby("rho", as_index=False).agg(
        clp=("clp_csgm_ridge", "mean"), ae=("ae_regression_ub", "mean"),
        prior_only=("prior_only", "mean"), delta_ae_minus_clp=("delta_ae_minus_clp", "mean"),
        clp_wins=("clp_wins", "sum"), n_pairs=("clp_wins", "size"))
    blocks = paired.groupby(["fold_id", "rho"], as_index=False).agg(
        clp=("clp_csgm_ridge", "mean"), ae=("ae_regression_ub", "mean"),
        prior_only=("prior_only", "mean"), n_pairs=("clp_wins", "size"))
    return pooled, blocks


def self_test():
    rows = []
    for fold, seed, rho, method in itertools.product(range(3), protocol.SEEDS, protocol.RHOS, METHODS):
        # Constructed fixture: CLP wins two blocks, AE wins one;
        # pooled AE is better. No field data or published scores.
        v = {"clp_csgm_ridge": (1., 1., 5.)[fold], "ae_regression_ub": 2., "prior_only": 6.}[method]
        rows.append(dict(zip(KEYS, (fold, seed, rho, method)), rmse_holdout=v))
    pooled, blocks = aggregate(pd.DataFrame(rows))
    assert len(blocks) == 6 and (blocks.n_pairs == 3).all()
    assert (pooled.clp_wins == 6).all() and (pooled.n_pairs == 9).all()
    assert np.allclose(pooled.delta_ae_minus_clp, -1/3)
    try:
        aggregate(pd.DataFrame(rows[:-1]))
    except ValueError:
        pass
    else:
        raise AssertionError("Incomplete grid was accepted")
    print("861 table aggregation self-test passed.")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-dir", type=Path, default=protocol.OUT_DIR)
    p.add_argument("--self-test", action="store_true")
    args = p.parse_args()
    if args.self_test:
        self_test()
        return
    meta = json.loads((args.run_dir / "protocol.json").read_text())
    if (meta["n_rows"] != 87 or meta["window_len"] != 8 or meta["step"] != 1
        or meta["n_blocks"] != 3 or meta["val_rows"] != 16 or meta["seeds"] != list(protocol.SEEDS)
        or meta["rhos"] != list(protocol.RHOS) or meta["measurement_noise_std"] != 0
        or meta["u_columns"] != list(protocol.U_COLUMNS)):
        raise ValueError("Input is not the published 861 protocol.")
    pooled, blocks = aggregate(pd.read_csv(args.run_dir / "heldout_rmse_detail.csv"))
    pooled.to_csv(args.run_dir / "table23_paired.csv", index=False)
    blocks.to_csv(args.run_dir / "table24_by_block.csv", index=False)
    print(pooled.to_string(index=False))
    print(blocks.to_string(index=False))


if __name__ == "__main__":
    main()
