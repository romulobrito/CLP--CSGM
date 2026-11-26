#!/usr/bin/env python3
"""Run the four B.7 channel/embargo cases and aggregate Table 13 locally."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "outputs/reviewer_revision/b7_gr_embargo"
CASES = (("Emb4+GR", 4, True), ("Emb4-noGR", 4, False),
         ("NoEmb+GR", 0, True), ("NoEmb-noGR", 0, False))
RHOS = (0.05, 0.10, 0.20)
SEEDS = (7, 23, 41)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--train-path", default=os.environ.get("CLP_CROSS_TRAIN", "data/F02-1,F03-2,F06-1_6logs_30dB.txt"))
    p.add_argument("--test-path", default=os.environ.get("CLP_CROSS_TEST", "data/F03-4_6logs_30dB.txt"))
    p.add_argument("--aggregate-only", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()
    rows = []
    for label, embargo, gr in CASES:
        channels = "sonic,rhob,gr,ai,vp" if gr else "sonic,rhob,ai,vp"
        command = [sys.executable, "sir_cs_benchmark_multi_well_vc.py",
                   "--train-path", args.train_path, "--test-path", args.test_path,
                   "--channels", channels, "--target", "vc", "--window-len", "64",
                   "--step", "16", "--rhos", "0.05,0.10,0.20", "--seeds", "7,23,41",
                   "--val-frac", "0.1", "--val-split-mode", "tail",
                   "--val-embargo-windows", str(embargo), "--measurement-noise-std", "0.01",
                   "--run-csgm-m2", "--run-csgm-ablations", "--csgm-prior-type", "ridge",
                   "--no-lfista", "--no-plots", "--no-parity", "--base-dir", str(BASE), "--run-id", label]
        if args.dry_run:
            print(json.dumps(command))
            continue
        if not args.aggregate_only:
            subprocess.run(command, cwd=ROOT, check=True)
        run = BASE / "runs" / label
        cfg = json.loads((run / "config.json").read_text())
        meta = cfg["cross_well_vc"]
        if (meta["channels"] != channels.split(",") or meta["val_embargo_windows"] != embargo
            or meta["step"] != 16 or meta["window_len"] != 64
            or meta["val_split_mode"] != "tail" or meta["val_frac"] != 0.1
            or meta["train_wells"] != ["F02-1", "F03-2", "F06-1"] or meta["test_wells"] != ["F03-4"]
            or (meta["n_train"], meta["n_val"], meta["n_test"]) != (41 if embargo else 52, 6, 49)
            or cfg["seeds"] != list(SEEDS) or cfg["measurement_ratios"] != list(RHOS)
            or cfg["measurement_noise_std"] != 0.01 or cfg["csgm_prior_type"] != "ridge"):
            raise RuntimeError(f"Protocol mismatch: {label}")
        per = pd.read_csv(run / "tables/summary_by_seed.csv")
        for rho in RHOS:
            sub = per[np.isclose(per.measurement_ratio, rho)]
            values = {}
            for method in ("ridge_prior_csgm", "ae_regression_ub", "measurement_only_csgm"):
                part = sub[sub.method == method]
                if len(part) != 3 or set(part.seed) != set(SEEDS) or not np.isfinite(part.rmse_mean).all():
                    raise RuntimeError(f"Missing/invalid seed cells: {label}/{rho}/{method}")
                values[method] = part.rmse_mean.to_numpy()
            clp, ae, meas = (values[k] for k in ("ridge_prior_csgm", "ae_regression_ub", "measurement_only_csgm"))
            c, a, m = map(np.mean, (clp, ae, meas))
            rows.append({"scenario": label, "rho": rho, "ridge_rmse_mean": c,
                         "ridge_rmse_std": clp.std(ddof=1), "ae_rmse": a,
                         "measurement_only_rmse": m,
                         "ridge_gain_vs_ae_pct": 100*(a-c)/a,
                         "ridge_gain_vs_meas_only_pct": 100*(m-c)/m})
    if args.dry_run:
        return
    out = BASE / "aggregate"
    out.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows)
    frame.to_csv(out / "table13.csv", index=False)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6, 4))
    for label, _, _ in CASES:
        part = frame[frame.scenario == label]
        ax.errorbar(part.rho, part.ridge_rmse_mean, yerr=part.ridge_rmse_std, label=label, marker="o", capsize=3)
    ax.set(xlabel="Measurement ratio", ylabel="RMSE")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "figure20.png", dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    main()
