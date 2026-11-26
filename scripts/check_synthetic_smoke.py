#!/usr/bin/env python3
"""Validate end-to-end synthetic smoke artifacts; not a scientific ranking test."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "outputs/synthetic_smoke"
METHODS = {"ml_only", "mlp_concat_ub", "pca_regression_ub", "ae_regression_ub",
           "ridge_prior_csgm", "ridge_prior_only_decoder", "measurement_only_csgm"}


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def inspect_run(relative, meta_key, channels_key, channels, rhos):
    run = BASE / relative
    cfg = json.loads((run / "config.json").read_text())
    meta = cfg[meta_key]
    require(meta[channels_key] == channels, f"{relative}: channels")
    require(cfg["seeds"] == [7] and cfg["measurement_ratios"] == rhos, f"{relative}: smoke grid")
    require(cfg["run_csgm_m2"] and cfg["run_csgm_ablations"] and not cfg["run_lfista"],
            f"{relative}: method switches")
    require(cfg["csgm_prior_type"] == "ridge" and cfg["csgm_ae_epochs"] == 8
            and cfg["csgm_iters"] == 20 and cfg["csgm_restarts"] == 1,
            f"{relative}: smoke budgets")
    require(all(int(meta[k]) > 0 for k in ("n_train", "n_val", "n_test")),
            f"{relative}: empty partition")
    require(int(meta["window_len"]) == 64 and int(meta["p_input"]) == 64 * len(channels),
            f"{relative}: input dimensions")
    if meta_key == "real_well_f03":
        require(meta["split_before_windowing"], "F03 split must precede windowing")
    else:
        require(set(meta["train_wells"]).isdisjoint(meta["test_wells"]), "Well leakage")
    for name in ("DATASET_MANIFEST.txt", "PROTOCOL.txt", "RUN_MANIFEST.txt"):
        require((run / name).stat().st_size > 0, f"{relative}: empty {name}")
    tables = run / "tables"
    detail = pd.read_csv(tables / "detailed_results.csv")
    per = pd.read_csv(tables / "summary_by_seed.csv")
    summary = pd.read_csv(tables / "summary.csv")
    keys = {(7, rho, method) for rho in rhos for method in METHODS}
    actual = set(zip(per.seed, per.measurement_ratio, per.method))
    require(keys <= actual and not per.duplicated(["seed", "measurement_ratio", "method"]).any(),
            f"{relative}: missing or duplicated method cells")
    n_test = int(meta["n_test"])
    for seed, rho, method in sorted(keys):
        d = detail[(detail.seed == seed) & (detail.measurement_ratio == rho) & (detail.method == method)]
        p = per[(per.seed == seed) & (per.measurement_ratio == rho) & (per.method == method)]
        s = summary[(summary.measurement_ratio == rho) & (summary.method == method)]
        require(len(d) == n_test and d.sample_id.nunique() == n_test and len(p) == len(s) == 1,
                f"{relative}: incorrect sample count for {method}/{rho}")
        require(np.isfinite(d[["rmse", "mae", "relative_l2"]].to_numpy()).all(),
                f"{relative}: nonfinite metrics")
        require(np.isclose(d.rmse.mean(), p.rmse_mean.iloc[0], rtol=1e-10, atol=1e-12)
                and np.isclose(p.rmse_mean.iloc[0], s.rmse_mean.iloc[0], rtol=1e-10, atol=1e-12),
                f"{relative}: aggregation mismatch")
        if method == "ridge_prior_csgm":
            require(np.isfinite(d["val_score"]).all() and np.isfinite(d["ae_recon_train_rmse"]).all()
                    and np.allclose(d["lambda"], 0.01), f"{relative}: refinement diagnostics")
        if method == "measurement_only_csgm":
            require(np.allclose(d["lambda"], 0), f"{relative}: ablation lambda")
    with np.load(tables / "parity_pooled.npz", allow_pickle=False) as archive:
        ref = archive["y_true"]
        require(ref.size > 0 and np.isfinite(ref).all(), f"{relative}: invalid parity reference")
        for method in ("ml_only", "mlp_concat_ub", "pca_regression_ub", "ae_regression_ub", "ridge_prior_csgm"):
            v = archive[method]
            require(v.shape == ref.shape and np.isfinite(v).all(), f"{relative}: invalid parity {method}")
    return {"run": relative, "channels": channels, "seeds": [7], "rhos": rhos,
            "test_windows": n_test, "required_method_cells": len(keys),
            "detailed_rows": len(detail),
            "rmse": summary[summary.method.isin(METHODS)][["measurement_ratio", "method", "rmse_mean"]].to_dict("records")}


def main():
    report_path = ROOT / "outputs/validation/synthetic.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report = {"status": "fail", "scope": "Synthetic execution only; not manuscript numerical reproduction.", "runs": []}
    try:
        report["runs"].append(inspect_run("cross_well_vc/runs/synthetic_crosswell_smoke",
            "cross_well_vc", "channels", ["sonic", "rhob", "ai", "vp"], [0.1, 0.2]))
        report["runs"].append(inspect_run("real_well/runs/synthetic_porosity_smoke",
            "real_well_f03", "u_channels", ["gr"], [0.3, 0.5]))
        report["status"] = "pass"
        print("Synthetic artifact verification passed: 2 benchmarks, 28 required method cells.")
    except Exception as exc:
        report["error"] = str(exc)
        raise
    finally:
        report_path.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
