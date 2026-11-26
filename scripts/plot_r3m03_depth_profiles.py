#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Depth tracks for two R3-03 cells. ASCII only.

Seed 7, published measurement draw. Step 32 at rho 0.20 and step 8 at rho 0.05.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Dict

os.environ["CUDA_VISIBLE_DEVICES"] = ""
ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

import direct_ub_baselines as dub
import multi_well_vc as mwv
from csgm_m2_module import (
    _make_prior,
    csgm_recover_with_prior,
    encode_y,
    train_ae_generator,
)
from real_well_f03 import reconstruct_depth_profile
from scripts.run_r3m03_simple_baselines import (
    CHANNELS,
    TEST_PATH,
    TRAIN_PATH,
    VAL_FRAC,
    WINDOW_LEN,
    _draw_measurements,
    _fit_linear_family,
    _interpolate,
    _mean_window_rmse,
    _omega_positions,
)
from sir_cs_pipeline_optimized import Config

OUT_DIR = ROOT_DIR / "outputs" / "reviewer_revision" / "r3m03_simple_baselines" / "figures"
CURVE_NAMES = ("truth", "interp_sparse", "linear_ub", "ridge_direct_ub", "clp")
CASES = (
    {"step": 32, "rho": 0.20, "seed": 7, "tag": "step32_rho020_seed7"},
    {"step": 8, "rho": 0.05, "seed": 7, "tag": "step08_rho005_seed7"},
)


def _load_test_depth() -> np.ndarray:
    """Depth samples of the held-out well, in stored order."""
    segments = mwv.load_6log_file(
        str(ROOT_DIR / TEST_PATH), target_name="vc", channels=CHANNELS
    )
    if len(segments) != 1:
        raise RuntimeError("Expected one test well, got {0}.".format(len(segments)))
    return np.asarray(segments[0].depth, dtype=np.float64).ravel()


def _paper_config() -> Config:
    """CSGM settings used by the published cross-well grid."""
    cfg = Config()
    cfg.csgm_device = "cpu"
    cfg.csgm_prior_type = "ridge"
    cfg.csgm_latent_dim = 16
    cfg.csgm_hidden_dim = 128
    cfg.csgm_ae_epochs = 200
    cfg.csgm_batch_size = 64
    cfg.csgm_ae_lr = 1.0e-3
    cfg.csgm_weight_decay = 1.0e-5
    cfg.csgm_iters = 400
    cfg.csgm_restarts = 3
    cfg.csgm_opt_lr = 0.05
    cfg.csgm_lambda_grid = [1.0e-4, 3.0e-4, 1.0e-3, 3.0e-3, 1.0e-2, 3.0e-2, 1.0e-1]
    return cfg


def _clp_windows(
    cfg: Config,
    seed: int,
    data: Dict[str, np.ndarray],
    measurement: np.ndarray,
    b_val: np.ndarray,
    b_test: np.ndarray,
) -> np.ndarray:
    """CLP test windows. Lambda is the validation RMSE minimum."""
    y_scaler = StandardScaler().fit(data["Y_train"])
    y_mean = np.asarray(y_scaler.mean_, dtype=np.float64)
    y_scale = np.asarray(y_scaler.scale_, dtype=np.float64)
    y_train_n = y_scaler.transform(data["Y_train"])
    ae = train_ae_generator(y_train_n, cfg, seed=int(seed), device="cpu")
    z_train = encode_y(ae, y_train_n, device="cpu")
    prior = _make_prior(cfg, seed=int(seed) + 7000).fit(data["X_train"], z_train)
    z0_val = prior.predict(data["X_val"])
    z0_test = prior.predict(data["X_test"])
    best_lam = float(cfg.csgm_lambda_grid[0])
    best_score = float("inf")
    for lam in list(cfg.csgm_lambda_grid):
        y_val_hat = csgm_recover_with_prior(
            ae=ae,
            M=measurement,
            B=b_val,
            z0_np=z0_val,
            y_mean=y_mean,
            y_scale=y_scale,
            lam=float(lam),
            n_iters=int(cfg.csgm_iters),
            opt_lr=float(cfg.csgm_opt_lr),
            n_restarts=int(cfg.csgm_restarts),
            device="cpu",
            seed=int(seed),
        )
        score = _mean_window_rmse(data["Y_val"], y_val_hat)
        if score < best_score:
            best_score = float(score)
            best_lam = float(lam)
    print("selected lambda", best_lam, "val rmse", best_score)
    return csgm_recover_with_prior(
        ae=ae,
        M=measurement,
        B=b_test,
        z0_np=z0_test,
        y_mean=y_mean,
        y_scale=y_scale,
        lam=best_lam,
        n_iters=int(cfg.csgm_iters),
        opt_lr=float(cfg.csgm_opt_lr),
        n_restarts=int(cfg.csgm_restarts),
        device="cpu",
        seed=int(seed),
    )


def _predict_case(step: int, rho: float, seed: int) -> Dict[str, np.ndarray]:
    """Window-level curves for one published cell."""
    data = mwv.build_cross_well_data_dict(
        train_path=str(ROOT_DIR / TRAIN_PATH),
        test_path=str(ROOT_DIR / TEST_PATH),
        target_name="vc",
        channels=CHANNELS,
        window_len=WINDOW_LEN,
        step=int(step),
        val_frac=VAL_FRAC,
        val_embargo_windows=0,
        val_split_mode="tail",
        val_split_seed=0,
        residual_basis="dct",
    )
    measurement, b_train, b_val, b_test = _draw_measurements(
        data["Y_train"],
        data["Y_val"],
        data["Y_test"],
        int(seed),
        float(rho),
        int(data["Y_train"].shape[1]),
    )
    x_train = dub.concat_ub(data["X_train"], b_train)
    x_val = dub.concat_ub(data["X_val"], b_val)
    x_test = dub.concat_ub(data["X_test"], b_test)
    linear_pred, _ = _fit_linear_family(
        x_train, data["Y_train"], x_val, data["Y_val"], x_test, "linear_ub"
    )
    ridge_pred, _ = _fit_linear_family(
        x_train, data["Y_train"], x_val, data["Y_val"], x_test, "ridge_direct_ub"
    )
    positions = _omega_positions(measurement)
    interp_pred = _interpolate(b_test, positions, int(data["Y_test"].shape[1]))
    clp_pred = _clp_windows(_paper_config(), int(seed), data, measurement, b_val, b_test)
    n_rows = int(list(data["meta"]["test_wells_rows"].values())[0])
    starts = np.asarray(data["meta"]["test_row_start_full"], dtype=np.int64)
    if starts.shape[0] != int(data["Y_test"].shape[0]):
        raise RuntimeError("Test window starts do not match the test windows.")
    return {
        "truth": np.asarray(data["Y_test"], dtype=np.float64),
        "linear_ub": linear_pred,
        "ridge_direct_ub": ridge_pred,
        "interp_sparse": interp_pred,
        "clp": np.asarray(clp_pred, dtype=np.float64),
        "starts": starts,
        "n_rows": np.asarray([n_rows]),
        "positions": positions,
        "b_test": np.asarray(b_test, dtype=np.float64),
    }


def _stitch(windows: np.ndarray, starts: np.ndarray, n_rows: int) -> np.ndarray:
    """Overlap-average windows onto the well sample axis."""
    profile, _weights = reconstruct_depth_profile(
        windows, starts, WINDOW_LEN, int(n_rows), overlap_agg="uniform_mean"
    )
    return profile


def _measurement_profile(
    starts: np.ndarray,
    positions: np.ndarray,
    observations: np.ndarray,
    n_rows: int,
) -> np.ndarray:
    """Mean noisy measurement at each sample that at least one window observed."""
    acc = np.zeros(int(n_rows), dtype=np.float64)
    count = np.zeros(int(n_rows), dtype=np.float64)
    for i in range(int(starts.shape[0])):
        for j, pos in enumerate(positions):
            row = int(starts[i]) + int(pos)
            if 0 <= row < int(n_rows):
                acc[row] += float(observations[i, j])
                count[row] += 1.0
    profile = np.full(int(n_rows), np.nan, dtype=np.float64)
    hit = count > 0.0
    profile[hit] = acc[hit] / count[hit]
    return profile


def _plot_case(
    depth: np.ndarray,
    profiles: Dict[str, np.ndarray],
    measurements: np.ndarray,
    step: int,
    rho: float,
    path: Path,
) -> None:
    """One depth track. Deeper samples point downward."""
    fig, ax = plt.subplots(figsize=(6.2, 9.0))
    series = (
        ("truth", "Processed Vc", "black", 1.6),
        ("interp_sparse", "Interpolation", "#d95f02", 1.2),
        ("linear_ub", "Linear [u, b]", "#1b9e77", 1.0),
        ("ridge_direct_ub", "Ridge direct [u, b]", "#7570b3", 1.0),
        ("clp", "CLP-CSGM Ridge", "#e7298a", 1.2),
    )
    for key, label, color, width in series:
        ax.plot(profiles[key], depth, color=color, lw=width, label=label)
    observed = np.isfinite(measurements)
    ax.scatter(
        measurements[observed],
        depth[observed],
        s=8,
        c="0.25",
        zorder=5,
        label="Sparse measurements",
    )
    ax.set_xlabel("Clay volume")
    ax.set_ylabel("Depth")
    ax.set_title("F03-4, seed 7, step {0}, rho {1:.2f}".format(int(step), float(rho)))
    ax.invert_yaxis()
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> None:
    """Write one figure and one table per case."""
    depth = _load_test_depth()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for case in CASES:
        print("case", case["tag"])
        windows = _predict_case(int(case["step"]), float(case["rho"]), int(case["seed"]))
        n_rows = int(windows["n_rows"][0])
        if depth.shape[0] != n_rows:
            raise RuntimeError("Depth length does not match the test well.")
        profiles = {
            name: _stitch(windows[name], windows["starts"], n_rows)
            for name in CURVE_NAMES
        }
        measurements = _measurement_profile(
            windows["starts"], windows["positions"], windows["b_test"], n_rows
        )
        table = pd.DataFrame({"depth": depth, "sparse_measurement": measurements})
        for name in CURVE_NAMES:
            table[name] = profiles[name]
        csv_path = OUT_DIR / "{0}.csv".format(case["tag"])
        png_path = OUT_DIR / "{0}.png".format(case["tag"])
        table.to_csv(csv_path, index=False)
        _plot_case(depth, profiles, measurements, int(case["step"]), float(case["rho"]), png_path)
        finite = np.isfinite(profiles["truth"])
        for name in CURVE_NAMES:
            if name == "truth":
                continue
            err = profiles[name][finite] - profiles["truth"][finite]
            score = float(np.sqrt(np.mean(err * err)))
            print(name, "stitched rmse", round(score, 5))
        print("wrote", png_path)


if __name__ == "__main__":
    main()
