#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Two-panel depth preview: small noise and large noise. ASCII only.

One held-out interval, seed 7, step 32, rho 0.20. Each panel shows the
processed curve, depth interpolation, CLP, and the sparse measurements.
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

from csgm_m2_module import csgm_recover_with_prior
from scripts.plot_r3m03_depth_profiles import (
    _load_test_depth,
    _measurement_profile,
    _stitch,
)
from scripts.run_r3m03_noise_sweep import (
    _draw_unit_noise,
    _fit_generator,
    _load_split,
    _paper_config,
    _scale_noise,
    _select_lambda,
)
from scripts.run_r3m03_simple_baselines import _interpolate, _omega_positions

OUT_PATH = (
    ROOT_DIR
    / "outputs"
    / "reviewer_revision"
    / "r3m03_noise_sweep"
    / "figures"
    / "preview_step32_rho020_seed7.png"
)
STEP = 32
RHO = 0.20
SEED = 7
NOISE_LEVELS = (0.01, 0.05)


def _recover(
    ae: object,
    cfg: object,
    measurement: np.ndarray,
    b_val: np.ndarray,
    b_test: np.ndarray,
    y_val: np.ndarray,
    z0_val: np.ndarray,
    z0_test: np.ndarray,
    y_mean: np.ndarray,
    y_scale: np.ndarray,
) -> np.ndarray:
    """CLP test windows after validation chooses lambda."""
    lam, val_score = _select_lambda(
        ae,
        measurement,
        b_val,
        y_val,
        z0_val,
        y_mean,
        y_scale,
        SEED,
        cfg,
    )
    print("selected lambda", lam, "val rmse", round(val_score, 5))
    y_hat = csgm_recover_with_prior(
        ae=ae,
        M=measurement,
        B=b_test,
        z0_np=z0_test,
        y_mean=y_mean,
        y_scale=y_scale,
        lam=float(lam),
        n_iters=int(cfg.csgm_iters),
        opt_lr=float(cfg.csgm_opt_lr),
        n_restarts=int(cfg.csgm_restarts),
        device="cpu",
        seed=SEED,
    )
    return np.asarray(y_hat, dtype=np.float64)


def main() -> None:
    """Write the two-panel preview."""
    cfg = _paper_config()
    data = _load_split(STEP)
    depth = _load_test_depth()
    n_rows = int(list(data["meta"]["test_wells_rows"].values())[0])
    if depth.shape[0] != n_rows:
        raise RuntimeError("Depth length does not match the test well.")
    starts = np.asarray(data["meta"]["test_row_start_full"], dtype=np.int64)
    print("fit generator", flush=True)
    ae, y_mean, y_scale, z0_val, z0_test = _fit_generator(cfg, SEED, data)
    (
        measurement,
        _clean_train,
        clean_val,
        clean_test,
        _unit_train,
        unit_val,
        unit_test,
    ) = _draw_unit_noise(
        data["Y_train"], data["Y_val"], data["Y_test"], SEED, RHO
    )
    positions = _omega_positions(measurement)
    length = int(data["Y_test"].shape[1])
    truth = _stitch(np.asarray(data["Y_test"], dtype=np.float64), starts, n_rows)
    panels: Dict[float, Dict[str, np.ndarray]] = {}
    for noise_std in NOISE_LEVELS:
        b_val = _scale_noise(clean_val, unit_val, float(noise_std))
        b_test = _scale_noise(clean_test, unit_test, float(noise_std))
        interp = _interpolate(b_test, positions, length)
        clp = _recover(
            ae,
            cfg,
            measurement,
            b_val,
            b_test,
            data["Y_val"],
            z0_val,
            z0_test,
            y_mean,
            y_scale,
        )
        panels[float(noise_std)] = {
            "interp": _stitch(interp, starts, n_rows),
            "clp": _stitch(clp, starts, n_rows),
            "measurements": _measurement_profile(starts, positions, b_test, n_rows),
        }
        finite = np.isfinite(truth)
        for name in ("interp", "clp"):
            err = panels[float(noise_std)][name][finite] - truth[finite]
            print(
                "noise",
                noise_std,
                name,
                "stitched rmse",
                round(float(np.sqrt(np.mean(err * err))), 5),
            )

    fig, axes = plt.subplots(1, 2, figsize=(7.4, 8.2), sharey=True)
    titles = {0.01: "(a) Noise std 0.01", 0.05: "(b) Noise std 0.05"}
    for ax, noise_std in zip(axes, NOISE_LEVELS):
        panel = panels[float(noise_std)]
        ax.plot(truth, depth, color="black", lw=1.5, label="Processed Vc")
        ax.plot(panel["interp"], depth, color="#d95f02", lw=1.2, label="Interpolation")
        ax.plot(panel["clp"], depth, color="#e7298a", lw=1.2, label="CLP-CSGM Ridge")
        observed = np.isfinite(panel["measurements"])
        ax.scatter(
            panel["measurements"][observed],
            depth[observed],
            s=10,
            c="0.35",
            zorder=5,
            label="Sparse measurements",
        )
        ax.set_title(titles[float(noise_std)])
        ax.set_xlabel("Clay volume")
        pieces = [
            truth,
            panel["interp"],
            panel["clp"],
            panel["measurements"][np.isfinite(panel["measurements"])],
        ]
        stacked = np.concatenate([np.asarray(item, dtype=np.float64).ravel() for item in pieces])
        x_lo = float(np.min(stacked))
        x_hi = float(np.max(stacked))
        pad = 0.04 * (x_hi - x_lo)
        ax.set_xlim(x_lo - pad, x_hi + pad)
        ax.invert_yaxis()
    axes[0].set_ylabel("Depth")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, fontsize=8, frameon=False)
    fig.tight_layout(rect=(0.0, 0.04, 1.0, 1.0))
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PATH, dpi=150)
    plt.close(fig)
    print("wrote", OUT_PATH)


if __name__ == "__main__":
    main()
