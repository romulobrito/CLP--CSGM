#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Build illustrative figures: one test window with ground truth, sparse noisy
measurements b at subsampled coordinates, and CLP-CSGM Ridge (optional AE).

Figures are written under paper_clp_csgm/figures/sparse_reconstruction/ for LaTeX.

This script re-runs a single (seed, measurement_ratio) benchmark cell using the
same data assembly and run_direct_ub_from_data entry point as the real-well and
cross-well launchers. It then replays the RNG prefix used to build M and B_test
so scatter coordinates match the training run.

ASCII-only.

Example:
  python scripts/paper_sparse_reconstruction_figures.py --mode f03 --skip-lfista
  python scripts/paper_sparse_reconstruction_figures.py --mode crosswell --skip-lfista --rho 0.1 --cross-step 16
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Dict, List, Optional, Tuple

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import matplotlib.pyplot as plt
import numpy as np

import direct_ub_baselines as dub
import multi_well_vc as mwv
import real_well_f03 as rwf
from sir_cs_benchmark_direct_ub import run_direct_ub_from_data
from sir_cs_benchmark_real_well_direct_ub import _contiguous_row_split
from sir_cs_pipeline_optimized import (
    Config,
    apply_config_profile,
    build_measurement_matrix,
    power_iteration_lipschitz,
)


def _parse_float_list(s: str) -> List[float]:
    out: List[float] = []
    for p in s.split(","):
        p = p.strip()
        if p:
            out.append(float(p))
    if not out:
        raise ValueError("empty float list")
    return out


def _parse_int_list(s: str) -> List[int]:
    out: List[int] = []
    for p in s.split(","):
        p = p.strip()
        if p:
            out.append(int(p))
    if not out:
        raise ValueError("empty int list")
    return out


def build_f03_data_dict(
    data_path: str,
    window_len: int,
    step: int,
    train_frac: float,
    val_frac: float,
    u_channels: Tuple[str, ...],
    residual_basis: str,
) -> Dict[str, np.ndarray]:
    tab = rwf.load_f03_table(os.path.abspath(data_path))
    n_rows_total = int(tab.n_rows)
    tr_row_sl, va_row_sl, te_row_sl, _, _, _ = _contiguous_row_split(
        n_rows_total, float(train_frac), float(val_frac), int(window_len)
    )
    tab_tr = rwf.F03Table(
        depth=tab.depth[tr_row_sl].copy(),
        ac=tab.ac[tr_row_sl].copy(),
        gr=tab.gr[tr_row_sl].copy(),
        porosity=tab.porosity[tr_row_sl].copy(),
    )
    tab_va = rwf.F03Table(
        depth=tab.depth[va_row_sl].copy(),
        ac=tab.ac[va_row_sl].copy(),
        gr=tab.gr[va_row_sl].copy(),
        porosity=tab.porosity[va_row_sl].copy(),
    )
    tab_te = rwf.F03Table(
        depth=tab.depth[te_row_sl].copy(),
        ac=tab.ac[te_row_sl].copy(),
        gr=tab.gr[te_row_sl].copy(),
        porosity=tab.porosity[te_row_sl].copy(),
    )
    x_tr, y_tr, _, _ = rwf.build_sliding_windows(tab_tr, int(window_len), int(step), channels=u_channels)
    x_va, y_va, _, _ = rwf.build_sliding_windows(tab_va, int(window_len), int(step), channels=u_channels)
    x_te, y_te, _, _ = rwf.build_sliding_windows(tab_te, int(window_len), int(step), channels=u_channels)
    n_tr = int(x_tr.shape[0])
    n_va = int(x_va.shape[0])
    n_te = int(x_te.shape[0])
    n_win = n_tr + n_va + n_te
    sl_tr = slice(0, n_tr)
    sl_va = slice(n_tr, n_tr + n_va)
    sl_te = slice(n_tr + n_va, n_win)
    x_all = np.concatenate([x_tr, x_va, x_te], axis=0)
    y_all = np.concatenate([y_tr, y_va, y_te], axis=0)
    return rwf.build_direct_ub_data_dict(x_all, y_all, sl_tr, sl_va, sl_te, str(residual_basis))


def configure_f03_cfg(
    data: Dict[str, np.ndarray],
    *,
    measurement_noise_std: float,
    residual_k: int,
    residual_basis: str,
    measurement_kind: str,
    csgm_prior_type: str,
    skip_lfista: bool,
) -> Config:
    p_in = int(data["X_train"].shape[1])
    l = int(data["Y_train"].shape[1])
    n_tr = int(data["X_train"].shape[0])
    n_va = int(data["X_val"].shape[0])
    n_te = int(data["X_test"].shape[0])
    cfg = Config()
    cfg.log_progress = False
    cfg.config_profile = "real_well_f03_direct_ub"
    apply_config_profile(cfg)
    cfg.p_input = p_in
    cfg.n_output = l
    cfg.n_train = n_tr
    cfg.n_val = n_va
    cfg.n_test = n_te
    cfg.residual_k = int(residual_k)
    cfg.residual_basis = str(residual_basis)
    cfg.measurement_kind = str(measurement_kind)
    cfg.measurement_noise_std = float(measurement_noise_std)
    cfg.run_lfista = not bool(skip_lfista)
    cfg.run_csgm_m2 = True
    cfg.run_csgm_ablations = False
    cfg.csgm_prior_type = str(csgm_prior_type).strip().lower()
    cfg.paper_strict_paired_b = True
    cfg.n_example_plots = min(4, max(1, n_te))
    return cfg


def configure_crosswell_cfg(
    data: Dict[str, np.ndarray],
    *,
    measurement_noise_std: float,
    residual_k: int,
    residual_basis: str,
    measurement_kind: str,
    csgm_prior_type: str,
    skip_lfista: bool,
    lfista_bg_epochs: int,
    bg_type: str,
    bg_hidden: Tuple[int, int],
) -> Config:
    p_in = int(data["X_train"].shape[1])
    l = int(data["Y_train"].shape[1])
    n_tr = int(data["X_train"].shape[0])
    n_va = int(data["X_val"].shape[0])
    n_te = int(data["X_test"].shape[0])
    cfg = Config()
    cfg.log_progress = False
    cfg.config_profile = "cross_well_vc_direct_ub"
    apply_config_profile(cfg)
    cfg.p_input = p_in
    cfg.n_output = l
    cfg.n_train = n_tr
    cfg.n_val = n_va
    cfg.n_test = n_te
    cfg.residual_k = int(residual_k)
    cfg.residual_basis = str(residual_basis)
    cfg.measurement_kind = str(measurement_kind)
    cfg.measurement_noise_std = float(measurement_noise_std)
    cfg.run_lfista = not bool(skip_lfista)
    cfg.run_csgm_m2 = True
    cfg.run_csgm_ablations = False
    cfg.csgm_prior_type = str(csgm_prior_type).strip().lower()
    cfg.paper_strict_paired_b = True
    cfg.n_example_plots = min(4, max(1, n_te))
    if int(lfista_bg_epochs) > 0:
        cfg.lfista_num_epochs_bg = int(lfista_bg_epochs)
    cfg.lfista_bg_type = str(bg_type).strip().lower()
    cfg.lfista_bg_hidden = (int(bg_hidden[0]), int(bg_hidden[1]))
    return cfg


def replay_M_and_B_test(
    cfg: Config,
    data: Dict[str, np.ndarray],
    seed: int,
    measurement_ratio: float,
) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(int(seed))
    m = max(4, int(round(float(measurement_ratio) * float(cfg.n_output))))
    M = build_measurement_matrix(m, int(cfg.n_output), str(cfg.measurement_kind), rng)
    Psi = data["Psi"]
    A = M @ Psi
    _ = power_iteration_lipschitz(A, n_iter=int(cfg.power_iteration_n_iter))
    B_train = dub.make_B(data["Y_train"], M, float(cfg.measurement_noise_std), rng)
    B_val = dub.make_B(data["Y_val"], M, float(cfg.measurement_noise_std), rng)
    B_test = dub.make_B(data["Y_test"], M, float(cfg.measurement_noise_std), rng)
    return M, B_test


def subsample_coord_indices(M: np.ndarray) -> np.ndarray:
    if int(M.shape[0]) == 0:
        return np.zeros((0,), dtype=np.int64)
    return np.asarray(np.argmax(M, axis=1), dtype=np.int64)


def plot_sparse_window(
    out_path: str,
    y_true: np.ndarray,
    y_csgm: np.ndarray,
    M: np.ndarray,
    b_vec: np.ndarray,
    y_ae: Optional[np.ndarray],
    title: str,
    ylabel: str,
) -> None:
    L = int(y_true.shape[0])
    x_idx = np.arange(L, dtype=np.float64)
    fig, ax = plt.subplots(figsize=(8.0, 3.8))
    ax.plot(x_idx, y_true, color="#1f77b4", linewidth=1.8, label="ground truth y")
    ax.plot(x_idx, y_csgm, color="#d62728", linewidth=1.4, label="CLP-CSGM Ridge")
    if y_ae is not None and y_ae.shape == y_true.shape:
        ax.plot(x_idx, y_ae, color="#2ca02c", linewidth=1.1, alpha=0.9, label="AE [u,b]")
    if M.ndim == 2 and int(M.shape[1]) == L and int(M.shape[0]) == int(b_vec.shape[0]):
        jj = subsample_coord_indices(M)
        ax.scatter(
            jj.astype(np.float64),
            b_vec,
            s=36.0,
            color="#000000",
            zorder=5,
            label="noisy measurements b",
        )
    ax.set_title(title, fontsize=10.5)
    ax.set_xlabel("window coordinate (depth index in window)")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser(description="Paper sparse-reconstruction window figures.")
    p.add_argument("--mode", type=str, choices=("f03", "crosswell"), required=True)
    p.add_argument("--out-dir", type=str, default="paper_clp_csgm/figures/sparse_reconstruction")
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--rho", type=float, default=0.2)
    p.add_argument("--example-index", type=int, default=0)
    p.add_argument("--skip-lfista", action="store_true", help="Skip LFISTA branch (faster).")
    p.add_argument("--measurement-noise-std", type=float, default=-1.0, help="Override; -1 uses mode default.")
    p.add_argument("--residual-k", type=int, default=-1, help="Override; -1 uses mode default.")
    # F03
    p.add_argument("--data-path", type=str, default="data/F03-4_AC+GR+Porosity.txt")
    p.add_argument("--window-len", type=int, default=64)
    p.add_argument("--f03-step", type=int, default=1)
    p.add_argument("--train-frac", type=float, default=0.6)
    p.add_argument("--f03-val-frac", type=float, default=0.2)
    p.add_argument("--u-channels", type=str, default="gr")
    p.add_argument("--f03-residual-basis", type=str, default="dct")
    # Cross-well
    p.add_argument("--train-path", type=str, default="data/F02-1,F03-2,F06-1_6logs_30dB.txt")
    p.add_argument("--test-path", type=str, default="data/F03-4_6logs_30dB.txt")
    p.add_argument("--channels", type=str, default="sonic,rhob,ai,vp")
    p.add_argument("--cross-step", type=int, default=16)
    p.add_argument("--cw-val-frac", type=float, default=0.1)
    p.add_argument("--val-embargo-windows", type=int, default=0)
    p.add_argument("--cross-residual-basis", type=str, default="dct")
    p.add_argument("--lfista-bg-epochs", type=int, default=150)
    p.add_argument("--bg-type", type=str, default="mlp2")
    p.add_argument("--bg-hidden", type=str, default="128,128")
    args = p.parse_args()

    mode = str(args.mode).strip().lower()
    out_dir = os.path.abspath(str(args.out_dir).strip())
    os.makedirs(out_dir, exist_ok=True)
    seed = int(args.seed)
    rho = float(args.rho)
    ex = int(args.example_index)
    skip_lf = bool(args.skip_lfista)
    dub_cfg = dub.DirectUBTrainConfig()
    joint_only = True
    include_hf = False
    run_ae = True
    include_lfista = not skip_lf

    if mode == "f03":
        noise = 0.02 if float(args.measurement_noise_std) < 0.0 else float(args.measurement_noise_std)
        rk = 6 if int(args.residual_k) < 0 else int(args.residual_k)
        u_channels_raw = tuple(c.strip() for c in str(args.u_channels).split(",") if c.strip())
        u_channels = rwf.normalize_channels(u_channels_raw)
        data_path = os.path.abspath(str(args.data_path).strip())
        if not os.path.isfile(data_path):
            print("Missing data file: " + data_path, file=sys.stderr)
            sys.exit(2)
        data = build_f03_data_dict(
            data_path,
            int(args.window_len),
            int(args.f03_step),
            float(args.train_frac),
            float(args.f03_val_frac),
            u_channels,
            str(args.f03_residual_basis),
        )
        cfg = configure_f03_cfg(
            data,
            measurement_noise_std=noise,
            residual_k=rk,
            residual_basis=str(args.f03_residual_basis),
            measurement_kind="subsample",
            csgm_prior_type="ridge",
            skip_lfista=skip_lf,
        )
        cfg.seeds = [seed]
        cfg.measurement_ratios = [rho]
        stem = "f03_gr_only"
        y_label = "porosity (processed curve scale)"
    else:
        noise = 0.01 if float(args.measurement_noise_std) < 0.0 else float(args.measurement_noise_std)
        rk = 6 if int(args.residual_k) < 0 else int(args.residual_k)
        train_path = os.path.abspath(str(args.train_path).strip())
        test_path = os.path.abspath(str(args.test_path).strip())
        if not os.path.isfile(train_path) or not os.path.isfile(test_path):
            print("Missing train or test file.", file=sys.stderr)
            sys.exit(2)
        channels = tuple(c.strip().lower() for c in str(args.channels).split(",") if c.strip())
        data = mwv.build_cross_well_data_dict(
            train_path=train_path,
            test_path=test_path,
            target_name="vc",
            channels=channels,
            window_len=int(args.window_len),
            step=int(args.cross_step),
            val_frac=float(args.cw_val_frac),
            val_embargo_windows=int(args.val_embargo_windows),
            residual_basis=str(args.cross_residual_basis),
        )
        pipe_data: Dict[str, np.ndarray] = {
            k: data[k]
            for k in (
                "X_train",
                "X_val",
                "X_test",
                "Y_train",
                "Y_val",
                "Y_test",
                "Alpha_train",
                "Alpha_val",
                "Alpha_test",
                "Psi",
            )
        }
        data = pipe_data
        hidden_sizes = _parse_int_list(str(args.bg_hidden))
        if len(hidden_sizes) == 1:
            hidden_sizes = [int(hidden_sizes[0]), int(hidden_sizes[0])]
        if len(hidden_sizes) < 2:
            raise ValueError("--bg-hidden expects at least one positive int.")
        bg_pair = (int(hidden_sizes[0]), int(hidden_sizes[1]))
        cfg = configure_crosswell_cfg(
            data,
            measurement_noise_std=noise,
            residual_k=rk,
            residual_basis=str(args.cross_residual_basis),
            measurement_kind="subsample",
            csgm_prior_type="ridge",
            skip_lfista=skip_lf,
            lfista_bg_epochs=int(args.lfista_bg_epochs),
            bg_type=str(args.bg_type),
            bg_hidden=bg_pair,
        )
        cfg.seeds = [seed]
        cfg.measurement_ratios = [rho]
        stem = "crosswell_vc_step{}".format(int(args.cross_step))
        y_label = "clay volume Vc (processed curve scale)"

    if str(cfg.measurement_kind) != "subsample":
        print("This figure script supports measurement_kind=subsample only.", file=sys.stderr)
        sys.exit(2)

    _df, _pfrag, line_ex = run_direct_ub_from_data(
        cfg,
        dub_cfg,
        data,
        seed,
        rho,
        include_hf,
        run_ae,
        include_lfista,
        joint_only,
    )
    if line_ex is None or "Y_true" not in line_ex:
        print("run_direct_ub_from_data returned no line examples; increase n_test or n_example_plots.", file=sys.stderr)
        sys.exit(3)
    y_true_all = np.asarray(line_ex["Y_true"], dtype=np.float64)
    n_ex = int(y_true_all.shape[0])
    if ex < 0 or ex >= n_ex:
        print("example_index out of range [0, {}).".format(n_ex), file=sys.stderr)
        sys.exit(4)
    key_csgm = "ridge_prior_csgm"
    if key_csgm not in line_ex:
        alt = [k for k in line_ex if str(k).endswith("_prior_csgm")]
        if not alt:
            print("Missing ridge_prior_csgm (or mlp_prior_csgm) in line examples.", file=sys.stderr)
            sys.exit(5)
        key_csgm = str(alt[0])
    y_true = y_true_all[ex]
    y_csgm = np.asarray(line_ex[key_csgm], dtype=np.float64)[ex]
    y_ae: Optional[np.ndarray] = None
    if "ae_regression_ub" in line_ex:
        y_ae = np.asarray(line_ex["ae_regression_ub"], dtype=np.float64)[ex]

    M, B_test = replay_M_and_B_test(cfg, data, seed, rho)
    b_vec = np.asarray(B_test[ex], dtype=np.float64).ravel()

    rho_tag = str(rho).replace(".", "p")
    fname = "{}_seed{}_rho{}_ex{}.png".format(stem, seed, rho_tag, ex)
    out_path = os.path.join(out_dir, fname)
    title = "{} | seed={} rho={:.2f} | test window index {}".format(stem, seed, rho, ex)
    plot_sparse_window(out_path, y_true, y_csgm, M, b_vec, y_ae, title, y_label)
    print("Wrote " + out_path, flush=True)


if __name__ == "__main__":
    main()
