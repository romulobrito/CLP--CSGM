#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Path A: real well F03-4 (AC, GR, Porosity) -> same direct [u,b]->y benchmark as
sir_cs_benchmark_direct_ub.py, with fixed contiguous depth splits and Alpha = Y @ Psi.

Artifacts mirror the synthetic benchmark: tables/, figures/, logs/, PROTOCOL.txt,
DATASET_MANIFEST.txt, config.json, RUN_MANIFEST.txt.

Usage (from repo root):
  python sir_cs_benchmark_real_well_direct_ub.py \\
    --data-path data/F03-4_AC+GR+Porosity.txt \\
    --base-dir outputs/real_well_f03/direct_ub

ASCII-only.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import warnings
from dataclasses import asdict
from typing import Dict, List, Optional, TextIO, Tuple

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning

import direct_ub_baselines as dub
import real_well_f03 as rwf
from sir_cs_benchmark_direct_ub import (
    _Tee,
    _log,
    merge_gt_pred_bundles,
    plot_parity_ground_truth_vs_predictions,
    run_direct_ub_from_data,
    save_focus_tables,
    write_protocol,
    write_run_manifest,
)
from sir_cs_pipeline_optimized import (
    Config,
    apply_config_profile,
    method_display_name,
    plot_direct_ub_ground_truth_vs_models,
    plot_real_well_depth_profile,
    save_all_comparison_plots,
    summarize_results_across_seeds,
    summarize_results_per_seed,
)


def write_dataset_manifest(
    run_root: str,
    data_path: str,
    window_len: int,
    step: int,
    train_frac: float,
    val_frac: float,
    n_tr: int,
    n_va: int,
    n_te: int,
    depth_tr: str,
    depth_va: str,
    depth_te: str,
    u_channels: Tuple[str, ...],
) -> str:
    ch_str = ", ".join(c.upper() for c in u_channels)
    u_dim = len(u_channels)
    lines = [
        "F03-4 real well: depth-first split, then sliding-window dataset.",
        "",
        "data_path: " + str(data_path),
        "window_len (L, n_output): " + str(window_len),
        "step: " + str(step),
        "train_frac: " + str(train_frac) + "  val_frac: " + str(val_frac),
        "n_train n_val n_test: " + str(n_tr) + " " + str(n_va) + " " + str(n_te),
        "",
        "u_channels (in order): " + ch_str,
        "u = ["
        + " || ".join(c.upper() + " segment" for c in u_channels)
        + "] in R^{" + str(u_dim) + "*L}; y = Porosity segment in R^L.",
        "No shuffle: depth rows are split contiguously before windowing; train is shallower, test is deeper.",
        "",
        "approx depth range train (m): " + depth_tr,
        "approx depth range val (m): " + depth_va,
        "approx depth range test (m): " + depth_te,
        "",
        "Alpha_train/val/test = Y @ Psi (DCT/identity of y per window).",
    ]
    path = os.path.join(run_root, "DATASET_MANIFEST.txt")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    return path


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


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Real well F03 direct [u,b] benchmark (joint baselines + LFISTA joint)."
    )
    p.add_argument(
        "--data-path",
        type=str,
        default="data/F03-4_AC+GR+Porosity.txt",
        help="Tab-separated file with Depth, AC, GR, Porosity.",
    )
    p.add_argument("--window-len", type=int, default=64, help="L = n_output (porosity window length).")
    p.add_argument("--step", type=int, default=1, help="Sliding window step in samples.")
    p.add_argument(
        "--u-channels",
        type=str,
        default="ac,gr",
        help="Comma-separated list of u channels (ordered). Valid: ac, gr. "
        "Examples: 'ac,gr' (default, strong); 'gr' (ablation: weakened u).",
    )
    p.add_argument("--train-frac", type=float, default=0.6, help="Fraction of windows for training.")
    p.add_argument("--val-frac", type=float, default=0.2, help="Fraction of windows for validation.")
    p.add_argument("--base-dir", type=str, default="outputs/real_well_f03/direct_ub", help="Base output dir.")
    p.add_argument("--run-id", type=str, default="", help="Run folder name (default: timestamp).")
    p.add_argument(
        "--seeds",
        type=str,
        default="7,23,41",
        help="Comma-separated RNG seeds (M and b noise; same u,y split for all).",
    )
    p.add_argument(
        "--rhos",
        type=str,
        default="0.2,0.3,0.4,0.5,0.6",
        help="Comma-separated measurement_ratio grid.",
    )
    p.add_argument(
        "--residual-basis",
        type=str,
        default="dct",
        choices=("identity", "dct", "haar", "db4", "sym4", "fd1"),
        help="Basis Psi; Alpha = Y @ Psi for LFISTA/alpha MLP targets. "
        "Wavelet bases (haar, db4, sym4) require L = power of 2.",
    )
    p.add_argument(
        "--lfista-bg-epochs",
        type=int,
        default=0,
        help="Override lfista_num_epochs_bg (0 keeps profile default).",
    )
    p.add_argument(
        "--measurement-kind",
        type=str,
        default="subsample",
        choices=("gaussian", "subsample"),
        help="M construction (subsample is default for 1D profiles).",
    )
    p.add_argument(
        "--measurement-noise-std",
        type=float,
        default=0.02,
        help="Std of noise on b (tune to porosity scale).",
    )
    p.add_argument(
        "--residual-k",
        type=int,
        default=6,
        help="k for CS/LFISTA (operational hyperparameter; not a generative sparsity).",
    )
    p.add_argument("--no-ae", action="store_true", help="Skip AE baseline.")
    p.add_argument(
        "--no-lfista",
        action="store_true",
        help="Skip hybrid_lfista_joint (useful for CSGM-focused benchmark runs).",
    )
    p.add_argument(
        "--run-csgm-m2",
        action="store_true",
        help="Enable optional conditional CSGM M2 branch.",
    )
    p.add_argument(
        "--run-csgm-ablations",
        action="store_true",
        help="Also report prior-only and measurement-only CSGM ablations.",
    )
    p.add_argument(
        "--csgm-prior-type",
        type=str,
        default="ridge",
        choices=("ridge", "mlp"),
        help="CSGM M2 prior h(u)->z0.",
    )
    p.add_argument("--csgm-latent-dim", type=int, default=16)
    p.add_argument("--csgm-ae-epochs", type=int, default=200)
    p.add_argument("--csgm-iters", type=int, default=400)
    p.add_argument("--csgm-restarts", type=int, default=3)
    p.add_argument("--csgm-opt-lr", type=float, default=0.05)
    p.add_argument(
        "--csgm-lambda-grid",
        type=str,
        default="0.0001,0.0003,0.001,0.003,0.01,0.03,0.1",
        help="Comma-separated lambda grid for CSGM M2 validation selection.",
    )
    p.add_argument("--no-plots", action="store_true", help="Skip dashboard figures.")
    p.add_argument("--no-parity", action="store_true", help="Skip parity scatter and parity npz.")
    p.add_argument(
        "--profile-overlap-agg",
        type=str,
        default=rwf.PROFILE_OVERLAP_AGG_UNIFORM_MEAN,
        choices=rwf.PROFILE_OVERLAP_AGG_CHOICES,
        help=(
            "How overlapping test-window predictions are fused into a full-depth profile "
            "for depth figures and npz exports: uniform_mean (default) or "
            "center_weighted_mean (triangular weights peaking at window center)."
        ),
    )
    p.add_argument(
        "--save-profile-overlap-compare",
        action="store_true",
        help=(
            "Write overlap_profile_agg/ with tables and figures comparing "
            "uniform_mean vs center_weighted_mean stitching (visualization path only)."
        ),
    )
    p.add_argument(
        "--fast",
        action="store_true",
        help="Shorter run: one seed, two rhos (0.3, 0.5).",
    )
    return p.parse_args()


def _depth_str(lo: float, hi: float) -> str:
    if not (np.isfinite(lo) and np.isfinite(hi)):
        return "n/a"
    return "[{:.2f}, {:.2f}]".format(lo, hi)


def main() -> None:
    args = parse_args()
    warnings.filterwarnings("ignore", category=ConvergenceWarning)
    seeds_str = "7" if bool(args.fast) else str(args.seeds)
    rhos_str = "0.3,0.5" if bool(args.fast) else str(args.rhos)
    data_path = os.path.abspath(str(args.data_path).strip())
    if not os.path.isfile(data_path):
        print("Missing data file: {}".format(data_path), file=sys.stderr)
        sys.exit(2)

    st = int(args.step)
    tab = rwf.load_f03_table(data_path)
    u_channels_raw = tuple(c.strip() for c in str(args.u_channels).split(",") if c.strip())
    u_channels = rwf.normalize_channels(u_channels_raw)
    n_rows_total = int(tab.n_rows)
    tr_row_sl, va_row_sl, te_row_sl, n_tr_rows, n_va_rows, n_te_rows = _contiguous_row_split(
        n_rows_total, float(args.train_frac), float(args.val_frac), int(args.window_len)
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
    x_tr, y_tr, centers_tr, _ = rwf.build_sliding_windows(
        tab_tr, int(args.window_len), st, channels=u_channels
    )
    x_va, y_va, centers_va, _ = rwf.build_sliding_windows(
        tab_va, int(args.window_len), st, channels=u_channels
    )
    x_te, y_te, centers_te, _ = rwf.build_sliding_windows(
        tab_te, int(args.window_len), st, channels=u_channels
    )
    n_tr = int(x_tr.shape[0])
    n_va = int(x_va.shape[0])
    n_te = int(x_te.shape[0])
    n_win = n_tr + n_va + n_te
    l = int(args.window_len)
    p_in = len(u_channels) * l
    d_lo_tr = _ranges_for_values(centers_tr)
    d_lo_va = _ranges_for_values(centers_va)
    d_lo_te = _ranges_for_values(centers_te)

    cfg = Config()
    cfg.log_progress = False
    cfg.config_profile = "real_well_f03_direct_ub"
    apply_config_profile(cfg)
    cfg.p_input = p_in
    cfg.n_output = l
    cfg.n_train = n_tr
    cfg.n_val = n_va
    cfg.n_test = n_te
    cfg.residual_k = int(args.residual_k)
    cfg.residual_basis = str(args.residual_basis)
    cfg.measurement_kind = str(args.measurement_kind)
    cfg.measurement_noise_std = float(args.measurement_noise_std)
    cfg.seeds = _parse_int_list(seeds_str)
    cfg.measurement_ratios = _parse_float_list(rhos_str)
    cfg.run_lfista = not bool(args.no_lfista)
    cfg.run_csgm_m2 = bool(args.run_csgm_m2)
    cfg.run_csgm_ablations = bool(args.run_csgm_ablations)
    cfg.csgm_prior_type = str(args.csgm_prior_type).strip().lower()
    cfg.csgm_latent_dim = int(args.csgm_latent_dim)
    cfg.csgm_ae_epochs = int(args.csgm_ae_epochs)
    cfg.csgm_iters = int(args.csgm_iters)
    cfg.csgm_restarts = int(args.csgm_restarts)
    cfg.csgm_opt_lr = float(args.csgm_opt_lr)
    cfg.csgm_lambda_grid = _parse_float_list(str(args.csgm_lambda_grid))
    cfg.paper_strict_paired_b = True
    bg_override = int(args.lfista_bg_epochs)
    if bg_override > 0:
        cfg.lfista_num_epochs_bg = bg_override

    x_all = np.concatenate([x_tr, x_va, x_te], axis=0)
    y_all = np.concatenate([y_tr, y_va, y_te], axis=0)
    sl_tr = slice(0, n_tr)
    sl_va = slice(n_tr, n_tr + n_va)
    sl_te = slice(n_tr + n_va, n_win)
    data = rwf.build_direct_ub_data_dict(x_all, y_all, sl_tr, sl_va, sl_te, str(args.residual_basis))

    dub_cfg = dub.DirectUBTrainConfig()
    joint_only = True
    include_lfista = not bool(args.no_lfista)
    include_hf = False
    run_ae = not bool(args.no_ae)

    run_id = str(args.run_id).strip() or time.strftime("%Y%m%d_%H%M%S")
    base_dir = os.path.abspath(str(args.base_dir).strip())
    run_root = os.path.join(base_dir, "runs", run_id)
    tables_dir = os.path.join(run_root, "tables")
    logs_dir = os.path.join(run_root, "logs")
    os.makedirs(tables_dir, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)
    cfg.save_dir = run_root
    cfg.plots_subdir = "figures"

    log_path = os.path.join(logs_dir, "run_console.log")
    old_stdout: TextIO = sys.stdout
    tee = _Tee(old_stdout, log_path)
    sys.stdout = tee  # type: ignore[assignment]

    all_dfs: List[pd.DataFrame] = []
    parity_bundles: List[Dict[str, np.ndarray]] = []
    first_line_examples: Optional[Dict[str, np.ndarray]] = None
    first_parity_fragment: Optional[Dict[str, np.ndarray]] = None
    first_job_info: Dict[str, float] = {}
    t0 = time.time()
    job_idx = 0
    total = len(cfg.seeds) * len(cfg.measurement_ratios)
    try:
        man_path = write_dataset_manifest(
            run_root,
            data_path,
            l,
            st,
            float(args.train_frac),
            float(args.val_frac),
            n_tr,
            n_va,
            n_te,
            _depth_str(d_lo_tr[0], d_lo_tr[1]),
            _depth_str(d_lo_va[0], d_lo_va[1]),
            _depth_str(d_lo_te[0], d_lo_te[1]),
            u_channels,
        )
        _log(tee, "Run root: " + run_root)
        _log(
            tee,
            "Real well F03 | n_windows=" + str(n_win) + " L=" + str(l)
            + " u_channels=[" + ",".join(c.upper() for c in u_channels) + "]"
            + " p_input=" + str(p_in)
            + " split=depth-first-windowing",
        )
        _log(tee, "DATASET_MANIFEST: " + man_path)
        _log(
            tee,
            "seeds=" + str(cfg.seeds) + " rhos=" + str(cfg.measurement_ratios)
            + " k=" + str(cfg.residual_k) + " noise=" + str(cfg.measurement_noise_std)
            + " csgm_m2=" + str(cfg.run_csgm_m2)
            + " csgm_ablations=" + str(cfg.run_csgm_ablations)
            + " csgm_prior=" + str(cfg.csgm_prior_type),
        )
        for seed in cfg.seeds:
            for mr in cfg.measurement_ratios:
                job_idx += 1
                _log(
                    tee,
                    "--- job {}/{} seed={} measurement_ratio={:.2f} ---".format(
                        job_idx, total, seed, mr
                    ),
                )
                df, pfrag, line_ex = run_direct_ub_from_data(
                    cfg,
                    dub_cfg,
                    data,
                    int(seed),
                    float(mr),
                    include_hf,
                    run_ae,
                    include_lfista,
                    joint_only,
                )
                all_dfs.append(df)
                parity_bundles.append(pfrag)
                if first_line_examples is None and line_ex is not None:
                    first_line_examples = line_ex
                if first_parity_fragment is None:
                    first_parity_fragment = pfrag
                    first_job_info = {"seed": float(seed), "rho": float(mr)}
        detailed = pd.concat(all_dfs, ignore_index=True)
        per_seed = summarize_results_per_seed(detailed)
        summary = summarize_results_across_seeds(per_seed)
        detailed["method_label"] = detailed["method"].map(method_display_name)
        per_seed["method_label"] = per_seed["method"].map(method_display_name)
        summary["method_label"] = summary["method"].map(method_display_name)
        detailed.to_csv(os.path.join(tables_dir, "detailed_results.csv"), index=False)
        per_seed.to_csv(os.path.join(tables_dir, "summary_by_seed.csv"), index=False)
        summary.to_csv(os.path.join(tables_dir, "summary.csv"), index=False)
        focus_paths = save_focus_tables(
            run_root,
            summary,
            per_seed,
            include_hf,
            run_ae,
            include_lfista,
            joint_only,
            bool(cfg.run_csgm_m2),
        )
        proto_path = write_protocol(
            run_root,
            joint_only,
            str(cfg.residual_basis),
            str(cfg.measurement_kind),
            float(cfg.measurement_noise_std),
            int(cfg.residual_k),
            bool(cfg.run_csgm_m2),
            include_lfista,
        )
        plot_paths: List[str] = []
        overlap_paths: List[str] = []
        if not bool(args.no_plots):
            plot_paths = save_all_comparison_plots(cfg, summary, per_seed)
            _log(tee, "Figures: {} files under {}".format(
                len(plot_paths), os.path.join(run_root, "figures")
            ))
        if (not bool(args.no_plots)) and first_line_examples is not None:
            p08 = os.path.join(run_root, cfg.plots_subdir, "08_example_ground_truth_vs_models.png")
            plot_direct_ub_ground_truth_vs_models(first_line_examples, p08)
            plot_paths.append(p08)
            _log(tee, "Figure: " + p08)
        if (not bool(args.no_plots)) and first_parity_fragment is not None:
            try:
                row_offset = int(n_tr_rows + n_va_rows)
                row_starts = np.arange(int(n_te), dtype=np.int64) * int(st) + row_offset
                depth_axis = np.asarray(tab.depth, dtype=np.float64).ravel()
                nrows = int(depth_axis.shape[0])
                known_model_keys = [
                    "ml_only",
                    "mlp_concat_ub",
                    "pca_regression_ub",
                    "ae_regression_ub",
                    "ridge_prior_csgm",
                    "mlp_prior_csgm",
                    "hybrid_fista",
                    "hybrid_lfista_joint",
                ]
                overlap_primary = rwf.validate_profile_overlap_agg(str(args.profile_overlap_agg))
                profiles: Dict[str, np.ndarray] = {}
                obs_stack = np.asarray(first_parity_fragment["y_true"], dtype=np.float64).reshape(n_te, l)
                obs_profile, _ = rwf.reconstruct_depth_profile(
                    obs_stack, row_starts, l, nrows, overlap_agg=overlap_primary
                )
                profiles["observed"] = obs_profile
                for k in known_model_keys:
                    if k not in first_parity_fragment:
                        continue
                    arr = np.asarray(first_parity_fragment[k], dtype=np.float64)
                    if arr.size != n_te * l:
                        continue
                    stack = arr.reshape(n_te, l)
                    prof, _ = rwf.reconstruct_depth_profile(
                        stack, row_starts, l, nrows, overlap_agg=overlap_primary
                    )
                    profiles[k] = prof
                if bool(args.save_profile_overlap_compare):
                    overlap_paths = rwf.save_overlap_profile_agg_bundle(
                        run_root,
                        overlap_primary,
                        depth_axis,
                        row_starts,
                        l,
                        nrows,
                        obs_stack,
                        first_parity_fragment,
                        known_model_keys,
                        log_line=lambda s: _log(tee, s),
                        profile_x_label="porosity",
                    )
                tt_boundary = 0.5 * (depth_axis[int(row_starts[0]) - 1] + depth_axis[int(row_starts[0])]) \
                    if int(row_starts[0]) - 1 >= 0 and int(row_starts[0]) < nrows else float("nan")
                title = (
                    "F03-4 porosity profile (test block) | rho={:.2f}, seed={}".format(
                        float(first_job_info.get("rho", float("nan"))),
                        int(first_job_info.get("seed", 0)),
                    )
                )
                p10 = os.path.join(run_root, cfg.plots_subdir, "10_depth_profile_porosity.png")
                plot_real_well_depth_profile(
                    depth_axis,
                    profiles,
                    p10,
                    train_test_boundary=float(tt_boundary) if np.isfinite(tt_boundary) else None,
                    title=title,
                )
                plot_paths.append(p10)
                _log(tee, "Figure: " + p10)
                prof_npz = os.path.join(tables_dir, "depth_profile_test_block.npz")
                np.savez_compressed(
                    prof_npz,
                    depth_axis=depth_axis,
                    **{("profile_" + k): v for k, v in profiles.items()},
                )
                plot_paths.append(prof_npz)
                plot_paths.extend(overlap_paths)
            except (ValueError, KeyError) as ex:
                _log(tee, "Warning: depth-profile plot skipped: " + str(ex))
        if (not bool(args.no_plots)) and (not bool(args.no_parity)) and parity_bundles:
            merged_parity = merge_gt_pred_bundles(parity_bundles)
            npz_path = os.path.join(tables_dir, "parity_pooled.npz")
            np.savez_compressed(npz_path, **merged_parity)
            parity_png = os.path.join(run_root, cfg.plots_subdir, "09_parity_ground_truth_vs_prediction.png")
            plot_parity_ground_truth_vs_predictions(cfg, merged_parity, parity_png)
            plot_paths.append(npz_path)
            plot_paths.append(parity_png)
            _log(tee, "Parity: " + parity_png)
        cfg_dump = asdict(cfg)
        dub_dump = asdict(dub_cfg)
        dub_dump["pca_r_grid"] = list(dub_dump["pca_r_grid"])
        cfg_dump["real_well_f03"] = {
            "run_id": run_id,
            "data_path": data_path,
            "window_len": l,
            "step": st,
            "train_frac": float(args.train_frac),
            "val_frac": float(args.val_frac),
            "n_train": n_tr,
            "n_val": n_va,
            "n_test": n_te,
            "n_windows": n_win,
            "n_rows_total": n_rows_total,
            "n_rows_train": n_tr_rows,
            "n_rows_val": n_va_rows,
            "n_rows_test": n_te_rows,
            "u_channels": list(u_channels),
            "p_input": p_in,
            "contiguous_split": True,
            "split_before_windowing": True,
            "dataset_manifest": man_path,
            "protocol_txt": proto_path,
            "tables_dir": tables_dir,
            "figures_dir": os.path.join(run_root, "figures"),
            "profile_overlap_agg": str(args.profile_overlap_agg),
            "save_profile_overlap_compare": bool(args.save_profile_overlap_compare),
        }
        with open(os.path.join(run_root, "config.json"), "w", encoding="utf-8") as f:
            json.dump(cfg_dump, f, indent=2)
        elapsed = time.time() - t0
        manifest_path = write_run_manifest(
            run_root,
            run_id,
            elapsed,
            plot_paths,
            focus_paths + [proto_path, man_path],
            joint_only,
            str(cfg.residual_basis),
            str(cfg.measurement_kind),
            float(cfg.measurement_noise_std),
            int(cfg.residual_k),
            bool(cfg.run_csgm_m2),
            include_lfista,
        )
        _log(tee, "Done in {:.1f}s | manifest: {}".format(elapsed, manifest_path))
    finally:
        sys.stdout = old_stdout
        tee.close()
    print("Artifacts: " + run_root, flush=True)


def _ranges_for_values(centers: List[float]) -> tuple[float, float]:
    c = np.asarray(centers, dtype=np.float64)
    if c.size < 1:
        return (float("nan"), float("nan"))
    return (float(np.min(c)), float(np.max(c)))


def _contiguous_row_split(
    n_rows: int, train_frac: float, val_frac: float, window_len: int
) -> tuple[slice, slice, slice, int, int, int]:
    if not (0.0 < float(train_frac) < 1.0) or not (0.0 < float(val_frac) < 1.0):
        raise ValueError("train_frac and val_frac must be in (0,1).")
    if float(train_frac) + float(val_frac) >= 1.0:
        raise ValueError("train_frac + val_frac must be < 1.0.")
    n_tr = int(np.floor(float(train_frac) * float(n_rows)))
    n_va = int(np.floor(float(val_frac) * float(n_rows)))
    n_te = int(n_rows - n_tr - n_va)
    l = int(window_len)
    if n_tr < l or n_va < l or n_te < l:
        raise ValueError(
            "Row split too small for window_len={}: n_tr={} n_va={} n_te={}".format(
                l, n_tr, n_va, n_te
            )
        )
    sl_tr = slice(0, n_tr)
    sl_va = slice(n_tr, n_tr + n_va)
    sl_te = slice(n_tr + n_va, n_rows)
    return sl_tr, sl_va, sl_te, n_tr, n_va, n_te


if __name__ == "__main__":
    main()
