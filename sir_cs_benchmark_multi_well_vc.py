#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Cross-well Vc prediction from noisy wireline logs with sparse core-sample
assimilation. Direct [u, b] -> y benchmark with a held-out test well.

- Target y: Vc (clay volume) segments per window.
- u: concatenated channel segments (default: sonic, rhob, ai, vp; i.e. no GR).
- b: artificially subsampled y with additive measurement noise (core analogue).
- Train set: windows from the three training wells concatenated.
- Val set: tail val_frac of windows of each training well (no leakage).
- Test set: all windows of the held-out well.

Usage (from repo root):
  python sir_cs_benchmark_multi_well_vc.py \\
    --train-path data/F02-1,F03-2,F06-1_6logs_30dB.txt \\
    --test-path  data/F03-4_6logs_30dB.txt \\
    --channels sonic,rhob,ai,vp \\
    --target vc \\
    --base-dir outputs/cross_well_vc/direct_ub

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
from typing import Any, Dict, List, Optional, TextIO, Tuple

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning

import direct_ub_baselines as dub
import multi_well_vc as mwv
import real_well_f03 as rwf03
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
        description="Cross-well Vc direct [u,b] benchmark (joint baselines + LFISTA joint)."
    )
    p.add_argument(
        "--train-path",
        type=str,
        default="data/F02-1,F03-2,F06-1_6logs_30dB.txt",
        help="Tab-separated 6-log file with training wells (may be concatenated).",
    )
    p.add_argument(
        "--test-path",
        type=str,
        default="data/F03-4_6logs_30dB.txt",
        help="Tab-separated 6-log file with the held-out test well.",
    )
    p.add_argument(
        "--target",
        type=str,
        default="vc",
        choices=("vc", "porosity"),
        help="Target variable (canonical name).",
    )
    p.add_argument(
        "--channels",
        type=str,
        default="sonic,rhob,ai,vp",
        help="Comma-separated canonical channels for u. Valid: sonic,rhob,gr,ai,vp.",
    )
    p.add_argument("--window-len", type=int, default=64, help="L = n_output per window.")
    p.add_argument("--step", type=int, default=4, help="Sliding window step in samples.")
    p.add_argument(
        "--val-frac",
        type=float,
        default=0.1,
        help="Tail-per-well validation fraction (held out inside training wells).",
    )
    p.add_argument(
        "--val-embargo-windows",
        type=int,
        default=0,
        help=(
            "Optional count of boundary windows dropped between train and val "
            "inside each training well to reduce overlap leakage."
        ),
    )
    p.add_argument("--base-dir", type=str, default="outputs/cross_well_vc/direct_ub")
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
        default="0.05,0.1,0.15,0.2,0.3",
        help="Comma-separated measurement_ratio grid (core-sampling density).",
    )
    p.add_argument(
        "--residual-basis",
        type=str,
        default="dct",
        choices=("identity", "dct", "haar", "db4", "sym4", "fd1"),
        help="Basis Psi; Alpha = Y @ Psi for LFISTA/alpha MLP targets.",
    )
    p.add_argument(
        "--lfista-bg-epochs",
        type=int,
        default=150,
        help="Override lfista_num_epochs_bg (0 keeps profile default).",
    )
    p.add_argument(
        "--bg-type",
        type=str,
        default="mlp2",
        choices=("mlp2", "shallow", "linear"),
        help=(
            "LFISTA background architecture: mlp2 (two hidden), "
            "shallow (one hidden), linear (single Linear). Lower capacity "
            "preserves residual sparsity."
        ),
    )
    p.add_argument(
        "--bg-hidden",
        type=str,
        default="128,128",
        help="Comma-separated hidden sizes. For shallow only the first is used.",
    )
    p.add_argument(
        "--measurement-kind",
        type=str,
        default="subsample",
        choices=("gaussian", "subsample"),
        help="M construction (subsample = sparse cores).",
    )
    p.add_argument(
        "--measurement-noise-std",
        type=float,
        default=0.01,
        help="Std of noise on b (Vc laboratory scale, default ~0.01 in Vc units).",
    )
    p.add_argument(
        "--residual-k",
        type=int,
        default=6,
        help="k for CS/LFISTA (operational hyperparameter).",
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
        help="Enable optional conditional CSGM M2 branch (generative prior + b consistency).",
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
        default=rwf03.PROFILE_OVERLAP_AGG_UNIFORM_MEAN,
        choices=rwf03.PROFILE_OVERLAP_AGG_CHOICES,
        help=(
            "How overlapping test-window predictions are fused into a full-depth profile "
            "for depth figures: uniform_mean (default) or center_weighted_mean."
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
        help="Shorter run: one seed, two rhos (0.1, 0.2).",
    )
    return p.parse_args()


def _depth_str(lo: float, hi: float) -> str:
    if not (np.isfinite(lo) and np.isfinite(hi)):
        return "n/a"
    return "[{:.2f}, {:.2f}]".format(lo, hi)


def write_dataset_manifest(
    run_root: str,
    meta: Dict[str, Any],
    train_path: str,
    test_path: str,
    noise_level: str,
) -> str:
    """Human-readable manifest describing the cross-well dataset composition."""
    channels = list(meta.get("channels", []))
    ch_str = ", ".join(c.upper() for c in channels)
    u_dim = len(channels)
    train_wells = list(meta.get("train_wells", []))
    test_wells = list(meta.get("test_wells", []))
    n_tr = int(meta.get("n_train", 0))
    n_va = int(meta.get("n_val", 0))
    n_te = int(meta.get("n_test", 0))
    lines = [
        "Cross-well Vc benchmark: multi-well sliding-window dataset.",
        "",
        "train_path: " + str(train_path),
        "test_path:  " + str(test_path),
        "noise_level: " + noise_level,
        "",
        "target: " + str(meta.get("target", "vc")).upper(),
        "train_wells: " + ", ".join(train_wells),
        "test_wells:  " + ", ".join(test_wells),
        "",
        "u_channels (in order): " + ch_str,
        "u = ["
        + " || ".join(c.upper() + " segment" for c in channels)
        + "] in R^{" + str(u_dim) + "*L}; y = "
        + str(meta.get("target", "vc")).upper() + " segment in R^L.",
        "",
        "window_len (L, n_output): " + str(meta.get("window_len", 0)),
        "step: " + str(meta.get("step", 0)),
        "val_frac (tail per train well): " + str(meta.get("val_frac", 0.0)),
        "val_embargo_windows (train/val boundary): "
        + str(meta.get("val_embargo_windows", 0)),
        "n_train n_val n_test (windows): " + str(n_tr) + " " + str(n_va) + " " + str(n_te),
        "",
        "train_wells_rows: " + json.dumps(meta.get("train_wells_rows", {})),
        "test_wells_rows:  " + json.dumps(meta.get("test_wells_rows", {})),
        "",
        "Alpha_{train,val,test} = Y_{...} @ Psi (orthonormal L x L basis).",
    ]
    path = os.path.join(run_root, "DATASET_MANIFEST.txt")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    return path


def _detect_noise_level(path: str) -> str:
    base = os.path.basename(path).lower()
    if "30db" in base:
        return "30dB"
    return "clean"


def _reconstruct_test_depth_axis(
    test_path: str,
    target_name: str,
    channels: Tuple[str, ...],
) -> Tuple[np.ndarray, Dict[str, int]]:
    """Read the test file once more to obtain the canonical sorted depth axis
    (one per test well). Returns the concatenated depth array of all test wells
    in the order produced by build_windows_for_segment (same well ordering).
    """
    segs = mwv.load_6log_file(
        test_path, target_name=target_name, channels=channels, well_names=None
    )
    if not segs:
        raise RuntimeError("No test segments in " + test_path)
    depth_list = []
    lengths: Dict[str, int] = {}
    for s in segs:
        depth_list.append(np.asarray(s.depth, dtype=np.float64).ravel())
        lengths[s.name] = int(s.depth.shape[0])
    return np.concatenate(depth_list, axis=0), lengths


def main() -> None:
    args = parse_args()
    warnings.filterwarnings("ignore", category=ConvergenceWarning)
    seeds_str = "7" if bool(args.fast) else str(args.seeds)
    rhos_str = "0.1,0.2" if bool(args.fast) else str(args.rhos)

    train_path = os.path.abspath(str(args.train_path).strip())
    test_path = os.path.abspath(str(args.test_path).strip())
    if not os.path.isfile(train_path):
        print("Missing train file: " + train_path, file=sys.stderr)
        sys.exit(2)
    if not os.path.isfile(test_path):
        print("Missing test file: " + test_path, file=sys.stderr)
        sys.exit(2)

    channels_raw = tuple(
        c.strip().lower() for c in str(args.channels).split(",") if c.strip()
    )
    target_name = str(args.target).strip().lower()

    data = mwv.build_cross_well_data_dict(
        train_path=train_path,
        test_path=test_path,
        target_name=target_name,
        channels=channels_raw,
        window_len=int(args.window_len),
        step=int(args.step),
        val_frac=float(args.val_frac),
        val_embargo_windows=int(args.val_embargo_windows),
        residual_basis=str(args.residual_basis),
    )
    meta = data["meta"]
    p_in = int(meta["p_input"])
    l = int(meta["n_output"])
    n_tr = int(meta["n_train"])
    n_va = int(meta["n_val"])
    n_te = int(meta["n_test"])

    cfg = Config()
    cfg.log_progress = False
    cfg.config_profile = "cross_well_vc_direct_ub"
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
    cfg.lfista_bg_type = str(args.bg_type).strip().lower()
    hidden_sizes: List[int] = _parse_int_list(str(args.bg_hidden))
    if len(hidden_sizes) == 1:
        hidden_sizes = [int(hidden_sizes[0]), int(hidden_sizes[0])]
    if len(hidden_sizes) < 2:
        raise ValueError("--bg-hidden expects at least one positive int.")
    cfg.lfista_bg_hidden = (int(hidden_sizes[0]), int(hidden_sizes[1]))

    pipe_data: Dict[str, np.ndarray] = {
        k: data[k]
        for k in (
            "X_train", "X_val", "X_test",
            "Y_train", "Y_val", "Y_test",
            "Alpha_train", "Alpha_val", "Alpha_test",
            "Psi",
        )
    }

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
    noise_level = _detect_noise_level(train_path) + "/" + _detect_noise_level(test_path)

    try:
        man_path = write_dataset_manifest(run_root, meta, train_path, test_path, noise_level)
        _log(tee, "Run root: " + run_root)
        _log(
            tee,
            "Cross-well Vc | train_wells=" + str(meta["train_wells"])
            + " test_wells=" + str(meta["test_wells"])
            + " channels=" + str(meta["channels"])
            + " target=" + str(meta["target"]).upper()
            + " L=" + str(l)
            + " p_input=" + str(p_in)
            + " noise=" + noise_level,
        )
        _log(
            tee,
            "n_train/n_val/n_test = " + str(n_tr) + "/" + str(n_va) + "/" + str(n_te),
        )
        _log(tee, "DATASET_MANIFEST: " + man_path)
        _log(
            tee,
            "seeds=" + str(cfg.seeds) + " rhos=" + str(cfg.measurement_ratios)
            + " k=" + str(cfg.residual_k) + " noise=" + str(cfg.measurement_noise_std)
            + " basis=" + str(cfg.residual_basis)
            + " bg_type=" + str(cfg.lfista_bg_type)
            + " bg_hidden=" + str(cfg.lfista_bg_hidden)
            + " bg_epochs=" + str(cfg.lfista_num_epochs_bg)
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
                    pipe_data,
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
                depth_axis, test_well_rows = _reconstruct_test_depth_axis(
                    test_path, target_name, channels_raw
                )
                nrows_total = int(depth_axis.shape[0])
                test_row_starts = np.asarray(
                    meta["test_row_start_full"], dtype=np.int64
                )
                test_well_of_window = np.asarray(meta["test_well_of_window_full"], dtype=object)
                well_order = []
                seen: List[str] = []
                for name in test_well_of_window.tolist():
                    if name not in seen:
                        seen.append(name)
                        well_order.append(name)
                offsets: Dict[str, int] = {}
                t = 0
                for name in well_order:
                    offsets[name] = t
                    t += int(test_well_rows.get(name, 0))
                absolute_starts = np.zeros_like(test_row_starts)
                for i in range(test_row_starts.shape[0]):
                    name = str(test_well_of_window[i])
                    absolute_starts[i] = int(test_row_starts[i]) + offsets.get(name, 0)

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
                overlap_primary = rwf03.validate_profile_overlap_agg(str(args.profile_overlap_agg))
                profiles: Dict[str, np.ndarray] = {}
                obs_stack = np.asarray(
                    first_parity_fragment["y_true"], dtype=np.float64
                ).reshape(n_te, l)
                obs_profile, _ = rwf03.reconstruct_depth_profile(
                    obs_stack,
                    absolute_starts,
                    l,
                    nrows_total,
                    overlap_agg=overlap_primary,
                )
                profiles["observed"] = obs_profile
                for k in known_model_keys:
                    if k not in first_parity_fragment:
                        continue
                    arr = np.asarray(first_parity_fragment[k], dtype=np.float64)
                    if arr.size != n_te * l:
                        continue
                    stack = arr.reshape(n_te, l)
                    prof, _ = rwf03.reconstruct_depth_profile(
                        stack,
                        absolute_starts,
                        l,
                        nrows_total,
                        overlap_agg=overlap_primary,
                    )
                    profiles[k] = prof
                if bool(args.save_profile_overlap_compare):
                    overlap_paths = rwf03.save_overlap_profile_agg_bundle(
                        run_root,
                        overlap_primary,
                        depth_axis,
                        absolute_starts,
                        l,
                        nrows_total,
                        obs_stack,
                        first_parity_fragment,
                        known_model_keys,
                        log_line=lambda s: _log(tee, s),
                        profile_x_label=str(meta["target"]).upper(),
                    )

                title = (
                    "{} {} profile (test block: {}) | rho={:.2f}, seed={}".format(
                        ",".join(meta["test_wells"]),
                        str(meta["target"]).upper(),
                        noise_level,
                        float(first_job_info.get("rho", float("nan"))),
                        int(first_job_info.get("seed", 0)),
                    )
                )
                p10 = os.path.join(run_root, cfg.plots_subdir, "10_depth_profile_target.png")
                plot_real_well_depth_profile(
                    depth_axis,
                    profiles,
                    p10,
                    train_test_boundary=None,
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
            except (ValueError, KeyError, RuntimeError) as ex:
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
        cfg_dump["cross_well_vc"] = {
            "run_id": run_id,
            "train_path": train_path,
            "test_path": test_path,
            "target": target_name,
            "channels": list(channels_raw),
            "train_wells": list(meta["train_wells"]),
            "test_wells": list(meta["test_wells"]),
            "window_len": l,
            "step": int(meta["step"]),
            "val_frac": float(meta["val_frac"]),
            "n_train": n_tr,
            "n_val": n_va,
            "n_test": n_te,
            "p_input": p_in,
            "noise_level": noise_level,
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


if __name__ == "__main__":
    main()
