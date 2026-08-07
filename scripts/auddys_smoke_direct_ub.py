#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Smoke benchmark for Auddys Excel data using the direct-UB core runner.

This script keeps the benchmark protocol close to the existing repository flow:
- Build sliding windows from the Logs sheet.
- Use contiguous train/val/test split (depth-ordered windows).
- Run direct [u,b]->y baselines plus optional CLP-CSGM branch via
  sir_cs_benchmark_direct_ub.run_direct_ub_from_data.

Artifacts (aligned with sir_cs_benchmark_real_well_direct_ub.py):
  <base_dir>/runs/<run_id>/
    tables/{detailed_results.csv,summary_by_seed.csv,summary.csv,
            summary_focus_clp_csgm_vs_ub.csv,parity_pooled.npz,
            depth_profile_test_block.npz}
    figures/{01..07}_*.png,08_example_ground_truth_vs_models.png,
            09_parity_ground_truth_vs_prediction.png,
            10_depth_profile_porosity.png}
    logs/run_console.log
    PROTOCOL.txt
    DATASET_MANIFEST.txt
    RUN_MANIFEST.txt
    config.json
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from dataclasses import asdict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import direct_ub_baselines as dub
import real_well_f03 as rwf
from sir_cs_benchmark_direct_ub import (
    _Tee,
    _log,
    run_direct_ub_from_data,
    save_focus_tables,
    write_protocol,
    write_run_manifest,
)
from sir_cs_pipeline_optimized import (
    Config,
    apply_config_profile,
    get_basis,
    merge_gt_pred_bundles,
    plot_direct_ub_ground_truth_vs_models,
    plot_parity_ground_truth_vs_predictions,
    plot_real_well_depth_profile,
    save_all_comparison_plots,
    summarize_results_across_seeds,
    summarize_results_per_seed,
)

_LINE_EXAMPLE_BASE_KEYS = frozenset(
    {
        "Y_true",
        "ml_only",
        "mlp_concat_ub",
        "pca_regression_ub",
        "ae_regression_ub",
        "hybrid_fista",
        "hybrid_lfista_joint",
        "hybrid_lfista_frozen",
    }
)

_DEPTH_PROFILE_KNOWN_MODEL_KEYS = (
    "ml_only",
    "mlp_concat_ub",
    "pca_regression_ub",
    "ae_regression_ub",
    "ridge_prior_csgm",
    "mlp_prior_csgm",
    "hybrid_fista",
    "hybrid_lfista_joint",
)


def _parse_float_list(s: str) -> List[float]:
    out: List[float] = []
    for p in str(s).split(","):
        q = p.strip()
        if q:
            out.append(float(q))
    if not out:
        raise ValueError("Empty float list.")
    return out


def _parse_int_list(s: str) -> List[int]:
    out: List[int] = []
    for p in str(s).split(","):
        q = p.strip()
        if q:
            out.append(int(q))
    if not out:
        raise ValueError("Empty int list.")
    return out


def _parse_csgm_prior_types(args: argparse.Namespace) -> Tuple[str, ...]:
    """
    Single prior: use --csgm-prior-type.
    Multiple priors: --csgm-prior-types ridge,mlp (runs same baselines twice;
    detailed rows are de-duplicated; parity merges CSGM keys).
    """
    raw = str(getattr(args, "csgm_prior_types", "")).strip()
    if raw:
        out: List[str] = []
        for p in raw.split(","):
            q = p.strip().lower()
            if q:
                if q not in ("ridge", "mlp"):
                    raise ValueError("Invalid CSGM prior type {!r}; use ridge or mlp.".format(q))
                out.append(q)
        if not out:
            raise ValueError("Empty --csgm-prior-types.")
        # preserve order, drop duplicates
        uniq: List[str] = []
        for q in out:
            if q not in uniq:
                uniq.append(q)
        return tuple(uniq)
    pt = str(args.csgm_prior_type).strip().lower()
    if pt not in ("ridge", "mlp"):
        raise ValueError("Invalid --csgm-prior-type {!r}.".format(pt))
    return (pt,)


def _merge_parity_fragments_csgm(
    fragments: List[Dict[str, np.ndarray]],
) -> Dict[str, np.ndarray]:
    """Baselines from the first fragment; CSGM / ablation keys from all fragments."""
    if not fragments:
        return {}
    if len(fragments) == 1:
        return dict(fragments[0])
    base_keys = frozenset(
        {
            "y_true",
            "ml_only",
            "mlp_concat_ub",
            "pca_regression_ub",
            "ae_regression_ub",
            "hybrid_fista",
            "hybrid_lfista_joint",
        }
    )
    out = dict(fragments[0])
    for frag in fragments[1:]:
        for k, v in frag.items():
            if k in base_keys:
                continue
            out[k] = v
    return out


def _parse_optional_float(text: str) -> Optional[float]:
    """Parse CLI optional float; empty string -> None."""
    q = str(text).strip()
    if not q:
        return None
    val = float(q)
    if not np.isfinite(val):
        raise ValueError("Non-finite float: {!r}".format(q))
    return val


def apply_depth_bounds(
    df: pd.DataFrame,
    depth_min_m: Optional[float],
    depth_max_m: Optional[float],
) -> pd.DataFrame:
    """Filter rows by depth before windowing (inclusive bounds)."""
    out = df
    if depth_min_m is not None:
        out = out[out["depth_m"] >= float(depth_min_m)]
    if depth_max_m is not None:
        out = out[out["depth_m"] <= float(depth_max_m)]
    out = out.sort_values("depth_m").reset_index(drop=True)
    if out.empty:
        raise ValueError(
            "Depth filter removed all rows (depth_min_m={}, depth_max_m={}).".format(
                depth_min_m, depth_max_m
            )
        )
    return out


def split_depth_summary_table(
    df: pd.DataFrame,
    starts: np.ndarray,
    sl_tr: slice,
    sl_va: slice,
    sl_te: slice,
    window_len: int,
    target: str,
) -> pd.DataFrame:
    """Per-split depth and target statistics for audit tables."""
    rows: List[Dict[str, object]] = []
    for name, sl in [("train", sl_tr), ("val", sl_va), ("test", sl_te)]:
        n_win = int(sl.stop - sl.start)
        if n_win <= 0:
            continue
        idx: List[int] = []
        for wi in range(sl.start, sl.stop):
            s = int(starts[wi])
            idx.extend(range(s, s + int(window_len)))
        band = df.iloc[idx]
        corr = band[["depth_m", target]].corr().iloc[0, 1]
        rows.append(
            {
                "split": name,
                "n_windows": n_win,
                "depth_min_m": float(band["depth_m"].min()),
                "depth_max_m": float(band["depth_m"].max()),
                "y_mean": float(band[target].mean()),
                "y_std": float(band[target].std()),
                "abs_corr_gr_y": float("nan"),
            }
        )
        if "gr" in band.columns:
            c = band[["gr", target]].corr().iloc[0, 1]
            if np.isfinite(c):
                rows[-1]["abs_corr_gr_y"] = float(abs(c))
    return pd.DataFrame(rows)


def _parse_channels(s: str) -> Tuple[str, ...]:
    out: List[str] = []
    for p in str(s).split(","):
        q = p.strip().lower()
        if q:
            out.append(q)
    if not out:
        raise ValueError("At least one channel is required.")
    # preserve order, remove duplicates
    uniq: List[str] = []
    for c in out:
        if c not in uniq:
            uniq.append(c)
    return tuple(uniq)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Auddys Excel smoke direct-UB benchmark with optional CLP-CSGM.")
    p.add_argument("--excel-path", type=str, default="data/Auddys_table.xlsx")
    p.add_argument("--sheet", type=str, default="Logs")
    p.add_argument(
        "--u-channels",
        type=str,
        default="density,gr,res_deep,res_shallow",
        help="Comma-separated channels from Logs sheet after canonical rename.",
    )
    p.add_argument(
        "--target",
        type=str,
        default="k_lab",
        choices=("k_lab", "phi_lab", "phi_neutron", "phi_sonic", "phi_nd"),
        help="Target column after canonical rename (e.g. phi_neutron for NMR-CMR export).",
    )
    p.add_argument("--target-log10", action="store_true", help="Apply log10 transform to target before windowing.")
    p.add_argument("--window-len", type=int, default=16)
    p.add_argument("--step", type=int, default=1)
    p.add_argument("--train-frac", type=float, default=0.6)
    p.add_argument("--val-frac", type=float, default=0.2)
    p.add_argument("--base-dir", type=str, default="outputs/auddys_smoke_direct_ub")
    p.add_argument(
        "--run-prefix",
        type=str,
        default="auddys_smoke_clp_csgm",
        help="Prefix used when --run-id is not provided.",
    )
    p.add_argument("--run-id", type=str, default="")
    p.add_argument(
        "--copy-figures-to",
        type=str,
        default="",
        help="Optional destination root for figures copy. Files are copied to <dest>/<run_id>/.",
    )
    p.add_argument("--seeds", type=str, default="7")
    p.add_argument("--rhos", type=str, default="0.3,0.5")
    p.add_argument("--measurement-kind", type=str, default="subsample", choices=("gaussian", "subsample"))
    p.add_argument("--measurement-noise-std", type=float, default=0.01)
    p.add_argument("--residual-basis", type=str, default="dct", choices=("identity", "dct", "haar", "db4", "sym4", "fd1"))
    p.add_argument("--run-csgm-m2", action="store_true")
    p.add_argument("--run-csgm-ablations", action="store_true")
    p.add_argument("--csgm-prior-type", type=str, default="ridge", choices=("ridge", "mlp"))
    p.add_argument(
        "--csgm-prior-types",
        type=str,
        default="",
        help=(
            "Optional comma list ridge,mlp to run both CLP priors in one benchmark. "
            "When non-empty, overrides --csgm-prior-type."
        ),
    )
    p.add_argument("--csgm-latent-dim", type=int, default=16)
    p.add_argument("--csgm-ae-epochs", type=int, default=200)
    p.add_argument("--csgm-iters", type=int, default=400)
    p.add_argument("--csgm-restarts", type=int, default=3)
    p.add_argument("--csgm-opt-lr", type=float, default=0.05)
    p.add_argument("--csgm-lambda-grid", type=str, default="0.0001,0.0003,0.001,0.003,0.01,0.03,0.1")
    p.add_argument("--no-ae", action="store_true", help="Skip AE baseline for faster smoke.")
    p.add_argument("--no-plots", action="store_true", help="Skip all dashboard figures (01-10).")
    p.add_argument("--no-parity", action="store_true", help="Skip parity scatter and parity_pooled.npz.")
    p.add_argument(
        "--profile-overlap-agg",
        type=str,
        default=rwf.PROFILE_OVERLAP_AGG_UNIFORM_MEAN,
        choices=rwf.PROFILE_OVERLAP_AGG_CHOICES,
        help="Overlap fusion for depth-profile stitching (same as real_well_f03 runner).",
    )
    p.add_argument(
        "--save-profile-overlap-compare",
        action="store_true",
        help="Export uniform_mean vs center_weighted_mean overlap comparison bundle.",
    )
    p.add_argument(
        "--well-label",
        type=str,
        default="",
        help="Title label for depth-profile figure (default: derived from excel path).",
    )
    p.add_argument(
        "--benchmark-scenario",
        type=str,
        default="nmr_cmr_full",
        help="Scenario id for manifests (e.g. nmr_cmr_full, nmr_cmr_variable).",
    )
    p.add_argument(
        "--depth-min-m",
        type=str,
        default="",
        help="Inclusive lower depth bound (m) applied before windowing.",
    )
    p.add_argument(
        "--depth-max-m",
        type=str,
        default="",
        help="Inclusive upper depth bound (m) applied before windowing.",
    )
    p.add_argument(
        "--min-windows",
        type=int,
        default=30,
        help="Minimum sliding windows required after depth filter.",
    )
    return p.parse_args()


def _merge_line_examples_csgm(
    base: Optional[Dict[str, np.ndarray]],
    extra: Optional[Dict[str, np.ndarray]],
) -> Optional[Dict[str, np.ndarray]]:
    """Merge CSGM / ablation keys from a second prior run into line examples."""
    if extra is None:
        return base
    if base is None:
        return dict(extra)
    out = dict(base)
    for key, val in extra.items():
        if key not in _LINE_EXAMPLE_BASE_KEYS:
            out[key] = val
    return out


def _profile_x_label_for_target(target: str, target_log10: bool = False) -> str:
    if target in ("phi_neutron", "phi_lab", "phi_sonic", "phi_nd"):
        return "porosity (pu)"
    if target == "k_lab":
        if target_log10:
            return "log10 permeability (mD)"
        return "permeability (mD)"
    return str(target)


def _depth_profile_figure_name(target: str) -> str:
    if target in ("phi_neutron", "phi_lab", "phi_sonic", "phi_nd"):
        return "10_depth_profile_porosity.png"
    if target == "k_lab":
        return "10_depth_profile_permeability.png"
    return "10_depth_profile_{}.png".format(target)


def _well_label_from_excel(excel_path: str, override: str) -> str:
    if str(override).strip():
        return str(override).strip()
    base = os.path.basename(str(excel_path))
    if "1045" in base:
        return "1-BRSA-1045-SPS"
    stem = os.path.splitext(base)[0]
    return stem if stem else "Auddys Excel"


def _save_depth_profile_artifacts(
    run_root: str,
    figures_dir: str,
    tables_dir: str,
    parity_fragment: Dict[str, np.ndarray],
    depth_axis: np.ndarray,
    row_starts_test: np.ndarray,
    window_len: int,
    n_te: int,
    overlap_agg: str,
    save_overlap_compare: bool,
    profile_x_label: str,
    depth_profile_name: str,
    profile_title: str,
    log_fn: Any,
) -> List[str]:
    """Build figure 10 and depth_profile_test_block.npz (F03 real-well protocol)."""
    paths: List[str] = []
    l = int(window_len)
    d = np.asarray(depth_axis, dtype=np.float64).ravel()
    nrows = int(d.shape[0])
    rs = np.asarray(row_starts_test, dtype=np.int64).ravel()
    if rs.shape[0] != int(n_te):
        raise ValueError("row_starts_test length must match n_test windows.")

    overlap_primary = rwf.validate_profile_overlap_agg(str(overlap_agg))
    profiles: Dict[str, np.ndarray] = {}
    obs_stack = np.asarray(parity_fragment["y_true"], dtype=np.float64).reshape(int(n_te), l)
    obs_profile, _ = rwf.reconstruct_depth_profile(
        obs_stack, rs, l, nrows, overlap_agg=overlap_primary
    )
    profiles["observed"] = obs_profile
    for key in _DEPTH_PROFILE_KNOWN_MODEL_KEYS:
        if key not in parity_fragment:
            continue
        arr = np.asarray(parity_fragment[key], dtype=np.float64)
        if arr.size != int(n_te) * l:
            continue
        stack = arr.reshape(int(n_te), l)
        prof, _ = rwf.reconstruct_depth_profile(
            stack, rs, l, nrows, overlap_agg=overlap_primary
        )
        profiles[key] = prof

    overlap_paths: List[str] = []
    if bool(save_overlap_compare):
        overlap_paths = rwf.save_overlap_profile_agg_bundle(
            run_root,
            overlap_primary,
            d,
            rs,
            l,
            nrows,
            obs_stack,
            parity_fragment,
            list(_DEPTH_PROFILE_KNOWN_MODEL_KEYS),
            log_line=log_fn,
            profile_x_label=str(profile_x_label),
        )
        paths.extend(overlap_paths)

    rs0 = int(rs[0]) if rs.size > 0 else 0
    if rs0 > 0 and rs0 < nrows:
        tt_boundary = 0.5 * (float(d[rs0 - 1]) + float(d[rs0]))
    else:
        tt_boundary = float("nan")

    p10 = os.path.join(figures_dir, depth_profile_name)
    plot_real_well_depth_profile(
        d,
        profiles,
        p10,
        train_test_boundary=float(tt_boundary) if np.isfinite(tt_boundary) else None,
        title=str(profile_title),
        profile_x_label=str(profile_x_label),
    )
    paths.append(p10)
    log_fn("Figure: " + p10)

    prof_npz = os.path.join(tables_dir, "depth_profile_test_block.npz")
    np.savez_compressed(
        prof_npz,
        depth_axis=d,
        row_starts_test=rs,
        **{("profile_" + k): v for k, v in profiles.items()},
    )
    paths.append(prof_npz)
    log_fn("Table: " + prof_npz)
    return paths


def load_logs_table(excel_path: str, sheet_name: str) -> pd.DataFrame:
    if not os.path.isfile(excel_path):
        raise FileNotFoundError(excel_path)
    df = pd.read_excel(excel_path, sheet_name=sheet_name)
    col_map = {
        "Depth(m)": "depth_m",
        "Density (g/cc)": "density",
        "GR (API)": "gr",
        "Res_Deep": "res_deep",
        "Res_Shallow": "res_shallow",
        "Phi_Neutron (pu)": "phi_neutron",
        "Phi_Sonic (pu)": "phi_sonic",
        "Phi_ND (pu)": "phi_nd",
        "Phi_lab (pu)": "phi_lab",
        "k_lab (mD)": "k_lab",
        "RQI": "rqi",
        "FZI_lab": "fzi_lab",
        "Lithotype": "lithotype",
        "HFU": "hfu",
    }
    missing = [c for c in col_map if c not in df.columns]
    if missing:
        raise ValueError("Missing expected columns in sheet {}: {}".format(sheet_name, missing))
    out = df.rename(columns=col_map).copy()
    out = out.sort_values("depth_m").reset_index(drop=True)
    return out


def build_windows(
    df: pd.DataFrame,
    channels: Tuple[str, ...],
    target: str,
    window_len: int,
    step: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    for c in channels:
        if c not in df.columns:
            raise ValueError("Unknown channel: {}".format(c))
    if target not in df.columns:
        raise ValueError("Unknown target: {}".format(target))
    l = int(window_len)
    st = max(1, int(step))
    n = int(df.shape[0])
    if n < l:
        raise ValueError("Need at least {} rows, got {}.".format(l, n))

    x_rows: List[np.ndarray] = []
    y_rows: List[np.ndarray] = []
    starts: List[int] = []

    values = {c: df[c].to_numpy(dtype=np.float64) for c in channels}
    y_all = df[target].to_numpy(dtype=np.float64)
    i = 0
    while i + l <= n:
        u = np.concatenate([values[c][i : i + l] for c in channels], axis=0)
        y = y_all[i : i + l]
        x_rows.append(u.astype(np.float64, copy=False))
        y_rows.append(y.astype(np.float64, copy=False))
        starts.append(i)
        i += st
    if not x_rows:
        raise ValueError("No windows produced; check window_len and step.")
    return np.stack(x_rows, axis=0), np.stack(y_rows, axis=0), np.asarray(starts, dtype=np.int64)


def build_data_dict(x_all: np.ndarray, y_all: np.ndarray, sl_tr: slice, sl_va: slice, sl_te: slice, residual_basis: str) -> Dict[str, np.ndarray]:
    y_tr = y_all[sl_tr]
    y_va = y_all[sl_va]
    y_te = y_all[sl_te]
    l = int(y_tr.shape[1])
    psi = get_basis(l, residual_basis)
    return {
        "X_train": x_all[sl_tr],
        "X_val": x_all[sl_va],
        "X_test": x_all[sl_te],
        "Y_train": y_tr,
        "Y_val": y_va,
        "Y_test": y_te,
        "Alpha_train": y_tr @ psi,
        "Alpha_val": y_va @ psi,
        "Alpha_test": y_te @ psi,
        "Psi": psi,
    }


def write_dataset_manifest(
    run_root: str,
    excel_path: str,
    channels: Tuple[str, ...],
    target: str,
    target_log10: bool,
    n_rows_raw: int,
    n_rows_after_depth_filter: int,
    depth_min_m: Optional[float],
    depth_max_m: Optional[float],
    benchmark_scenario: str,
    n_windows: int,
    n_tr: int,
    n_va: int,
    n_te: int,
    window_len: int,
    step: int,
    paper_strict_paired_b: bool,
    split_depth_csv: str,
) -> str:
    lines = [
        "Auddys Excel CLP-CSGM benchmark dataset manifest.",
        "",
        "benchmark_scenario: {}".format(benchmark_scenario),
        "excel_path: {}".format(excel_path),
        "u_channels: {}".format(", ".join(channels)),
        "target: {}".format(target),
        "target_log10: {}".format(target_log10),
        "depth_min_m: {}".format(depth_min_m if depth_min_m is not None else "none"),
        "depth_max_m: {}".format(depth_max_m if depth_max_m is not None else "none"),
        "n_rows_excel: {}".format(n_rows_raw),
        "n_rows_after_depth_filter: {}".format(n_rows_after_depth_filter),
        "n_windows: {}".format(n_windows),
        "window_len: {}".format(window_len),
        "step: {}".format(step),
        "split_windows_n_train: {}".format(n_tr),
        "split_windows_n_val: {}".format(n_va),
        "split_windows_n_test: {}".format(n_te),
        "paper_strict_paired_b: {}".format(bool(paper_strict_paired_b)),
        "split_depth_summary_csv: {}".format(split_depth_csv),
        "",
        "Protocol notes:",
        "  - Depth filter applied before sliding windows (if depth_min/max set).",
        "  - Windows follow depth order (contiguous split; no random split).",
        "  - Alpha arrays are built as Y @ Psi.",
        "  - See experiments/nmr_cmr_variable/PROTOCOL_NMR_CMR_variable.md when scenario is nmr_cmr_variable.",
    ]
    path = os.path.join(run_root, "DATASET_MANIFEST.txt")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    return path


def main() -> None:
    args = parse_args()
    run_id = args.run_id.strip() or "{}_{}".format(str(args.run_prefix).strip(), time.strftime("%Y%m%d_%H%M%S"))
    run_root = os.path.join(args.base_dir, "runs", run_id)
    tables_dir = os.path.join(run_root, "tables")
    figures_dir = os.path.join(run_root, "figures")
    logs_dir = os.path.join(run_root, "logs")
    os.makedirs(tables_dir, exist_ok=True)
    os.makedirs(figures_dir, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)

    tee = _Tee(sys.stdout, os.path.join(logs_dir, "run_console.log"))
    t0 = time.time()
    old_stdout = sys.stdout
    sys.stdout = tee
    try:
        channels = _parse_channels(args.u_channels)
        seeds = _parse_int_list(args.seeds)
        rhos = _parse_float_list(args.rhos)

        df = load_logs_table(args.excel_path, args.sheet)
        n_rows_excel = int(df.shape[0])
        y_col = str(args.target)
        if bool(args.target_log10):
            y_raw = np.clip(df[y_col].to_numpy(dtype=np.float64), 1e-6, None)
            df = df.copy()
            df[y_col] = np.log10(y_raw)
            _log(tee, "Target log10 transform applied to {}".format(y_col))

        depth_min_m = _parse_optional_float(str(args.depth_min_m))
        depth_max_m = _parse_optional_float(str(args.depth_max_m))
        benchmark_scenario = str(args.benchmark_scenario).strip() or "nmr_cmr_full"
        if depth_min_m is not None or depth_max_m is not None:
            df = apply_depth_bounds(df, depth_min_m, depth_max_m)
            _log(
                tee,
                "Depth filter: min={} max={} -> {} rows".format(
                    depth_min_m, depth_max_m, int(df.shape[0])
                ),
            )
        n_rows_filtered = int(df.shape[0])

        x_all, y_all, starts = build_windows(df, channels, y_col, int(args.window_len), int(args.step))
        min_windows = int(args.min_windows)
        if int(x_all.shape[0]) < min_windows:
            raise ValueError(
                "Only {} windows after depth filter; need >= {} (min-windows).".format(
                    int(x_all.shape[0]), min_windows
                )
            )
        sl_tr, sl_va, sl_te, n_tr, n_va, n_te = rwf.contiguous_split(
            int(x_all.shape[0]), float(args.train_frac), float(args.val_frac)
        )
        split_depth_df = split_depth_summary_table(
            df, starts, sl_tr, sl_va, sl_te, int(args.window_len), y_col
        )
        split_depth_path = os.path.join(tables_dir, "split_depth_summary.csv")
        split_depth_df.to_csv(split_depth_path, index=False)
        _log(tee, "Wrote split depth summary: " + split_depth_path)
        data = build_data_dict(x_all, y_all, sl_tr, sl_va, sl_te, str(args.residual_basis))

        cfg = Config(config_profile="real_well_f03_direct_ub")
        apply_config_profile(cfg)
        cfg.n_output = int(y_all.shape[1])
        cfg.p_input = int(x_all.shape[1])
        cfg.n_train = int(n_tr)
        cfg.n_val = int(n_va)
        cfg.n_test = int(n_te)
        cfg.seeds = list(seeds)
        cfg.measurement_ratios = list(rhos)
        cfg.measurement_kind = str(args.measurement_kind)
        cfg.measurement_noise_std = float(args.measurement_noise_std)
        cfg.residual_basis = str(args.residual_basis)
        cfg.run_lfista = False
        cfg.log_progress = False
        cfg.run_csgm_m2 = bool(args.run_csgm_m2)
        cfg.run_csgm_ablations = bool(args.run_csgm_ablations)
        prior_types = _parse_csgm_prior_types(args)
        cfg.csgm_prior_type = str(prior_types[0])
        cfg.csgm_latent_dim = int(args.csgm_latent_dim)
        cfg.csgm_ae_epochs = int(args.csgm_ae_epochs)
        cfg.csgm_iters = int(args.csgm_iters)
        cfg.csgm_restarts = int(args.csgm_restarts)
        cfg.csgm_opt_lr = float(args.csgm_opt_lr)
        cfg.csgm_lambda_grid = _parse_float_list(args.csgm_lambda_grid)
        # Align with main paper runners: same sparse measurement vector b for CSGM and direct [u,b] baselines.
        cfg.paper_strict_paired_b = True

        dub_cfg = dub.DirectUBTrainConfig(ae_epochs=80)
        run_ae = not bool(args.no_ae)
        include_hybrid_fista = False
        include_lfista = False
        joint_only = True

        detailed_parts: List[pd.DataFrame] = []
        parity_bundles: List[Dict[str, np.ndarray]] = []
        first_line_examples: Optional[Dict[str, np.ndarray]] = None
        first_parity_fragment: Optional[Dict[str, np.ndarray]] = None
        first_job_info: Dict[str, float] = {}
        for seed in seeds:
            for rho in rhos:
                frags_this_rho: List[Dict[str, np.ndarray]] = []
                line_ex_this_rho: Optional[Dict[str, np.ndarray]] = None
                for pt in prior_types:
                    cfg.csgm_prior_type = str(pt)
                    _log(tee, "[run] seed={} rho={} csgm_prior={}".format(seed, rho, pt))
                    df_res, parity_fragment, line_ex = run_direct_ub_from_data(
                        cfg=cfg,
                        dub_cfg=dub_cfg,
                        data=data,
                        seed=int(seed),
                        measurement_ratio=float(rho),
                        include_hybrid_fista=include_hybrid_fista,
                        run_ae=run_ae,
                        include_lfista=include_lfista,
                        joint_only=joint_only,
                    )
                    detailed_parts.append(df_res)
                    frags_this_rho.append(parity_fragment)
                    line_ex_this_rho = _merge_line_examples_csgm(line_ex_this_rho, line_ex)
                parity_bundles.append(_merge_parity_fragments_csgm(frags_this_rho))
                if first_parity_fragment is None:
                    first_parity_fragment = _merge_parity_fragments_csgm(frags_this_rho)
                    first_line_examples = line_ex_this_rho
                    first_job_info = {"seed": float(seed), "rho": float(rho)}

        cfg.csgm_prior_type = str(prior_types[0])

        if not detailed_parts:
            raise RuntimeError("No results generated.")
        detailed = pd.concat(detailed_parts, ignore_index=True)
        if len(prior_types) > 1:
            detailed = detailed.drop_duplicates(
                subset=["seed", "measurement_ratio", "method", "sample_id"],
                keep="first",
            ).reset_index(drop=True)
        per_seed = summarize_results_per_seed(detailed)
        summary = summarize_results_across_seeds(per_seed)

        detailed.to_csv(os.path.join(tables_dir, "detailed_results.csv"), index=False)
        per_seed.to_csv(os.path.join(tables_dir, "summary_by_seed.csv"), index=False)
        summary.to_csv(os.path.join(tables_dir, "summary.csv"), index=False)

        cfg.save_dir = run_root
        cfg.plots_subdir = "figures"
        plot_paths: List[str] = []
        if not bool(args.no_plots):
            plot_paths = save_all_comparison_plots(cfg, summary, per_seed)
            if first_line_examples is not None:
                p08 = os.path.join(figures_dir, "08_example_ground_truth_vs_models.png")
                plot_direct_ub_ground_truth_vs_models(first_line_examples, p08)
                plot_paths.append(p08)
                _log(tee, "Figure: " + p08)

        if (not bool(args.no_plots)) and first_parity_fragment is not None:
            try:
                row_starts_test = np.asarray(starts[sl_te], dtype=np.int64)
                depth_axis = df["depth_m"].to_numpy(dtype=np.float64)
                well_label = _well_label_from_excel(args.excel_path, str(args.well_label))
                profile_x_label = _profile_x_label_for_target(y_col, bool(args.target_log10))
                depth_profile_name = _depth_profile_figure_name(y_col)
                profile_title = (
                    "{} {} profile (test block) | rho={:.2f}, seed={}".format(
                        well_label,
                        y_col,
                        float(first_job_info.get("rho", float("nan"))),
                        int(first_job_info.get("seed", 0)),
                    )
                )
                plot_paths.extend(
                    _save_depth_profile_artifacts(
                        run_root=run_root,
                        figures_dir=figures_dir,
                        tables_dir=tables_dir,
                        parity_fragment=first_parity_fragment,
                        depth_axis=depth_axis,
                        row_starts_test=row_starts_test,
                        window_len=int(args.window_len),
                        n_te=int(n_te),
                        overlap_agg=str(args.profile_overlap_agg),
                        save_overlap_compare=bool(args.save_profile_overlap_compare),
                        profile_x_label=profile_x_label,
                        depth_profile_name=depth_profile_name,
                        profile_title=profile_title,
                        log_fn=lambda msg: _log(tee, msg),
                    )
                )
            except (ValueError, KeyError) as ex:
                _log(tee, "Warning: depth-profile plot skipped: " + str(ex))

        if (not bool(args.no_parity)) and parity_bundles:
            merged = merge_gt_pred_bundles(parity_bundles)
            if "y_true" in merged:
                if not bool(args.no_plots):
                    parity_path = os.path.join(figures_dir, "09_parity_ground_truth_vs_prediction.png")
                    plot_parity_ground_truth_vs_predictions(cfg, merged, parity_path)
                    plot_paths.append(parity_path)
                np.savez_compressed(os.path.join(tables_dir, "parity_pooled.npz"), **merged)

        copied_plot_paths: List[str] = []
        if str(args.copy_figures_to).strip():
            dst_root = os.path.join(str(args.copy_figures_to).strip(), run_id)
            os.makedirs(dst_root, exist_ok=True)
            for src in sorted(plot_paths):
                dst = os.path.join(dst_root, os.path.basename(src))
                shutil.copy2(src, dst)
                copied_plot_paths.append(dst)
            _log(tee, "Copied {} figure(s) to {}".format(len(copied_plot_paths), dst_root))

        focus_paths = save_focus_tables(
            run_root=run_root,
            summary=summary,
            per_seed=per_seed,
            include_hybrid_fista=include_hybrid_fista,
            run_ae=run_ae,
            include_lfista=include_lfista,
            joint_only=joint_only,
            include_csgm_m2=bool(cfg.run_csgm_m2),
        )

        with open(os.path.join(run_root, "config.json"), "w", encoding="utf-8") as f:
            json.dump(
                {
                    "cfg": asdict(cfg),
                    "dub_cfg": asdict(dub_cfg),
                    "args": vars(args),
                    "resolved": {
                        "benchmark_scenario": benchmark_scenario,
                        "depth_min_m": depth_min_m,
                        "depth_max_m": depth_max_m,
                        "n_rows_excel": n_rows_excel,
                        "n_rows_after_depth_filter": n_rows_filtered,
                        "csgm_prior_types": list(prior_types),
                        "channels": list(channels),
                        "seeds": list(seeds),
                        "rhos": list(rhos),
                        "n_windows": int(x_all.shape[0]),
                        "n_train": int(n_tr),
                        "n_val": int(n_va),
                        "n_test": int(n_te),
                        "split_depth_summary_csv": split_depth_path,
                    },
                },
                f,
                indent=2,
            )

        protocol_path = write_protocol(
            run_root=run_root,
            joint_only=joint_only,
            residual_basis=str(args.residual_basis),
            measurement_kind=str(args.measurement_kind),
            measurement_noise_std=float(args.measurement_noise_std),
            residual_k=int(cfg.residual_k),
            include_csgm_m2=bool(cfg.run_csgm_m2),
            include_lfista=False,
        )
        manifest_path = write_dataset_manifest(
            run_root=run_root,
            excel_path=args.excel_path,
            channels=channels,
            target=y_col,
            target_log10=bool(args.target_log10),
            n_rows_raw=n_rows_excel,
            n_rows_after_depth_filter=n_rows_filtered,
            depth_min_m=depth_min_m,
            depth_max_m=depth_max_m,
            benchmark_scenario=benchmark_scenario,
            n_windows=int(x_all.shape[0]),
            n_tr=int(n_tr),
            n_va=int(n_va),
            n_te=int(n_te),
            window_len=int(args.window_len),
            step=int(args.step),
            paper_strict_paired_b=bool(cfg.paper_strict_paired_b),
            split_depth_csv=split_depth_path,
        )
        elapsed = float(time.time() - t0)
        run_manifest_path = write_run_manifest(
            run_root=run_root,
            run_id=run_id,
            elapsed_s=elapsed,
            plot_paths=plot_paths,
            focus_paths=focus_paths,
            joint_only=joint_only,
            residual_basis=str(args.residual_basis),
            measurement_kind=str(args.measurement_kind),
            measurement_noise_std=float(args.measurement_noise_std),
            residual_k=int(cfg.residual_k),
            include_csgm_m2=bool(cfg.run_csgm_m2),
            include_lfista=False,
        )
        _log(tee, "DONE run_root={}".format(run_root))
        _log(tee, "WROTE protocol={}".format(protocol_path))
        _log(tee, "WROTE dataset_manifest={}".format(manifest_path))
        _log(tee, "WROTE run_manifest={}".format(run_manifest_path))
        _log(tee, "WROTE summary={}".format(os.path.join(tables_dir, "summary.csv")))
        if copied_plot_paths:
            _log(tee, "WROTE copied_figures={}".format(len(copied_plot_paths)))
    finally:
        sys.stdout = old_stdout
        tee.close()


if __name__ == "__main__":
    main()
