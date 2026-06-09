#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Load F03-4 well log (AC, GR, Porosity vs depth) and build sliding windows for
direct [u,b]->y benchmark (Path A, real data).

u: [AC window || GR window] in R^{2L}. y: Porosity in R^L. Split is contiguous
along depth (no shuffle) to reduce leakage.

ASCII-only.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from sir_cs_pipeline_optimized import get_basis

# Overlap stitching when step < window_len: how window predictions are fused per depth row.
PROFILE_OVERLAP_AGG_UNIFORM_MEAN = "uniform_mean"
PROFILE_OVERLAP_AGG_CENTER_WEIGHTED_MEAN = "center_weighted_mean"
PROFILE_OVERLAP_AGG_CHOICES: Tuple[str, ...] = (
    PROFILE_OVERLAP_AGG_UNIFORM_MEAN,
    PROFILE_OVERLAP_AGG_CENTER_WEIGHTED_MEAN,
)


def validate_profile_overlap_agg(name: str) -> str:
    """Return canonical overlap-aggregation name or raise ValueError."""
    s = str(name).strip().lower().replace("-", "_")
    if s in PROFILE_OVERLAP_AGG_CHOICES:
        return s
    raise ValueError(
        "profile_overlap_agg must be one of {}, got {!r}.".format(
            list(PROFILE_OVERLAP_AGG_CHOICES),
            name,
        )
    )


def overlap_position_weights(window_len: int, overlap_agg: str) -> np.ndarray:
    """
    Per-within-window weights of shape (L,) for overlap aggregation.

    uniform_mean: all ones (simple overlap average).
    center_weighted_mean: symmetric triangular weights min(j+1, L-j), max at center.
    """
    l = int(window_len)
    if l < 1:
        raise ValueError("window_len must be >= 1.")
    agg = validate_profile_overlap_agg(overlap_agg)
    if agg == PROFILE_OVERLAP_AGG_UNIFORM_MEAN:
        return np.ones(l, dtype=np.float64)
    w = np.empty(l, dtype=np.float64)
    for j in range(l):
        w[j] = float(min(j + 1, l - j))
    w = np.maximum(w, 1.0e-12)
    return w


@dataclass
class F03Table:
    """Indexed rows sorted by increasing depth."""

    depth: np.ndarray
    ac: np.ndarray
    gr: np.ndarray
    porosity: np.ndarray

    @property
    def n_rows(self) -> int:
        return int(self.depth.shape[0])


def load_f03_table(path: str) -> F03Table:
    """Read tab-separated file with header Depth, AC, GR, Porosity."""
    p = os.path.abspath(path.strip())
    if not os.path.isfile(p):
        raise FileNotFoundError(p)
    df = pd.read_csv(p, sep="\t")
    dcol = _find_col(df, "depth")
    acol = _find_col(df, "ac")
    gcol = _find_col(df, "gr")
    pcol = _find_col(df, "porosity")
    depth = np.asarray(df[dcol].values, dtype=np.float64)
    ac = np.asarray(df[acol].values, dtype=np.float64)
    gr = np.asarray(df[gcol].values, dtype=np.float64)
    poro = np.asarray(df[pcol].values, dtype=np.float64)
    order = np.argsort(depth)
    depth = depth[order]
    ac = ac[order]
    gr = gr[order]
    poro = poro[order]
    mask = np.isfinite(depth) & np.isfinite(ac) & np.isfinite(gr) & np.isfinite(poro)
    if not bool(np.all(mask)):
        depth = depth[mask]
        ac = ac[mask]
        gr = gr[mask]
        poro = poro[mask]
    if depth.size < 8:
        raise ValueError("Too few rows after cleaning.")
    return F03Table(depth=depth, ac=ac, gr=gr, porosity=poro)


def _norm_name(c: str) -> str:
    s = str(c).strip().lower()
    for ch in " ()/":
        s = s.replace(ch, "")
    return s


def _find_col(df: pd.DataFrame, key: str) -> str:
    """
    key: 'depth' | 'ac' | 'gr' | 'porosity'
    """
    for c in df.columns:
        n = _norm_name(c)
        if key == "depth" and n.startswith("depth"):
            return str(c)
        if key == "ac" and n.startswith("ac"):
            return str(c)
        if key == "gr" and n.startswith("gr"):
            return str(c)
        if key == "porosity" and (n == "porosity" or n.startswith("porosity") or n == "phi"):
            return str(c)
    names = list(df.columns)
    if len(names) >= 4 and key == "depth":
        return str(names[0])
    if len(names) >= 4 and key == "ac":
        return str(names[1])
    if len(names) >= 4 and key == "gr":
        return str(names[2])
    if len(names) >= 4 and key == "porosity":
        return str(names[3])
    raise KeyError("Could not find column for key={} in {}".format(key, list(df.columns)))


_VALID_CHANNELS = ("ac", "gr")


def _select_channel_series(tab: F03Table, name: str) -> np.ndarray:
    """Return the raw series from tab by canonical channel name."""
    key = name.strip().lower()
    if key == "ac":
        return tab.ac
    if key == "gr":
        return tab.gr
    raise ValueError(
        "Unknown channel '{}'. Valid channels: {}.".format(name, _VALID_CHANNELS)
    )


def normalize_channels(channels: Tuple[str, ...]) -> Tuple[str, ...]:
    """Return a validated, order-preserving tuple of unique channel names."""
    seen: List[str] = []
    for c in channels:
        key = c.strip().lower()
        if key not in _VALID_CHANNELS:
            raise ValueError(
                "Unknown channel '{}'. Valid channels: {}.".format(c, _VALID_CHANNELS)
            )
        if key not in seen:
            seen.append(key)
    if not seen:
        raise ValueError("At least one channel must be provided.")
    return tuple(seen)


def build_sliding_windows(
    tab: F03Table,
    window_len: int,
    step: int,
    channels: Tuple[str, ...] = ("ac", "gr"),
) -> Tuple[
    np.ndarray,
    np.ndarray,
    List[float],
    List[Tuple[float, float]],
]:
    """
    Return X (n_win, C*L), Y (n_win, L), center_depths, depth_range per window.
    Here C = len(channels); channels in order define the block structure of each u.
    """
    ch = normalize_channels(channels)
    series = [_select_channel_series(tab, c) for c in ch]
    l = int(window_len)
    st = max(1, int(step))
    n = tab.n_rows
    if n < l:
        raise ValueError(f"Need at least L={l} depth samples, got n={n}.")
    xs: List[np.ndarray] = []
    ys: List[np.ndarray] = []
    centers: List[float] = []
    ranges: List[Tuple[float, float]] = []
    t = 0
    while t + l <= n:
        sw = tab.depth[t : t + l]
        centers.append(float(0.5 * (sw[0] + sw[-1])))
        ranges.append((float(sw[0]), float(sw[-1])))
        yseg = tab.porosity[t : t + l]
        segs = [s[t : t + l] for s in series]
        u = np.concatenate(segs, axis=0)
        xs.append(u.astype(np.float64, copy=False))
        ys.append(yseg.astype(np.float64, copy=False))
        t += st
    if not xs:
        raise ValueError("No windows: check window_len and step.")
    x_arr = np.stack(xs, axis=0)
    y_arr = np.stack(ys, axis=0)
    return x_arr, y_arr, centers, ranges


def contiguous_split(
    n_samples: int, train_frac: float, val_frac: float
) -> Tuple[slice, slice, slice, int, int, int]:
    if not (0.0 < train_frac < 1.0) or not (0.0 < val_frac < 1.0):
        raise ValueError("train_frac and val_frac must be in (0,1).")
    if train_frac + val_frac >= 1.0:
        raise ValueError("train_frac + val_frac must be < 1.0 to leave a test set.")
    n_tr = int(np.floor(float(train_frac) * float(n_samples)))
    n_va = int(np.floor(float(val_frac) * float(n_samples)))
    n_te = n_samples - n_tr - n_va
    if n_tr < 4 or n_va < 2 or n_te < 2:
        raise ValueError(
            f"Split too small: n={n_samples} n_tr={n_tr} n_va={n_va} n_te={n_te}. "
            "Increase data or adjust fracs / windowing."
        )
    sl_tr = slice(0, n_tr)
    sl_va = slice(n_tr, n_tr + n_va)
    sl_te = slice(n_tr + n_va, n_samples)
    return sl_tr, sl_va, sl_te, n_tr, n_va, n_te


def reconstruct_depth_profile(
    window_preds: np.ndarray,
    window_starts: np.ndarray,
    window_len: int,
    n_rows_total: int,
    overlap_agg: str = PROFILE_OVERLAP_AGG_UNIFORM_MEAN,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Reconstruct a point-wise profile of length n_rows_total from overlapping window
    predictions by weighted fusion of all windows that cover each row.

    window_preds: array (n_win, L) of predicted y-windows.
    window_starts: array (n_win,) of absolute starting row indices for each window.
    window_len: L (number of rows per window).
    n_rows_total: total number of rows in the full well (depth axis).
    overlap_agg: uniform_mean (default) or center_weighted_mean; see module constants.

    Returns (profile, weight_sum) both of length n_rows_total. weight_sum is the
    accumulated overlap weight per row (integer counts for uniform_mean). Rows with
    weight_sum == 0 are left as NaN in profile.
    """
    agg = validate_profile_overlap_agg(overlap_agg)
    wp = np.asarray(window_preds, dtype=np.float64)
    ws = np.asarray(window_starts, dtype=np.int64).ravel()
    l = int(window_len)
    nr = int(n_rows_total)
    if wp.ndim != 2 or wp.shape[1] != l:
        raise ValueError("window_preds shape must be (n_win, L).")
    if ws.shape[0] != wp.shape[0]:
        raise ValueError("window_starts must align with window_preds.")
    wpos = overlap_position_weights(l, agg)
    acc = np.zeros(nr, dtype=np.float64)
    wsum = np.zeros(nr, dtype=np.float64)
    for j in range(wp.shape[0]):
        t = int(ws[j])
        if t < 0 or t + l > nr:
            continue
        acc[t : t + l] += wp[j] * wpos
        wsum[t : t + l] += wpos
    profile = np.full(nr, np.nan, dtype=np.float64)
    mask = wsum > 0.0
    profile[mask] = acc[mask] / wsum[mask]
    return profile, wsum


def build_profile_overlap_sensitivity_frame(
    obs_profile_uniform: np.ndarray,
    obs_profile_weighted: np.ndarray,
    profiles_uniform: Dict[str, np.ndarray],
    profiles_weighted: Dict[str, np.ndarray],
    method_keys: Sequence[str],
) -> pd.DataFrame:
    """
    Tabulate how much stitched depth profiles differ between uniform_mean and
    center_weighted_mean aggregation for each method, plus RMSE to observed profiles
    under matched aggregation.
    """
    rows: List[Dict[str, Any]] = []
    ou = np.asarray(obs_profile_uniform, dtype=np.float64).ravel()
    ow = np.asarray(obs_profile_weighted, dtype=np.float64).ravel()
    for key in method_keys:
        if key not in profiles_uniform or key not in profiles_weighted:
            continue
        pu = np.asarray(profiles_uniform[key], dtype=np.float64).ravel()
        pw = np.asarray(profiles_weighted[key], dtype=np.float64).ravel()
        if pu.shape[0] != pw.shape[0] or pu.shape[0] != ou.shape[0]:
            continue
        m_uw = np.isfinite(pu) & np.isfinite(pw)
        m_uo = np.isfinite(pu) & np.isfinite(ou)
        m_wo = np.isfinite(pw) & np.isfinite(ow)
        n_uw = int(np.sum(m_uw))
        n_uo = int(np.sum(m_uo))
        n_wo = int(np.sum(m_wo))
        def _rmse(a: np.ndarray, b: np.ndarray, m: np.ndarray) -> float:
            if not bool(np.any(m)):
                return float("nan")
            d = a[m] - b[m]
            return float(np.sqrt(np.mean(d * d)))

        def _mae(a: np.ndarray, b: np.ndarray, m: np.ndarray) -> float:
            if not bool(np.any(m)):
                return float("nan")
            return float(np.mean(np.abs(a[m] - b[m])))

        rows.append(
            {
                "method": str(key),
                "n_finite_uniform_vs_weighted": n_uw,
                "rmse_profile_uniform_vs_weighted": _rmse(pu, pw, m_uw),
                "mae_profile_uniform_vs_weighted": _mae(pu, pw, m_uw),
                "n_finite_uniform_vs_obs_uniform": n_uo,
                "rmse_profile_uniform_vs_obs_uniform": _rmse(pu, ou, m_uo),
                "n_finite_weighted_vs_obs_weighted": n_wo,
                "rmse_profile_weighted_vs_obs_weighted": _rmse(pw, ow, m_wo),
            }
        )
    return pd.DataFrame(rows)


def write_overlap_profile_agg_readme(dst_readme: str, primary_agg: str) -> None:
    lines = [
        "Overlap profile sensitivity pack (visualization path only).",
        "",
        "Primary stitched profile for the run depth figure uses overlap_agg="
        + str(primary_agg)
        + " (CLI --profile-overlap-agg).",
        "",
        "This folder compares uniform_mean vs center_weighted_mean row fusion:",
        "same overlapping window predictions, different per-position weights.",
        "",
        "Tabular benchmark metrics in detailed_results.csv remain window-level paired;",
        "they are not recomputed from stitched profiles.",
    ]
    with open(dst_readme, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def save_overlap_profile_agg_bundle(
    run_root: str,
    primary_agg: str,
    depth_axis: np.ndarray,
    row_starts: np.ndarray,
    l: int,
    nrows: int,
    obs_stack: np.ndarray,
    first_parity_fragment: Dict[str, np.ndarray],
    known_model_keys: Sequence[str],
    log_line: Optional[Callable[[str], None]] = None,
    profile_x_label: str = "target",
) -> List[str]:
    """
    Write run_root/overlap_profile_agg/{README.txt,tables/,figures/} comparing
    uniform_mean vs center_weighted_mean overlap stitching (visualization only).
    """
    import matplotlib.pyplot as plt

    def _emit(msg: str) -> None:
        if log_line is not None:
            log_line(msg)

    root = os.path.join(run_root, "overlap_profile_agg")
    tdir = os.path.join(root, "tables")
    fdir = os.path.join(root, "figures")
    os.makedirs(tdir, exist_ok=True)
    os.makedirs(fdir, exist_ok=True)
    out_paths: List[str] = []
    readme_path = os.path.join(root, "README.txt")
    write_overlap_profile_agg_readme(readme_path, primary_agg)
    out_paths.append(readme_path)

    ou, _ = reconstruct_depth_profile(
        obs_stack,
        row_starts,
        l,
        nrows,
        overlap_agg=PROFILE_OVERLAP_AGG_UNIFORM_MEAN,
    )
    ow, _ = reconstruct_depth_profile(
        obs_stack,
        row_starts,
        l,
        nrows,
        overlap_agg=PROFILE_OVERLAP_AGG_CENTER_WEIGHTED_MEAN,
    )
    profiles_u: Dict[str, np.ndarray] = {}
    profiles_w: Dict[str, np.ndarray] = {}
    for k in known_model_keys:
        if k not in first_parity_fragment:
            continue
        arr = np.asarray(first_parity_fragment[k], dtype=np.float64)
        if arr.size != int(obs_stack.shape[0]) * l:
            continue
        stack = arr.reshape(int(obs_stack.shape[0]), l)
        pu, _ = reconstruct_depth_profile(
            stack,
            row_starts,
            l,
            nrows,
            overlap_agg=PROFILE_OVERLAP_AGG_UNIFORM_MEAN,
        )
        pw, _ = reconstruct_depth_profile(
            stack,
            row_starts,
            l,
            nrows,
            overlap_agg=PROFILE_OVERLAP_AGG_CENTER_WEIGHTED_MEAN,
        )
        profiles_u[k] = pu
        profiles_w[k] = pw

    frame = build_profile_overlap_sensitivity_frame(ou, ow, profiles_u, profiles_w, known_model_keys)
    csv_path = os.path.join(tdir, "profile_overlap_sensitivity.csv")
    frame.to_csv(csv_path, index=False)
    out_paths.append(csv_path)

    npz_path = os.path.join(tdir, "profiles_uniform_vs_weighted.npz")
    payload: Dict[str, np.ndarray] = {
        "depth_axis": np.asarray(depth_axis, dtype=np.float64).ravel(),
        "obs_profile_uniform_mean": ou,
        "obs_profile_center_weighted_mean": ow,
    }
    for k in profiles_u:
        payload["pred_" + k + "_uniform_mean"] = profiles_u[k]
        payload["pred_" + k + "_center_weighted_mean"] = profiles_w[k]
    np.savez_compressed(npz_path, **payload)
    out_paths.append(npz_path)

    if frame.shape[0] > 0:
        methods = frame["method"].tolist()
        yvals = frame["rmse_profile_uniform_vs_weighted"].to_numpy(dtype=np.float64)
        fig, ax = plt.subplots(figsize=(max(6.0, 0.35 * len(methods) + 2.0), 4.0))
        xpos = np.arange(len(methods))
        ax.bar(xpos, yvals, color="#4c72b0")
        ax.set_xticks(xpos)
        ax.set_xticklabels(methods, rotation=35, ha="right")
        ax.set_ylabel("RMSE (uniform vs weighted profile)")
        ax.set_title("Overlap aggregation sensitivity (profile stitch only)")
        ax.grid(True, axis="y", alpha=0.3)
        fig.tight_layout()
        bar_path = os.path.join(fdir, "01_bar_rmse_uniform_vs_weighted_profile.png")
        fig.savefig(bar_path, dpi=160)
        plt.close(fig)
        out_paths.append(bar_path)

    overlay_key = None
    for cand in ("ridge_prior_csgm", "mlp_prior_csgm", "hybrid_fista", "ae_regression_ub"):
        if cand in profiles_u:
            overlay_key = cand
            break
    if overlay_key is None and profiles_u:
        overlay_key = sorted(profiles_u.keys())[0]
    if overlay_key is not None:
        d = np.asarray(depth_axis, dtype=np.float64).ravel()
        pu = profiles_u[overlay_key]
        pw = profiles_w[overlay_key]
        m = np.isfinite(ou) & np.isfinite(pu) & np.isfinite(pw)
        fig, ax = plt.subplots(figsize=(5.0, 7.0))
        if bool(np.any(m)):
            ax.plot(ou[m], d[m], color="#404040", linewidth=1.0, label="observed (uniform stitch)")
            ax.plot(pu[m], d[m], color="#1f77b4", linewidth=1.0, alpha=0.9, label=overlay_key + " uniform")
            ax.plot(pw[m], d[m], color="#d62728", linewidth=1.0, alpha=0.9, label=overlay_key + " weighted")
        ax.set_xlabel(profile_x_label)
        ax.set_ylabel("depth")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best", fontsize=8)
        ax.set_title("Depth overlay | " + overlay_key)
        fig.tight_layout()
        ov_path = os.path.join(fdir, "02_depth_overlay_uniform_vs_weighted.png")
        fig.savefig(ov_path, dpi=160)
        plt.close(fig)
        out_paths.append(ov_path)

    _emit("Overlap profile sensitivity: " + str(len(out_paths)) + " files under " + root)
    return out_paths


def test_window_row_starts(
    n_train: int, n_val: int, n_test: int, step: int
) -> np.ndarray:
    """Absolute starting row indices (into tab.depth) for each test-block window."""
    st = max(1, int(step))
    start = (int(n_train) + int(n_val)) * st
    return np.arange(int(n_test), dtype=np.int64) * st + start


def build_direct_ub_data_dict(
    x_all: np.ndarray,
    y_all: np.ndarray,
    sl_tr: slice,
    sl_va: slice,
    sl_te: slice,
    residual_basis: str,
) -> Dict[str, np.ndarray]:
    """Tensors for run_direct_ub_from_data: Alpha = Y @ Psi (DCT/identity)."""
    y_tr = y_all[sl_tr]
    y_va = y_all[sl_va]
    y_te = y_all[sl_te]
    l = y_tr.shape[1]
    p_in = int(x_all.shape[1])
    if p_in % l != 0 or p_in < l:
        raise ValueError(
            "Expected p_input = C * L with C >= 1 and C integer; got p_in={} L={}.".format(
                p_in, l
            )
        )
    psi = get_basis(l, residual_basis)
    alpha_tr = y_tr @ psi
    alpha_va = y_va @ psi
    alpha_te = y_te @ psi
    return {
        "X_train": x_all[sl_tr].copy(),
        "X_val": x_all[sl_va].copy(),
        "X_test": x_all[sl_te].copy(),
        "Y_train": y_tr.copy(),
        "Y_val": y_va.copy(),
        "Y_test": y_te.copy(),
        "Alpha_train": alpha_tr,
        "Alpha_val": alpha_va,
        "Alpha_test": alpha_te,
        "Psi": psi,
    }
