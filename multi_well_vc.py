#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Multi-well dataset for cross-well Vc (clay volume) prediction.

Reads the 6-log files (Sonic, Density, Gamma_Ray, P_Impedance, Vp, Vc) from the
F3 Block wells (F02-1, F03-2, F03-4, F06-1), including clean and 30dB-noisy
twins, detects well boundaries in concatenated files by depth discontinuity,
builds sliding windows per well (no window crosses a well boundary), and
assembles (X, Y) tensors suitable for the direct [u, b] -> y benchmark pipeline.

Canonical channel names (lowercase):
    sonic, rhob, gr, ai, vp
Target:
    vc

Split convention for the cross-well study:
    - train/val wells: a list of wells read from one or more files
    - test well: a single well read from a separate file
    - default val is the tail (last val_frac) of each train well
    - optional val_split_mode=random_per_well redraws the same val budget
      with val_split_seed (R1-M03 supplement; test well unchanged)

ASCII-only.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from sir_cs_pipeline_optimized import get_basis


# ---------------------------------------------------------------------------
# Canonical schema
# ---------------------------------------------------------------------------

VALID_CHANNELS: Tuple[str, ...] = ("sonic", "rhob", "gr", "ai", "vp")
VALID_TARGETS: Tuple[str, ...] = ("vc", "porosity")

_COL_ALIASES: Dict[str, Tuple[str, ...]] = {
    "depth": ("depth", "depthm", "depth(m)"),
    "sonic": ("sonic", "ac"),
    "rhob": ("rhob", "density", "rhob(g/cc)", "rho_b"),
    "gr": ("gr", "gammaray", "gamma_ray", "gr(api)"),
    "ai": ("ai", "pimpedance", "p_impedance"),
    "vp": ("vp",),
    "vc": ("vc",),
    "porosity": ("porosity", "phi"),
}


def _norm(name: str) -> str:
    s = str(name).strip().lower()
    for ch in " ()/_":
        s = s.replace(ch, "")
    return s


def _resolve_column(df: pd.DataFrame, key: str) -> Optional[str]:
    aliases = _COL_ALIASES.get(key, (key,))
    norm_cols = {_norm(c): c for c in df.columns}
    for a in aliases:
        if a in norm_cols:
            return norm_cols[a]
    return None


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class WellSegment:
    """Contiguous sorted-by-depth slice of one well with multiple channels."""

    name: str
    depth: np.ndarray
    channels: Dict[str, np.ndarray]
    target: np.ndarray
    target_name: str

    @property
    def n_rows(self) -> int:
        return int(self.depth.shape[0])


@dataclass
class MultiWellWindows:
    """Sliding-window tensors from one or more wells."""

    x: np.ndarray  # (n_win, C*L)
    y: np.ndarray  # (n_win, L)
    well_of_window: np.ndarray  # (n_win,) str
    row_start_in_well: np.ndarray  # (n_win,) int
    depth_range: np.ndarray  # (n_win, 2) float
    channels: Tuple[str, ...]
    target_name: str
    window_len: int
    step: int
    well_n_rows: Dict[str, int] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Readers and boundary detection
# ---------------------------------------------------------------------------


def _parse_well_names_from_filename(path: str, fallback: str = "W") -> List[str]:
    """Infer well names from a filename like 'F02-1,F03-2,F06-1_6logs_30dB.txt'.

    Returns a list of well names in the order they were concatenated.
    """
    base = os.path.basename(path)
    stem = base.split("_", 1)[0]
    tokens = [t.strip() for t in stem.split(",") if t.strip()]
    if not tokens:
        return [fallback]
    return tokens


def _detect_well_boundaries(depth: np.ndarray, step_tol_factor: float = 3.0) -> List[int]:
    """Return the start index of each well segment (always includes 0).

    Boundary is declared where diff(depth) is negative or > step_tol_factor * median_step.
    """
    d = np.asarray(depth, dtype=np.float64).ravel()
    if d.size < 2:
        return [0]
    dd = np.diff(d)
    med = float(np.median(dd[dd > 0])) if np.any(dd > 0) else 0.15
    jumps = np.where((dd < 0) | (dd > step_tol_factor * med))[0]
    starts = [0] + [int(j) + 1 for j in jumps.tolist()]
    return starts


def load_6log_file(
    path: str,
    target_name: str = "vc",
    channels: Sequence[str] = VALID_CHANNELS,
    well_names: Optional[Sequence[str]] = None,
) -> List[WellSegment]:
    """Read a 6- or 7-log file and return the per-well segments.

    Parameters
    ----------
    path : path to a tab-separated file with a header row.
    target_name : 'vc' or 'porosity'.
    channels : which canonical channels to keep (must all be present in the file).
    well_names : if provided, overrides filename-based well name parsing. Must
                 have the same length as the number of detected segments.
    """
    p = os.path.abspath(path.strip())
    if not os.path.isfile(p):
        raise FileNotFoundError(p)
    tkey = str(target_name).strip().lower()
    if tkey not in VALID_TARGETS:
        raise ValueError("Unknown target '{}'. Valid: {}.".format(target_name, VALID_TARGETS))
    ch_keys = tuple(str(c).strip().lower() for c in channels)
    for c in ch_keys:
        if c not in VALID_CHANNELS:
            raise ValueError("Unknown channel '{}'. Valid: {}.".format(c, VALID_CHANNELS))

    df = pd.read_csv(p, sep="\t", encoding="latin1")

    depth_col = _resolve_column(df, "depth")
    if depth_col is None:
        raise KeyError("depth column not found in {}".format(list(df.columns)))
    tgt_col = _resolve_column(df, tkey)
    if tgt_col is None:
        raise KeyError(
            "target '{}' not found in {}. Columns: {}".format(tkey, p, list(df.columns))
        )

    ch_cols: Dict[str, str] = {}
    for c in ch_keys:
        col = _resolve_column(df, c)
        if col is None:
            raise KeyError(
                "channel '{}' not found in {}. Columns: {}".format(c, p, list(df.columns))
            )
        ch_cols[c] = col

    depth_all = np.asarray(df[depth_col].values, dtype=np.float64)
    tgt_all = np.asarray(df[tgt_col].values, dtype=np.float64)
    ch_all: Dict[str, np.ndarray] = {
        c: np.asarray(df[col].values, dtype=np.float64) for c, col in ch_cols.items()
    }

    finite = np.isfinite(depth_all) & np.isfinite(tgt_all)
    for c in ch_keys:
        finite &= np.isfinite(ch_all[c])
    if not bool(np.all(finite)):
        depth_all = depth_all[finite]
        tgt_all = tgt_all[finite]
        for c in ch_keys:
            ch_all[c] = ch_all[c][finite]

    starts = _detect_well_boundaries(depth_all)
    if well_names is None:
        names = _parse_well_names_from_filename(p)
    else:
        names = list(well_names)
    if len(names) < len(starts):
        names = names + ["W{}".format(i) for i in range(len(names), len(starts))]

    segs: List[WellSegment] = []
    for i, s0 in enumerate(starts):
        s1 = starts[i + 1] if i + 1 < len(starts) else depth_all.size
        if s1 - s0 < 2:
            continue
        # Sort within segment to guarantee monotonic depth.
        seg_d = depth_all[s0:s1]
        order = np.argsort(seg_d)
        seg_channels = {c: ch_all[c][s0:s1][order] for c in ch_keys}
        segs.append(
            WellSegment(
                name=str(names[i]),
                depth=seg_d[order],
                channels=seg_channels,
                target=tgt_all[s0:s1][order],
                target_name=tkey,
            )
        )
    if not segs:
        raise ValueError("No well segments found in {}.".format(p))
    return segs


# ---------------------------------------------------------------------------
# Sliding windows
# ---------------------------------------------------------------------------


def build_windows_for_segment(
    seg: WellSegment,
    window_len: int,
    step: int,
    channels: Sequence[str],
) -> MultiWellWindows:
    """Sliding windows of a single well segment. u = concat(channels in order)."""
    ch_keys = tuple(str(c).strip().lower() for c in channels)
    for c in ch_keys:
        if c not in seg.channels:
            raise ValueError(
                "Channel '{}' not present in segment '{}'. Have: {}.".format(
                    c, seg.name, list(seg.channels.keys())
                )
            )
    l = int(window_len)
    st = max(1, int(step))
    n = seg.n_rows
    if n < l:
        raise ValueError(
            "Well '{}' has n={} rows, fewer than L={}.".format(seg.name, n, l)
        )
    xs: List[np.ndarray] = []
    ys: List[np.ndarray] = []
    wells: List[str] = []
    starts: List[int] = []
    dranges: List[Tuple[float, float]] = []
    t = 0
    while t + l <= n:
        u = np.concatenate([seg.channels[c][t : t + l] for c in ch_keys], axis=0)
        xs.append(u.astype(np.float64, copy=False))
        ys.append(seg.target[t : t + l].astype(np.float64, copy=False))
        wells.append(seg.name)
        starts.append(int(t))
        dranges.append((float(seg.depth[t]), float(seg.depth[t + l - 1])))
        t += st
    if not xs:
        raise ValueError(
            "No windows for well '{}' with L={}, step={}.".format(seg.name, l, st)
        )
    return MultiWellWindows(
        x=np.stack(xs, axis=0),
        y=np.stack(ys, axis=0),
        well_of_window=np.array(wells, dtype=object),
        row_start_in_well=np.array(starts, dtype=np.int64),
        depth_range=np.array(dranges, dtype=np.float64),
        channels=ch_keys,
        target_name=seg.target_name,
        window_len=l,
        step=st,
        well_n_rows={seg.name: n},
    )


def concat_windows(batches: Sequence[MultiWellWindows]) -> MultiWellWindows:
    """Concatenate window batches from multiple segments (same L, step, channels)."""
    if not batches:
        raise ValueError("concat_windows: empty batches.")
    ref = batches[0]
    for b in batches[1:]:
        if b.window_len != ref.window_len or b.step != ref.step or b.channels != ref.channels:
            raise ValueError("Inconsistent batches in concat_windows.")
        if b.target_name != ref.target_name:
            raise ValueError("Inconsistent target_name in concat_windows.")
    x = np.concatenate([b.x for b in batches], axis=0)
    y = np.concatenate([b.y for b in batches], axis=0)
    wells = np.concatenate([b.well_of_window for b in batches], axis=0)
    starts = np.concatenate([b.row_start_in_well for b in batches], axis=0)
    dranges = np.concatenate([b.depth_range for b in batches], axis=0)
    nrows: Dict[str, int] = {}
    for b in batches:
        nrows.update(b.well_n_rows)
    return MultiWellWindows(
        x=x,
        y=y,
        well_of_window=wells,
        row_start_in_well=starts,
        depth_range=dranges,
        channels=ref.channels,
        target_name=ref.target_name,
        window_len=ref.window_len,
        step=ref.step,
        well_n_rows=nrows,
    )


# ---------------------------------------------------------------------------
# Train/val split: tail per well
# ---------------------------------------------------------------------------


def random_per_well_val_mask(
    wells: np.ndarray, val_frac: float, split_seed: int
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Partition window indices by drawing val_frac of each well at random.

    The held-out test well is not in `wells`. Only development windows are
    split. Embargo is not applied here: combine with tail mode instead.

    Returns (is_train, is_val) boolean arrays of shape wells.shape.
    """
    if not (0.0 < float(val_frac) < 1.0):
        raise ValueError("val_frac must be in (0,1).")
    names = np.asarray(wells, dtype=object).ravel()
    n = names.size
    is_train = np.ones(n, dtype=bool)
    is_val = np.zeros(n, dtype=bool)
    rng = np.random.default_rng(int(split_seed))
    unique_order: List[str] = []
    for name in names.tolist():
        if name not in unique_order:
            unique_order.append(str(name))
    for name in unique_order:
        idx = np.where(names == name)[0]
        if idx.size < 4:
            continue
        n_val = max(1, int(np.floor(float(val_frac) * idx.size)))
        if n_val >= idx.size:
            n_val = idx.size - 1
        chosen = rng.choice(idx, size=int(n_val), replace=False)
        is_val[chosen] = True
        is_train[chosen] = False
    if not bool(np.any(is_val)):
        raise ValueError("random_per_well val mask is empty; check window counts.")
    return is_train, is_val


def development_val_mask(
    wells: np.ndarray,
    val_frac: float,
    *,
    val_split_mode: str = "tail",
    val_split_seed: int = 0,
    val_embargo_windows: int = 0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Select the train/val mask for development wells."""
    mode = str(val_split_mode).strip().lower()
    if mode == "tail":
        return tail_per_well_val_mask(
            wells, val_frac=val_frac, val_embargo_windows=int(val_embargo_windows)
        )
    if mode == "random_per_well":
        if int(val_embargo_windows) != 0:
            raise ValueError(
                "val_embargo_windows must be 0 when val_split_mode=random_per_well"
            )
        return random_per_well_val_mask(
            wells, val_frac=val_frac, split_seed=int(val_split_seed)
        )
    raise ValueError(
        "val_split_mode must be 'tail' or 'random_per_well', got {}".format(mode)
    )


def tail_per_well_val_mask(
    wells: np.ndarray, val_frac: float, val_embargo_windows: int = 0
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Partition window indices by marking the last val_frac of each well as val.

    Windows are assumed to be ordered with non-decreasing row_start_in_well within
    each well (true for the output of build_windows_for_segment).

    Optional embargo:
      - If val_embargo_windows > 0, the last `val_embargo_windows` windows
        before the validation tail are dropped from train/val (neither split).

    Returns (is_train, is_val) boolean arrays of shape wells.shape.
    """
    if not (0.0 < float(val_frac) < 1.0):
        raise ValueError("val_frac must be in (0,1).")
    names = np.asarray(wells, dtype=object).ravel()
    emb = max(0, int(val_embargo_windows))
    n = names.size
    is_train = np.ones(n, dtype=bool)
    is_val = np.zeros(n, dtype=bool)
    unique_order: List[str] = []
    for name in names.tolist():
        if name not in unique_order:
            unique_order.append(name)
    for name in unique_order:
        idx = np.where(names == name)[0]
        if idx.size < 4:
            continue
        n_val = max(1, int(np.floor(float(val_frac) * idx.size)))
        val_idx = idx[-n_val:]
        is_val[val_idx] = True
        is_train[val_idx] = False
        if emb > 0:
            lo = max(0, int(idx.size - n_val - emb))
            hi = max(0, int(idx.size - n_val))
            if hi > lo:
                emb_idx = idx[lo:hi]
                is_val[emb_idx] = False
                is_train[emb_idx] = False
    return is_train, is_val


# ---------------------------------------------------------------------------
# Build cross-well data_dict for the direct [u,b] -> y pipeline
# ---------------------------------------------------------------------------


def build_cross_well_data_dict(
    train_path: str,
    test_path: str,
    *,
    target_name: str = "vc",
    channels: Sequence[str] = ("sonic", "rhob", "ai", "vp"),
    window_len: int = 64,
    step: int = 4,
    val_frac: float = 0.1,
    val_embargo_windows: int = 0,
    val_split_mode: str = "tail",
    val_split_seed: int = 0,
    residual_basis: str = "dct",
    scale_x: bool = True,
    train_well_names: Optional[Sequence[str]] = None,
    test_well_names: Optional[Sequence[str]] = None,
) -> Dict[str, object]:
    """Assemble the data dict consumed by run_direct_ub_from_data.

    The result has the same tensor keys used by the real-well launcher, plus a
    `meta` entry describing the cross-well configuration.

    Tensor shapes:
        X_* : (n, C * L)
        Y_* : (n, L)
        Alpha_* : (n, L)  equal to Y_* @ Psi (orthonormal L x L basis)
        Psi : (L, L)

    If scale_x is True, X_train/X_val/X_test are standardized per-feature using
    StandardScaler fit on X_train. Required for stable LFISTA training when u
    channels span many orders of magnitude (AI ~1e6 vs Density ~1, etc).
    sklearn-based baselines (MultiOutputMLP, mlp_concat_ub, pca_regression_ub,
    ae_regression_ub) apply their own StandardScaler, so an extra rescaling is
    idempotent in their pipelines.
    """
    train_segs = load_6log_file(
        train_path, target_name=target_name, channels=channels, well_names=train_well_names
    )
    test_segs = load_6log_file(
        test_path, target_name=target_name, channels=channels, well_names=test_well_names
    )

    train_batches = [
        build_windows_for_segment(s, window_len=window_len, step=step, channels=channels)
        for s in train_segs
    ]
    tr_all = concat_windows(train_batches)
    te_batches = [
        build_windows_for_segment(s, window_len=window_len, step=step, channels=channels)
        for s in test_segs
    ]
    te_all = concat_windows(te_batches)

    is_train, is_val = development_val_mask(
        tr_all.well_of_window,
        val_frac=val_frac,
        val_split_mode=str(val_split_mode),
        val_split_seed=int(val_split_seed),
        val_embargo_windows=int(val_embargo_windows),
    )
    x_tr_raw = tr_all.x[is_train]
    y_tr = tr_all.y[is_train]
    x_va_raw = tr_all.x[is_val]
    y_va = tr_all.y[is_val]
    x_te_raw = te_all.x
    y_te = te_all.y

    if bool(scale_x):
        scaler = StandardScaler()
        x_tr = scaler.fit_transform(x_tr_raw)
        x_va = scaler.transform(x_va_raw)
        x_te = scaler.transform(x_te_raw)
        x_scaler_mean = scaler.mean_.tolist()
        x_scaler_scale = scaler.scale_.tolist()
    else:
        x_tr = x_tr_raw
        x_va = x_va_raw
        x_te = x_te_raw
        x_scaler_mean = []
        x_scaler_scale = []

    l = int(window_len)
    psi = get_basis(l, residual_basis)
    alpha_tr = y_tr @ psi
    alpha_va = y_va @ psi
    alpha_te = y_te @ psi

    meta = {
        "train_path": os.path.abspath(train_path),
        "test_path": os.path.abspath(test_path),
        "train_wells": [s.name for s in train_segs],
        "test_wells": [s.name for s in test_segs],
        "channels": list(channels),
        "target": str(target_name).strip().lower(),
        "window_len": int(window_len),
        "step": int(step),
        "val_frac": float(val_frac),
        "val_embargo_windows": int(val_embargo_windows),
        "val_split_mode": str(val_split_mode).strip().lower(),
        "val_split_seed": int(val_split_seed),
        "residual_basis": str(residual_basis),
        "scale_x": bool(scale_x),
        "n_train": int(x_tr.shape[0]),
        "n_val": int(x_va.shape[0]),
        "n_test": int(x_te.shape[0]),
        "p_input": int(x_tr.shape[1]),
        "n_output": int(y_tr.shape[1]),
        "train_wells_rows": tr_all.well_n_rows,
        "test_wells_rows": te_all.well_n_rows,
        "train_well_of_window_full": np.asarray(tr_all.well_of_window).tolist(),
        "test_well_of_window_full": np.asarray(te_all.well_of_window).tolist(),
        "train_row_start_full": tr_all.row_start_in_well.tolist(),
        "test_row_start_full": te_all.row_start_in_well.tolist(),
        "train_is_val_mask": is_val.tolist(),
        "x_scaler_mean": x_scaler_mean,
        "x_scaler_scale": x_scaler_scale,
    }

    return {
        "X_train": x_tr.copy(),
        "X_val": x_va.copy(),
        "X_test": x_te.copy(),
        "Y_train": y_tr.copy(),
        "Y_val": y_va.copy(),
        "Y_test": y_te.copy(),
        "Alpha_train": alpha_tr,
        "Alpha_val": alpha_va,
        "Alpha_test": alpha_te,
        "Psi": psi,
        "meta": meta,
    }


# ---------------------------------------------------------------------------
# CLI sanity test
# ---------------------------------------------------------------------------


def _main() -> None:
    import argparse

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--train-path",
        default="data/F02-1,F03-2,F06-1_6logs_30dB.txt",
    )
    ap.add_argument("--test-path", default="data/F03-4_6logs_30dB.txt")
    ap.add_argument("--target", default="vc")
    ap.add_argument(
        "--channels", default="sonic,rhob,ai,vp",
        help="comma-separated list of canonical channels",
    )
    ap.add_argument("--window-len", type=int, default=64)
    ap.add_argument("--step", type=int, default=4)
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--residual-basis", default="dct")
    args = ap.parse_args()

    channels = tuple(c.strip().lower() for c in str(args.channels).split(",") if c.strip())

    data = build_cross_well_data_dict(
        train_path=args.train_path,
        test_path=args.test_path,
        target_name=args.target,
        channels=channels,
        window_len=args.window_len,
        step=args.step,
        val_frac=args.val_frac,
        residual_basis=args.residual_basis,
    )
    meta = data["meta"]
    print("=== cross-well data dict ===")
    for k in [
        "train_wells",
        "test_wells",
        "channels",
        "target",
        "window_len",
        "step",
        "val_frac",
        "residual_basis",
        "n_train",
        "n_val",
        "n_test",
        "p_input",
        "n_output",
        "train_wells_rows",
        "test_wells_rows",
    ]:
        print("  {:20s}: {}".format(k, meta[k]))
    print("X_train shape:", data["X_train"].shape, "Y_train shape:", data["Y_train"].shape)
    print("X_test  shape:", data["X_test"].shape, " Y_test shape:", data["Y_test"].shape)
    print("Psi shape:", data["Psi"].shape)

    print()
    print("Sanity: y_train stats")
    y = data["Y_train"]
    print("  mean={:.5f} std={:.5f} min={:.5f} max={:.5f}".format(
        float(y.mean()), float(y.std()), float(y.min()), float(y.max())
    ))
    print("Sanity: y_test stats")
    y = data["Y_test"]
    print("  mean={:.5f} std={:.5f} min={:.5f} max={:.5f}".format(
        float(y.mean()), float(y.std()), float(y.min()), float(y.max())
    ))


if __name__ == "__main__":
    _main()
