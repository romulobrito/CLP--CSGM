#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Redraw the manuscript RMSE and depth figures from saved summaries.

The RMSE panels read the no-GR copies under
outputs/reviewer_revision/r3m06_no_gr_summaries. They do not read the
five-channel CSVs in outputs/cross_well_vc/direct_ub.
The depth profile still requires the local NPZ named in PROFILE_NPZ.
That file is a processed well curve and is not part of the summary bundle.
Does not refit models. ASCII only.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path
from typing import List, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PAPER_FIG = ROOT / "paper_clp_csgm" / "figures"
BACKUP = PAPER_FIG / "r3m06_originals"
PROFILE_NPZ = (
    ROOT
    / "outputs"
    / "paper_regen"
    / "crosswell_depth_profile"
    / "runs"
    / "step16_seed7_rho0p2"
    / "tables"
    / "depth_profile_test_block.npz"
)
SUMMARY_DIR = ROOT / "outputs" / "reviewer_revision" / "r3m06_no_gr_summaries"
PROVENANCE_PATH = SUMMARY_DIR / "provenance.json"
STEP_SUMMARIES: Tuple[Tuple[int, str], ...] = (
    (8, "step08_summary_focus_clp_csgm_vs_ub.csv"),
    (16, "step16_summary_focus_clp_csgm_vs_ub.csv"),
    (32, "step32_summary_focus_clp_csgm_vs_ub.csv"),
)
NO_GR_CHANNELS = ("sonic", "rhob", "ai", "vp")
# Published cell means used only as a consistency check, not as a data source.
PUBLISHED_CLP = {
    (8, 0.05): 0.04483,
    (8, 0.10): 0.03646,
    (8, 0.20): 0.02267,
    (16, 0.05): 0.03344,
    (16, 0.10): 0.02689,
    (16, 0.20): 0.01771,
    (32, 0.05): 0.03217,
    (32, 0.10): 0.02640,
    (32, 0.20): 0.01705,
}
METHODS: Tuple[Tuple[str, str, str], ...] = (
    ("ml_only", "ML only", "#1f77b4"),
    ("mlp_concat_ub", "MLP [u,b]", "#2ca02c"),
    ("pca_regression_ub", "PCA [u,b]", "#9467bd"),
    ("ae_regression_ub", "AE [u,b]", "#17becf"),
    ("ridge_prior_csgm", "CLP-CSGM Ridge", "#d62728"),
)
ZOOM_METHODS = ("ml_only", "ae_regression_ub", "ridge_prior_csgm")
MAIN_PROFILES = (
    ("profile_observed", "reference"),
    ("profile_ml_only", "ML only"),
    ("profile_ae_regression_ub", "AE [u,b]"),
    ("profile_ridge_prior_csgm", "CLP-CSGM Ridge"),
)
WIDE_PROFILES = (
    ("profile_mlp_concat_ub", "MLP [u,b]"),
    ("profile_pca_regression_ub", "PCA [u,b]"),
)
T_CRIT_DF2 = 4.302652729911275


def _backup(name: str) -> None:
    """Keep the previous paper PNG under an identified folder."""
    source = PAPER_FIG / name
    if not source.is_file():
        raise FileNotFoundError(str(source))
    BACKUP.mkdir(parents=True, exist_ok=True)
    destination = BACKUP / name
    if not destination.is_file():
        shutil.copy2(source, destination)


def _sha256(path: Path) -> str:
    """Hash a file without loading it as text."""
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def _checked_summary_paths() -> List[Tuple[int, Path]]:
    """Return the three no-GR summaries after checking protocol and hashes."""
    if not PROVENANCE_PATH.is_file():
        raise FileNotFoundError(str(PROVENANCE_PATH))
    payload = json.loads(PROVENANCE_PATH.read_text())
    records = payload.get("files")
    if not isinstance(records, list) or len(records) != 3:
        raise RuntimeError("Provenance does not list three summary files.")
    by_step = {}
    for record in records:
        step = int(record["step"])
        channels = tuple(record["worktree_channels"])
        if channels != NO_GR_CHANNELS:
            raise RuntimeError("Step {0} is not the four-channel protocol.".format(step))
        if "gr" in channels:
            raise RuntimeError("Step {0} includes gamma ray.".format(step))
        if int(record["worktree_p_input"]) != 256:
            raise RuntimeError("Step {0} p_input is not 4 x 64.".format(step))
        by_step[step] = record
    checked: List[Tuple[int, Path]] = []
    for step, name in STEP_SUMMARIES:
        record = by_step[step]
        if str(record["file"]) != name:
            raise RuntimeError("Provenance filename does not match step {0}.".format(step))
        path = SUMMARY_DIR / name
        if not path.is_file():
            raise FileNotFoundError(str(path))
        digest = _sha256(path)
        if digest != str(record["sha256"]):
            raise RuntimeError("Hash mismatch for {0}.".format(name))
        if digest == str(record["git_head_same_path_sha256"]):
            raise RuntimeError("Step {0} matches the five-channel git blob.".format(step))
        checked.append((step, path))
    return checked


def _load_rmse() -> pd.DataFrame:
    """Stack the three no-GR step summaries. Seeds and splits stay as stored."""
    frames: List[pd.DataFrame] = []
    for step, path in _checked_summary_paths():
        frame = pd.read_csv(path)
        required = {
            "measurement_ratio",
            "method",
            "rmse_mean",
            "rmse_sem",
            "n_seeds",
        }
        missing = sorted(required.difference(frame.columns))
        if missing:
            raise ValueError("Missing columns in {0}: {1}".format(path, ",".join(missing)))
        part = frame.copy()
        part["step"] = int(step)
        frames.append(part)
    data = pd.concat(frames, ignore_index=True)
    for (step, rho), expected in PUBLISHED_CLP.items():
        row = data[
            (data["step"] == int(step))
            & np.isclose(data["measurement_ratio"], float(rho))
            & (data["method"] == "ridge_prior_csgm")
        ]
        if int(row.shape[0]) != 1:
            raise RuntimeError("Missing CLP row for step {0}, rho {1}.".format(step, rho))
        got = float(row["rmse_mean"].iloc[0])
        if abs(got - float(expected)) > 5.0e-5:
            raise RuntimeError(
                "CLP mean {0} at step {1}, rho {2} differs from {3}.".format(
                    got, step, rho, expected
                )
            )
    return data


def _half_width(row: pd.Series) -> float:
    """Student-t 95 percent half-width from the stored standard error."""
    n_seeds = int(row["n_seeds"])
    if n_seeds != 3:
        raise RuntimeError("Expected three seeds, found {0}.".format(n_seeds))
    return float(row["rmse_sem"]) * T_CRIT_DF2


def _plot_rmse(data: pd.DataFrame, out_dir: Path) -> None:
    """Full five-method view above a zoom of the three lower-RMSE methods."""
    steps = (8, 16, 32)
    rhos = (0.05, 0.10, 0.20)
    figure, axes = plt.subplots(2, 3, figsize=(7.4, 6.6), sharex=True)
    axes[0, 1].sharey(axes[0, 0])
    axes[0, 2].sharey(axes[0, 0])
    axes[1, 1].sharey(axes[1, 0])
    axes[1, 2].sharey(axes[1, 0])
    zoom_top = 0.0
    for method in ZOOM_METHODS:
        subset = data[data["method"] == method]
        for _index, row in subset.iterrows():
            zoom_top = max(zoom_top, float(row["rmse_mean"]) + _half_width(row))
    zoom_top = zoom_top * 1.12
    for column, step in enumerate(steps):
        for row_index, method_names in enumerate(
            (("ml_only", "mlp_concat_ub", "pca_regression_ub", "ae_regression_ub", "ridge_prior_csgm"), ZOOM_METHODS)
        ):
            axis = axes[row_index, column]
            for method, label, color in METHODS:
                if method not in method_names:
                    continue
                subset = data[
                    (data["step"] == int(step)) & (data["method"] == method)
                ].sort_values("measurement_ratio")
                if int(subset.shape[0]) != 3:
                    raise RuntimeError(
                        "Step {0}, method {1} does not have three ratios.".format(step, method)
                    )
                ratios = subset["measurement_ratio"].to_numpy(dtype=np.float64)
                means = subset["rmse_mean"].to_numpy(dtype=np.float64)
                errors = np.asarray([_half_width(item) for _i, item in subset.iterrows()], dtype=np.float64)
                if not np.allclose(ratios, np.asarray(rhos, dtype=np.float64)):
                    raise RuntimeError("Unexpected ratios at step {0}.".format(step))
                axis.errorbar(
                    ratios,
                    means,
                    yerr=errors,
                    marker="o",
                    markersize=4.5,
                    capsize=3,
                    linewidth=1.6 if method == "ridge_prior_csgm" else 1.2,
                    color=color,
                    label=label,
                )
            axis.set_title("step {0}".format(int(step)), fontsize=10)
            axis.set_xticks(list(rhos))
            axis.set_xticklabels(["0.05", "0.10", "0.20"], fontsize=8)
            axis.tick_params(axis="y", labelsize=8)
            axis.grid(True, alpha=0.3)
            if row_index == 1:
                axis.set_ylim(0.0, zoom_top)
                axis.set_xlabel("measurement ratio", fontsize=9)
            if column == 0:
                axis.set_ylabel("full-window RMSE", fontsize=9)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        loc="upper center",
        ncol=5,
        frameon=False,
        fontsize=8,
        bbox_to_anchor=(0.5, 1.02),
    )
    axes[0, 0].annotate(
        "all five methods",
        xy=(0.0, 1.18),
        xycoords="axes fraction",
        fontsize=8,
    )
    axes[1, 0].annotate(
        "lower-RMSE methods only",
        xy=(0.0, 1.18),
        xycoords="axes fraction",
        fontsize=8,
    )
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.94))
    figure.savefig(out_dir / "results_crosswell_rmse_vs_rho.png", dpi=200)
    plt.close(figure)


def _finite_limits(arrays: Sequence[np.ndarray]) -> Tuple[float, float]:
    """Shared limits covering every finite value, with a small margin."""
    low = min(float(np.nanmin(item)) for item in arrays)
    high = max(float(np.nanmax(item)) for item in arrays)
    span = high - low
    if span <= 0.0:
        raise RuntimeError("Profile range has no width.")
    pad = 0.04 * span
    return low - pad, high + pad


def _plot_profiles(out_dir: Path) -> None:
    """Shared-scale main profile and a full-range MLP/PCA complement."""
    if not PROFILE_NPZ.is_file():
        raise FileNotFoundError(
            "Local depth profile is missing: {0}. It is not in the no-GR summary bundle. "
            "Regeneration refits the step-16 seed-7 rho-0.20 run and was not executed here.".format(
                PROFILE_NPZ
            )
        )
    archive = np.load(PROFILE_NPZ)
    depth = np.asarray(archive["depth_axis"], dtype=np.float64)
    if depth.ndim != 1 or int(depth.size) < 2:
        raise RuntimeError("Depth axis is empty.")
    if not bool(np.all(np.diff(depth) >= -1.0e-9)):
        raise RuntimeError("Depth axis is not nondecreasing.")
    main_arrays = [np.asarray(archive[key], dtype=np.float64) for key, _label in MAIN_PROFILES]
    for array in main_arrays:
        if array.shape != depth.shape:
            raise RuntimeError("A main profile does not match the depth axis.")
    low, high = _finite_limits(main_arrays)
    figure, axes = plt.subplots(1, 4, figsize=(7.6, 7.2), sharex=True, sharey=True)
    colors = {
        "reference": "#404040",
        "ML only": "#1f77b4",
        "AE [u,b]": "#17becf",
        "CLP-CSGM Ridge": "#d62728",
    }
    reference = main_arrays[0]
    for axis, (key, label), values in zip(axes, MAIN_PROFILES, main_arrays):
        axis.plot(reference, depth, color="#404040", linewidth=1.0, label="reference")
        if key != "profile_observed":
            axis.plot(values, depth, color=colors[label], linewidth=1.15, label=label)
        axis.set_xlim(low, high)
        axis.set_title(label, fontsize=9)
        axis.set_xlabel("Vc", fontsize=9)
        axis.grid(True, alpha=0.3)
        axis.tick_params(labelsize=7)
    axes[0].set_ylabel("depth (m), increasing downward", fontsize=9)
    axes[0].invert_yaxis()
    figure.suptitle(
        "F03-4, step 16, seed 7, rho 0.20. Shared Vc scale.",
        fontsize=10,
    )
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.96))
    figure.savefig(out_dir / "10_depth_profile_crosswell_vc.png", dpi=200)
    plt.close(figure)

    wide_arrays = [np.asarray(archive[key], dtype=np.float64) for key, _label in WIDE_PROFILES]
    for array in wide_arrays:
        if array.shape != depth.shape:
            raise RuntimeError("A wide profile does not match the depth axis.")
    wide_low, wide_high = _finite_limits([reference] + wide_arrays)
    wide_figure, wide_axes = plt.subplots(1, 2, figsize=(6.4, 7.0), sharex=True, sharey=True)
    wide_colors = {"MLP [u,b]": "#2ca02c", "PCA [u,b]": "#9467bd"}
    for axis, (key, label), values in zip(wide_axes, WIDE_PROFILES, wide_arrays):
        axis.plot(reference, depth, color="#404040", linewidth=1.0, label="reference")
        axis.plot(values, depth, color=wide_colors[label], linewidth=1.05, label=label)
        axis.set_xlim(wide_low, wide_high)
        axis.set_title(label, fontsize=9)
        axis.set_xlabel("Vc", fontsize=9)
        axis.grid(True, alpha=0.3)
        axis.tick_params(labelsize=7)
    wide_handles: List[plt.Artist] = []
    wide_labels: List[str] = []
    seen: set[str] = set()
    for axis in wide_axes:
        handles, labels = axis.get_legend_handles_labels()
        for handle, label in zip(handles, labels):
            if label in seen:
                continue
            seen.add(label)
            wide_handles.append(handle)
            wide_labels.append(label)
    wide_figure.legend(
        wide_handles,
        wide_labels,
        loc="upper center",
        ncol=3,
        frameon=False,
        fontsize=8,
        bbox_to_anchor=(0.5, 0.93),
    )
    wide_axes[0].set_ylabel("depth (m), increasing downward", fontsize=9)
    wide_axes[0].invert_yaxis()
    wide_figure.suptitle(
        "Same interval. Full Vc range of MLP and PCA.",
        fontsize=10,
        y=0.98,
    )
    wide_figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.90))
    wide_figure.savefig(out_dir / "10_depth_profile_crosswell_vc_wide.png", dpi=200)
    plt.close(wide_figure)


def main() -> None:
    """Write the figures. A test directory does not replace the manuscript PNGs."""
    parser = argparse.ArgumentParser(description="Redraw the manuscript cross-well figures.")
    parser.add_argument(
        "--out-dir",
        default=str(PAPER_FIG),
        help="Directory for the PNG files. The default is the manuscript figures folder.",
    )
    args = parser.parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if out_dir.resolve() == PAPER_FIG.resolve():
        _backup("results_crosswell_rmse_vs_rho.png")
        _backup("10_depth_profile_crosswell_vc.png")
    _plot_rmse(_load_rmse(), out_dir)
    _plot_profiles(out_dir)
    print("wrote revised figures under", out_dir)


if __name__ == "__main__":
    main()
