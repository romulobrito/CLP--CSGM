#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate tables and figures for the CLP-CSGM paper.

The script is intentionally read-only with respect to experiment outputs: it
loads existing raw data and benchmark CSV files, then writes paper-ready assets
under paper_clp_csgm/.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence, Tuple, Union

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import multi_well_vc as mwv
import real_well_f03 as rwf

_SCRIPTS_DIR = os.path.join(REPO_ROOT, "scripts")
if _SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, _SCRIPTS_DIR)
from paper_figure_style import apply_paper_style, save_paper_figure


PAPER_DIR = os.path.join(REPO_ROOT, "paper_clp_csgm")
FIG_DIR = os.path.join(PAPER_DIR, "figures")
TAB_DIR = os.path.join(PAPER_DIR, "tables")

CROSS_TRAIN = os.path.join(REPO_ROOT, "data", "F02-1,F03-2,F06-1_6logs_30dB.txt")
CROSS_TEST = os.path.join(REPO_ROOT, "data", "F03-4_6logs_30dB.txt")
F03_REAL = os.path.join(REPO_ROOT, "data", "F03-4_AC+GR+Porosity.txt")

CROSS_SUMMARY = os.path.join(
    REPO_ROOT, "outputs", "cross_well_vc", "csgm", "m2_grid", "tables", "summary_focus_clp_csgm_vs_ub.csv"
)
CROSS_WINNERS = os.path.join(
    REPO_ROOT, "outputs", "cross_well_vc", "csgm", "m2_grid", "tables", "csgm_vs_ae_winners.csv"
)
CROSS_RANK = os.path.join(
    REPO_ROOT, "outputs", "cross_well_vc", "csgm", "m2_grid", "tables", "overall_rank.csv"
)
# Per-cell seed statistics for Table tab:crosswell_clp_vs_ae (Ridge ablation runs).
CROSSWELL_ABLATION_SUMMARIES: Tuple[Tuple[int, str], ...] = (
    (8, "crosswell_step08_clp_csgm_ablation_ridge"),
    (16, "crosswell_step16_clp_csgm_ablation_ridge"),
    (32, "crosswell_step32_clp_csgm_ablation_ridge"),
)
F03_SUMMARY = os.path.join(
    REPO_ROOT,
    "outputs",
    "real_well_f03",
    "direct_ub",
    "runs",
    "f03_full_gr_only_clp_csgm_ridge",
    "tables",
    "summary_focus_clp_csgm_vs_ub.csv",
)

# Display order for F03 main-paper RMSE table and figure.
F03_METHOD_ORDER: Tuple[str, ...] = (
    "CLP-CSGM Ridge",
    "AE [u,b]",
    "Prior-only Ridge",
    "Measurement-only CSGM",
    "ML only",
    "PCA [u,b]",
    "MLP [u,b]",
)

# Plot style: solid = direct baselines; dashed = structural CSGM-style references.
F03_PLOT_STYLE: Dict[str, Dict[str, object]] = {
    "CLP-CSGM Ridge": {"color": "#d62728", "linewidth": 2.2, "linestyle": "-", "alpha": 1.0},
    "AE [u,b]": {"color": "#17becf", "linewidth": 1.6, "linestyle": "-", "alpha": 1.0},
    "Prior-only Ridge": {"color": "#ff9896", "linewidth": 1.4, "linestyle": "--", "alpha": 0.95},
    "Measurement-only CSGM": {"color": "#7f7f7f", "linewidth": 1.4, "linestyle": "--", "alpha": 0.95},
    "ML only": {"color": "#1f77b4", "linewidth": 1.2, "linestyle": "-", "alpha": 0.85},
    "PCA [u,b]": {"color": "#9467bd", "linewidth": 1.2, "linestyle": "-", "alpha": 0.85},
    "MLP [u,b]": {"color": "#2ca02c", "linewidth": 1.2, "linestyle": "-", "alpha": 0.85},
}


@dataclass(frozen=True)
class TableSpec:
    """Container for a DataFrame and its output stem."""

    frame: pd.DataFrame
    stem: str
    float_format: str = "%.4f"


def ensure_dirs() -> None:
    """Create output folders for paper assets."""
    os.makedirs(FIG_DIR, exist_ok=True)
    os.makedirs(TAB_DIR, exist_ok=True)


def _latex_escape(value: object) -> str:
    """Escape a small subset of LaTeX-special characters for table cells."""
    text = str(value)
    repl = {
        "\\": r"\textbackslash{}",
        "_": r"\_",
        "%": r"\%",
        "&": r"\&",
        "#": r"\#",
    }
    for old, new in repl.items():
        text = text.replace(old, new)
    return text


def _channel_paper_label(channel: str) -> str:
    """Map internal channel keys to paper-facing log mnemonics."""
    key = str(channel).strip().lower()
    labels = {
        "sonic": "Sonic",
        "rhob": "RHOB",
        "gr": "GR",
        "ai": "AI",
        "vp": r"$V_p$",
        "ac": "AC",
    }
    if key not in labels:
        raise KeyError("Unknown channel label: " + str(channel))
    return labels[key]


def _target_paper_label(target: str) -> str:
    """Map internal target keys to paper-facing property names."""
    key = str(target).strip().lower()
    labels = {
        "vc": r"$V_c$",
        "porosity": "Porosity",
    }
    if key not in labels:
        raise KeyError("Unknown target label: " + str(target))
    return labels[key]


def _latex_table_frame(spec: TableSpec) -> pd.DataFrame:
    """Apply display labels for LaTeX tables that use log mnemonics."""
    frame = spec.frame.copy()
    if spec.stem == "eda_channel_target_correlations":
        frame["channel"] = frame["channel"].map(_channel_paper_label)
        frame["target"] = frame["target"].map(_target_paper_label)
    elif spec.stem == "eda_dataset_summary":
        frame["target"] = frame["target"].map(_target_paper_label)
    return frame


def write_table(spec: TableSpec) -> str:
    """Write CSV and LaTeX versions of a paper table."""
    csv_path = os.path.join(TAB_DIR, "{}.csv".format(spec.stem))
    tex_path = os.path.join(TAB_DIR, "{}.tex".format(spec.stem))
    spec.frame.to_csv(csv_path, index=False)
    latex_frame = _latex_table_frame(spec)
    escape_latex = spec.stem not in (
        "eda_channel_target_correlations",
        "eda_dataset_summary",
        "results_f03_gr_only_compact",
    )
    with open(tex_path, "w", encoding="ascii") as f:
        f.write(
            latex_frame.to_latex(
                index=False,
                escape=escape_latex,
                float_format=lambda x: spec.float_format % x,
            )
        )
    return tex_path


def _series_stats(values: np.ndarray) -> Dict[str, float]:
    """Return robust univariate statistics for a numeric vector."""
    v = np.asarray(values, dtype=np.float64)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return {
            "n": 0.0,
            "mean": np.nan,
            "std": np.nan,
            "min": np.nan,
            "q25": np.nan,
            "median": np.nan,
            "q75": np.nan,
            "max": np.nan,
        }
    return {
        "n": float(v.size),
        "mean": float(np.mean(v)),
        "std": float(np.std(v, ddof=1)) if v.size > 1 else 0.0,
        "min": float(np.min(v)),
        "q25": float(np.percentile(v, 25)),
        "median": float(np.median(v)),
        "q75": float(np.percentile(v, 75)),
        "max": float(np.max(v)),
    }


def _load_cross_segments() -> Tuple[List[mwv.WellSegment], List[mwv.WellSegment]]:
    """Load train and held-out cross-well Vc segments."""
    channels = ("sonic", "rhob", "gr", "ai", "vp")
    train = mwv.load_6log_file(CROSS_TRAIN, target_name="vc", channels=channels)
    test = mwv.load_6log_file(CROSS_TEST, target_name="vc", channels=channels)
    return train, test


def build_dataset_summary() -> pd.DataFrame:
    """Summarize rows, depth spans, targets, and channels used by the article."""
    train, test = _load_cross_segments()
    rows: List[Dict[str, object]] = []
    for split, segs in (("cross_train", train), ("cross_test", test)):
        for seg in segs:
            stats = _series_stats(seg.target)
            rows.append(
                {
                    "dataset": split,
                    "well": seg.name,
                    "target": seg.target_name,
                    "n_rows": int(seg.n_rows),
                    "depth_min": float(np.min(seg.depth)),
                    "depth_max": float(np.max(seg.depth)),
                    "target_mean": stats["mean"],
                    "target_std": stats["std"],
                    "target_min": stats["min"],
                    "target_max": stats["max"],
                }
            )

    f03 = rwf.load_f03_table(F03_REAL)
    stats = _series_stats(f03.porosity)
    rows.append(
        {
            "dataset": "real_f03_gr",
            "well": "F03-4",
            "target": "porosity",
            "n_rows": int(f03.n_rows),
            "depth_min": float(np.min(f03.depth)),
            "depth_max": float(np.max(f03.depth)),
            "target_mean": stats["mean"],
            "target_std": stats["std"],
            "target_min": stats["min"],
            "target_max": stats["max"],
        }
    )
    return pd.DataFrame(rows)


def build_correlation_table() -> pd.DataFrame:
    """Compute channel-target correlations for the cross-well and F03 data."""
    train, test = _load_cross_segments()
    rows: List[Dict[str, object]] = []
    for split, segs in (("cross_train", train), ("cross_test", test)):
        for channel in ("sonic", "rhob", "gr", "ai", "vp"):
            xs: List[np.ndarray] = []
            ys: List[np.ndarray] = []
            for seg in segs:
                xs.append(np.asarray(seg.channels[channel], dtype=np.float64))
                ys.append(np.asarray(seg.target, dtype=np.float64))
            x = np.concatenate(xs)
            y = np.concatenate(ys)
            corr = float(np.corrcoef(x, y)[0, 1])
            rows.append({"dataset": split, "channel": channel, "target": "vc", "pearson_r": corr})

    f03 = rwf.load_f03_table(F03_REAL)
    for channel, values in (("ac", f03.ac), ("gr", f03.gr)):
        corr = float(np.corrcoef(np.asarray(values, dtype=np.float64), f03.porosity)[0, 1])
        rows.append({"dataset": "real_f03", "channel": channel, "target": "porosity", "pearson_r": corr})
    return pd.DataFrame(rows)


def _crosswell_summary_row(summary: pd.DataFrame, rho: float, method: str) -> pd.Series:
    """Return the unique summary row for (measurement_ratio, method)."""
    rho_col = summary["measurement_ratio"].astype(float)
    mask = np.isclose(rho_col, float(rho), rtol=0.0, atol=1e-9) & (summary["method"] == method)
    hits = summary.loc[mask]
    if len(hits) != 1:
        raise ValueError(
            "Expected one row for rho={} method={}, found {} in summary".format(rho, method, len(hits))
        )
    return hits.iloc[0]


def build_crosswell_winners_table() -> pd.DataFrame:
    """
    Cross-well CLP-CSGM Ridge vs AE [u,b] with RMSE mean and std across seeds.

    Cell geometry (step, n_train_approx, rho) follows csgm_vs_ae_winners.csv;
    RMSE moments are taken from direct_ub ablation Ridge summary_focus CSVs.
    """
    winners = pd.read_csv(CROSS_WINNERS)
    rows: List[Dict[str, Union[float, int]]] = []
    for step, run_name in CROSSWELL_ABLATION_SUMMARIES:
        summary_path = os.path.join(
            REPO_ROOT,
            "outputs",
            "cross_well_vc",
            "direct_ub",
            "runs",
            run_name,
            "tables",
            "summary_focus_clp_csgm_vs_ub.csv",
        )
        if not os.path.isfile(summary_path):
            raise FileNotFoundError("Missing cross-well summary: {}".format(summary_path))
        summary = pd.read_csv(summary_path)
        sub = winners.loc[winners["step"] == step, :]
        for _, w in sub.iterrows():
            rho = float(w["measurement_ratio"])
            n_train = int(w["n_train_approx"])
            csgm = _crosswell_summary_row(summary, rho, "ridge_prior_csgm")
            ae = _crosswell_summary_row(summary, rho, "ae_regression_ub")
            c_mean = float(csgm["rmse_mean"])
            c_std = float(csgm["rmse_std_across_seeds"])
            a_mean = float(ae["rmse_mean"])
            a_std = float(ae["rmse_std_across_seeds"])
            if a_mean == 0.0:
                raise ValueError("AE RMSE mean is zero for step={} rho={}".format(step, rho))
            gap_pct = 100.0 * (c_mean - a_mean) / a_mean
            rows.append(
                {
                    "step": step,
                    "n_train_approx": n_train,
                    "rho": rho,
                    "clp_csgm_rmse_mean": c_mean,
                    "clp_csgm_rmse_std": c_std,
                    "ae_ub_rmse_mean": a_mean,
                    "ae_ub_rmse_std": a_std,
                    "gap_vs_ae_pct": gap_pct,
                }
            )
    frame = pd.DataFrame(rows)
    if len(frame) != len(winners):
        raise ValueError("Cross-well table row count mismatch: got {}, expected {}".format(len(frame), len(winners)))
    return frame


def build_crosswell_rank_table() -> pd.DataFrame:
    """Format overall cross-well ranking from the same ablation summaries used in Table 3."""
    keep_methods = {
        "ridge_prior_csgm",
        "ae_regression_ub",
        "ml_only",
        "pca_regression_ub",
        "mlp_concat_ub",
    }
    rows: List[pd.DataFrame] = []
    for _step, run_name in CROSSWELL_ABLATION_SUMMARIES:
        summary_path = os.path.join(
            REPO_ROOT,
            "outputs",
            "cross_well_vc",
            "direct_ub",
            "runs",
            run_name,
            "tables",
            "summary_focus_clp_csgm_vs_ub.csv",
        )
        if not os.path.isfile(summary_path):
            raise FileNotFoundError("Missing cross-well summary: {}".format(summary_path))
        df = pd.read_csv(summary_path)
        rows.append(
            df.loc[
                df["method"].isin(keep_methods),
                ["method", "method_label", "rmse_mean", "relative_l2_mean"],
            ].copy()
        )
    all_cells = pd.concat(rows, ignore_index=True)
    out = (
        all_cells.groupby(["method_label"], as_index=False)
        .agg(
            mean_rmse=("rmse_mean", "mean"),
            mean_relative_l2=("relative_l2_mean", "mean"),
            cells=("rmse_mean", "size"),
        )
        .sort_values("mean_rmse")
        .reset_index(drop=True)
    )
    return out.rename(columns={"method_label": "method"})[
        ["method", "mean_rmse", "mean_relative_l2", "cells"]
    ]


def build_f03_table() -> pd.DataFrame:
    """Format the single-well F03 GR-only benchmark summary (means plus RMSE seed variability)."""
    df = pd.read_csv(F03_SUMMARY)
    out = df[
        [
            "measurement_ratio",
            "method_label",
            "rmse_mean",
            "rmse_std_across_seeds",
            "rmse_sem",
            "n_seeds",
            "mae_mean",
            "relative_l2_mean",
        ]
    ].copy()
    # Student-t 95% half-width (df = n_seeds - 1); replaces legacy z=1.96 half-width.
    tcrit = {
        1: 12.706204736432095,
        2: 4.302652729911275,
        3: 3.182446305284263,
        4: 2.7764451051977987,
        5: 2.57058183563631,
        6: 2.4469118511449682,
        7: 2.3646242515927846,
        8: 2.306004135204166,
        9: 2.2621571627982046,
    }
    n0 = int(out["n_seeds"].iloc[0]) if len(out) else 3
    out["rmse_t95_half"] = out["rmse_sem"].astype(float) * float(tcrit.get(max(n0 - 1, 1), 1.96))
    out = out.rename(
        columns={
            "measurement_ratio": "rho",
            "method_label": "method",
            "rmse_mean": "rmse",
            "rmse_std_across_seeds": "rmse_std",
            "mae_mean": "mae",
            "relative_l2_mean": "relative_l2",
        }
    )
    out = out[["rho", "method", "rmse", "rmse_std", "rmse_t95_half", "mae", "relative_l2"]]
    return out.sort_values(["rho", "rmse"]).reset_index(drop=True)


def build_f03_table_compact() -> pd.DataFrame:
    """Pivot mean RMSE for the main-paper F03 table (methods x rho)."""
    df = pd.read_csv(F03_SUMMARY)
    sub = df[df["method_label"].isin(F03_METHOD_ORDER)].copy()
    pivot = sub.pivot_table(
        index="method_label",
        columns="measurement_ratio",
        values="rmse_mean",
        aggfunc="mean",
    )
    pivot = pivot.reindex(list(F03_METHOD_ORDER))
    rhos = sorted(pivot.columns.astype(float))
    pivot = pivot[rhos]
    out = pivot.reset_index()
    out.columns = ["method"] + [r"$\rho={:.2f}$".format(float(r)) for r in rhos]
    return out


def plot_crosswell_target_hist() -> str:
    """Plot target distributions for train and held-out cross-well V_c data."""
    apply_paper_style()
    train, test = _load_cross_segments()
    y_train = np.concatenate([seg.target for seg in train])
    y_test = np.concatenate([seg.target for seg in test])
    out = os.path.join(FIG_DIR, "eda_crosswell_vc_distribution.png")
    fig, ax = plt.subplots(figsize=(8.0, 4.6))
    ax.hist(y_train, bins=50, alpha=0.65, density=True, label="train wells")
    ax.hist(y_test, bins=50, alpha=0.65, density=True, label="held-out F03-4")
    ax.set_xlabel(r"$V_c$ (clay-volume fraction)")
    ax.set_ylabel("density")
    ax.set_title(r"Cross-well $V_c$ target distribution")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right")
    fig.tight_layout()
    save_paper_figure(fig, out)
    plt.close(fig)
    return out


def plot_f03_depth_profile() -> str:
    """Plot F03 GR and porosity versus depth for EDA."""
    apply_paper_style()
    f03 = rwf.load_f03_table(F03_REAL)
    out = os.path.join(FIG_DIR, "eda_f03_gr_porosity_depth.png")
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 6.4), sharey=True)
    axes[0].plot(f03.gr, f03.depth, color="#1f77b4", linewidth=0.9)
    axes[0].set_xlabel("GR")
    axes[0].set_ylabel("depth (m)")
    axes[0].set_title("Gamma ray")
    axes[1].plot(f03.porosity, f03.depth, color="#d62728", linewidth=0.9)
    axes[1].set_xlabel("porosity")
    axes[1].set_title("Porosity")
    for ax in axes:
        ax.grid(True, alpha=0.3)
        ax.invert_yaxis()
    fig.suptitle("F03-4 single-well profile used in the GR-only benchmark")
    fig.tight_layout()
    save_paper_figure(fig, out)
    plt.close(fig)
    return out


def plot_correlations(corr: pd.DataFrame) -> str:
    """Plot channel-target correlations from EDA."""
    apply_paper_style()
    out = os.path.join(FIG_DIR, "eda_channel_target_correlations.png")
    fig, ax = plt.subplots(figsize=(10.0, 4.8))
    labels = [
        "{}:{}".format(str(row.dataset), _channel_paper_label(str(row.channel)))
        for row in corr.itertuples(index=False)
    ]
    ax.bar(np.arange(len(labels)), corr["pearson_r"].values, color="#4c78a8")
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.set_xticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=35, ha="right")
    ax.set_ylabel("Pearson r")
    ax.set_title("Channel-target linear correlation")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    save_paper_figure(fig, out)
    plt.close(fig)
    return out


def plot_eda_target_summary(summary: pd.DataFrame) -> str:
    """Plot target means and standard deviations for each benchmark split."""
    apply_paper_style()
    out = os.path.join(FIG_DIR, "eda_target_summary.png")
    short_labels = []
    for row in summary.itertuples(index=False):
        ds = str(row.dataset)
        well = str(row.well)
        if ds == "cross_train":
            short_labels.append("{} train".format(well))
        elif ds == "cross_test":
            short_labels.append("{} test".format(well))
        else:
            short_labels.append("{} GR-only".format(well))
    x = np.arange(len(short_labels))
    y = summary["target_mean"].values
    yerr = summary["target_std"].values
    colors = ["#4c78a8"] * int(x.size)
    if int(x.size) > 0:
        colors[-1] = "#f58518"
    fig, ax = plt.subplots(figsize=(10.0, 4.8))
    ax.bar(x, y, yerr=yerr, capsize=4, color=colors)
    ax.set_xticks(x)
    ax.set_xticklabels(short_labels, rotation=25, ha="right")
    ax.set_ylabel("target mean +/- std")
    ax.set_title("Target distribution summary")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    save_paper_figure(fig, out)
    plt.close(fig)
    return out


def plot_crosswell_results() -> str:
    """Plot RMSE versus measurement ratio for the cross-well low-data grid."""
    df = pd.read_csv(CROSS_SUMMARY)
    out = os.path.join(FIG_DIR, "results_crosswell_rmse_vs_rho.png")
    labels = {
        "ridge_prior_csgm": "CLP-CSGM Ridge",
        "ae_regression_ub": "AE [u,b]",
        "ml_only": "ML only",
        "mlp_concat_ub": "MLP [u,b]",
        "pca_regression_ub": "PCA [u,b]",
    }
    colors = {
        "ridge_prior_csgm": "#d62728",
        "ae_regression_ub": "#17becf",
        "ml_only": "#1f77b4",
        "mlp_concat_ub": "#2ca02c",
        "pca_regression_ub": "#9467bd",
    }
    steps = sorted(df["step"].unique())
    fig, axes = plt.subplots(1, len(steps), figsize=(4.2 * len(steps), 4.0), sharey=True)
    if len(steps) == 1:
        axes = np.array([axes])
    for ax, step in zip(axes, steps):
        sub_step = df[df["step"] == step]
        for method in labels:
            sub = sub_step[sub_step["method"] == method].sort_values("measurement_ratio")
            if sub.empty:
                continue
            # Student-t half-width from SEM (df = n_seeds - 1); convert legacy z half-width if needed.
            if "rmse_sem" in sub.columns and "n_seeds" in sub.columns:
                n = int(sub["n_seeds"].iloc[0])
                tcrit = {
                    1: 12.706204736432095,
                    2: 4.302652729911275,
                    3: 3.182446305284263,
                    4: 2.7764451051977987,
                    5: 2.57058183563631,
                    6: 2.4469118511449682,
                    7: 2.3646242515927846,
                    8: 2.306004135204166,
                    9: 2.2621571627982046,
                }.get(max(n - 1, 1), 1.96)
                yerr = sub["rmse_sem"].astype(float) * float(tcrit)
            else:
                n = int(sub["n_seeds"].iloc[0]) if "n_seeds" in sub.columns else 3
                factor = {
                    1: 12.706204736432095,
                    2: 4.302652729911275,
                }.get(max(n - 1, 1), 1.96) / 1.96
                yerr = sub["rmse_ci95_half"].fillna(0.0).astype(float) * float(factor)
            ax.errorbar(
                sub["measurement_ratio"],
                sub["rmse_mean"],
                yerr=yerr,
                marker="o",
                capsize=3,
                linewidth=2.2 if method == "ridge_prior_csgm" else 1.4,
                color=colors[method],
                label=labels[method],
            )
        ax.set_title("step = {}".format(step))
        ax.set_xlabel("measurement ratio rho")
        ax.grid(True, alpha=0.3)
    axes[0].set_ylabel("RMSE")
    axes[-1].legend(loc="upper right", fontsize=7)
    fig.suptitle("Cross-well Vc low-data benchmark", y=1.02)
    fig.tight_layout()
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_f03_results() -> str:
    """Plot RMSE versus measurement ratio for the single-well F03 benchmark."""
    df = pd.read_csv(F03_SUMMARY)
    out = os.path.join(FIG_DIR, "results_f03_gr_only_rmse_vs_rho.png")
    fig, ax = plt.subplots(figsize=(7.6, 4.6))
    for label in F03_METHOD_ORDER:
        sub = df[df["method_label"] == label].sort_values("measurement_ratio")
        if sub.empty:
            continue
        style = F03_PLOT_STYLE[label]
        if "rmse_sem" in sub.columns and "n_seeds" in sub.columns:
            n = int(sub["n_seeds"].iloc[0])
            tcrit = {
                1: 12.706204736432095,
                2: 4.302652729911275,
                3: 3.182446305284263,
                4: 2.7764451051977987,
                5: 2.57058183563631,
                6: 2.4469118511449682,
                7: 2.3646242515927846,
                8: 2.306004135204166,
                9: 2.2621571627982046,
            }.get(max(n - 1, 1), 1.96)
            yerr = sub["rmse_sem"].astype(float) * float(tcrit)
        else:
            yerr = sub["rmse_ci95_half"].fillna(0.0)
        ax.errorbar(
            sub["measurement_ratio"],
            sub["rmse_mean"],
            yerr=yerr,
            marker="o",
            capsize=3,
            markersize=4 if label in ("Prior-only Ridge", "Measurement-only CSGM") else 5,
            color=str(style["color"]),
            linewidth=float(style["linewidth"]),
            linestyle=str(style["linestyle"]),
            alpha=float(style["alpha"]),
            label=label,
        )
    ax.set_xlabel("measurement ratio rho")
    ax.set_ylabel("RMSE")
    ax.set_title("Single-well F03 GR-only porosity benchmark")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right", fontsize=7, ncol=2, framealpha=0.92)
    fig.tight_layout()
    fig.savefig(out, dpi=180)
    plt.close(fig)
    return out


def write_asset_manifest(paths: Iterable[str]) -> None:
    """Write a compact manifest for generated paper assets."""
    manifest = os.path.join(PAPER_DIR, "ASSET_MANIFEST.txt")
    rels = [os.path.relpath(p, PAPER_DIR) for p in paths]
    with open(manifest, "w", encoding="ascii") as f:
        f.write("CLP-CSGM paper assets generated by scripts/clp_csgm_paper_assets.py\n")
        f.write("\n")
        for rel in sorted(rels):
            f.write("{}\n".format(rel))


def main() -> None:
    """Generate all paper assets."""
    ensure_dirs()
    generated: List[str] = []

    dataset_summary = build_dataset_summary()
    corr = build_correlation_table()
    tables = [
        TableSpec(dataset_summary, "eda_dataset_summary", "%.4f"),
        TableSpec(corr, "eda_channel_target_correlations", "%.4f"),
        TableSpec(build_crosswell_winners_table(), "results_crosswell_clp_vs_ae", "%.5f"),
        TableSpec(build_crosswell_rank_table(), "results_crosswell_overall_rank", "%.5f"),
        TableSpec(build_f03_table(), "results_f03_gr_only", "%.5f"),
        TableSpec(build_f03_table_compact(), "results_f03_gr_only_compact", "%.5f"),
    ]
    for spec in tables:
        generated.append(write_table(spec))
        generated.append(os.path.join(TAB_DIR, "{}.csv".format(spec.stem)))

    generated.extend(
        [
            plot_eda_target_summary(dataset_summary),
            plot_crosswell_target_hist(),
            plot_f03_depth_profile(),
            plot_correlations(corr),
            plot_crosswell_results(),
            plot_f03_results(),
        ]
    )
    write_asset_manifest(generated)
    print("Generated {} CLP-CSGM paper assets under {}".format(len(generated), PAPER_DIR))


if __name__ == "__main__":
    main()
