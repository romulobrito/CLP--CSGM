#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate short-horizon uncertainty assets for the CLP-CSGM paper.

1) Paired seed-level Delta RMSE = RMSE(AE) - RMSE(CLP-CSGM Ridge), with
   Student-t 95% half-widths (df = n_seeds - 1).
2) Rewrite results_f03_gr_only.tex using Student-t half-widths instead of z=1.96.
3) Regenerate F03 / cross-well RMSE figures with the same Student-t error bars.

ASCII-only source.
"""

from __future__ import annotations

import math
import os
from typing import Dict, List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
PAPER_DIR = os.path.join(REPO_ROOT, "paper_clp_csgm")
TAB_DIR = os.path.join(PAPER_DIR, "tables")
FIG_DIR = os.path.join(PAPER_DIR, "figures")

F03_SEED_SUMMARY = os.path.join(
    REPO_ROOT,
    "outputs",
    "real_well_f03",
    "direct_ub",
    "runs",
    "f03_full_gr_only_clp_csgm_ridge",
    "tables",
    "summary_by_seed.csv",
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
CROSS_SUMMARY = os.path.join(
    REPO_ROOT,
    "outputs",
    "cross_well_vc",
    "csgm",
    "m2_grid",
    "tables",
    "summary_focus_clp_csgm_vs_ub.csv",
)
CROSSWELL_STEPS: Tuple[Tuple[int, str], ...] = (
    (8, "crosswell_step08_clp_csgm_ablation_ridge"),
    (16, "crosswell_step16_clp_csgm_ablation_ridge"),
    (32, "crosswell_step32_clp_csgm_ablation_ridge"),
)

CLP_METHOD = "ridge_prior_csgm"
AE_METHOD = "ae_regression_ub"

F03_METHOD_ORDER: Tuple[str, ...] = (
    "CLP-CSGM Ridge",
    "AE [u,b]",
    "Prior-only Ridge",
    "Measurement-only CSGM",
    "ML only",
    "PCA [u,b]",
    "MLP [u,b]",
)
F03_PLOT_STYLE: Dict[str, Dict[str, object]] = {
    "CLP-CSGM Ridge": {"color": "#d62728", "linewidth": 2.2, "linestyle": "-", "alpha": 1.0},
    "AE [u,b]": {"color": "#17becf", "linewidth": 1.6, "linestyle": "-", "alpha": 1.0},
    "Prior-only Ridge": {"color": "#ff9896", "linewidth": 1.4, "linestyle": "--", "alpha": 0.95},
    "Measurement-only CSGM": {"color": "#7f7f7f", "linewidth": 1.4, "linestyle": "--", "alpha": 0.95},
    "ML only": {"color": "#1f77b4", "linewidth": 1.2, "linestyle": "-", "alpha": 0.85},
    "PCA [u,b]": {"color": "#9467bd", "linewidth": 1.2, "linestyle": "-", "alpha": 0.85},
    "MLP [u,b]": {"color": "#2ca02c", "linewidth": 1.2, "linestyle": "-", "alpha": 0.85},
}


def student_t_critical_975(df: int) -> float:
    """Two-sided 95% Student-t critical value for the given degrees of freedom."""
    if df < 1:
        return float("nan")
    return float(stats.t.ppf(0.975, df=int(df)))


def t95_half_width(std: float, n: int) -> float:
    """Student-t 95% CI half-width from sample std with n observations."""
    if n <= 1 or not np.isfinite(std):
        return 0.0
    se = float(std) / math.sqrt(float(n))
    return student_t_critical_975(n - 1) * se


def paired_delta_frame(seed_csv: str) -> pd.DataFrame:
    """Build paired Delta RMSE = AE - CLP rows for one summary_by_seed.csv."""
    df = pd.read_csv(seed_csv)
    rows: List[Dict[str, float]] = []
    for rho, g in df.groupby("measurement_ratio"):
        ae = g.loc[g["method"] == AE_METHOD].set_index("seed")["rmse_mean"]
        clp = g.loc[g["method"] == CLP_METHOD].set_index("seed")["rmse_mean"]
        seeds = sorted(set(ae.index).intersection(set(clp.index)))
        if not seeds:
            continue
        delta = np.asarray([float(ae[s] - clp[s]) for s in seeds], dtype=np.float64)
        n = int(delta.size)
        mean = float(np.mean(delta))
        std = float(np.std(delta, ddof=1)) if n > 1 else 0.0
        se = std / math.sqrt(n) if n > 1 else 0.0
        rows.append(
            {
                "rho": float(rho),
                "n_seeds": float(n),
                "mean_delta_rmse": mean,
                "delta_std": std,
                "delta_se": se,
                "delta_t95_half": t95_half_width(std, n),
                "clp_wins": float(np.sum(delta > 0.0)),
                "ae_wins": float(np.sum(delta < 0.0)),
            }
        )
    return pd.DataFrame(rows).sort_values("rho").reset_index(drop=True)


def write_latex_table(frame: pd.DataFrame, stem: str, float_format: str = "%.5f") -> str:
    """Write CSV + ASCII LaTeX table under paper_clp_csgm/tables/."""
    os.makedirs(TAB_DIR, exist_ok=True)
    csv_path = os.path.join(TAB_DIR, "{}.csv".format(stem))
    tex_path = os.path.join(TAB_DIR, "{}.tex".format(stem))
    frame.to_csv(csv_path, index=False)
    with open(tex_path, "w", encoding="ascii") as f:
        f.write(
            frame.to_latex(
                index=False,
                escape=True,
                float_format=lambda x: float_format % x,
            )
        )
    return tex_path


def build_f03_full_table_t() -> pd.DataFrame:
    """Rewrite F03 full multi-metric table with Student-t half-widths."""
    df = pd.read_csv(F03_SUMMARY)
    n = int(df["n_seeds"].iloc[0]) if "n_seeds" in df.columns and len(df) else 3
    out = df[
        [
            "measurement_ratio",
            "method_label",
            "rmse_mean",
            "rmse_std_across_seeds",
            "rmse_sem",
            "mae_mean",
            "relative_l2_mean",
        ]
    ].copy()
    out["rmse_t95_half"] = out["rmse_sem"].astype(float) * student_t_critical_975(max(n - 1, 1))
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


def _yerr_t_from_summary(sub: pd.DataFrame) -> np.ndarray:
    """Compute Student-t half-widths from rmse_sem / n_seeds columns."""
    if "rmse_sem" in sub.columns and "n_seeds" in sub.columns:
        halves = []
        for _, row in sub.iterrows():
            n = int(row["n_seeds"])
            halves.append(float(row["rmse_sem"]) * student_t_critical_975(max(n - 1, 1)))
        return np.asarray(halves, dtype=np.float64)
    if "rmse_std_across_seeds" in sub.columns and "n_seeds" in sub.columns:
        halves = []
        for _, row in sub.iterrows():
            n = int(row["n_seeds"])
            halves.append(t95_half_width(float(row["rmse_std_across_seeds"]), n))
        return np.asarray(halves, dtype=np.float64)
    # Legacy fallback: convert old z half-width to Student-t (n=3 => factor ~2.195).
    if "rmse_ci95_half" in sub.columns and "n_seeds" in sub.columns:
        halves = []
        for _, row in sub.iterrows():
            n = int(row["n_seeds"])
            z_half = float(row["rmse_ci95_half"])
            if n <= 1 or not np.isfinite(z_half):
                halves.append(0.0)
            else:
                factor = student_t_critical_975(n - 1) / 1.96
                halves.append(z_half * factor)
        return np.asarray(halves, dtype=np.float64)
    return np.zeros(len(sub), dtype=np.float64)


def plot_f03_results_t() -> str:
    """Regenerate F03 RMSE figure with Student-t error bars."""
    df = pd.read_csv(F03_SUMMARY)
    out = os.path.join(FIG_DIR, "results_f03_gr_only_rmse_vs_rho.png")
    fig, ax = plt.subplots(figsize=(7.6, 4.6))
    for label in F03_METHOD_ORDER:
        sub = df[df["method_label"] == label].sort_values("measurement_ratio")
        if sub.empty:
            continue
        style = F03_PLOT_STYLE[label]
        ax.errorbar(
            sub["measurement_ratio"],
            sub["rmse_mean"],
            yerr=_yerr_t_from_summary(sub),
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
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_crosswell_results_t() -> str:
    """Regenerate cross-well RMSE figure with Student-t error bars."""
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
            ax.errorbar(
                sub["measurement_ratio"],
                sub["rmse_mean"],
                yerr=_yerr_t_from_summary(sub),
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


def build_crosswell_paired_table() -> pd.DataFrame:
    """Stack paired Delta RMSE over the three cross-well steps."""
    frames: List[pd.DataFrame] = []
    for step, run_id in CROSSWELL_STEPS:
        path = os.path.join(
            REPO_ROOT,
            "outputs",
            "cross_well_vc",
            "direct_ub",
            "runs",
            run_id,
            "tables",
            "summary_by_seed.csv",
        )
        part = paired_delta_frame(path)
        part.insert(0, "step", int(step))
        frames.append(part)
    out = pd.concat(frames, ignore_index=True)
    # Compact display columns for the paper table.
    return out[
        [
            "step",
            "rho",
            "mean_delta_rmse",
            "delta_se",
            "delta_t95_half",
            "clp_wins",
            "ae_wins",
        ]
    ]


def build_f03_paired_table() -> pd.DataFrame:
    """Paired Delta RMSE table for the F03 GR-only benchmark."""
    out = paired_delta_frame(F03_SEED_SUMMARY)
    return out[
        [
            "rho",
            "mean_delta_rmse",
            "delta_se",
            "delta_t95_half",
            "clp_wins",
            "ae_wins",
        ]
    ]


def main() -> None:
    """Generate paired-delta tables, t-CI F03 table, and updated RMSE figures."""
    os.makedirs(TAB_DIR, exist_ok=True)
    os.makedirs(FIG_DIR, exist_ok=True)

    f03_paired = build_f03_paired_table()
    cw_paired = build_crosswell_paired_table()
    f03_full = build_f03_full_table_t()

    generated: Sequence[str] = [
        write_latex_table(f03_paired, "paired_delta_f03_clp_vs_ae", "%.5f"),
        write_latex_table(cw_paired, "paired_delta_crosswell_clp_vs_ae", "%.5f"),
        write_latex_table(f03_full, "results_f03_gr_only", "%.5f"),
        plot_f03_results_t(),
        plot_crosswell_results_t(),
    ]
    print("Generated {} uncertainty assets:".format(len(generated)))
    for path in generated:
        print(" - {}".format(path))


if __name__ == "__main__":
    main()
