#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Generate lightweight figures for the CLP-CSGM manuscript from CSV assets."""

from __future__ import annotations

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
PAPER = os.path.join(ROOT, "outputs/paper_assets")
FIG = os.path.join(PAPER, "figures")
TAB = os.path.join(PAPER, "tables")


def _ensure() -> None:
    os.makedirs(FIG, exist_ok=True)


def plot_crosswell() -> str:
    """Plot CLP-CSGM and AE RMSE for the low-data cross-well grid."""
    df = pd.read_csv(os.path.join(TAB, "results_crosswell_clp_vs_ae.csv"))
    out = os.path.join(FIG, "results_crosswell_clp_vs_ae.png")
    steps = sorted(df["step"].unique())
    fig, axes = plt.subplots(1, len(steps), figsize=(4.2 * len(steps), 3.6), sharey=True)
    if len(steps) == 1:
        axes = np.array([axes])
    for ax, step in zip(axes, steps):
        sub = df[df["step"] == step].sort_values("rho")
        ax.plot(
            sub["rho"],
            sub["clp_csgm_rmse_mean"],
            marker="o",
            label="CLP-CSGM Ridge",
            color="#d62728",
        )
        ax.plot(
            sub["rho"],
            sub["ae_ub_rmse_mean"],
            marker="o",
            label="AE [u,b]",
            color="#17becf",
        )
        ax.set_title("step = {}".format(step))
        ax.set_xlabel("rho")
        ax.grid(True, alpha=0.3)
    axes[0].set_ylabel("RMSE")
    axes[-1].legend(loc="best", fontsize=8)
    fig.suptitle("Cross-well Vc: CLP-CSGM Ridge vs AE [u,b]", y=1.02)
    fig.tight_layout()
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_f03() -> str:
    """Plot F03 GR-only RMSE by measurement ratio."""
    df = pd.read_csv(os.path.join(TAB, "results_f03_gr_only.csv"))
    out = os.path.join(FIG, "results_f03_gr_only_rmse.png")
    keep = ["CLP-CSGM Ridge", "AE [u,b]", "ML only", "PCA [u,b]", "MLP [u,b]"]
    colors = {
        "CLP-CSGM Ridge": "#d62728",
        "AE [u,b]": "#17becf",
        "ML only": "#1f77b4",
        "PCA [u,b]": "#9467bd",
        "MLP [u,b]": "#2ca02c",
    }
    fig, ax = plt.subplots(figsize=(7.0, 4.0))
    for method in keep:
        sub = df[df["method"] == method].sort_values("rho")
        if sub.empty:
            continue
        ax.plot(sub["rho"], sub["rmse"], marker="o", label=method, color=colors[method])
    ax.set_xlabel("rho")
    ax.set_ylabel("RMSE")
    ax.set_title("F03-4 GR-only porosity benchmark")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(out, dpi=180)
    plt.close(fig)
    return out


def plot_eda() -> str:
    """Plot target means and ranges from the EDA summary."""
    df = pd.read_csv(os.path.join(TAB, "eda_dataset_summary.csv"))
    out = os.path.join(FIG, "eda_target_summary.png")
    labels = ["{} {}".format(a, b) for a, b in zip(df["dataset"], df["well"])]
    x = np.arange(len(labels))
    y = df["target_mean"].values
    yerr = df["target_std"].values
    fig, ax = plt.subplots(figsize=(8.0, 4.0))
    ax.bar(x, y, yerr=yerr, capsize=3, color="#4c78a8")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_ylabel("target mean +/- std")
    ax.set_title("Target distribution summary")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=180)
    plt.close(fig)
    return out


def main() -> None:
    """Generate all lightweight figures and a manifest."""
    _ensure()
    paths = [plot_crosswell(), plot_f03(), plot_eda()]
    manifest = os.path.join(PAPER, "FIGURE_MANIFEST.txt")
    with open(manifest, "w", encoding="ascii") as f:
        for path in paths:
            f.write("{}\n".format(os.path.relpath(path, PAPER)))
    print("Generated {} quick figures".format(len(paths)))


if __name__ == "__main__":
    main()
