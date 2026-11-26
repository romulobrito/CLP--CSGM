#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""RMSE versus noise for interpolation and CLP-CSGM. ASCII only.

One panel per step. One curve pair per measurement fraction.
Means come from the finished noise sweep and are not refit here.
"""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

ROOT_DIR = Path(__file__).resolve().parents[1]
SUMMARY_PATH = (
    ROOT_DIR
    / "outputs"
    / "reviewer_revision"
    / "r3m03_noise_sweep"
    / "tables"
    / "summary_by_cell.csv"
)
PAPER_PATH = (
    ROOT_DIR
    / "paper_clp_csgm"
    / "figures"
    / "crosswell_vc_rmse_vs_noise.png"
)
PREVIEW_PATH = (
    ROOT_DIR
    / "outputs"
    / "reviewer_revision"
    / "r3m03_noise_sweep"
    / "figures"
    / "rmse_vs_noise_by_step.png"
)

STEPS = (8, 16, 32)
RHOS = (0.05, 0.10, 0.20)
CURVE_STD = 0.035
RHO_COLOR = {
    0.05: "#0072B2",
    0.10: "#D55E00",
    0.20: "#009E73",
}


def _load_means(path: Path) -> dict[tuple[int, float, float], tuple[float, float]]:
    """Map (step, rho, noise) to (interpolation RMSE, CLP RMSE)."""
    if not path.is_file():
        raise FileNotFoundError("Missing noise-sweep summary: {0}".format(path))
    means: dict[tuple[int, float, float], tuple[float, float]] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required = {"step", "measurement_ratio", "noise_std", "interp_rmse", "clp_rmse"}
        if reader.fieldnames is None or not required.issubset(set(reader.fieldnames)):
            raise ValueError("Noise-sweep summary is missing required columns.")
        for row in reader:
            key = (
                int(row["step"]),
                float(row["measurement_ratio"]),
                float(row["noise_std"]),
            )
            means[key] = (float(row["interp_rmse"]), float(row["clp_rmse"]))
    return means


def _series(
    means: dict[tuple[int, float, float], tuple[float, float]],
    step: int,
    rho: float,
) -> tuple[list[float], list[float], list[float]]:
    """Noise levels and the two RMSE series for one cell family."""
    noises = sorted(
        noise
        for (row_step, row_rho, noise) in means
        if row_step == step and row_rho == rho
    )
    if not noises:
        raise ValueError(
            "No rows for step {0} and rho {1}.".format(step, rho)
        )
    interp = [means[(step, rho, noise)][0] for noise in noises]
    clp = [means[(step, rho, noise)][1] for noise in noises]
    return noises, interp, clp


def main() -> None:
    """Write the paper figure and a copy beside the sweep tables."""
    means = _load_means(SUMMARY_PATH)
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 3.15), sharey=True)
    for axis, step in zip(axes, STEPS):
        axis.axvspan(0.07, 0.32, color="#F0F0F0", zorder=0)
        axis.axvline(CURVE_STD, color="#444444", linewidth=0.8, linestyle=":")
        if step == STEPS[0]:
            axis.text(
                CURVE_STD,
                0.178,
                "0.035",
                rotation=90,
                va="top",
                ha="right",
                fontsize=6.5,
                color="#444444",
            )
        for rho in RHOS:
            noises, interp, clp = _series(means, step, rho)
            color = RHO_COLOR[rho]
            axis.plot(
                noises,
                interp,
                color=color,
                linestyle="--",
                marker="o",
                markersize=4.5,
                markerfacecolor="white",
                markeredgewidth=1.1,
                linewidth=1.1,
                zorder=2,
            )
            axis.plot(
                noises,
                clp,
                color=color,
                linestyle="-",
                marker="s",
                markersize=4.2,
                linewidth=1.1,
                zorder=3,
            )
        axis.set_xscale("log")
        axis.set_xlim(0.007, 0.30)
        axis.set_xticks([0.01, 0.05, 0.10, 0.20])
        axis.set_xticklabels(["0.01", "0.05", "0.10", "0.20"])
        axis.set_title("Step {0}".format(step), fontsize=9)
        axis.set_xlabel("Noise standard deviation (log scale)", fontsize=8)
        axis.tick_params(labelsize=7.5)
        axis.grid(True, which="major", axis="y", linewidth=0.4, color="#DDDDDD")
        axis.set_axisbelow(True)
    axes[0].set_ylabel("Mean full-window RMSE", fontsize=8)
    axes[0].set_ylim(0.0, 0.19)

    method_handles = [
        Line2D(
            [0],
            [0],
            color="#333333",
            linestyle="--",
            marker="o",
            markersize=4.5,
            markerfacecolor="white",
            markeredgewidth=1.1,
            label="Interpolation",
        ),
        Line2D(
            [0],
            [0],
            color="#333333",
            linestyle="-",
            marker="s",
            markersize=4.2,
            label="CLP-CSGM Ridge",
        ),
    ]
    rho_handles = [
        Line2D([0], [0], color=RHO_COLOR[rho], linewidth=2.0, label=r"$\rho={0:.2f}$".format(rho))
        for rho in RHOS
    ]
    fig.legend(
        handles=method_handles + rho_handles,
        loc="lower center",
        ncol=5,
        frameon=False,
        fontsize=7.5,
        bbox_to_anchor=(0.5, -0.02),
    )
    fig.tight_layout(rect=(0.0, 0.08, 1.0, 1.0))
    for path in (PAPER_PATH, PREVIEW_PATH):
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    main()
