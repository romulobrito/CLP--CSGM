#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Generate CLP-CSGM diagnostic figures from saved benchmark CSV files.

The saved runs do not contain epoch-wise train/validation loss histories. This
script therefore builds indirect diagnostics from values that are already saved:
validation score, selected lambda, test RMSE by sample, and AE reconstruction
train RMSE.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter, MaxNLocator
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / "outputs/paper_assets"
FIG_DIR = PAPER / "figures"
TAB_DIR = PAPER / "tables"

CROSS_RUNS: Dict[int, Path] = {
    8: ROOT / "outputs" / "cross_well_vc" / "direct_ub" / "runs" / "crosswell_step08_clp_csgm_ablation_ridge",
    16: ROOT / "outputs" / "cross_well_vc" / "direct_ub" / "runs" / "crosswell_step16_clp_csgm_ablation_ridge",
    32: ROOT / "outputs" / "cross_well_vc" / "direct_ub" / "runs" / "crosswell_step32_clp_csgm_ablation_ridge",
}
F03_RUN = (
    ROOT
    / "outputs"
    / "real_well_f03"
    / "direct_ub"
    / "runs"
    / "f03_full_gr_only_clp_csgm_ridge"
)

METHOD_ORDER = [
    "ridge_prior_csgm",
    "ridge_prior_only_decoder",
    "measurement_only_csgm",
    "ae_regression_ub",
]

METHOD_LABELS = {
    "ridge_prior_csgm": "CLP-CSGM Ridge",
    "ridge_prior_only_decoder": "Prior-only",
    "measurement_only_csgm": "Measurement-only",
    "ae_regression_ub": "AE [u,b]",
}


def _ensure_dirs() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    TAB_DIR.mkdir(parents=True, exist_ok=True)


def _read_detailed(run_path: Path, step: int | None = None) -> pd.DataFrame:
    path = run_path / "tables" / "detailed_results.csv"
    if not path.is_file():
        raise FileNotFoundError(str(path))
    df = pd.read_csv(path)
    required = {
        "seed",
        "measurement_ratio",
        "method",
        "sample_id",
        "rmse",
        "lambda",
        "val_score",
        "ae_recon_train_rmse",
    }
    missing = required.difference(df.columns)
    if missing:
        raise ValueError("Missing columns in {}: {}".format(path, sorted(missing)))
    if step is not None:
        df["step"] = int(step)
    return df


def _aggregate_by_seed(df: pd.DataFrame, include_step: bool) -> pd.DataFrame:
    keys = ["seed", "measurement_ratio", "method"]
    if include_step:
        keys.insert(1, "step")
    agg = (
        df.groupby(keys, as_index=False)
        .agg(
            test_rmse=("rmse", "mean"),
            test_rmse_std=("rmse", "std"),
            val_score=("val_score", "first"),
            selected_lambda=("lambda", "first"),
            ae_recon_train_rmse=("ae_recon_train_rmse", "first"),
            n_test_samples=("sample_id", "nunique"),
        )
        .sort_values(keys)
    )
    return agg


def _plot_val_vs_test(ax: plt.Axes, agg: pd.DataFrame, title: str, include_step: bool) -> None:
    sub = agg[agg["method"] == "ridge_prior_csgm"].copy()
    sub = sub.dropna(subset=["val_score", "test_rmse"])
    if sub.empty:
        ax.set_title(title)
        ax.text(0.5, 0.5, "No validation scores", ha="center", va="center")
        return
    if include_step:
        for step, part in sub.groupby("step"):
            ax.scatter(part["val_score"], part["test_rmse"], label="step {}".format(step), alpha=0.85)
    else:
        scatter = ax.scatter(
            sub["val_score"],
            sub["test_rmse"],
            c=sub["measurement_ratio"],
            alpha=0.85,
        )
        plt.colorbar(scatter, ax=ax, label="rho")
    lo = float(min(sub["val_score"].min(), sub["test_rmse"].min()))
    hi = float(max(sub["val_score"].max(), sub["test_rmse"].max()))
    pad = 0.05 * (hi - lo if hi > lo else 1.0)
    ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], linestyle="--", linewidth=1.0, color="0.4")
    ax.set_xlabel("Validation RMSE used for lambda selection")
    ax.set_ylabel("Mean test RMSE")
    ax.set_title(title)
    if include_step:
        ax.legend(fontsize=8)
    ax.grid(True, alpha=0.25)


def _plot_lambda(ax: plt.Axes, agg: pd.DataFrame, title: str, include_step: bool) -> None:
    sub = agg[agg["method"] == "ridge_prior_csgm"].copy()
    sub = sub.dropna(subset=["selected_lambda"])
    if sub.empty:
        ax.set_title(title)
        ax.text(0.5, 0.5, "No selected lambda values", ha="center", va="center")
        return
    if include_step:
        for step, part in sub.groupby("step"):
            med = part.groupby("measurement_ratio", as_index=False)["selected_lambda"].median()
            ax.plot(med["measurement_ratio"], med["selected_lambda"], marker="o", label="step {}".format(step))
            ax.scatter(part["measurement_ratio"], part["selected_lambda"], alpha=0.25)
        ax.legend(fontsize=8)
    else:
        med = sub.groupby("measurement_ratio", as_index=False)["selected_lambda"].median()
        ax.plot(med["measurement_ratio"], med["selected_lambda"], marker="o", label="median")
        ax.scatter(sub["measurement_ratio"], sub["selected_lambda"], alpha=0.35)
    ax.set_yscale("log")
    ax.set_xlabel("Measurement ratio rho")
    ax.set_ylabel("Selected lambda")
    ax.set_title(title)
    ax.grid(True, alpha=0.25, which="both")


def _plot_rmse_box(ax: plt.Axes, df: pd.DataFrame, title: str) -> None:
    data: List[np.ndarray] = []
    labels: List[str] = []
    for method in METHOD_ORDER:
        vals = df.loc[df["method"] == method, "rmse"].dropna().to_numpy(dtype=float)
        if vals.size:
            data.append(vals)
            labels.append(METHOD_LABELS[method])
    if not data:
        ax.set_title(title)
        ax.text(0.5, 0.5, "No RMSE values", ha="center", va="center")
        return
    ax.boxplot(data, labels=labels, showfliers=False)
    ax.set_ylabel("Per-window RMSE")
    ax.set_title(title)
    ax.tick_params(axis="x", rotation=20)
    ax.grid(True, axis="y", alpha=0.25)


def _plot_ae_recon(ax: plt.Axes, agg: pd.DataFrame, title: str, include_step: bool) -> None:
    sub = agg[agg["method"] == "ridge_prior_csgm"].copy()
    sub = sub.dropna(subset=["ae_recon_train_rmse", "test_rmse"])
    if sub.empty:
        ax.set_title(title)
        ax.text(0.5, 0.5, "No AE reconstruction values", ha="center", va="center")
        return
    if include_step:
        for step, part in sub.groupby("step"):
            ax.scatter(part["ae_recon_train_rmse"], part["test_rmse"], label="step {}".format(step), alpha=0.85)
        ax.legend(fontsize=8)
    else:
        scatter = ax.scatter(
            sub["ae_recon_train_rmse"],
            sub["test_rmse"],
            c=sub["measurement_ratio"],
            alpha=0.85,
        )
        plt.colorbar(scatter, ax=ax, label="rho")
    ax.set_xlabel("AE train reconstruction RMSE")
    ax.set_ylabel("Mean test RMSE")
    ax.set_title(title)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=4))
    ax.xaxis.set_major_formatter(FormatStrFormatter("%.5f"))
    ax.grid(True, alpha=0.25)


def _make_diagnostic_figure(
    df: pd.DataFrame,
    agg: pd.DataFrame,
    output_path: Path,
    title: str,
    include_step: bool,
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12.0, 8.5))
    _plot_val_vs_test(axes[0, 0], agg, "Validation RMSE vs test RMSE", include_step)
    _plot_lambda(axes[0, 1], agg, "Selected lambda across rho", include_step)
    _plot_rmse_box(axes[1, 0], df, "Per-window RMSE distribution")
    _plot_ae_recon(axes[1, 1], agg, "Decoder train reconstruction vs test RMSE", include_step)
    fig.suptitle(title, y=1.02)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _write_method_summary(df: pd.DataFrame, path: Path, include_step: bool) -> None:
    keys = ["measurement_ratio", "method"]
    if include_step:
        keys.insert(0, "step")
    out = (
        df[df["method"].isin(METHOD_ORDER)]
        .groupby(keys, as_index=False)
        .agg(
            rmse_mean=("rmse", "mean"),
            rmse_median=("rmse", "median"),
            rmse_q25=("rmse", lambda x: float(np.quantile(x, 0.25))),
            rmse_q75=("rmse", lambda x: float(np.quantile(x, 0.75))),
            n=("rmse", "size"),
        )
    )
    out["method_label"] = out["method"].map(METHOD_LABELS).fillna(out["method"])
    out.to_csv(path, index=False)


def build_crosswell() -> Tuple[Path, Path, Path]:
    frames = [_read_detailed(path, step=step) for step, path in CROSS_RUNS.items()]
    df = pd.concat(frames, ignore_index=True)
    agg = _aggregate_by_seed(df, include_step=True)
    agg_path = TAB_DIR / "diagnostic_crosswell_by_seed.csv"
    box_path = TAB_DIR / "diagnostic_crosswell_rmse_distribution.csv"
    fig_path = FIG_DIR / "diagnostic_crosswell_generalization.png"
    agg.to_csv(agg_path, index=False)
    _write_method_summary(df, box_path, include_step=True)
    _make_diagnostic_figure(
        df=df,
        agg=agg,
        output_path=fig_path,
        title="Cross-well Vc CLP-CSGM diagnostics",
        include_step=True,
    )
    return fig_path, agg_path, box_path


def build_f03() -> Tuple[Path, Path, Path]:
    df = _read_detailed(F03_RUN, step=None)
    agg = _aggregate_by_seed(df, include_step=False)
    agg_path = TAB_DIR / "diagnostic_f03_by_seed.csv"
    box_path = TAB_DIR / "diagnostic_f03_rmse_distribution.csv"
    fig_path = FIG_DIR / "diagnostic_f03_generalization.png"
    agg.to_csv(agg_path, index=False)
    _write_method_summary(df, box_path, include_step=False)
    _make_diagnostic_figure(
        df=df,
        agg=agg,
        output_path=fig_path,
        title="F03-4 GR-only CLP-CSGM diagnostics",
        include_step=False,
    )
    return fig_path, agg_path, box_path


def main() -> None:
    _ensure_dirs()
    outputs: Iterable[Path] = [*build_crosswell(), *build_f03()]
    for path in outputs:
        print(path.relative_to(ROOT))


if __name__ == "__main__":
    main()
