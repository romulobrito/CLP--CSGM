#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Measure representative computational cost for CLP-CSGM experiments.

The study is intentionally separated from the main benchmark runners. It uses
the same data loaders, model settings, seeds, measurement matrices, and metrics,
but records end-to-end wall-clock time per method for a representative protocol.
The reported time includes model fitting and validation/test prediction for the
corresponding seed and measurement ratio.
"""

from __future__ import annotations

import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple
import sys

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import csgm_m2_module as csgm
import direct_ub_baselines as dub
import multi_well_vc as mwv
import real_well_f03 as rwf
from sir_cs_pipeline_optimized import (
    Config,
    MultiOutputMLP,
    apply_config_profile,
    build_measurement_matrix,
)


OUT_DIR = ROOT / "outputs" / "runtime_clp_csgm"
TAB_DIR = OUT_DIR / "tables"
FIG_DIR = OUT_DIR / "figures"
PAPER = ROOT / "outputs/paper_assets"
PAPER_TAB_DIR = PAPER / "tables"
PAPER_FIG_DIR = PAPER / "figures"
PAPER_RUNTIME_TABLE = PAPER_TAB_DIR / "runtime_cost_summary.tex"

CROSS_TRAIN = ROOT / "data" / "F02-1,F03-2,F06-1_6logs_30dB.txt"
CROSS_TEST = ROOT / "data" / "F03-4_6logs_30dB.txt"
F03_DATA = ROOT / "data" / "F03-4_AC+GR+Porosity.txt"


@dataclass(frozen=True)
class RuntimeCase:
    """Container for one dataset configuration."""

    dataset: str
    data: Dict[str, np.ndarray]
    cfg: Config
    measurement_ratios: Tuple[float, ...]
    seeds: Tuple[int, ...]


def _ensure_dirs() -> None:
    for path in (TAB_DIR, FIG_DIR, PAPER_TAB_DIR, PAPER_FIG_DIR):
        path.mkdir(parents=True, exist_ok=True)


def _rmse_mean(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    diff = np.asarray(y_pred, dtype=np.float64) - np.asarray(y_true, dtype=np.float64)
    per_sample = np.sqrt(np.mean(diff * diff, axis=1))
    return float(np.mean(per_sample))


def _base_cfg(profile: str, p_input: int, n_output: int, n_train: int, n_val: int, n_test: int) -> Config:
    cfg = Config()
    cfg.log_progress = False
    cfg.config_profile = profile
    apply_config_profile(cfg)
    cfg.p_input = int(p_input)
    cfg.n_output = int(n_output)
    cfg.n_train = int(n_train)
    cfg.n_val = int(n_val)
    cfg.n_test = int(n_test)
    cfg.residual_basis = "dct"
    cfg.measurement_kind = "subsample"
    cfg.measurement_noise_std = 0.02
    cfg.run_lfista = False
    cfg.run_csgm_m2 = True
    cfg.run_csgm_ablations = False
    cfg.csgm_prior_type = "ridge"
    cfg.csgm_latent_dim = 16
    cfg.csgm_ae_epochs = 200
    cfg.csgm_iters = 400
    cfg.csgm_restarts = 3
    cfg.csgm_opt_lr = 0.05
    cfg.csgm_lambda_grid = [0.0001, 0.0003, 0.001, 0.003, 0.01, 0.03, 0.1]
    return cfg


def build_cross_case() -> RuntimeCase:
    """Build the representative cross-well Vc runtime case."""
    if not CROSS_TRAIN.is_file() or not CROSS_TEST.is_file():
        missing = [str(p) for p in (CROSS_TRAIN, CROSS_TEST) if not p.is_file()]
        raise FileNotFoundError("Missing cross-well data files: {}".format(missing))
    raw = mwv.build_cross_well_data_dict(
        train_path=str(CROSS_TRAIN),
        test_path=str(CROSS_TEST),
        target_name="vc",
        channels=("sonic", "rhob", "gr", "ai", "vp"),
        window_len=64,
        step=16,
        val_frac=0.1,
        residual_basis="dct",
    )
    meta = raw["meta"]
    cfg = _base_cfg(
        profile="cross_well_vc_direct_ub",
        p_input=int(meta["p_input"]),
        n_output=int(meta["n_output"]),
        n_train=int(meta["n_train"]),
        n_val=int(meta["n_val"]),
        n_test=int(meta["n_test"]),
    )
    data = {
        k: raw[k]
        for k in (
            "X_train",
            "X_val",
            "X_test",
            "Y_train",
            "Y_val",
            "Y_test",
            "Alpha_train",
            "Alpha_val",
            "Alpha_test",
            "Psi",
        )
    }
    return RuntimeCase(
        dataset="crosswell_vc_step16",
        data=data,
        cfg=cfg,
        measurement_ratios=(0.05, 0.10, 0.20),
        seeds=(7,),
    )


def build_f03_case() -> RuntimeCase:
    """Build the representative F03-4 GR-only runtime case."""
    if not F03_DATA.is_file():
        raise FileNotFoundError(str(F03_DATA))
    tab = rwf.load_f03_table(str(F03_DATA))
    n_rows = int(tab.n_rows)
    n_tr_rows = int(np.floor(0.6 * n_rows))
    n_va_rows = int(np.floor(0.2 * n_rows))
    n_te_rows = int(n_rows - n_tr_rows - n_va_rows)
    if min(n_tr_rows, n_va_rows, n_te_rows) < 64:
        raise ValueError(
            "Row split too small for F03 depth-first runtime case: "
            "train={} val={} test={}".format(n_tr_rows, n_va_rows, n_te_rows)
        )
    tr = rwf.F03Table(
        depth=tab.depth[:n_tr_rows].copy(),
        ac=tab.ac[:n_tr_rows].copy(),
        gr=tab.gr[:n_tr_rows].copy(),
        porosity=tab.porosity[:n_tr_rows].copy(),
    )
    va = rwf.F03Table(
        depth=tab.depth[n_tr_rows : n_tr_rows + n_va_rows].copy(),
        ac=tab.ac[n_tr_rows : n_tr_rows + n_va_rows].copy(),
        gr=tab.gr[n_tr_rows : n_tr_rows + n_va_rows].copy(),
        porosity=tab.porosity[n_tr_rows : n_tr_rows + n_va_rows].copy(),
    )
    te = rwf.F03Table(
        depth=tab.depth[n_tr_rows + n_va_rows :].copy(),
        ac=tab.ac[n_tr_rows + n_va_rows :].copy(),
        gr=tab.gr[n_tr_rows + n_va_rows :].copy(),
        porosity=tab.porosity[n_tr_rows + n_va_rows :].copy(),
    )
    x_tr, y_tr, _c1, _r1 = rwf.build_sliding_windows(tr, 64, 1, channels=("gr",))
    x_va, y_va, _c2, _r2 = rwf.build_sliding_windows(va, 64, 1, channels=("gr",))
    x_te, y_te, _c3, _r3 = rwf.build_sliding_windows(te, 64, 1, channels=("gr",))
    n_tr = int(x_tr.shape[0])
    n_va = int(x_va.shape[0])
    n_te = int(x_te.shape[0])
    x_all = np.concatenate([x_tr, x_va, x_te], axis=0)
    y_all = np.concatenate([y_tr, y_va, y_te], axis=0)
    sl_tr = slice(0, n_tr)
    sl_va = slice(n_tr, n_tr + n_va)
    sl_te = slice(n_tr + n_va, n_tr + n_va + n_te)
    data = rwf.build_direct_ub_data_dict(x_all, y_all, sl_tr, sl_va, sl_te, "dct")
    cfg = _base_cfg(
        profile="real_well_f03_direct_ub",
        p_input=64,
        n_output=64,
        n_train=n_tr,
        n_val=n_va,
        n_test=n_te,
    )
    return RuntimeCase(
        dataset="f03_gr_only",
        data=data,
        cfg=cfg,
        measurement_ratios=(0.20, 0.40, 0.60),
        seeds=(7,),
    )


def _append_row(
    rows: List[Dict[str, float | int | str]],
    dataset: str,
    method: str,
    seed: int,
    rho: float,
    n_train: int,
    n_val: int,
    n_test: int,
    elapsed_sec: float,
    rmse_mean: float,
) -> None:
    per_window = float(elapsed_sec) / float(max(n_test, 1))
    rows.append(
        {
            "dataset": dataset,
            "method": method,
            "seed": int(seed),
            "measurement_ratio": float(rho),
            "n_train": int(n_train),
            "n_val": int(n_val),
            "n_test": int(n_test),
            "wall_time_sec": float(elapsed_sec),
            "time_per_test_window_sec": per_window,
            "rmse_mean": float(rmse_mean),
        }
    )


def run_case(case: RuntimeCase) -> pd.DataFrame:
    """Run runtime measurements for one dataset case."""
    rows: List[Dict[str, float | int | str]] = []
    data = case.data
    cfg = case.cfg
    dub_cfg = dub.DirectUBTrainConfig()
    n_train = int(cfg.n_train)
    n_val = int(cfg.n_val)
    n_test = int(cfg.n_test)

    for seed in case.seeds:
        for rho in case.measurement_ratios:
            rng = np.random.default_rng(int(seed))
            m = max(4, int(round(float(rho) * int(cfg.n_output))))
            m_mat = build_measurement_matrix(m, int(cfg.n_output), cfg.measurement_kind, rng)
            b_train = dub.make_B(data["Y_train"], m_mat, cfg.measurement_noise_std, rng)
            b_val = dub.make_B(data["Y_val"], m_mat, cfg.measurement_noise_std, rng)
            b_test = dub.make_B(data["Y_test"], m_mat, cfg.measurement_noise_std, rng)
            xb_train = dub.concat_ub(data["X_train"], b_train)
            xb_val = dub.concat_ub(data["X_val"], b_val)
            xb_test = dub.concat_ub(data["X_test"], b_test)
            ub_scaler = dub.fit_scaler_ub(xb_train)

            start = time.perf_counter()
            baseline = MultiOutputMLP(
                hidden_layer_sizes=cfg.baseline_hidden,
                max_iter=cfg.baseline_max_iter,
                learning_rate_init=cfg.baseline_learning_rate_init,
                alpha=cfg.baseline_alpha,
                early_stopping=cfg.baseline_early_stopping,
                random_state=int(seed),
            )
            baseline.fit(data["X_train"], data["Y_train"])
            pred_ml = baseline.predict(data["X_test"])
            elapsed = time.perf_counter() - start
            _append_row(
                rows,
                case.dataset,
                "ml_only",
                seed,
                rho,
                n_train,
                n_val,
                n_test,
                elapsed,
                _rmse_mean(data["Y_test"], pred_ml),
            )

            start = time.perf_counter()
            _pred_val_mlp, pred_test_mlp = dub.fit_predict_mlp_concat(
                cfg,
                int(seed),
                xb_train,
                data["Y_train"],
                xb_val,
                data["Y_val"],
                xb_test,
                ub_scaler,
            )
            elapsed = time.perf_counter() - start
            _append_row(
                rows,
                case.dataset,
                "mlp_concat_ub",
                seed,
                rho,
                n_train,
                n_val,
                n_test,
                elapsed,
                _rmse_mean(data["Y_test"], pred_test_mlp),
            )

            start = time.perf_counter()
            _pred_val_pca, pred_test_pca, _best_r = dub.fit_predict_pca_regression_ub(
                cfg,
                int(seed),
                dub_cfg,
                xb_train,
                data["Y_train"],
                xb_val,
                data["Y_val"],
                xb_test,
                ub_scaler,
            )
            elapsed = time.perf_counter() - start
            _append_row(
                rows,
                case.dataset,
                "pca_regression_ub",
                seed,
                rho,
                n_train,
                n_val,
                n_test,
                elapsed,
                _rmse_mean(data["Y_test"], pred_test_pca),
            )

            start = time.perf_counter()
            _pred_val_ae, pred_test_ae = dub.fit_predict_ae_regression_ub(
                cfg,
                int(seed),
                dub_cfg,
                xb_train,
                data["Y_train"],
                xb_val,
                data["Y_val"],
                xb_test,
                ub_scaler,
            )
            elapsed = time.perf_counter() - start
            _append_row(
                rows,
                case.dataset,
                "ae_regression_ub",
                seed,
                rho,
                n_train,
                n_val,
                n_test,
                elapsed,
                _rmse_mean(data["Y_test"], pred_test_ae),
            )

            start = time.perf_counter()
            result = csgm.run_csgm_m2_experiment_dataframe(
                cfg=cfg,
                seed=int(seed),
                measurement_ratio=float(rho),
                X_train=data["X_train"],
                X_val=data["X_val"],
                X_test=data["X_test"],
                Y_train=data["Y_train"],
                Y_val=data["Y_val"],
                Y_test=data["Y_test"],
                Alpha_test=data["Alpha_test"],
                M=m_mat,
                B_val=b_val,
                B_test=b_test,
            )
            elapsed = time.perf_counter() - start
            method = "{}_prior_csgm".format(str(cfg.csgm_prior_type).strip().lower())
            csgm_rows = result.df[result.df["method"] == method]
            _append_row(
                rows,
                case.dataset,
                method,
                seed,
                rho,
                n_train,
                n_val,
                n_test,
                elapsed,
                float(csgm_rows["rmse"].mean()),
            )

    return pd.DataFrame(rows)


def _summarize(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate runtime by dataset, method, and measurement ratio."""
    out = (
        df.groupby(["dataset", "method", "measurement_ratio"], as_index=False)
        .agg(
            wall_time_sec_mean=("wall_time_sec", "mean"),
            wall_time_sec_std=("wall_time_sec", "std"),
            time_per_test_window_sec_mean=("time_per_test_window_sec", "mean"),
            rmse_mean=("rmse_mean", "mean"),
            n_train=("n_train", "first"),
            n_val=("n_val", "first"),
            n_test=("n_test", "first"),
            n_runs=("seed", "nunique"),
        )
    )
    refs = (
        out[out["method"] == "ae_regression_ub"][
            ["dataset", "measurement_ratio", "time_per_test_window_sec_mean", "rmse_mean"]
        ]
        .rename(
            columns={
                "time_per_test_window_sec_mean": "ae_time_per_test_window_sec",
                "rmse_mean": "ae_rmse_mean",
            }
        )
    )
    out = out.merge(refs, on=["dataset", "measurement_ratio"], how="left")
    out["relative_time_vs_ae"] = (
        out["time_per_test_window_sec_mean"] / out["ae_time_per_test_window_sec"]
    )
    out["rmse_delta_vs_ae"] = out["rmse_mean"] - out["ae_rmse_mean"]
    return out


def _plot_runtime_tradeoff(summary: pd.DataFrame) -> Path:
    labels = {
        "ml_only": "ML only",
        "mlp_concat_ub": "MLP [u,b]",
        "pca_regression_ub": "PCA [u,b]",
        "ae_regression_ub": "AE [u,b]",
        "ridge_prior_csgm": "CLP-CSGM Ridge",
    }
    datasets = list(summary["dataset"].drop_duplicates())
    fig, axes = plt.subplots(1, len(datasets), figsize=(6.0 * len(datasets), 4.4), squeeze=False)
    for ax, dataset in zip(axes[0], datasets):
        sub = summary[summary["dataset"] == dataset].copy()
        for method, part in sub.groupby("method"):
            ax.scatter(
                part["time_per_test_window_sec_mean"],
                part["rmse_mean"],
                label=labels.get(method, method),
                alpha=0.85,
            )
            for _, row in part.iterrows():
                ax.annotate(
                    "{:.2f}".format(float(row["measurement_ratio"])),
                    (float(row["time_per_test_window_sec_mean"]), float(row["rmse_mean"])),
                    fontsize=7,
                    xytext=(3, 3),
                    textcoords="offset points",
                )
        ax.set_xscale("log")
        ax.set_xlabel("Wall time per test window (s, log scale)")
        ax.set_ylabel("Mean RMSE")
        ax.set_title(dataset)
        ax.grid(True, alpha=0.25, which="both")
    axes[0, -1].legend(fontsize=8, loc="best")
    fig.suptitle("Computational cost versus RMSE", y=1.02)
    fig.tight_layout()
    out = FIG_DIR / "runtime_vs_rmse.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out


def _plot_relative_runtime(summary: pd.DataFrame) -> Path:
    order = ["ml_only", "mlp_concat_ub", "pca_regression_ub", "ae_regression_ub", "ridge_prior_csgm"]
    labels = ["ML only", "MLP [u,b]", "PCA [u,b]", "AE [u,b]", "CLP-CSGM Ridge"]
    datasets = list(summary["dataset"].drop_duplicates())
    fig, axes = plt.subplots(1, len(datasets), figsize=(6.0 * len(datasets), 4.2), squeeze=False)
    for ax, dataset in zip(axes[0], datasets):
        sub = summary[summary["dataset"] == dataset].copy()
        means = []
        for method in order:
            vals = sub.loc[sub["method"] == method, "relative_time_vs_ae"].dropna()
            means.append(float(vals.mean()) if not vals.empty else np.nan)
        ax.bar(labels, means)
        ax.axhline(1.0, color="0.3", linestyle="--", linewidth=1.0)
        ax.set_yscale("log")
        ax.set_ylabel("Relative time vs AE [u,b]")
        ax.set_title(dataset)
        ax.tick_params(axis="x", rotation=25)
        ax.grid(True, axis="y", alpha=0.25, which="both")
    fig.suptitle("Relative computational cost", y=1.02)
    fig.tight_layout()
    out = FIG_DIR / "runtime_relative_vs_ae.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out


def _write_latex_table(summary: pd.DataFrame) -> Path:
    """Write the compact runtime table used by the manuscript."""
    dataset_labels = {
        "crosswell_vc_step16": "Cross-well Vc",
        "f03_gr_only": "F03-4 GR-only",
    }
    method_labels = {
        "ml_only": "ML only",
        "mlp_concat_ub": "MLP \\([u,b]\\)",
        "pca_regression_ub": "PCA \\([u,b]\\)",
        "ae_regression_ub": "AE \\([u,b]\\)",
        "ridge_prior_csgm": "CLP-CSGM Ridge",
    }
    method_order = [
        "ml_only",
        "mlp_concat_ub",
        "pca_regression_ub",
        "ae_regression_ub",
        "ridge_prior_csgm",
    ]
    grouped = (
        summary.groupby(["dataset", "method"], as_index=False)
        .agg(
            time_per_window=("time_per_test_window_sec_mean", "mean"),
            relative_time=("relative_time_vs_ae", "mean"),
            rmse=("rmse_mean", "mean"),
        )
    )
    lines = [
        "\\begin{table}[htbp]",
        "  \\centering",
        "  \\caption{Representative computational-cost summary. Time is reported as",
        "  average wall-clock seconds per test window across the tested measurement",
        "  ratios. Relative time is normalized by AE \\([u,b]\\) within each dataset.}",
        "  \\label{tab:runtime_cost}",
        "  \\resizebox{\\linewidth}{!}{%",
        "  \\begin{tabular}{llccc}",
        "    \\toprule",
        "    Dataset & Method & Time/window (s) & Relative time vs. AE & Mean RMSE \\\\",
        "    \\midrule",
    ]
    first_dataset = True
    for dataset in ("crosswell_vc_step16", "f03_gr_only"):
        if not first_dataset:
            lines.append("    \\midrule")
        first_dataset = False
        for method in method_order:
            row = grouped[(grouped["dataset"] == dataset) & (grouped["method"] == method)]
            if row.empty:
                continue
            rec = row.iloc[0]
            lines.append(
                "    {} & {} & {:.5f} & {:.2f} & {:.5f} \\\\".format(
                    dataset_labels[dataset],
                    method_labels[method],
                    float(rec["time_per_window"]),
                    float(rec["relative_time"]),
                    float(rec["rmse"]),
                )
            )
    lines.extend(
        [
            "    \\bottomrule",
            "  \\end{tabular}%",
            "  }",
            "\\end{table}",
        ]
    )
    PAPER_RUNTIME_TABLE.write_text("\n".join(lines) + "\n", encoding="ascii")
    return PAPER_RUNTIME_TABLE


def _copy_for_paper(paths: Iterable[Path]) -> None:
    for path in paths:
        target = PAPER_FIG_DIR / path.name
        target.write_bytes(path.read_bytes())


def main() -> None:
    warnings.filterwarnings("ignore", category=ConvergenceWarning)
    _ensure_dirs()
    frames = [run_case(build_cross_case()), run_case(build_f03_case())]
    detailed = pd.concat(frames, ignore_index=True)
    summary = _summarize(detailed)
    detailed_path = TAB_DIR / "runtime_detailed.csv"
    summary_path = TAB_DIR / "runtime_summary.csv"
    paper_summary_path = PAPER_TAB_DIR / "runtime_summary.csv"
    detailed.to_csv(detailed_path, index=False)
    summary.to_csv(summary_path, index=False)
    summary.to_csv(paper_summary_path, index=False)
    paper_table_path = _write_latex_table(summary)
    fig_paths = [_plot_runtime_tradeoff(summary), _plot_relative_runtime(summary)]
    _copy_for_paper(fig_paths)
    for path in [detailed_path, summary_path, paper_summary_path, paper_table_path, *fig_paths]:
        print(path.relative_to(ROOT))


if __name__ == "__main__":
    main()
