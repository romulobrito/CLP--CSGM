#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Build the controlled R1-M06 Ridge-versus-MLP prior tables.

Reads the published Ridge runs and the controlled MLP runs only.
The older MLP runs that added gamma ray, or that split windows before
depth rows, are not inputs. There is no fallback to those directories.

ASCII-only source.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / "paper_clp_csgm"
FIG_DIR = PAPER / "figures"
TAB_DIR = PAPER / "tables"
CONTROLLED = ROOT / "outputs" / "reviewer_revision" / "r1m06_prior_class_controlled"

RIDGE_CROSS = {
    8: ROOT / "outputs" / "cross_well_vc" / "direct_ub" / "runs" / "crosswell_step08_clp_csgm_ablation_ridge",
    16: ROOT / "outputs" / "cross_well_vc" / "direct_ub" / "runs" / "crosswell_step16_clp_csgm_ablation_ridge",
    32: ROOT / "outputs" / "cross_well_vc" / "direct_ub" / "runs" / "crosswell_step32_clp_csgm_ablation_ridge",
}
MLP_CROSS = {
    8: CONTROLLED / "crosswell" / "runs" / "step08_mlp",
    16: CONTROLLED / "crosswell" / "runs" / "step16_mlp",
    32: CONTROLLED / "crosswell" / "runs" / "step32_mlp",
}
F03_RIDGE = (
    ROOT / "outputs" / "real_well_f03" / "direct_ub" / "runs" / "f03_full_gr_only_clp_csgm_ridge"
)
F03_MLP = CONTROLLED / "f03_gr_only" / "runs" / "mlp_depth_first"

FORBIDDEN_DIR_NAMES = (
    "crosswell_step08_clp_csgm_mlp_prior_full",
    "crosswell_step16_clp_csgm_mlp_prior_full",
    "crosswell_step32_clp_csgm_mlp_prior_full",
    "f03_gr_only_clp_csgm_mlp_prior_full",
)

CROSS_CHANNELS = ("sonic", "rhob", "ai", "vp")
CROSS_WELLS = ("F02-1", "F03-2", "F06-1")
CROSS_TEST = ("F03-4",)
CROSS_COUNTS = {8: (104, 11, 97), 16: (52, 6, 49), 32: (27, 3, 25)}
CROSS_RHOS = (0.05, 0.10, 0.20)
F03_COUNTS = (798, 224, 224)
F03_RHOS = (0.20, 0.30, 0.40, 0.50, 0.60)
SEEDS = (7, 23, 41)
SUMMARY_COLUMNS = (
    "measurement_ratio",
    "method",
    "rmse_mean",
    "rmse_std_across_seeds",
    "rmse_sem",
    "mae_mean",
    "mae_std_across_seeds",
    "relative_l2_mean",
    "relative_l2_std_across_seeds",
    "support_f1_mean",
    "support_f1_std_across_seeds",
    "n_seeds",
    "n_test_samples_per_run",
    "rmse_ci95_half",
    "method_label",
)


def _reject_forbidden(path: Path) -> None:
    """Stop if a path points at an uncontrolled historical MLP run."""
    if path.name in FORBIDDEN_DIR_NAMES:
        raise RuntimeError("refusing uncontrolled MLP run: {}".format(path))


def _require_dir(path: Path) -> None:
    """Require a controlled or published Ridge run directory."""
    _reject_forbidden(path)
    if not path.is_dir():
        raise FileNotFoundError("missing controlled run: {}".format(path))
    for name in ("config.json", "DATASET_MANIFEST.txt"):
        if not (path / name).is_file():
            raise FileNotFoundError("missing {} in {}".format(name, path))
    if not (path / "tables" / "summary.csv").is_file():
        raise FileNotFoundError("missing summary.csv in {}".format(path))
    if not (path / "tables" / "summary_by_seed.csv").is_file():
        raise FileNotFoundError("missing summary_by_seed.csv in {}".format(path))


def _load_config(path: Path) -> dict:
    """Load one run config."""
    _require_dir(path)
    return json.loads((path / "config.json").read_text(encoding="utf-8"))


def _channels(values: Sequence[str]) -> Tuple[str, ...]:
    """Normalize channel names."""
    return tuple(str(item).strip().lower() for item in values)


def _close_set(values: Sequence[float], expected: Sequence[float]) -> None:
    """Require the same measurement ratios, independent of order."""
    left = sorted(float(item) for item in values)
    right = sorted(float(item) for item in expected)
    if len(left) != len(right):
        raise RuntimeError("measurement ratios differ: {} vs {}".format(left, right))
    for got, want in zip(left, right):
        if abs(got - want) > 1e-12:
            raise RuntimeError("measurement ratios differ: {} vs {}".format(left, right))


def _pair_keys(path: Path, method: str) -> set:
    """Return the seed and ratio keys present for one method."""
    frame = pd.read_csv(path / "tables" / "summary_by_seed.csv")
    sub = frame[frame["method"] == method]
    if sub.empty:
        raise RuntimeError("method {} missing in {}".format(method, path))
    keys = set()
    for row in sub.itertuples(index=False):
        keys.add((int(row.seed), round(float(row.measurement_ratio), 10)))
    return keys


def _check_cross_pair(step: int, ridge_dir: Path, mlp_dir: Path) -> None:
    """Require the same cross-well protocol for Ridge and MLP."""
    ridge = _load_config(ridge_dir)
    mlp = _load_config(mlp_dir)
    ridge_meta = ridge["cross_well_vc"]
    mlp_meta = mlp["cross_well_vc"]
    ridge_channels = _channels(ridge_meta["channels"])
    mlp_channels = _channels(mlp_meta["channels"])
    if ridge_channels != CROSS_CHANNELS or mlp_channels != CROSS_CHANNELS:
        raise RuntimeError("cross-well channels differ from sonic,rhob,ai,vp")
    if "gr" in ridge_channels or "gr" in mlp_channels:
        raise RuntimeError("cross-well comparison must exclude gamma ray")
    if ridge_channels != mlp_channels:
        raise RuntimeError("Ridge and MLP auxiliary channels differ")
    if tuple(ridge_meta["train_wells"]) != CROSS_WELLS or tuple(mlp_meta["train_wells"]) != CROSS_WELLS:
        raise RuntimeError("development wells differ")
    if tuple(ridge_meta["test_wells"]) != CROSS_TEST or tuple(mlp_meta["test_wells"]) != CROSS_TEST:
        raise RuntimeError("test well differs")
    if Path(ridge_meta["test_path"]).name != Path(mlp_meta["test_path"]).name:
        raise RuntimeError("test files differ")
    counts = (
        int(ridge_meta["n_train"]),
        int(ridge_meta["n_val"]),
        int(ridge_meta["n_test"]),
    )
    mlp_counts = (
        int(mlp_meta["n_train"]),
        int(mlp_meta["n_val"]),
        int(mlp_meta["n_test"]),
    )
    if counts != CROSS_COUNTS[step] or mlp_counts != counts:
        raise RuntimeError("window counts differ at step {}: {} vs {}".format(step, counts, mlp_counts))
    if list(ridge["seeds"]) != list(SEEDS) or list(mlp["seeds"]) != list(SEEDS):
        raise RuntimeError("seeds differ at step {}".format(step))
    _close_set(ridge["measurement_ratios"], CROSS_RHOS)
    _close_set(mlp["measurement_ratios"], CROSS_RHOS)
    if abs(float(ridge["measurement_noise_std"]) - 0.01) > 1e-12:
        raise RuntimeError("Ridge cross-well noise is not 0.01")
    if abs(float(mlp["measurement_noise_std"]) - float(ridge["measurement_noise_std"])) > 1e-12:
        raise RuntimeError("measurement noise differs")
    if str(ridge.get("csgm_prior_type")) != "ridge" or str(mlp.get("csgm_prior_type")) != "mlp":
        raise RuntimeError("prior classes are not Ridge versus MLP")
    ridge_keys = _pair_keys(ridge_dir, "ridge_prior_csgm")
    mlp_keys = _pair_keys(mlp_dir, "mlp_prior_csgm")
    if ridge_keys != mlp_keys:
        raise RuntimeError("step/rho/seed pairs differ at step {}".format(step))
    expected = {(seed, round(rho, 10)) for seed in SEEDS for rho in CROSS_RHOS}
    if ridge_keys != expected:
        raise RuntimeError("cross-well pairs are not the 3 by 3 seed grid")


def _check_f03_pair(ridge_dir: Path, mlp_dir: Path) -> None:
    """Require the same depth-first GR-only protocol."""
    ridge = _load_config(ridge_dir)["real_well_f03"]
    mlp = _load_config(mlp_dir)["real_well_f03"]
    if _channels(ridge["u_channels"]) != ("gr",) or _channels(mlp["u_channels"]) != ("gr",):
        raise RuntimeError("F03 auxiliary channel must be GR only for both priors")
    if not bool(ridge["split_before_windowing"]) or not bool(mlp["split_before_windowing"]):
        raise RuntimeError("F03 split must be depth-first for both priors")
    if Path(ridge["data_path"]).name != Path(mlp["data_path"]).name:
        raise RuntimeError("F03 data files differ")
    ridge_counts = (int(ridge["n_train"]), int(ridge["n_val"]), int(ridge["n_test"]))
    mlp_counts = (int(mlp["n_train"]), int(mlp["n_val"]), int(mlp["n_test"]))
    if ridge_counts != F03_COUNTS or mlp_counts != ridge_counts:
        raise RuntimeError("F03 window counts differ: {} vs {}".format(ridge_counts, mlp_counts))
    ridge_cfg = _load_config(ridge_dir)
    mlp_cfg = _load_config(mlp_dir)
    if list(ridge_cfg["seeds"]) != list(SEEDS) or list(mlp_cfg["seeds"]) != list(SEEDS):
        raise RuntimeError("F03 seeds differ")
    _close_set(ridge_cfg["measurement_ratios"], F03_RHOS)
    _close_set(mlp_cfg["measurement_ratios"], F03_RHOS)
    if abs(float(ridge_cfg["measurement_noise_std"]) - 0.02) > 1e-12:
        raise RuntimeError("F03 Ridge noise is not 0.02")
    if abs(float(mlp_cfg["measurement_noise_std"]) - 0.02) > 1e-12:
        raise RuntimeError("F03 MLP noise is not 0.02")
    ridge_keys = _pair_keys(ridge_dir, "ridge_prior_csgm")
    mlp_keys = _pair_keys(mlp_dir, "mlp_prior_csgm")
    if ridge_keys != mlp_keys:
        raise RuntimeError("F03 step/rho/seed pairs differ")
    expected = {(seed, round(rho, 10)) for seed in SEEDS for rho in F03_RHOS}
    if ridge_keys != expected:
        raise RuntimeError("F03 pairs are not the five-ratio seed grid")


def _read_method(path: Path, method: str) -> pd.DataFrame:
    """Read one method from summary.csv and require matching test counts later."""
    frame = pd.read_csv(path / "tables" / "summary.csv")
    missing = [name for name in SUMMARY_COLUMNS if name not in frame.columns]
    if missing:
        raise RuntimeError("missing summary columns in {}: {}".format(path, missing))
    sub = frame[frame["method"] == method][list(SUMMARY_COLUMNS)].copy()
    if sub.empty:
        raise RuntimeError("method {} missing in summary {}".format(method, path))
    return sub


def _matched_counts(left: pd.DataFrame, right: pd.DataFrame, keys: List[str]) -> None:
    """Require the same test-window count inside every compared cell."""
    merged = left[keys + ["n_test_samples_per_run"]].merge(
        right[keys + ["n_test_samples_per_run"]],
        on=keys,
        suffixes=("_ridge", "_mlp"),
    )
    if len(merged) != len(left) or len(merged) != len(right):
        raise RuntimeError("Ridge and MLP cells do not match")
    mismatch = merged["n_test_samples_per_run_ridge"] != merged["n_test_samples_per_run_mlp"]
    if bool(mismatch.any()):
        raise RuntimeError("test-window counts differ between Ridge and MLP")


def _write_latex(frame: pd.DataFrame, path: Path) -> None:
    """Write a compact LaTeX table fragment."""
    text = frame.to_latex(index=False, escape=False, float_format="{:.4f}".format)
    path.write_text(text, encoding="utf-8")


def _gain(wide: pd.DataFrame) -> pd.DataFrame:
    """Signed Ridge gain. Negative values favor MLP."""
    wide = wide.copy()
    wide["ridge_gain_vs_mlp_pct"] = (
        100.0 * (wide["mlp_prior_csgm"] - wide["ridge_prior_csgm"]) / wide["mlp_prior_csgm"]
    )
    return wide


def build_controlled_prior_class_assets() -> None:
    """Write the seven published prior-class artifacts from controlled runs."""
    for step in (8, 16, 32):
        _check_cross_pair(step, RIDGE_CROSS[step], MLP_CROSS[step])
    _check_f03_pair(F03_RIDGE, F03_MLP)

    cross_rows: List[pd.DataFrame] = []
    for step in (8, 16, 32):
        ridge = _read_method(RIDGE_CROSS[step], "ridge_prior_csgm")
        mlp = _read_method(MLP_CROSS[step], "mlp_prior_csgm")
        _matched_counts(ridge, mlp, ["measurement_ratio"])
        ridge.insert(0, "step", int(step))
        mlp.insert(0, "step", int(step))
        cross_rows.extend([ridge, mlp])
    cross_long = pd.concat(cross_rows, ignore_index=True)
    cross_long.to_csv(TAB_DIR / "ablation_prior_class_crosswell_summary.csv", index=False)
    cross_wide = (
        cross_long.pivot_table(
            index=["step", "measurement_ratio"],
            columns="method",
            values="rmse_mean",
        )
        .reset_index()
        .sort_values(["step", "measurement_ratio"])
    )
    cross_wide = _gain(cross_wide)
    if int((cross_wide["ridge_gain_vs_mlp_pct"] < 0).sum()) != 7:
        raise RuntimeError("controlled cross-well table is not seven MLP cells")
    if int((cross_wide["ridge_gain_vs_mlp_pct"] > 0).sum()) != 2:
        raise RuntimeError("controlled cross-well table is not two Ridge cells")
    cross_wide.to_csv(TAB_DIR / "ablation_prior_class_crosswell_wide.csv", index=False)
    cross_table = cross_wide.copy()
    cross_table.columns = ["Step", "$\\rho$", "CLP-CSGM MLP", "CLP-CSGM Ridge", "Ridge gain (\\%)"]
    _write_latex(cross_table, TAB_DIR / "ablation_prior_class_crosswell_wide.tex")

    f03_ridge = _read_method(F03_RIDGE, "ridge_prior_csgm")
    f03_mlp = _read_method(F03_MLP, "mlp_prior_csgm")
    _matched_counts(f03_ridge, f03_mlp, ["measurement_ratio"])
    if set(int(value) for value in f03_ridge["n_test_samples_per_run"]) != {224}:
        raise RuntimeError("F03 Ridge test windows are not 224")
    if set(int(value) for value in f03_mlp["n_test_samples_per_run"]) != {224}:
        raise RuntimeError("F03 MLP test windows are not 224")
    f03_long = pd.concat([f03_ridge, f03_mlp], ignore_index=True)
    f03_long.to_csv(TAB_DIR / "ablation_prior_class_f03_summary.csv", index=False)
    f03_wide = (
        f03_long.pivot_table(index=["measurement_ratio"], columns="method", values="rmse_mean")
        .reset_index()
        .sort_values("measurement_ratio")
    )
    f03_wide = _gain(f03_wide)
    if not bool((f03_wide["ridge_gain_vs_mlp_pct"] > 0).all()):
        raise RuntimeError("controlled F03 table does not favor Ridge in every row")
    f03_wide.to_csv(TAB_DIR / "ablation_prior_class_f03_wide.csv", index=False)
    f03_table = f03_wide.copy()
    f03_table.columns = ["$\\rho$", "CLP-CSGM MLP", "CLP-CSGM Ridge", "Ridge gain (\\%)"]
    _write_latex(f03_table, TAB_DIR / "ablation_prior_class_f03_wide.tex")

    _write_figure(cross_long, f03_long)


def _write_figure(cross_long: pd.DataFrame, f03_long: pd.DataFrame) -> None:
    """Redraw the prior-class sensitivity figure."""
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.0))
    for method, label, color in [
        ("ridge_prior_csgm", "CLP-CSGM Ridge", "#d62728"),
        ("mlp_prior_csgm", "CLP-CSGM MLP", "#8c564b"),
    ]:
        for step in [8, 16, 32]:
            sub = cross_long[
                (cross_long["step"] == step) & (cross_long["method"] == method)
            ].sort_values("measurement_ratio")
            axes[0].plot(
                sub["measurement_ratio"],
                sub["rmse_mean"],
                marker="o",
                linestyle="-" if method == "ridge_prior_csgm" else "--",
                color=color,
                alpha=0.35 + 0.2 * ([8, 16, 32].index(step)),
                label="{} step {}".format(label, step),
            )
    axes[0].set_title("Cross-well Vc")
    axes[0].set_xlabel("Measurement ratio")
    axes[0].set_ylabel("RMSE")
    axes[0].grid(alpha=0.25)
    axes[0].legend(fontsize=7, ncol=2)
    for method, label, color in [
        ("ridge_prior_csgm", "CLP-CSGM Ridge", "#d62728"),
        ("mlp_prior_csgm", "CLP-CSGM MLP", "#8c564b"),
    ]:
        sub = f03_long[f03_long["method"] == method].sort_values("measurement_ratio")
        axes[1].plot(
            sub["measurement_ratio"],
            sub["rmse_mean"],
            marker="o",
            linestyle="-" if method == "ridge_prior_csgm" else "--",
            color=color,
            label=label,
        )
    axes[1].set_title("F03 GR-only")
    axes[1].set_xlabel("Measurement ratio")
    axes[1].grid(alpha=0.25)
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "ablation_prior_class_ridge_vs_mlp.png", dpi=200)
    plt.close(fig)


def main() -> None:
    """Generate only the controlled prior-class assets."""
    build_controlled_prior_class_assets()
    print("controlled prior-class assets written")


if __name__ == "__main__":
    main()
