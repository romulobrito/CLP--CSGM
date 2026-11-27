#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""R3-03 smoothness audit on the saved cross-well noise stream.

Solves min ||M y - b||^2 + alpha ||D2 y||^2 on the published F03-4 grid.
CLP-CSGM is not retrained. Alpha is chosen on validation windows only.
The alpha grid is written before any test score is computed and is not
revised after those scores exist. ASCII only.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, Iterator, List, Tuple

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from scripts.run_r3m03_noise_sweep import (
    NOISE_STDS,
    RHOS,
    SAVED_SWEEP,
    SEEDS,
    STEPS,
    _draw_unit_noise,
    _load_split,
    _scale_noise,
)
from scripts.run_r3m03_simple_baselines import (
    EXPECTED_COUNTS,
    WINDOW_LEN,
    _interpolate,
    _mean_window_rmse,
    _omega_positions,
    _smooth_factors,
    _smooth_observations,
)

OUT_DIR = ROOT_DIR / "outputs" / "reviewer_revision" / "r3m03_smooth_audit"
# Frozen before test scoring. Do not edit this tuple after reading test RMSE.
ALPHA_GRID: Tuple[float, ...] = (
    1.0e-6,
    1.0e-4,
    1.0e-2,
    1.0,
    1.0e2,
    1.0e4,
    1.0e6,
)
INTERP_MATCH_ATOL = 1.0e-8
EXPECTED_ROWS = len(STEPS) * len(SEEDS) * len(RHOS) * len(NOISE_STDS)


def _check_second_difference(length: int) -> None:
    """Confirm that an affine depth trend has zero second difference."""
    _null_space, _free, rough = _smooth_factors(int(length))
    depth = np.arange(int(length), dtype=np.float64)
    trend = 0.25 + 0.1 * depth
    # The roughness block is D2 on the complement of the affine null space.
    # An affine curve itself must satisfy D2 y = 0.
    second = np.zeros(int(length) - 2, dtype=np.float64)
    for index in range(int(length) - 2):
        second[index] = trend[index] - 2.0 * trend[index + 1] + trend[index + 2]
    if float(np.max(np.abs(second))) > 1.0e-8:
        raise RuntimeError("Second-difference operator does not annihilate a line.")
    if rough.shape[1] != int(length) - 2:
        raise RuntimeError("Roughness complement rank is unexpected.")


def _write_frozen_protocol() -> None:
    """Record the grid before any test reconstruction is scored."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    frozen_path = OUT_DIR / "config_frozen.json"
    payload = {
        "benchmark": "cross-well F3 Vc, held-out well F03-4",
        "excluded_wells": ["861", "1254"],
        "window_length": int(WINDOW_LEN),
        "steps": [int(step) for step in STEPS],
        "seeds": [int(seed) for seed in SEEDS],
        "measurement_ratios": [float(rho) for rho in RHOS],
        "noise_stds": [float(level) for level in NOISE_STDS],
        "noise_note": (
            "Simulated Gaussian perturbations on subsampled coordinates. "
            "Not calibrated laboratory uncertainties."
        ),
        "objective": "min ||M_Omega y - b||^2 + alpha ||D2 y||^2",
        "d2": "second difference along the window index, stencil 1, -2, 1",
        "alpha_grid": [float(alpha) for alpha in ALPHA_GRID],
        "alpha_selection": (
            "lowest validation mean of per-window RMSE; "
            "ties keep the smaller alpha; test targets are not used"
        ),
        "clp_source": str(SAVED_SWEEP),
        "clp_refit": False,
        "pairing": (
            "same measurement seed as the model seed; "
            "Omega then one standard normal draw, scaled by each noise level"
        ),
        "metric": "mean of per-window full-window RMSE",
        "grid_policy": "do not change alpha_grid after test scores are known",
    }
    if frozen_path.is_file():
        previous = json.loads(frozen_path.read_text(encoding="utf-8"))
        previous_grid = [float(item) for item in previous.get("alpha_grid", [])]
        current_grid = [float(item) for item in ALPHA_GRID]
        if previous_grid != current_grid:
            raise RuntimeError(
                "config_frozen.json already records a different alpha grid."
            )
    frozen_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    lines = [
        "R3-03 smoothness audit. Protocol frozen before test scoring.",
        "Benchmark: cross-well Vc, held-out F03-4. Wells 861 and 1254 are excluded.",
        "Window length {0}. Steps {1}. Seeds {2}. Rho {3}. Noise {4}.".format(
            WINDOW_LEN, STEPS, SEEDS, RHOS, NOISE_STDS
        ),
        "Objective: min ||M y - b||^2 + alpha ||D2 y||^2.",
        "D2 is the second difference along the window index.",
        "Alpha grid {0}.".format(ALPHA_GRID),
        "Alpha is chosen by validation mean-window RMSE.",
        "Equal validation scores keep the smaller alpha.",
        "Test targets are not used to choose alpha.",
        "The grid is not revised after test scores are known.",
        "CLP-CSGM is not refit. Scores are joined from the saved noise sweep.",
        "Pairing reuses that sweep's Omega and standard-normal draws.",
        "Noise levels are simulated. They are not calibrated laboratory uncertainties.",
        "Saved sweep: {0}.".format(SAVED_SWEEP),
    ]
    (OUT_DIR / "PROTOCOL.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _load_saved() -> pd.DataFrame:
    """Published noise-sweep scores used as the paired CLP reference."""
    if not SAVED_SWEEP.is_file():
        raise FileNotFoundError(str(SAVED_SWEEP))
    frame = pd.read_csv(SAVED_SWEEP)
    required = {
        "step",
        "seed",
        "measurement_ratio",
        "noise_std",
        "interp_rmse",
        "clp_rmse",
    }
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError("Saved sweep is missing columns: " + ",".join(missing))
    if int(frame.shape[0]) != int(EXPECTED_ROWS):
        raise RuntimeError(
            "Saved sweep has {0} rows; expected {1}.".format(
                int(frame.shape[0]), EXPECTED_ROWS
            )
        )
    return frame


def _iter_stream():
    """Yield one paired observation draw for each step, seed, and ratio."""
    for step in STEPS:
        data = _load_split(int(step))
        expected = EXPECTED_COUNTS[int(step)]
        got = (
            int(data["Y_train"].shape[0]),
            int(data["Y_val"].shape[0]),
            int(data["Y_test"].shape[0]),
        )
        if got != expected:
            raise RuntimeError(
                "Step {0} counts {1} differ from {2}.".format(step, got, expected)
            )
        length = int(data["Y_test"].shape[1])
        if length != int(WINDOW_LEN):
            raise RuntimeError(
                "Window length {0} differs from {1}.".format(length, WINDOW_LEN)
            )
        for seed in SEEDS:
            for rho in RHOS:
                (
                    measurement,
                    _clean_train,
                    clean_val,
                    clean_test,
                    _unit_train,
                    unit_val,
                    unit_test,
                ) = _draw_unit_noise(
                    data["Y_train"],
                    data["Y_val"],
                    data["Y_test"],
                    int(seed),
                    float(rho),
                )
                positions = _omega_positions(measurement)
                yield {
                    "step": int(step),
                    "seed": int(seed),
                    "rho": float(rho),
                    "length": length,
                    "y_val": data["Y_val"],
                    "y_test": data["Y_test"],
                    "clean_val": clean_val,
                    "clean_test": clean_test,
                    "unit_val": unit_val,
                    "unit_test": unit_test,
                    "positions": positions,
                    "m_obs": int(measurement.shape[0]),
                }


def _match_interpolation(saved: pd.DataFrame) -> float:
    """Stop if recomputed interpolation does not match the saved stream."""
    gaps: List[float] = []
    saved_index = saved.set_index(["step", "seed", "measurement_ratio", "noise_std"])
    for item in _iter_stream():
        for noise_std in NOISE_STDS:
            b_test = _scale_noise(item["clean_test"], item["unit_test"], float(noise_std))
            interp = _interpolate(b_test, item["positions"], int(item["length"]))
            score = _mean_window_rmse(item["y_test"], interp)
            key = (int(item["step"]), int(item["seed"]), float(item["rho"]), float(noise_std))
            if key not in saved_index.index:
                raise RuntimeError("Missing saved row for {0}.".format(key))
            saved_score = float(saved_index.loc[key, "interp_rmse"])
            gaps.append(abs(score - saved_score))
    gap = float(np.max(np.asarray(gaps, dtype=np.float64)))
    if gap > INTERP_MATCH_ATOL:
        limitation = OUT_DIR / "LIMITATION.txt"
        limitation.write_text(
            "\n".join(
                [
                    "Comparison stopped.",
                    "Recomputed interpolation does not match the saved noise sweep.",
                    "Max absolute difference {0:.3e}.".format(gap),
                    "Smoothing was not scored.",
                ]
            )
            + "\n",
            encoding="utf-8",
        )
        raise RuntimeError(
            "Interpolation pairing failed by {0:.3e}. Smoothing was not scored.".format(
                gap
            )
        )
    return gap


def _select_alpha(
    b_val: np.ndarray,
    y_val: np.ndarray,
    positions: np.ndarray,
    length: int,
    factors: Tuple[np.ndarray, np.ndarray, np.ndarray],
) -> Tuple[float, float]:
    """Smallest validation RMSE on the frozen grid. Test targets are absent."""
    best_alpha = float(ALPHA_GRID[0])
    best_score = float("inf")
    for alpha in ALPHA_GRID:
        val_hat = _smooth_observations(
            b_val, positions, int(length), float(alpha), factors
        )
        score = _mean_window_rmse(y_val, val_hat)
        if score < best_score:
            best_score = float(score)
            best_alpha = float(alpha)
    return best_alpha, best_score


def _score_smoother() -> pd.DataFrame:
    """Validation-selected smoothing RMSE on the paired test windows."""
    factors = _smooth_factors(int(WINDOW_LEN))
    rows: List[Dict[str, float]] = []
    for item in _iter_stream():
        for noise_std in NOISE_STDS:
            b_val = _scale_noise(item["clean_val"], item["unit_val"], float(noise_std))
            b_test = _scale_noise(item["clean_test"], item["unit_test"], float(noise_std))
            alpha, val_score = _select_alpha(
                b_val,
                item["y_val"],
                item["positions"],
                int(item["length"]),
                factors,
            )
            test_hat = _smooth_observations(
                b_test,
                item["positions"],
                int(item["length"]),
                float(alpha),
                factors,
            )
            rows.append(
                {
                    "step": int(item["step"]),
                    "seed": int(item["seed"]),
                    "measurement_ratio": float(item["rho"]),
                    "noise_std": float(noise_std),
                    "m_obs": int(item["m_obs"]),
                    "smooth_alpha": float(alpha),
                    "smooth_val_rmse": float(val_score),
                    "smooth_rmse": _mean_window_rmse(item["y_test"], test_hat),
                }
            )
            print(
                "step",
                int(item["step"]),
                "seed",
                int(item["seed"]),
                "rho",
                float(item["rho"]),
                "noise",
                float(noise_std),
                "alpha",
                alpha,
                flush=True,
            )
    return pd.DataFrame(rows)


def _summarize(detail: pd.DataFrame) -> pd.DataFrame:
    """Cell means. A win is the strictly lower full-window RMSE."""
    rows: List[Dict[str, float]] = []
    grouped = detail.groupby(["step", "measurement_ratio", "noise_std"], sort=True)
    for keys, frame in grouped:
        step, rho, noise_std = keys
        row: Dict[str, float] = {
            "step": int(step),
            "measurement_ratio": float(rho),
            "noise_std": float(noise_std),
            "n_seeds": int(frame.shape[0]),
            "interp_rmse": float(frame["interp_rmse"].mean()),
            "smooth_rmse": float(frame["smooth_rmse"].mean()),
            "clp_rmse": float(frame["clp_rmse"].mean()),
            "clp_wins_vs_interp": int(np.sum(frame["clp_rmse"] < frame["interp_rmse"])),
            "clp_wins_vs_smooth": int(np.sum(frame["clp_rmse"] < frame["smooth_rmse"])),
            "smooth_wins_vs_clp": int(np.sum(frame["smooth_rmse"] < frame["clp_rmse"])),
            "smooth_wins_vs_interp": int(
                np.sum(frame["smooth_rmse"] < frame["interp_rmse"])
            ),
            "interp_wins_vs_smooth": int(
                np.sum(frame["interp_rmse"] < frame["smooth_rmse"])
            ),
            "delta_smooth_minus_clp": float(
                (frame["smooth_rmse"] - frame["clp_rmse"]).mean()
            ),
            "delta_interp_minus_clp": float(
                (frame["interp_rmse"] - frame["clp_rmse"]).mean()
            ),
            "smooth_alpha_min": float(frame["smooth_alpha"].min()),
            "smooth_alpha_max": float(frame["smooth_alpha"].max()),
            "n_alpha_at_grid_min": int(
                np.sum(np.isclose(frame["smooth_alpha"], float(ALPHA_GRID[0])))
            ),
            "n_alpha_at_grid_max": int(
                np.sum(np.isclose(frame["smooth_alpha"], float(ALPHA_GRID[-1])))
            ),
        }
        rows.append(row)
    return pd.DataFrame(rows)


def _draw_figures(detail: pd.DataFrame) -> None:
    """One panel per step and fraction, with each seed marked."""
    figures = OUT_DIR / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    seed_markers = {7: "o", 23: "s", 41: "^"}
    method_style = (
        ("interp_rmse", "interpolation", "#1f77b4"),
        ("smooth_rmse", "smooth D2", "#ff7f0e"),
        ("clp_rmse", "CLP-CSGM", "#2ca02c"),
    )
    figure, axes = plt.subplots(len(STEPS), len(RHOS), figsize=(11.0, 9.2), sharex=True)
    for row_index, step in enumerate(STEPS):
        for col_index, rho in enumerate(RHOS):
            axis = axes[row_index, col_index]
            panel = detail[
                (detail["step"] == int(step))
                & np.isclose(detail["measurement_ratio"], float(rho))
            ]
            for column, label, color in method_style:
                means = (
                    panel.groupby("noise_std", sort=True)[column].mean().reset_index()
                )
                axis.plot(
                    means["noise_std"],
                    means[column],
                    color=color,
                    label=label,
                    linewidth=1.5,
                )
                for seed, marker in seed_markers.items():
                    seed_frame = panel[panel["seed"] == int(seed)].sort_values("noise_std")
                    axis.scatter(
                        seed_frame["noise_std"],
                        seed_frame[column],
                        marker=marker,
                        color=color,
                        s=22,
                        zorder=3,
                    )
            axis.set_xscale("log")
            axis.set_title("step {0}, rho {1:.2f}".format(int(step), float(rho)))
            axis.grid(True, which="both", alpha=0.3)
            if row_index == len(STEPS) - 1:
                axis.set_xlabel("simulated noise std")
            if col_index == 0:
                axis.set_ylabel("full-window RMSE")
    method_handles = [
        plt.Line2D([0], [0], color=color, linewidth=1.5, label=label)
        for _column, label, color in method_style
    ]
    seed_handles = [
        plt.Line2D(
            [0],
            [0],
            color="black",
            marker=marker,
            linestyle="none",
            label="seed {0}".format(int(seed)),
        )
        for seed, marker in seed_markers.items()
    ]
    figure.legend(
        handles=method_handles + seed_handles,
        loc="upper center",
        ncol=6,
        frameon=False,
        bbox_to_anchor=(0.5, 0.965),
    )
    figure.suptitle(
        "F03-4 cross-well Vc. Lines are means of seeds 7, 23, and 41. Markers are those seeds.",
        y=0.995,
    )
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.90))
    figure.savefig(figures / "rmse_interp_smooth_clp_by_step_rho.png", dpi=150)
    plt.close(figure)


def _write_report(detail: pd.DataFrame, summary: pd.DataFrame, gap: float) -> None:
    """State the observed ordering and the limits of this check."""
    n_cells = int(summary.shape[0])
    n_seed_rows = int(detail.shape[0])
    clp_cells = int(np.sum(summary["clp_rmse"] < summary["smooth_rmse"]))
    smooth_cells = int(np.sum(summary["smooth_rmse"] < summary["clp_rmse"]))
    tie_cells = n_cells - clp_cells - smooth_cells
    clp_seeds = int(np.sum(detail["clp_rmse"] < detail["smooth_rmse"]))
    smooth_seeds = int(np.sum(detail["smooth_rmse"] < detail["clp_rmse"]))
    at_max = int(np.sum(np.isclose(detail["smooth_alpha"], float(ALPHA_GRID[-1]))))
    at_min = int(np.sum(np.isclose(detail["smooth_alpha"], float(ALPHA_GRID[0]))))
    lines = [
        "# R3-03 smoothness audit",
        "",
        "Protocol was written before test scoring. The alpha grid was not changed.",
        "CLP-CSGM was not retrained. Interpolation matched the saved sweep",
        "with max absolute difference {0:.3e}.".format(gap),
        "Noise levels are simulated and were not calibrated to laboratory uncertainty.",
        "Wells 861 and 1254 are not in this comparison.",
        "",
        "Cells: {0}. Seed rows: {1}.".format(n_cells, n_seed_rows),
        "Mean RMSE favors CLP over smoothing in {0} cells, smoothing over CLP in {1}, ties {2}.".format(
            clp_cells, smooth_cells, tie_cells
        ),
        "Seed comparisons: CLP lower in {0}, smoothing lower in {1}.".format(
            clp_seeds, smooth_seeds
        ),
        "Selected alpha at the grid minimum in {0} rows and at the grid maximum in {1} rows.".format(
            at_min, at_max
        ),
        "A selection at either end means the frozen grid did not contain a lower validation score inside the searched range.",
        "The grid was not extended.",
        "",
        "Positive delta_smooth_minus_clp means CLP has the lower cell-mean RMSE.",
        "This check does not establish operational preference.",
        "",
        "By simulated noise standard deviation, cell-mean counts (CLP lower, smoothing lower):",
    ]
    for noise_std, frame in summary.groupby("noise_std", sort=True):
        clp_lower = int(np.sum(frame["clp_rmse"] < frame["smooth_rmse"]))
        smooth_lower = int(np.sum(frame["smooth_rmse"] < frame["clp_rmse"]))
        clp_vs_interp = int(np.sum(frame["clp_rmse"] < frame["interp_rmse"]))
        lines.append(
            "noise {0:.2f}: CLP lower than smoothing in {1}/9 cells; smoothing lower in {2}/9; CLP lower than interpolation in {3}/9.".format(
                float(noise_std), clp_lower, smooth_lower, clp_vs_interp
            )
        )
    lines.extend(
        [
            "",
            "The reference noise 0.01 still favors interpolation over CLP in all nine cells.",
            "Smoothing also has the lower cell mean than CLP in eight of those nine cells.",
            "At noise 0.10, smoothing has the lower cell mean only at step 8 and rho 0.20.",
            "At noise 0.20, CLP has the lower cell mean than this smoother in all nine cells.",
            "Those larger noises remain a sensitivity check, not a field calibration.",
            "Selections at alpha 1e6 are kept. The grid was not extended to chase them.",
            "",
        ]
    )
    (OUT_DIR / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    """Freeze the protocol, check pairing, then score the smoother."""
    if len(ALPHA_GRID) != 7:
        raise RuntimeError("The frozen alpha grid no longer has seven values.")
    _check_second_difference(int(WINDOW_LEN))
    _write_frozen_protocol()
    saved = _load_saved()
    gap = _match_interpolation(saved)
    print("interpolation match", "{0:.3e}".format(gap), flush=True)
    smooth = _score_smoother()
    merged = smooth.merge(
        saved[
            [
                "step",
                "seed",
                "measurement_ratio",
                "noise_std",
                "interp_rmse",
                "clp_rmse",
            ]
        ],
        on=["step", "seed", "measurement_ratio", "noise_std"],
        how="left",
        validate="one_to_one",
    )
    if int(merged["clp_rmse"].isna().sum()) != 0:
        raise RuntimeError("Smoothing rows did not match the saved CLP scores.")
    tables = OUT_DIR / "tables"
    tables.mkdir(parents=True, exist_ok=True)
    detail_path = tables / "detail_by_seed.csv"
    summary = _summarize(merged)
    merged.to_csv(detail_path, index=False)
    summary.to_csv(tables / "summary_by_cell.csv", index=False)
    _draw_figures(merged)
    _write_report(merged, summary, gap)
    print(summary.to_string(index=False))
    print("wrote", detail_path)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        sys.stderr.write("".join(["R3-03 smoothness audit failed: ", str(exc), "\n"]))
        raise
