#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Compare alpha=1e6 smoothing with the affine least-squares limit.

The frozen R3-03 smoother is not refit and its tables are not overwritten.
The affine fit has no penalty weight. It is the second-difference limit as
alpha tends to infinity. ASCII only.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from scripts.run_r3m03_noise_sweep import NOISE_STDS, _scale_noise
from scripts.run_r3m03_simple_baselines import (
    WINDOW_LEN,
    _mean_window_rmse,
    _smooth_factors,
    _smooth_observations,
)
from scripts.run_r3m03_smooth_audit import (
    ALPHA_GRID,
    OUT_DIR as SMOOTH_DIR,
    _iter_stream,
)

AUDIT_DIR = ROOT_DIR / "outputs" / "reviewer_revision" / "r3m03_affine_limit"
CEILING = float(ALPHA_GRID[-1])
SAVED_DETAIL = SMOOTH_DIR / "tables" / "detail_by_seed.csv"
PROFILE_ATOL = 1.0e-8


def _profile_gap(left: np.ndarray, right: np.ndarray) -> Dict[str, float]:
    """Mean per-window RMSE and the largest absolute coordinate gap."""
    first = np.asarray(left, dtype=np.float64)
    second = np.asarray(right, dtype=np.float64)
    if first.shape != second.shape or first.ndim != 2:
        raise ValueError("Profile arrays do not match.")
    gaps = np.sqrt(np.mean((first - second) ** 2, axis=1))
    return {
        "profile_rmse_mean": float(np.mean(gaps)),
        "profile_rmse_max": float(np.max(gaps)),
        "profile_abs_max": float(np.max(np.abs(first - second))),
    }


def _score_limits() -> pd.DataFrame:
    """Affine and alpha=1e6 curves on the saved observation stream."""
    factors = _smooth_factors(int(WINDOW_LEN))
    rows: List[Dict[str, float]] = []
    for item in _iter_stream():
        for noise_std in NOISE_STDS:
            b_test = _scale_noise(item["clean_test"], item["unit_test"], float(noise_std))
            curve_ceiling = _smooth_observations(
                b_test,
                item["positions"],
                int(item["length"]),
                CEILING,
                factors,
            )
            curve_affine = _smooth_observations(
                b_test,
                item["positions"],
                int(item["length"]),
                float("inf"),
                factors,
            )
            gap = _profile_gap(curve_ceiling, curve_affine)
            rows.append(
                {
                    "step": int(item["step"]),
                    "seed": int(item["seed"]),
                    "measurement_ratio": float(item["rho"]),
                    "noise_std": float(noise_std),
                    "m_obs": int(item["m_obs"]),
                    "smooth_1e6_rmse": _mean_window_rmse(item["y_test"], curve_ceiling),
                    "affine_rmse": _mean_window_rmse(item["y_test"], curve_affine),
                    "profile_rmse_mean": gap["profile_rmse_mean"],
                    "profile_rmse_max": gap["profile_rmse_max"],
                    "profile_abs_max": gap["profile_abs_max"],
                }
            )
    return pd.DataFrame(rows)


def _join_saved(limits: pd.DataFrame) -> pd.DataFrame:
    """Attach the frozen audit without replacing its selected-alpha scores."""
    if not SAVED_DETAIL.is_file():
        raise FileNotFoundError(str(SAVED_DETAIL))
    saved = pd.read_csv(SAVED_DETAIL)
    required = {
        "step",
        "seed",
        "measurement_ratio",
        "noise_std",
        "smooth_alpha",
        "smooth_rmse",
        "interp_rmse",
        "clp_rmse",
    }
    missing = sorted(required.difference(saved.columns))
    if missing:
        raise ValueError("Saved smoother is missing columns: " + ",".join(missing))
    merged = limits.merge(
        saved[
            [
                "step",
                "seed",
                "measurement_ratio",
                "noise_std",
                "smooth_alpha",
                "smooth_rmse",
                "interp_rmse",
                "clp_rmse",
            ]
        ],
        on=["step", "seed", "measurement_ratio", "noise_std"],
        how="left",
        validate="one_to_one",
    )
    if int(merged["smooth_rmse"].isna().sum()) != 0:
        raise RuntimeError("Affine rows did not match the frozen smoother.")
    merged["at_ceiling"] = np.isclose(merged["smooth_alpha"], CEILING)
    merged["rmse_gap_affine_minus_1e6"] = (
        merged["affine_rmse"] - merged["smooth_1e6_rmse"]
    )
    ceiling = merged[merged["at_ceiling"]]
    if int(ceiling.shape[0]) == 0:
        raise RuntimeError("No ceiling rows were found.")
    selected_gap = np.max(
        np.abs(
            ceiling["smooth_rmse"].to_numpy(dtype=np.float64)
            - ceiling["smooth_1e6_rmse"].to_numpy(dtype=np.float64)
        )
    )
    if float(selected_gap) > PROFILE_ATOL:
        raise RuntimeError(
            "Ceiling selection does not match the alpha=1e6 curve by {0:.3e}.".format(
                float(selected_gap)
            )
        )
    merged.attrs["ceiling_score_gap"] = float(selected_gap)
    return merged


def _group_summary(detail: pd.DataFrame, mask: np.ndarray, label: str) -> Dict[str, float]:
    """Profile and RMSE gaps for one pre-specified row group."""
    frame = detail.loc[mask]
    if int(frame.shape[0]) == 0:
        raise RuntimeError("Empty group: " + label)
    return {
        "group": label,
        "n": int(frame.shape[0]),
        "profile_rmse_mean_median": float(frame["profile_rmse_mean"].median()),
        "profile_rmse_mean_max": float(frame["profile_rmse_mean"].max()),
        "profile_abs_max": float(frame["profile_abs_max"].max()),
        "abs_rmse_gap_median": float(frame["rmse_gap_affine_minus_1e6"].abs().median()),
        "abs_rmse_gap_max": float(frame["rmse_gap_affine_minus_1e6"].abs().max()),
        "affine_lower_than_1e6": int(np.sum(frame["affine_rmse"] < frame["smooth_1e6_rmse"])),
        "one_e6_lower_than_affine": int(
            np.sum(frame["smooth_1e6_rmse"] < frame["affine_rmse"])
        ),
        "affine_lower_than_clp": int(np.sum(frame["affine_rmse"] < frame["clp_rmse"])),
        "clp_lower_than_affine": int(np.sum(frame["clp_rmse"] < frame["affine_rmse"])),
        "affine_lower_than_selected_smooth": int(
            np.sum(frame["affine_rmse"] < frame["smooth_rmse"])
        ),
        "selected_smooth_lower_than_affine": int(
            np.sum(frame["smooth_rmse"] < frame["affine_rmse"])
        ),
    }


def _cell_summary(detail: pd.DataFrame) -> pd.DataFrame:
    """All 36 cells. Affine is additional and does not replace smoothing."""
    rows: List[Dict[str, float]] = []
    grouped = detail.groupby(["step", "measurement_ratio", "noise_std"], sort=True)
    for keys, frame in grouped:
        step, rho, noise_std = keys
        rows.append(
            {
                "step": int(step),
                "measurement_ratio": float(rho),
                "noise_std": float(noise_std),
                "n_seeds": int(frame.shape[0]),
                "interp_rmse": float(frame["interp_rmse"].mean()),
                "smooth_rmse": float(frame["smooth_rmse"].mean()),
                "smooth_1e6_rmse": float(frame["smooth_1e6_rmse"].mean()),
                "affine_rmse": float(frame["affine_rmse"].mean()),
                "clp_rmse": float(frame["clp_rmse"].mean()),
                "n_at_ceiling": int(np.sum(frame["at_ceiling"])),
                "profile_rmse_mean": float(frame["profile_rmse_mean"].mean()),
                "profile_abs_max": float(frame["profile_abs_max"].max()),
                "clp_wins_vs_affine": int(np.sum(frame["clp_rmse"] < frame["affine_rmse"])),
                "affine_wins_vs_clp": int(np.sum(frame["affine_rmse"] < frame["clp_rmse"])),
                "clp_wins_vs_smooth": int(np.sum(frame["clp_rmse"] < frame["smooth_rmse"])),
                "smooth_wins_vs_clp": int(np.sum(frame["smooth_rmse"] < frame["clp_rmse"])),
            }
        )
    return pd.DataFrame(rows)


def _draw(detail: pd.DataFrame) -> None:
    """Profile gap of alpha=1e6 against the affine limit, for every seed row."""
    figures = AUDIT_DIR / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(7.2, 4.6))
    ceiling = detail[detail["at_ceiling"]]
    interior = detail[~detail["at_ceiling"]]
    axis.scatter(
        interior["noise_std"],
        interior["profile_rmse_mean"],
        marker="o",
        color="#1f77b4",
        label="alpha selected inside the grid (n={0})".format(int(interior.shape[0])),
        alpha=0.8,
    )
    axis.scatter(
        ceiling["noise_std"],
        ceiling["profile_rmse_mean"],
        marker="x",
        color="#d62728",
        label="alpha selected at 1e6 (n={0})".format(int(ceiling.shape[0])),
        alpha=0.8,
    )
    axis.set_xscale("log")
    axis.set_yscale("log")
    axis.set_xlabel("simulated noise std")
    axis.set_ylabel("mean per-window RMSE between alpha=1e6 and affine")
    axis.set_title("F03-4 cross-well Vc. All 108 seed rows. Affine is alpha to infinity.")
    axis.grid(True, which="both", alpha=0.3)
    axis.legend(frameon=False)
    figure.tight_layout()
    figure.savefig(figures / "profile_gap_1e6_vs_affine.png", dpi=150)
    plt.close(figure)


def _write_report(detail: pd.DataFrame, groups: pd.DataFrame, cells: pd.DataFrame) -> None:
    """Record whether the ceiling has reached the affine limit."""
    lines = [
        "# Affine limit of the frozen smoother",
        "",
        "The selected-alpha audit was not overwritten.",
        "Ceiling selected-alpha RMSE matches the recomputed alpha=1e6 curve",
        "with max absolute difference {0:.3e}.".format(float(detail.attrs["ceiling_score_gap"])),
        "The affine fit uses only the sparse noisy observations.",
        "It has no alpha. It is the least-squares line on the window index.",
        "",
        "Groups were fixed before this comparison: all rows, ceiling rows, and interior rows.",
        "",
    ]
    for _index, row in groups.iterrows():
        lines.append(
            "{0}: n={1}, median profile RMSE {2:.3e}, max profile RMSE {3:.3e}, max abs coordinate gap {4:.3e}, median |RMSE gap| {5:.3e}, max |RMSE gap| {6:.3e}.".format(
                row["group"],
                int(row["n"]),
                float(row["profile_rmse_mean_median"]),
                float(row["profile_rmse_mean_max"]),
                float(row["profile_abs_max"]),
                float(row["abs_rmse_gap_median"]),
                float(row["abs_rmse_gap_max"]),
            )
        )
        lines.append(
            "  affine lower than alpha=1e6 in {0}; alpha=1e6 lower in {1}; CLP lower than affine in {2}; affine lower than CLP in {3}.".format(
                int(row["affine_lower_than_1e6"]),
                int(row["one_e6_lower_than_affine"]),
                int(row["clp_lower_than_affine"]),
                int(row["affine_lower_than_clp"]),
            )
        )
    lines.append("")
    lines.append("Cell-mean ordering of the additional affine comparator against CLP:")
    for noise_std, frame in cells.groupby("noise_std", sort=True):
        clp_lower = int(np.sum(frame["clp_rmse"] < frame["affine_rmse"]))
        affine_lower = int(np.sum(frame["affine_rmse"] < frame["clp_rmse"]))
        lines.append(
            "noise {0:.2f}: CLP lower cell mean in {1}/9; affine lower cell mean in {2}/9.".format(
                float(noise_std), clp_lower, affine_lower
            )
        )
    lines.extend(
        [
            "",
            "A small profile gap means alpha=1e6 is already near the affine limit.",
            "A large profile gap means the frozen ceiling is not that limit.",
            "Neither result changes the selected-alpha smoother table.",
            "",
        ]
    )
    (AUDIT_DIR / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    """Score the affine limit beside the frozen smoother."""
    if not np.isclose(CEILING, 1.0e6):
        raise RuntimeError("The frozen ceiling is no longer 1e6.")
    AUDIT_DIR.mkdir(parents=True, exist_ok=True)
    detail = _join_saved(_score_limits())
    tables = AUDIT_DIR / "tables"
    tables.mkdir(parents=True, exist_ok=True)
    groups = pd.DataFrame(
        [
            _group_summary(detail, np.ones(detail.shape[0], dtype=bool), "all_108"),
            _group_summary(detail, detail["at_ceiling"].to_numpy(), "ceiling_62_expected"),
            _group_summary(detail, ~detail["at_ceiling"].to_numpy(), "interior"),
        ]
    )
    if int(groups.loc[groups["group"] == "ceiling_62_expected", "n"].iloc[0]) != 62:
        raise RuntimeError("Ceiling count is not the reported 62 rows.")
    cells = _cell_summary(detail)
    detail.to_csv(tables / "detail_by_seed.csv", index=False)
    groups.to_csv(tables / "group_summary.csv", index=False)
    cells.to_csv(tables / "summary_by_cell.csv", index=False)
    _draw(detail)
    _write_report(detail, groups, cells)
    frozen = {
        "ceiling_alpha": CEILING,
        "affine": "least squares onto span of 1 and the window index",
        "saved_smoother": str(SAVED_DETAIL),
        "saved_smoother_overwritten": False,
        "ceiling_score_gap": float(detail.attrs["ceiling_score_gap"]),
    }
    (AUDIT_DIR / "config_frozen.json").write_text(
        json.dumps(frozen, indent=2) + "\n", encoding="utf-8"
    )
    print(groups.to_string(index=False))
    print("ceiling score gap", "{0:.3e}".format(float(detail.attrs["ceiling_score_gap"])))


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        sys.stderr.write("".join(["Affine-limit audit failed: ", str(exc), "\n"]))
        raise
