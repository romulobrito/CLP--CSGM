#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Empirical S-REC-style diagnostics for CLP-CSGM.

This script estimates how well the coordinate-subsampling operator separates
decoder-generated target windows. It does not prove S-REC. Instead, it reports
percentiles of pairwise separation ratios on candidates produced by the learned
decoder:

    ||M (y_i - y_j)||_2 / ||y_i - y_j||_2

and the energy-normalized version obtained by multiplying M by sqrt(N / m).
"""

from __future__ import annotations

import sys
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.exceptions import ConvergenceWarning
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import csgm_m2_module as csgm
from scripts.clp_csgm_runtime_study import build_cross_case, build_f03_case
from sir_cs_pipeline_optimized import build_measurement_matrix


OUT_DIR = ROOT / "outputs" / "srec_diagnostics"
TAB_DIR = OUT_DIR / "tables"
FIG_DIR = OUT_DIR / "figures"
PAPER = ROOT / "outputs/paper_assets"
PAPER_TAB_DIR = PAPER / "tables"
PAPER_FIG_DIR = PAPER / "figures"

PAIR_SAMPLE_SIZE = 5000
PERCENTILES = (0.0, 1.0, 5.0, 10.0, 50.0, 90.0)


@dataclass(frozen=True)
class SrecCase:
    """Container for one empirical S-REC dataset case."""

    dataset: str
    data: Dict[str, np.ndarray]
    cfg: object
    measurement_ratios: Tuple[float, ...]
    seeds: Tuple[int, ...]


def _ensure_dirs() -> None:
    for path in (TAB_DIR, FIG_DIR, PAPER_TAB_DIR, PAPER_FIG_DIR):
        path.mkdir(parents=True, exist_ok=True)


def _decode_training_candidates(case: SrecCase, seed: int) -> np.ndarray:
    """Train the CLP-CSGM decoder and return decoded training candidates."""
    print(
        "Training decoder for {} seed={}...".format(case.dataset, int(seed)),
        flush=True,
    )
    cfg = case.cfg
    device_raw = str(getattr(cfg, "csgm_device", "")).strip()
    device = device_raw if device_raw else "cpu"
    scaler = StandardScaler().fit(case.data["Y_train"])
    y_train_n = scaler.transform(case.data["Y_train"])
    ae = csgm.train_ae_generator(y_train_n, cfg, seed=int(seed), device=device)
    z_train = csgm.encode_y(ae, y_train_n, device=device)
    ae.eval()
    with torch.no_grad():
        y_dec_n = ae.decode(torch.tensor(z_train, dtype=torch.float32, device=device))
    y_dec = y_dec_n.cpu().numpy()
    return y_dec * scaler.scale_[None, :] + scaler.mean_[None, :]


def _sample_pair_indices(n: int, rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray]:
    """Sample unordered candidate pairs, using all pairs when the set is small."""
    if n < 2:
        raise ValueError("At least two candidates are required.")
    all_pairs = n * (n - 1) // 2
    if all_pairs <= PAIR_SAMPLE_SIZE:
        i_idx: List[int] = []
        j_idx: List[int] = []
        for i in range(n):
            for j in range(i + 1, n):
                i_idx.append(i)
                j_idx.append(j)
        return np.asarray(i_idx, dtype=np.int64), np.asarray(j_idx, dtype=np.int64)
    i_vals = rng.integers(0, n, size=PAIR_SAMPLE_SIZE, endpoint=False)
    j_vals = rng.integers(0, n - 1, size=PAIR_SAMPLE_SIZE, endpoint=False)
    j_vals = j_vals + (j_vals >= i_vals)
    return i_vals.astype(np.int64), j_vals.astype(np.int64)


def _pairwise_ratios(
    candidates: np.ndarray,
    m_mat: np.ndarray,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute raw and energy-normalized pairwise separation ratios."""
    y = np.asarray(candidates, dtype=np.float64)
    n_candidates, n_out = y.shape
    i_idx, j_idx = _sample_pair_indices(n_candidates, rng)
    diff = y[i_idx] - y[j_idx]
    denom = np.linalg.norm(diff, axis=1)
    keep = denom > 1e-12
    if not np.any(keep):
        raise RuntimeError("All sampled decoder pairs are numerically identical.")
    diff = diff[keep]
    denom = denom[keep]
    measured = diff @ np.asarray(m_mat, dtype=np.float64).T
    raw = np.linalg.norm(measured, axis=1) / denom
    scale = np.sqrt(float(n_out) / float(m_mat.shape[0]))
    return raw, scale * raw


def _summary_rows(
    dataset: str,
    seed: int,
    rho: float,
    ratios: np.ndarray,
    ratio_type: str,
    n_candidates: int,
    n_pairs: int,
    n_output: int,
    m: int,
) -> List[Dict[str, float | int | str]]:
    """Build summary rows for requested percentiles."""
    rows: List[Dict[str, float | int | str]] = []
    for pct in PERCENTILES:
        rows.append(
            {
                "dataset": dataset,
                "seed": int(seed),
                "measurement_ratio": float(rho),
                "ratio_type": ratio_type,
                "percentile": float(pct),
                "value": float(np.percentile(ratios, pct)),
                "mean": float(np.mean(ratios)),
                "std": float(np.std(ratios)),
                "n_candidates": int(n_candidates),
                "n_pairs": int(n_pairs),
                "n_output": int(n_output),
                "m": int(m),
            }
        )
    return rows


def run_case(case: SrecCase) -> pd.DataFrame:
    """Run empirical S-REC diagnostics for one case."""
    rows: List[Dict[str, float | int | str]] = []
    for seed in case.seeds:
        candidates = _decode_training_candidates(case, seed)
        n_candidates, n_output = candidates.shape
        for rho in case.measurement_ratios:
            print(
                "Computing S-REC percentiles for {} seed={} rho={:.2f}...".format(
                    case.dataset,
                    int(seed),
                    float(rho),
                ),
                flush=True,
            )
            rng = np.random.default_rng(int(seed))
            m = max(4, int(round(float(rho) * int(n_output))))
            m_mat = build_measurement_matrix(m, int(n_output), "subsample", rng)
            pair_rng = np.random.default_rng(int(seed) + int(round(1000.0 * float(rho))) + 911)
            raw, normalized = _pairwise_ratios(candidates, m_mat, pair_rng)
            rows.extend(
                _summary_rows(
                    case.dataset,
                    seed,
                    rho,
                    raw,
                    "raw",
                    n_candidates,
                    int(raw.size),
                    n_output,
                    m,
                )
            )
            rows.extend(
                _summary_rows(
                    case.dataset,
                    seed,
                    rho,
                    normalized,
                    "energy_normalized",
                    n_candidates,
                    int(normalized.size),
                    n_output,
                    m,
                )
            )
    return pd.DataFrame(rows)


def _build_cases() -> List[SrecCase]:
    """Build representative cases using the same setup as the runtime study."""
    cross = build_cross_case()
    f03 = build_f03_case()
    return [
        SrecCase(cross.dataset, cross.data, cross.cfg, cross.measurement_ratios, cross.seeds),
        SrecCase(f03.dataset, f03.data, f03.cfg, f03.measurement_ratios, f03.seeds),
    ]


def _plot_percentiles(summary: pd.DataFrame) -> Path:
    """Plot selected S-REC-style percentiles by measurement ratio."""
    normalized = summary[summary["ratio_type"] == "energy_normalized"].copy()
    datasets = list(normalized["dataset"].drop_duplicates())
    fig, axes = plt.subplots(1, len(datasets), figsize=(6.0 * len(datasets), 4.2), squeeze=False)
    for ax, dataset in zip(axes[0], datasets):
        sub = normalized[normalized["dataset"] == dataset]
        for pct, label in ((1.0, "p1"), (5.0, "p5"), (10.0, "p10"), (50.0, "median")):
            part = sub[sub["percentile"] == pct].sort_values("measurement_ratio")
            ax.plot(part["measurement_ratio"], part["value"], marker="o", label=label)
        ax.axhline(1.0, color="0.4", linestyle="--", linewidth=1.0)
        ax.set_xlabel("Measurement ratio")
        ax.set_ylabel("Energy-normalized separation ratio")
        ax.set_title(dataset)
        ax.grid(True, alpha=0.25)
    axes[0, -1].legend(fontsize=8, loc="best")
    fig.suptitle("Empirical S-REC-style separation on decoder-generated windows", y=1.02)
    fig.tight_layout()
    out = FIG_DIR / "srec_percentiles.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out


def _copy_for_paper(paths: List[Path]) -> None:
    for path in paths:
        target = PAPER_FIG_DIR / path.name
        target.write_bytes(path.read_bytes())


def main() -> None:
    warnings.filterwarnings("ignore", category=ConvergenceWarning)
    _ensure_dirs()
    summary = pd.concat([run_case(case) for case in _build_cases()], ignore_index=True)
    summary_path = TAB_DIR / "srec_percentiles.csv"
    paper_summary_path = PAPER_TAB_DIR / "srec_percentiles.csv"
    summary.to_csv(summary_path, index=False)
    summary.to_csv(paper_summary_path, index=False)
    fig_path = _plot_percentiles(summary)
    _copy_for_paper([fig_path])
    for path in (summary_path, paper_summary_path, fig_path):
        print(path.relative_to(ROOT))


if __name__ == "__main__":
    main()
