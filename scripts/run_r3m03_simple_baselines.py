#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""R3-03 supplement: direct linear maps, sparse interpolation, decoder recon.

Does not retrain CLP-CSGM. Published CLP and direct-AE RMSE are read from
the offset-0 placement runs. ASCII only.
"""

from __future__ import annotations

import json
import math
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple

# CPU only. Set before torch is imported by the CSGM module.
os.environ["CUDA_VISIBLE_DEVICES"] = ""
ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import StandardScaler

import direct_ub_baselines as dub
import multi_well_vc as mwv
from csgm_m2_module import train_ae_generator
from sir_cs_pipeline import build_measurement_matrix
from sir_cs_pipeline_optimized import Config, rmse

ROOT = Path("outputs/reviewer_revision/r3m03_simple_baselines")
REF_ROOT = Path("outputs/reviewer_revision/r2m04_omega_placement/runs")
TRAIN_PATH = "data/F02-1,F03-2,F06-1_6logs_30dB.txt"
TEST_PATH = "data/F03-4_6logs_30dB.txt"
CHANNELS = ("sonic", "rhob", "ai", "vp")
STEPS = (8, 16, 32)
SEEDS = (7, 23, 41)
RHOS = (0.05, 0.10, 0.20)
NOISE_STD = 0.01
WINDOW_LEN = 64
VAL_FRAC = 0.1
RIDGE_ALPHAS = (
    1.0e-4,
    1.0e-3,
    1.0e-2,
    1.0e-1,
    1.0,
    10.0,
    100.0,
    1.0e3,
    1.0e4,
    1.0e5,
    1.0e6,
)
EXPECTED_COUNTS = {
    8: (104, 11, 97),
    16: (52, 6, 49),
    32: (27, 3, 25),
}
CLP = "ridge_prior_csgm"
AE = "ae_regression_ub"
REF_FOCUS = "summary_by_seed_focus_clp_csgm_vs_ub.csv"


def _mean_window_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean of the per-window RMSE used by the published grid."""
    truth = np.asarray(y_true, dtype=np.float64)
    pred = np.asarray(y_pred, dtype=np.float64)
    if truth.shape != pred.shape or truth.ndim != 2:
        raise ValueError("Prediction and target shapes do not match.")
    scores = [rmse(truth[i], pred[i]) for i in range(int(truth.shape[0]))]
    return float(np.mean(scores))


def _omega_positions(measurement: np.ndarray) -> np.ndarray:
    """Depth index of each row of the subsample matrix, in row order."""
    operator = np.asarray(measurement, dtype=np.float64)
    positions: List[int] = []
    for row in range(int(operator.shape[0])):
        hits = np.flatnonzero(operator[row])
        if hits.size != 1:
            raise ValueError("Each measurement row must select one coordinate.")
        positions.append(int(hits[0]))
    if len(set(positions)) != len(positions):
        raise ValueError("Repeated coordinates in Omega.")
    return np.asarray(positions, dtype=np.int64)


def _interpolate(observations: np.ndarray, positions: np.ndarray, length: int) -> np.ndarray:
    """Linear interpolation along depth. Ends keep the nearest measured value."""
    observed = np.asarray(observations, dtype=np.float64)
    order = np.argsort(positions)
    grid = np.arange(int(length), dtype=np.float64)
    xp = positions[order].astype(np.float64)
    out = np.empty((int(observed.shape[0]), int(length)), dtype=np.float64)
    for i in range(int(observed.shape[0])):
        out[i] = np.interp(grid, xp, observed[i, order])
    return out


# Discrete smoothing spline: ||M y - b||^2 + alpha ||D2 y||^2.
# The finite grid continues past the old ceiling of 1e6. The linear limit
# is the alpha -> infinity member: a straight line fit to the observations.
SMOOTH_ALPHAS = (
    1.0e-4,
    1.0e-3,
    1.0e-2,
    1.0e-1,
    1.0,
    10.0,
    100.0,
    1.0e3,
    1.0e4,
    1.0e5,
    1.0e6,
    1.0e7,
    1.0e8,
    1.0e9,
    1.0e10,
    1.0e11,
    1.0e12,
)
SMOOTH_LINEAR_LIMIT = float("inf")


def _smooth_factors(length: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Null space of D2, a complement, and D2 on that complement."""
    n_depth = int(length)
    if n_depth < 3:
        raise ValueError("Smoothing needs a window of length at least 3.")
    depth = np.arange(n_depth, dtype=np.float64)
    center = float(np.mean(depth))
    scale = float(np.std(depth))
    if scale <= 0.0:
        raise ValueError("Depth index has no spread.")
    depth = (depth - center) / scale
    null_space = np.column_stack(
        [np.ones(n_depth, dtype=np.float64), depth]
    )
    orthogonal, _r_factor = np.linalg.qr(null_space, mode="complete")
    free = orthogonal[:, 2:]
    diff = np.zeros((n_depth - 2, n_depth), dtype=np.float64)
    for i in range(n_depth - 2):
        diff[i, i] = 1.0
        diff[i, i + 1] = -2.0
        diff[i, i + 2] = 1.0
    rough = diff @ free
    if int(rough.shape[0]) != int(rough.shape[1]):
        raise RuntimeError("Roughness block is not square.")
    return null_space, free, rough


def _smooth_observations(
    observations: np.ndarray,
    positions: np.ndarray,
    length: int,
    alpha: float,
    factors: Tuple[np.ndarray, np.ndarray, np.ndarray],
) -> np.ndarray:
    """Fit one roughness-penalized curve per window, using only sparse b.

    A finite alpha solves ||M y - b||^2 + alpha ||D2 y||^2. The linear limit
    keeps only the straight line that best matches the observations.
    Dense logs are not used.
    """
    observed = np.asarray(observations, dtype=np.float64)
    index = np.asarray(positions, dtype=np.int64)
    n_depth = int(length)
    if observed.ndim != 2 or int(observed.shape[1]) != int(index.size):
        raise ValueError("Observations must have one column per Omega row.")
    if (not math.isinf(float(alpha))) and float(alpha) <= 0.0:
        raise ValueError("Smoothing alpha must be positive.")
    null_space, free, rough = factors
    if null_space.shape != (n_depth, 2):
        raise ValueError("Null-space shape does not match the window length.")
    n_windows = int(observed.shape[0])
    n_obs = int(index.size)
    if math.isinf(float(alpha)):
        design = null_space[index, :]
        coeff, _residual, rank, _singular = np.linalg.lstsq(
            design, observed.T, rcond=None
        )
        if int(rank) < 2:
            raise RuntimeError("Observed depths do not identify a straight line.")
        curve = null_space @ coeff
        return np.asarray(curve.T, dtype=np.float64)
    design = np.zeros((n_obs + n_depth - 2, n_depth), dtype=np.float64)
    design[:n_obs, :2] = null_space[index, :]
    design[:n_obs, 2:] = free[index, :]
    design[n_obs:, 2:] = math.sqrt(float(alpha)) * rough
    target = np.zeros((design.shape[0], n_windows), dtype=np.float64)
    target[:n_obs, :] = observed.T
    coeff, _residual, rank, _singular = np.linalg.lstsq(design, target, rcond=None)
    if int(rank) < n_depth:
        raise RuntimeError("Smoothing system lost rank at alpha {0}.".format(alpha))
    curve = null_space @ coeff[:2, :] + free @ coeff[2:, :]
    return np.asarray(curve.T, dtype=np.float64)


def _select_smooth_alpha(
    b_val: np.ndarray,
    y_val: np.ndarray,
    b_test: np.ndarray,
    positions: np.ndarray,
    length: int,
    alphas: Tuple[float, ...] = SMOOTH_ALPHAS,
) -> Tuple[np.ndarray, float, float]:
    """Alpha with the lowest validation mean-window RMSE, then the test curve.

    The candidates are the finite grid and the straight-line limit.
    """
    if len(alphas) == 0:
        raise ValueError("Smoothing alpha grid is empty.")
    factors = _smooth_factors(int(length))
    candidates = tuple(float(alpha) for alpha in alphas) + (SMOOTH_LINEAR_LIMIT,)
    best_alpha = float(candidates[0])
    best_score = float("inf")
    for alpha in candidates:
        val_hat = _smooth_observations(
            b_val, positions, int(length), float(alpha), factors
        )
        score = _mean_window_rmse(y_val, val_hat)
        if score < best_score:
            best_score = float(score)
            best_alpha = float(alpha)
    test_hat = _smooth_observations(
        b_test, positions, int(length), best_alpha, factors
    )
    return test_hat, best_alpha, best_score


def _fit_linear_family(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    x_test: np.ndarray,
    kind: str,
) -> Tuple[np.ndarray, float]:
    """Fit OLS or validation-selected Ridge on standardized [u, b]."""
    x_scaler = StandardScaler()
    y_scaler = StandardScaler()
    x_tr = x_scaler.fit_transform(x_train)
    y_tr = y_scaler.fit_transform(y_train)
    x_va = x_scaler.transform(x_val)
    x_te = x_scaler.transform(x_test)
    if kind == "linear_ub":
        model = LinearRegression()
        model.fit(x_tr, y_tr)
        pred = y_scaler.inverse_transform(model.predict(x_te))
        return np.asarray(pred, dtype=np.float64), float("nan")
    if kind != "ridge_direct_ub":
        raise ValueError("Unknown linear family: " + kind)
    best_alpha = float(RIDGE_ALPHAS[0])
    best_score = float("inf")
    best_pred = None
    for alpha in RIDGE_ALPHAS:
        model = Ridge(alpha=float(alpha), random_state=0)
        model.fit(x_tr, y_tr)
        val_pred = y_scaler.inverse_transform(model.predict(x_va))
        score = _mean_window_rmse(y_val, val_pred)
        if score < best_score:
            best_score = score
            best_alpha = float(alpha)
            best_pred = y_scaler.inverse_transform(model.predict(x_te))
    if best_pred is None:
        raise RuntimeError("Ridge selection did not produce a prediction.")
    return np.asarray(best_pred, dtype=np.float64), best_alpha


def _decoder_rmse(
    cfg: Config,
    seed: int,
    y_train: np.ndarray,
    y_test: np.ndarray,
) -> Tuple[float, float]:
    """Retrospective G(E(y)) RMSE. The test value uses the full target window."""
    y_scaler = StandardScaler().fit(y_train)
    scale = np.asarray(y_scaler.scale_, dtype=np.float64)
    mean = np.asarray(y_scaler.mean_, dtype=np.float64)
    y_train_n = y_scaler.transform(y_train)
    ae = train_ae_generator(y_train_n, cfg, seed=int(seed), device="cpu")
    ae.eval()

    def _reconstruct(windows: np.ndarray) -> np.ndarray:
        coded = y_scaler.transform(windows)
        with torch.no_grad():
            restored = ae(torch.tensor(coded, dtype=torch.float32)).cpu().numpy()
        return restored * scale[None, :] + mean[None, :]

    return (
        _mean_window_rmse(y_train, _reconstruct(y_train)),
        _mean_window_rmse(y_test, _reconstruct(y_test)),
    )


def _load_reference(step: int) -> pd.DataFrame:
    """Published offset-0 CLP and direct-AE means for one step."""
    path = REF_ROOT / "offset{0:05d}_step{1:02d}".format(0, step) / "tables" / REF_FOCUS
    if not path.is_file():
        raise FileNotFoundError(str(path))
    frame = pd.read_csv(path)
    keep = frame[frame["method"].isin([CLP, AE])].copy()
    keep["step"] = int(step)
    return keep


def _check_counts(step: int, n_train: int, n_val: int, n_test: int, reference: pd.DataFrame) -> None:
    """Stop the join when the window budget is not the published one."""
    expected = EXPECTED_COUNTS[int(step)]
    got = (int(n_train), int(n_val), int(n_test))
    if got != expected:
        raise RuntimeError(
            "Step {0} window counts {1} differ from {2}.".format(step, got, expected)
        )
    ref_n = int(reference["n_test_samples"].iloc[0])
    if ref_n != int(n_test):
        raise RuntimeError(
            "Step {0} reference n_test {1} differs from {2}.".format(step, ref_n, n_test)
        )


def _draw_measurements(
    y_train: np.ndarray,
    y_val: np.ndarray,
    y_test: np.ndarray,
    seed: int,
    rho: float,
    length: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Published Omega stream: measurement seed equals the model seed."""
    rng = np.random.default_rng(int(seed))
    m_obs = max(4, int(round(float(rho) * int(length))))
    measurement = build_measurement_matrix(m_obs, int(length), "subsample", rng)
    b_train = dub.make_B(y_train, measurement, NOISE_STD, rng)
    b_val = dub.make_B(y_val, measurement, NOISE_STD, rng)
    b_test = dub.make_B(y_test, measurement, NOISE_STD, rng)
    return measurement, b_train, b_val, b_test


def _run_step(step: int, cfg: Config) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Fit the cheap comparators and the retrospective decoder for one step."""
    data = mwv.build_cross_well_data_dict(
        train_path=TRAIN_PATH,
        test_path=TEST_PATH,
        target_name="vc",
        channels=CHANNELS,
        window_len=WINDOW_LEN,
        step=int(step),
        val_frac=VAL_FRAC,
        val_embargo_windows=0,
        val_split_mode="tail",
        val_split_seed=0,
        residual_basis="dct",
    )
    reference = _load_reference(int(step))
    n_train = int(data["Y_train"].shape[0])
    n_val = int(data["Y_val"].shape[0])
    n_test = int(data["Y_test"].shape[0])
    _check_counts(int(step), n_train, n_val, n_test, reference)
    rows: List[Dict[str, float]] = []
    oracle_rows: List[Dict[str, float]] = []
    length = int(data["Y_train"].shape[1])
    for seed in SEEDS:
        train_rmse, test_rmse = _decoder_rmse(
            cfg, int(seed), data["Y_train"], data["Y_test"]
        )
        oracle_rows.append(
            {
                "step": int(step),
                "seed": int(seed),
                "split": "train",
                "rmse": float(train_rmse),
                "n": int(n_train),
            }
        )
        oracle_rows.append(
            {
                "step": int(step),
                "seed": int(seed),
                "split": "test_retrospective",
                "rmse": float(test_rmse),
                "n": int(n_test),
            }
        )
        for rho in RHOS:
            measurement, b_train, b_val, b_test = _draw_measurements(
                data["Y_train"],
                data["Y_val"],
                data["Y_test"],
                int(seed),
                float(rho),
                length,
            )
            positions = _omega_positions(measurement)
            x_train = dub.concat_ub(data["X_train"], b_train)
            x_val = dub.concat_ub(data["X_val"], b_val)
            x_test = dub.concat_ub(data["X_test"], b_test)
            linear_pred, _ = _fit_linear_family(
                x_train, data["Y_train"], x_val, data["Y_val"], x_test, "linear_ub"
            )
            ridge_pred, ridge_alpha = _fit_linear_family(
                x_train,
                data["Y_train"],
                x_val,
                data["Y_val"],
                x_test,
                "ridge_direct_ub",
            )
            interp_pred = _interpolate(b_test, positions, length)
            scored = {
                "linear_ub": (linear_pred, float("nan")),
                "ridge_direct_ub": (ridge_pred, float(ridge_alpha)),
                "interp_sparse": (interp_pred, float("nan")),
            }
            for method, (pred, alpha) in scored.items():
                rows.append(
                    {
                        "step": int(step),
                        "seed": int(seed),
                        "measurement_ratio": float(rho),
                        "method": method,
                        "rmse": _mean_window_rmse(data["Y_test"], pred),
                        "ridge_alpha": float(alpha),
                        "n_test": int(n_test),
                    }
                )
    reference_rows = reference.rename(columns={"rmse_mean": "rmse"})
    for _, item in reference_rows.iterrows():
        rows.append(
            {
                "step": int(step),
                "seed": int(item["seed"]),
                "measurement_ratio": float(item["measurement_ratio"]),
                "method": str(item["method"]),
                "rmse": float(item["rmse"]),
                "ridge_alpha": float("nan"),
                "n_test": int(item["n_test_samples"]),
            }
        )
    return pd.DataFrame(rows), pd.DataFrame(oracle_rows)


def _cell_summary(detail: pd.DataFrame) -> pd.DataFrame:
    """One row per step and rho, with the mean RMSE of each comparator."""
    methods = ["linear_ub", "ridge_direct_ub", "interp_sparse", CLP, AE]
    rows: List[Dict[str, float]] = []
    grouped = detail.groupby(["step", "measurement_ratio"], sort=True)
    for (step, rho), group in grouped:
        record: Dict[str, float] = {
            "step": int(step),
            "rho": float(rho),
            "n_seeds": float(group[group["method"] == CLP]["seed"].nunique()),
        }
        for method in methods:
            part = group[group["method"] == method]
            if len(part) != 3:
                raise RuntimeError(
                    "Expected 3 seeds for {0} at step {1} rho {2}.".format(method, step, rho)
                )
            record[method] = float(part["rmse"].mean())
        clp = float(record[CLP])
        record["linear_minus_clp"] = float(record["linear_ub"] - clp)
        record["ridge_direct_minus_clp"] = float(record["ridge_direct_ub"] - clp)
        record["interp_minus_clp"] = float(record["interp_sparse"] - clp)
        record["ae_minus_clp"] = float(record[AE] - clp)
        names = {
            "linear_ub": float(record["linear_ub"]),
            "ridge_direct_ub": float(record["ridge_direct_ub"]),
            "interp_sparse": float(record["interp_sparse"]),
            CLP: clp,
            AE: float(record[AE]),
        }
        record["lowest"] = min(names, key=names.get)
        rows.append(record)
    return pd.DataFrame(rows)


def _write_protocol() -> None:
    """Save the frozen protocol next to the run artifacts."""
    text = "\n".join(
        [
            "R3-03 supplement. Cross-well Vc only.",
            "Train F02-1, F03-2, F06-1. Test F03-4.",
            "Channels sonic, rhob, ai, vp. Target vc.",
            "Steps 8, 16, 32. Ratios 0.05, 0.10, 0.20. Seeds 7, 23, 41.",
            "Tail validation. Measurement seed equals the model seed.",
            "Noise std 0.01. Full-window RMSE.",
            "linear_ub: ordinary least squares from [u, b] to the window.",
            "ridge_direct_ub: ridge from [u, b], alpha chosen on validation.",
            "interp_sparse: linear depth interpolation of noisy b only.",
            "CLP and direct AE RMSE are copied from offset 0 of the placement run.",
            "G(E(y)) on the test windows uses the full target and is not a method.",
            "",
        ]
    )
    (ROOT / "PROTOCOL.txt").write_text(text, encoding="utf-8")


def main() -> None:
    """Run the three steps and write the aggregate tables."""
    if not Path(TRAIN_PATH).is_file() or not Path(TEST_PATH).is_file():
        raise FileNotFoundError("Missing F3 log files.")
    ROOT.mkdir(parents=True, exist_ok=True)
    (ROOT / "runs").mkdir(parents=True, exist_ok=True)
    _write_protocol()
    cfg = Config()
    cfg.csgm_device = "cpu"
    cfg.csgm_latent_dim = 16
    cfg.csgm_hidden_dim = 128
    cfg.csgm_ae_epochs = 200
    cfg.csgm_batch_size = 64
    cfg.csgm_ae_lr = 1.0e-3
    cfg.csgm_weight_decay = 1.0e-5
    frozen = {
        "steps": list(STEPS),
        "seeds": list(SEEDS),
        "rhos": list(RHOS),
        "noise_std": NOISE_STD,
        "ridge_alphas": list(RIDGE_ALPHAS),
        "reference": str(REF_ROOT),
        "decoder": "CLP TinyAE k=16, train only, test recon is retrospective",
    }
    with (ROOT / "config_frozen.json").open("w", encoding="utf-8") as handle:
        json.dump(frozen, handle, indent=2)
    details: List[pd.DataFrame] = []
    oracles: List[pd.DataFrame] = []
    for step in STEPS:
        print("step", step)
        detail, oracle = _run_step(int(step), cfg)
        run_dir = ROOT / "runs" / "step{0:02d}".format(int(step)) / "tables"
        run_dir.mkdir(parents=True, exist_ok=True)
        detail.to_csv(run_dir / "simple_baselines_by_seed.csv", index=False)
        oracle.to_csv(run_dir / "decoder_recon.csv", index=False)
        details.append(detail)
        oracles.append(oracle)
    all_detail = pd.concat(details, ignore_index=True)
    all_oracle = pd.concat(oracles, ignore_index=True)
    summary = _cell_summary(all_detail)
    out = ROOT / "aggregate" / "tables"
    out.mkdir(parents=True, exist_ok=True)
    all_detail.to_csv(out / "detail_by_seed.csv", index=False)
    summary.to_csv(out / "summary_by_cell.csv", index=False)
    all_oracle.to_csv(out / "decoder_recon.csv", index=False)
    print(summary.to_string(index=False))
    print(all_oracle.groupby(["step", "split"])["rmse"].mean().to_string())


if __name__ == "__main__":
    main()
