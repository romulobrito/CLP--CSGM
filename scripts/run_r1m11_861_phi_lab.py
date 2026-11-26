#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
R1-M11 supplement: laboratory porosity on well 861, held-out coordinates.

The target is Phi_lab at every depth of the cored interval. Dense context is
gamma ray, deep resistivity, shallow resistivity, and lithotype. Neutron,
sonic, density, and Phi_ND stay out of u because they track porosity.

Within a fold, one coordinate set Omega is shared by every window. b is the
laboratory value on Omega, with no added noise. RMSE is computed only on the
complement, so a coordinate used as a measurement is not scored.

Depth blocks are split before windowing. Train and test windows do not share
rows. This script does not change the F3 grid.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import csgm_m2_module as csgm  # noqa: E402
import direct_ub_baselines as dub  # noqa: E402
from sir_cs_pipeline_optimized import Config  # noqa: E402

EXCEL_PATH = (
    REPO_ROOT
    / "methods_comparison"
    / "data"
    / "processed"
    / "861_integrated_logs_enriched.xlsx"
)
OUT_DIR = REPO_ROOT / "outputs" / "reviewer_revision" / "r1m11_861_phi_lab"

DEPTH_COL = "Depth(m)"
TARGET_COL = "Phi_lab (pu)"
U_COLUMNS: Tuple[str, ...] = (
    "GR (API)",
    "Res_Deep",
    "Res_Shallow",
    "Lithotype",
)
EXCLUDED_POROSITY_PROXIES: Tuple[str, ...] = (
    "Density (g/cc)",
    "Phi_Neutron (pu)",
    "Phi_Sonic (pu)",
    "Phi_ND (pu)",
)

WINDOW_LEN = 8
STEP = 1
N_BLOCKS = 3
VAL_ROWS = 16
RHOS: Tuple[float, ...] = (0.25, 0.50)
SEEDS: Tuple[int, ...] = (7, 23, 41)
LATENT_DIM = 4
HIDDEN_DIM = 32
AE_EPOCHS = 80
CSGM_ITERS = 200
CSGM_RESTARTS = 2
LAMBDA_GRID: Tuple[float, ...] = (1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1)


def _require_columns(df: pd.DataFrame, columns: Sequence[str]) -> None:
    """Raise if a required column is absent."""
    missing = [name for name in columns if name not in df.columns]
    if missing:
        raise ValueError("Missing columns: {}".format(", ".join(missing)))


def _load_frame(path: Path) -> pd.DataFrame:
    """Load the 861 enriched table, sorted by depth."""
    if not path.is_file():
        raise FileNotFoundError(str(path))
    frame = pd.read_excel(path)
    _require_columns(frame, (DEPTH_COL, TARGET_COL) + U_COLUMNS)
    frame = frame.sort_values(DEPTH_COL).reset_index(drop=True)
    values = frame[list(U_COLUMNS) + [TARGET_COL]].to_numpy(dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError("Non-finite values in u or Phi_lab.")
    if int(frame.shape[0]) < N_BLOCKS * WINDOW_LEN:
        raise ValueError("Not enough rows for the depth-block windows.")
    return frame


def _depth_blocks(n_rows: int, n_blocks: int) -> List[np.ndarray]:
    """Contiguous index blocks along depth order."""
    if n_blocks < 2:
        raise ValueError("n_blocks must be >= 2")
    return [np.asarray(part, dtype=np.int64) for part in np.array_split(np.arange(n_rows), n_blocks)]


def _window_starts(rows: np.ndarray, window_len: int, step: int) -> List[int]:
    """Starts whose full window lies inside one contiguous row set."""
    row_set = set(int(row) for row in rows.tolist())
    if len(row_set) < window_len:
        return []
    lo = min(row_set)
    hi = max(row_set)
    starts: List[int] = []
    for start in range(lo, hi - window_len + 2):
        if (start - lo) % step != 0:
            continue
        if all((start + offset) in row_set for offset in range(window_len)):
            starts.append(int(start))
    return starts


def _stack_windows(
    u_matrix: np.ndarray,
    y_vector: np.ndarray,
    starts: Sequence[int],
    window_len: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Build (n_windows, window_len * n_channels) and (n_windows, window_len)."""
    if len(starts) < 1:
        raise ValueError("Cannot stack an empty window list.")
    x_rows: List[np.ndarray] = []
    y_rows: List[np.ndarray] = []
    n_channels = int(u_matrix.shape[1])
    for start in starts:
        stop = int(start) + window_len
        y_rows.append(y_vector[start:stop])
        x_rows.append(u_matrix[start:stop, :].reshape(window_len * n_channels))
    x_out = np.vstack(x_rows)
    y_out = np.vstack(y_rows)
    if x_out.shape[1] != window_len * n_channels or y_out.shape[1] != window_len:
        raise ValueError("Window shapes do not match the configured length.")
    return x_out, y_out


def _sample_omega(window_len: int, rho: float, rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray]:
    """Shared coordinate restriction. M has shape (m, window_len)."""
    m_obs = int(round(float(rho) * window_len))
    m_obs = max(1, min(m_obs, window_len - 1))
    observed = np.sort(rng.choice(window_len, size=m_obs, replace=False).astype(np.int64))
    operator = np.zeros((m_obs, window_len), dtype=np.float64)
    for row_id, col_id in enumerate(observed.tolist()):
        operator[row_id, int(col_id)] = 1.0
    if int(np.sum(operator)) != m_obs:
        raise ValueError("Omega operator is not a coordinate restriction.")
    return observed, operator


def _holdout_rmse(y_true: np.ndarray, y_pred: np.ndarray, observed: np.ndarray) -> float:
    """RMSE on coordinates that were not used as measurements."""
    if y_true.shape != y_pred.shape:
        raise ValueError("Prediction and target shapes differ.")
    scored = np.ones(y_true.shape[1], dtype=bool)
    scored[observed] = False
    if int(np.sum(scored)) < 1:
        raise ValueError("No held-out coordinate left to score.")
    residual = y_pred[:, scored] - y_true[:, scored]
    return float(np.sqrt(np.mean(residual ** 2)))


def _measurements(y_windows: np.ndarray, operator: np.ndarray) -> np.ndarray:
    """Laboratory values on Omega. No synthetic noise."""
    return y_windows @ operator.T


def _fit_clp(
    x_train: np.ndarray,
    y_train: np.ndarray,
    seed: int,
    cfg: Config,
) -> Tuple[csgm.TinyAE, csgm.RidgePrior, np.ndarray, np.ndarray]:
    """Train the decoder on laboratory windows and the ridge anchor on u."""
    y_scaler = StandardScaler()
    y_train_n = y_scaler.fit_transform(y_train)
    y_mean = np.asarray(y_scaler.mean_, dtype=np.float64)
    y_scale = np.asarray(y_scaler.scale_, dtype=np.float64)
    if np.any(y_scale <= 0.0):
        raise ValueError("Target scale is not positive.")
    ae = csgm.train_ae_generator(y_train_n, cfg, seed=int(seed), device="cpu")
    z_train = csgm.encode_y(ae, y_train_n, device="cpu")
    prior = csgm.RidgePrior(alpha=float(cfg.csgm_ridge_alpha), random_state=int(seed))
    prior.fit(x_train, z_train)
    return ae, prior, y_mean, y_scale


def _select_lambda(
    ae: csgm.TinyAE,
    prior: csgm.RidgePrior,
    x_val: np.ndarray,
    y_val: np.ndarray,
    operator: np.ndarray,
    observed: np.ndarray,
    y_mean: np.ndarray,
    y_scale: np.ndarray,
    cfg: Config,
    seed: int,
) -> float:
    """Choose lambda by held-out laboratory RMSE on the validation windows."""
    z0 = prior.predict(x_val)
    b_val = _measurements(y_val, operator)
    best_lam = float(cfg.csgm_lambda_grid[0])
    best_score = float("inf")
    for lam in cfg.csgm_lambda_grid:
        y_hat = csgm.csgm_recover_with_prior(
            ae=ae,
            M=operator,
            B=b_val,
            z0_np=z0,
            y_mean=y_mean,
            y_scale=y_scale,
            lam=float(lam),
            n_iters=int(cfg.csgm_iters),
            opt_lr=float(cfg.csgm_opt_lr),
            n_restarts=int(cfg.csgm_restarts),
            device="cpu",
            seed=int(seed),
        )
        score = _holdout_rmse(y_val, y_hat, observed)
        if score < best_score:
            best_score = score
            best_lam = float(lam)
    return best_lam


def _predict_clp_and_prior(
    ae: csgm.TinyAE,
    prior: csgm.RidgePrior,
    x_test: np.ndarray,
    y_test: np.ndarray,
    operator: np.ndarray,
    y_mean: np.ndarray,
    y_scale: np.ndarray,
    lam: float,
    cfg: Config,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return CLP and prior-only predictions in laboratory porosity units."""
    z0 = prior.predict(x_test)
    b_test = _measurements(y_test, operator)
    y_clp = csgm.csgm_recover_with_prior(
        ae=ae,
        M=operator,
        B=b_test,
        z0_np=z0,
        y_mean=y_mean,
        y_scale=y_scale,
        lam=float(lam),
        n_iters=int(cfg.csgm_iters),
        opt_lr=float(cfg.csgm_opt_lr),
        n_restarts=int(cfg.csgm_restarts),
        device="cpu",
        seed=int(seed),
    )
    y_prior = csgm.decode_latent_prior(ae, z0, y_mean, y_scale, device="cpu")
    return y_clp, y_prior


def _run() -> None:
    """Fit the three methods and write held-out laboratory RMSE tables."""
    frame = _load_frame(EXCEL_PATH)
    u_matrix = frame.loc[:, list(U_COLUMNS)].to_numpy(dtype=np.float64)
    y_vector = frame[TARGET_COL].to_numpy(dtype=np.float64)
    blocks = _depth_blocks(int(frame.shape[0]), N_BLOCKS)

    cfg = Config()
    cfg.n_output = WINDOW_LEN
    cfg.p_input = WINDOW_LEN * len(U_COLUMNS)
    cfg.csgm_latent_dim = LATENT_DIM
    cfg.csgm_hidden_dim = HIDDEN_DIM
    cfg.csgm_ae_epochs = AE_EPOCHS
    cfg.csgm_iters = CSGM_ITERS
    cfg.csgm_restarts = CSGM_RESTARTS
    cfg.csgm_lambda_grid = list(LAMBDA_GRID)
    cfg.measurement_noise_std = 0.0
    cfg.csgm_device = "cpu"

    dub_cfg = dub.DirectUBTrainConfig(
        ae_latent_dim=LATENT_DIM,
        ae_hidden=HIDDEN_DIM,
        ae_epochs=AE_EPOCHS,
        ae_batch_size=16,
    )

    rows: List[Dict[str, float]] = []
    for fold_id, test_rows in enumerate(blocks):
        train_pool = np.concatenate([block for i, block in enumerate(blocks) if i != fold_id])
        train_pool = np.sort(train_pool.astype(np.int64))
        segments = []
        seg_start = int(train_pool[0])
        prev = int(train_pool[0])
        for value in train_pool[1:].tolist():
            if int(value) != prev + 1:
                segments.append(np.arange(seg_start, prev + 1, dtype=np.int64))
                seg_start = int(value)
            prev = int(value)
        segments.append(np.arange(seg_start, prev + 1, dtype=np.int64))
        deepest = max(segments, key=lambda seg: int(seg[-1]))
        if int(deepest.shape[0]) <= VAL_ROWS + WINDOW_LEN:
            raise ValueError("Training segment is too short for validation windows.")
        val_rows = deepest[-VAL_ROWS:]
        train_rows = np.setdiff1d(train_pool, val_rows, assume_unique=False)

        train_starts = _window_starts(train_rows, WINDOW_LEN, STEP)
        val_starts = _window_starts(val_rows, WINDOW_LEN, STEP)
        test_starts = _window_starts(test_rows, WINDOW_LEN, STEP)
        if not train_starts or not val_starts or not test_starts:
            raise ValueError("Fold {} produced an empty window split.".format(fold_id))
        overlap = set(train_rows.tolist()) & set(test_rows.tolist())
        if overlap:
            raise ValueError("Train and test rows overlap.")

        x_train, y_train = _stack_windows(u_matrix, y_vector, train_starts, WINDOW_LEN)
        x_val, y_val = _stack_windows(u_matrix, y_vector, val_starts, WINDOW_LEN)
        x_test, y_test = _stack_windows(u_matrix, y_vector, test_starts, WINDOW_LEN)
        x_scaler = StandardScaler().fit(x_train)
        x_train_s = x_scaler.transform(x_train)
        x_val_s = x_scaler.transform(x_val)
        x_test_s = x_scaler.transform(x_test)

        print(
            "fold",
            fold_id,
            "n_train",
            len(train_starts),
            "n_val",
            len(val_starts),
            "n_test",
            len(test_starts),
            flush=True,
        )

        for seed in SEEDS:
            ae, prior, y_mean, y_scale = _fit_clp(x_train_s, y_train, int(seed), cfg)
            for rho in RHOS:
                rng = np.random.default_rng(int(seed) * 1009 + fold_id * 17 + int(round(rho * 100)))
                observed, operator = _sample_omega(WINDOW_LEN, float(rho), rng)
                lam = _select_lambda(
                    ae, prior, x_val_s, y_val, operator, observed, y_mean, y_scale, cfg, int(seed)
                )
                y_clp, y_prior = _predict_clp_and_prior(
                    ae, prior, x_test_s, y_test, operator, y_mean, y_scale, lam, cfg, int(seed)
                )
                b_train = _measurements(y_train, operator)
                b_val = _measurements(y_val, operator)
                b_test = _measurements(y_test, operator)
                xb_scaler = StandardScaler().fit(dub.concat_ub(x_train_s, b_train))
                _pred_val, y_ae = dub.fit_predict_ae_regression_ub(
                    cfg,
                    int(seed),
                    dub_cfg,
                    dub.concat_ub(x_train_s, b_train),
                    y_train,
                    dub.concat_ub(x_val_s, b_val),
                    y_val,
                    dub.concat_ub(x_test_s, b_test),
                    xb_scaler,
                )
                scores = {
                    "clp_csgm_ridge": _holdout_rmse(y_test, y_clp, observed),
                    "prior_only": _holdout_rmse(y_test, y_prior, observed),
                    "ae_regression_ub": _holdout_rmse(y_test, np.asarray(y_ae, dtype=np.float64), observed),
                }
                for method, score in scores.items():
                    rows.append(
                        {
                            "fold_id": float(fold_id),
                            "seed": float(seed),
                            "rho": float(rho),
                            "m_obs": float(observed.shape[0]),
                            "n_holdout": float(WINDOW_LEN - int(observed.shape[0])),
                            "n_train_windows": float(len(train_starts)),
                            "n_val_windows": float(len(val_starts)),
                            "n_test_windows": float(len(test_starts)),
                            "lambda": float(lam) if method == "clp_csgm_ridge" else float("nan"),
                            "method": method,
                            "rmse_holdout": float(score),
                            "omega": ",".join(str(int(idx)) for idx in observed.tolist()),
                        }
                    )
                print(
                    "  seed",
                    seed,
                    "rho",
                    rho,
                    "clp",
                    round(scores["clp_csgm_ridge"], 5),
                    "ae",
                    round(scores["ae_regression_ub"], 5),
                    "prior",
                    round(scores["prior_only"], 5),
                    flush=True,
                )

    detail = pd.DataFrame(rows)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    detail_path = OUT_DIR / "heldout_rmse_detail.csv"
    detail.to_csv(detail_path, index=False)

    summary_rows: List[Dict[str, float]] = []
    for (rho, method), part in detail.groupby(["rho", "method"], sort=True):
        values = part["rmse_holdout"].to_numpy(dtype=np.float64)
        summary_rows.append(
            {
                "rho": float(rho),
                "method": str(method),
                "n": float(values.shape[0]),
                "rmse_mean": float(np.mean(values)),
                "rmse_std": float(np.std(values, ddof=1)) if values.shape[0] > 1 else 0.0,
            }
        )
    summary = pd.DataFrame(summary_rows)
    summary_path = OUT_DIR / "heldout_rmse_summary.csv"
    summary.to_csv(summary_path, index=False)

    meta = {
        "excel": str(EXCEL_PATH),
        "n_rows": int(frame.shape[0]),
        "depth_min_m": float(frame[DEPTH_COL].min()),
        "depth_max_m": float(frame[DEPTH_COL].max()),
        "target": TARGET_COL,
        "u_columns": list(U_COLUMNS),
        "excluded_porosity_proxies": list(EXCLUDED_POROSITY_PROXIES),
        "window_len": WINDOW_LEN,
        "step": STEP,
        "n_blocks": N_BLOCKS,
        "val_rows": VAL_ROWS,
        "rhos": list(RHOS),
        "seeds": list(SEEDS),
        "measurement_noise_std": 0.0,
        "score": "rmse on coordinates outside Omega",
        "device": "cpu",
    }
    (OUT_DIR / "protocol.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    print(summary.to_string(index=False))
    print("wrote", detail_path)


if __name__ == "__main__":
    _run()
