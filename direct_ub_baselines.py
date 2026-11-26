#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
direct_ub_baselines.py

Direct baselines mapping concatenated context and measurements [u, b] -> y,
methodologically aligned with sir_cs_pipeline_optimized synthetic protocol.

Methods:
    mlp_concat_ub: sklearn MLPRegressor on standardized [X, b].
    pca_regression_ub: PCA on Y_train, MLP from [X,b] to PCA scores; inverse transform.
    ae_regression_ub: PyTorch autoencoder on Y_train, MLP from [X,b] to latent; decode.

ASCII-only source.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from sklearn.decomposition import PCA
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

from sir_cs_pipeline_optimized import Config, evaluate_metric


@dataclass
class DirectUBTrainConfig:
    """Knobs for direct [u,b]->y baselines (separate from Config to avoid dataclass bloat)."""

    ae_latent_dim: int = 32
    ae_hidden: int = 128
    ae_epochs: int = 120
    ae_batch_size: int = 256
    ae_lr: float = 1e-3
    ae_weight_decay: float = 1e-5
    pca_r_grid: Tuple[int, ...] = (8, 16, 24, 32, 48, 64)


def make_B(
    Y: np.ndarray,
    M: np.ndarray,
    noise_std: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """b_i = M y_i + eta for each row of Y (same noise model as pipeline test)."""
    b_clean = Y @ M.T
    if noise_std > 0.0:
        return b_clean + noise_std * rng.normal(size=b_clean.shape)
    return b_clean


def concat_ub(X: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Stack inputs u and measurements b row-wise."""
    return np.hstack([X, B])


def embed_sparse_observation(B: np.ndarray, M: np.ndarray) -> np.ndarray:
    """Place noisy measurements on window coordinates and append a binary mask.

    B has shape (n_samples, m). M has shape (m, n_output) with one selected
    coordinate per row and no repeated columns. The return has shape
    (n_samples, 2 * n_output): placed values, then the same mask on every row.
    """
    measurements = np.asarray(B, dtype=np.float64)
    operator = np.asarray(M, dtype=np.float64)
    if measurements.ndim != 2 or operator.ndim != 2:
        raise ValueError("B and M must be two-dimensional arrays.")
    n_samples, n_meas = measurements.shape
    n_rows, n_output = operator.shape
    if n_rows != n_meas:
        raise ValueError("M rows ({}) must match B columns ({}).".format(n_rows, n_meas))
    if n_output < 1 or n_samples < 1:
        raise ValueError("embed_sparse_observation received an empty array.")
    row_sums = operator.sum(axis=1)
    if not np.allclose(row_sums, 1.0):
        raise ValueError("each measurement row must select exactly one coordinate.")
    column_sums = operator.sum(axis=0)
    if bool(np.any(column_sums > 1.0 + 1e-8)):
        raise ValueError("measurement coordinates must be unique.")
    placed = measurements @ operator
    mask = np.repeat(column_sums.reshape(1, n_output), n_samples, axis=0)
    return np.hstack([placed, mask])


def fit_scaler_ub(Xb_train: np.ndarray) -> StandardScaler:
    s = StandardScaler()
    s.fit(Xb_train)
    return s


def fit_predict_mlp_concat(
    cfg: Config,
    seed: int,
    Xb_train: np.ndarray,
    Y_train: np.ndarray,
    Xb_val: np.ndarray,
    Y_val: np.ndarray,
    Xb_test: np.ndarray,
    scaler: StandardScaler,
) -> Tuple[np.ndarray, np.ndarray]:
    """Train MLP on [u,b] -> y; return val preds and test preds (unscaled y space)."""
    Xt = scaler.transform(Xb_train)
    Xv = scaler.transform(Xb_val)
    Xs = scaler.transform(Xb_test)
    model = MLPRegressor(
        hidden_layer_sizes=cfg.baseline_hidden,
        max_iter=cfg.baseline_max_iter,
        learning_rate_init=cfg.baseline_learning_rate_init,
        alpha=cfg.baseline_alpha,
        early_stopping=cfg.baseline_early_stopping,
        random_state=seed,
    )
    model.fit(Xt, Y_train)
    return model.predict(Xv), model.predict(Xs)


def _max_pca_components(cfg: Config) -> int:
    return int(min(cfg.n_output - 1, cfg.n_train - 1, cfg.n_val + cfg.n_train - 2))


def fit_predict_pca_regression_ub(
    cfg: Config,
    seed: int,
    dub: DirectUBTrainConfig,
    Xb_train: np.ndarray,
    Y_train: np.ndarray,
    Xb_val: np.ndarray,
    Y_val: np.ndarray,
    Xb_test: np.ndarray,
    scaler: StandardScaler,
) -> Tuple[np.ndarray, np.ndarray, int]:
    """
    Select PCA rank r by validation RMSE in Y space; fit MLP [u,b] -> PCA scores.
    Returns (y_val_hat, y_test_hat, best_r).
    """
    Xt = scaler.transform(Xb_train)
    Xv = scaler.transform(Xb_val)
    Xs = scaler.transform(Xb_test)
    mx = _max_pca_components(cfg)
    grid = [r for r in dub.pca_r_grid if 1 <= r <= mx]
    if not grid:
        grid = [min(8, mx)]
    best_r = int(grid[0])
    best_score = float("inf")
    metric = cfg.model_selection_metric

    for r in grid:
        pca = PCA(n_components=int(r), svd_solver="randomized", random_state=seed)
        Zt = pca.fit_transform(Y_train)
        Zv = pca.transform(Y_val)
        head = MLPRegressor(
            hidden_layer_sizes=cfg.baseline_hidden,
            max_iter=cfg.baseline_max_iter,
            learning_rate_init=cfg.baseline_learning_rate_init,
            alpha=cfg.baseline_alpha,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=15,
            random_state=seed + 7,
        )
        head.fit(Xt, Zt)
        Zv_hat = head.predict(Xv)
        Yv_hat = pca.inverse_transform(Zv_hat)
        score = evaluate_metric(Y_val, Yv_hat, metric)
        if score < best_score:
            best_score = score
            best_r = int(r)

    pca = PCA(n_components=best_r, svd_solver="randomized", random_state=seed)
    Zt = pca.fit_transform(Y_train)
    Zv = pca.transform(Y_val)
    head = MLPRegressor(
        hidden_layer_sizes=cfg.baseline_hidden,
        max_iter=cfg.baseline_max_iter,
        learning_rate_init=cfg.baseline_learning_rate_init,
        alpha=cfg.baseline_alpha,
        early_stopping=cfg.baseline_early_stopping,
        random_state=seed + 7,
    )
    head.fit(Xt, Zt)
    Zv_hat = head.predict(Xv)
    Zs_hat = head.predict(Xs)
    Yv_hat = pca.inverse_transform(Zv_hat)
    Ys_hat = pca.inverse_transform(Zs_hat)
    return Yv_hat, Ys_hat, best_r


class _TinyAE(nn.Module):
    """MLP autoencoder on R^N."""

    def __init__(self, n_output: int, h_dim: int, hidden: int) -> None:
        super().__init__()
        h = int(hidden)
        self.enc = nn.Sequential(
            nn.Linear(n_output, h),
            nn.ReLU(),
            nn.Linear(h, h_dim),
        )
        self.dec = nn.Sequential(
            nn.Linear(h_dim, h),
            nn.ReLU(),
            nn.Linear(h, n_output),
        )

    def encode(self, y: torch.Tensor) -> torch.Tensor:
        return self.enc(y)

    def decode(self, h: torch.Tensor) -> torch.Tensor:
        return self.dec(h)

    def forward(self, y: torch.Tensor) -> torch.Tensor:
        return self.decode(self.encode(y))


def fit_predict_ae_regression_ub(
    cfg: Config,
    seed: int,
    dub: DirectUBTrainConfig,
    Xb_train: np.ndarray,
    Y_train: np.ndarray,
    Xb_val: np.ndarray,
    Y_val: np.ndarray,
    Xb_test: np.ndarray,
    scaler: StandardScaler,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Train AE on standardized Y_train; MLP maps scaled [u,b] to latent; decode to Y.
    """
    y_scaler = StandardScaler()
    Yt_n = y_scaler.fit_transform(Y_train)
    Yv_n = y_scaler.transform(Y_val)

    Xt = scaler.transform(Xb_train)
    Xv = scaler.transform(Xb_val)
    Xs = scaler.transform(Xb_test)

    n_out = int(cfg.n_output)
    h_dim = int(min(dub.ae_latent_dim, n_out - 1))
    h_dim = max(h_dim, 4)
    device = torch.device("cpu")
    torch.manual_seed(seed)

    ae = _TinyAE(n_out, h_dim, dub.ae_hidden).to(device)
    opt = torch.optim.Adam(ae.parameters(), lr=dub.ae_lr, weight_decay=dub.ae_weight_decay)
    y_t = torch.tensor(Yt_n, dtype=torch.float32, device=device)
    ds = TensorDataset(y_t)
    loader = DataLoader(ds, batch_size=min(dub.ae_batch_size, len(Yt_n)), shuffle=True, drop_last=False)
    ae.train()
    for _ in range(dub.ae_epochs):
        for (yb,) in loader:
            opt.zero_grad()
            recon = ae(yb)
            loss = torch.mean((recon - yb) ** 2)
            loss.backward()
            opt.step()

    ae.eval()
    with torch.no_grad():
        Ht = ae.encode(torch.tensor(Yt_n, dtype=torch.float32, device=device)).cpu().numpy()
        Hv = ae.encode(torch.tensor(Yv_n, dtype=torch.float32, device=device)).cpu().numpy()

    head = MLPRegressor(
        hidden_layer_sizes=cfg.baseline_hidden,
        max_iter=cfg.baseline_max_iter,
        learning_rate_init=cfg.baseline_learning_rate_init,
        alpha=cfg.baseline_alpha,
        early_stopping=cfg.baseline_early_stopping,
        random_state=seed + 11,
    )
    head.fit(Xt, Ht)
    Hv_hat = head.predict(Xv)
    Hs_hat = head.predict(Xs)

    with torch.no_grad():
        Yv_hat_n = ae.decode(torch.tensor(Hv_hat, dtype=torch.float32, device=device)).cpu().numpy()
        Ys_hat_n = ae.decode(torch.tensor(Hs_hat, dtype=torch.float32, device=device)).cpu().numpy()

    Yv_hat = y_scaler.inverse_transform(Yv_hat_n)
    Ys_hat = y_scaler.inverse_transform(Ys_hat_n)
    return Yv_hat, Ys_hat
