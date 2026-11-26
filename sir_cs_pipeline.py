
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
sir_cs_pipeline.py

Pipeline mínimo-ao-publicável para testar a viabilidade de:
    y = f_theta(u) + Psi alpha + xi
com inferência via
    b = M y + eta
e recuperação da inovação esparsa por compressed sensing.

O script executa:
1) geração sintética controlada;
2) treino de baseline multi-saída (ML-only);
3) treino opcional de preditor de coeficientes esparsos (para weighted-CS);
4) avaliação de:
   - ML-only
   - CS-only
   - Hybrid SIR-CS
   - Weighted Hybrid SIR-CS
5) varredura de hiperparâmetros e produção de tabelas/figuras.

Dependências:
    numpy, pandas, matplotlib, scikit-learn

Uso:
    python sir_cs_pipeline.py

Saídas:
    outputs/
        summary.csv
        detailed_results.csv
        config.json
        plot_rmse_vs_measurement_ratio.png
        example_reconstruction.png
"""

from __future__ import annotations

import json
import math
import os
import time
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler


# ============================================================
# 1) Configuração
# ============================================================

@dataclass
class Config:
    # reprodutibilidade
    seeds: List[int] = field(default_factory=lambda: [7, 13, 23])

    # dimensões do problema
    n_train: int = 1200
    n_val: int = 300
    n_test: int = 300
    p_input: int = 12
    n_output: int = 128

    # estrutura do residual
    residual_basis: str = "identity"   # "identity" ou "dct"
    residual_k: int = 6                # nº de coeficientes relevantes
    residual_amplitude: float = 1.2
    residual_mode: str = "support_from_u"  # "support_from_u" ou "random"

    # ruídos
    measurement_noise_std: float = 0.02
    output_noise_std: float = 0.01

    # medições comprimidas
    measurement_kind: str = "gaussian"  # "gaussian" ou "subsample"
    measurement_ratios: List[float] = field(default_factory=lambda: [0.20, 0.30, 0.40, 0.50])

    # baseline ML
    baseline_hidden: Tuple[int, int] = (128, 128)
    baseline_max_iter: int = 500
    baseline_learning_rate_init: float = 1e-3
    baseline_alpha: float = 1e-4
    baseline_early_stopping: bool = True

    # preditor auxiliar de alpha
    use_alpha_predictor: bool = True
    alpha_hidden: Tuple[int, int] = (128, 128)
    alpha_max_iter: int = 400
    alpha_learning_rate_init: float = 1e-3
    alpha_alpha: float = 1e-4
    alpha_early_stopping: bool = True

    # solver FISTA
    fista_max_iter: int = 500
    fista_tol: float = 1e-6
    l1_lambda_grid: List[float] = field(default_factory=lambda: [1e-3, 3e-3, 1e-2, 3e-2, 1e-1])
    weight_mode: str = "inverse_magnitude"  # como construir pesos do weighted l1
    weight_eps: float = 1e-3
    weight_clip_min: float = 0.15
    weight_clip_max: float = 2.5
    weight_power: float = 0.6

    # estudos adicionais
    run_cs_only: bool = True
    run_weighted_hybrid: bool = True

    # visualização
    save_dir: str = "outputs"
    n_example_plots: int = 3

    # métrica de seleção de lambda
    model_selection_metric: str = "rmse"  # "rmse" ou "mae"


# ============================================================
# 2) Utilidades matemáticas
# ============================================================

def set_seed(seed: int) -> None:
    np.random.seed(seed)


def orthonormal_dct_matrix(n: int) -> np.ndarray:
    """
    Matriz DCT-II ortonormal.
    Se x = Psi @ alpha, então alpha = Psi.T @ x.
    """
    Psi = np.zeros((n, n), dtype=float)
    factor0 = math.sqrt(1.0 / n)
    factor = math.sqrt(2.0 / n)
    k = np.arange(n)
    for i in range(n):
        if i == 0:
            Psi[:, i] = factor0
        else:
            Psi[:, i] = factor * np.cos(np.pi * (k + 0.5) * i / n)
    return Psi


def get_basis(n: int, basis_name: str) -> np.ndarray:
    if basis_name == "identity":
        return np.eye(n)
    if basis_name == "dct":
        return orthonormal_dct_matrix(n)
    raise ValueError(f"Base desconhecida: {basis_name}")


def build_measurement_matrix(m: int, n: int, kind: str, rng: np.random.Generator) -> np.ndarray:
    if kind == "gaussian":
        M = rng.normal(0.0, 1.0 / math.sqrt(m), size=(m, n))
        return M
    if kind == "subsample":
        idx = rng.choice(n, size=m, replace=False)
        M = np.zeros((m, n))
        M[np.arange(m), idx] = 1.0
        return M
    raise ValueError(f"measurement_kind desconhecido: {kind}")


def soft_threshold(x: np.ndarray, thr: np.ndarray) -> np.ndarray:
    return np.sign(x) * np.maximum(np.abs(x) - thr, 0.0)


def power_iteration_lipschitz(A: np.ndarray, n_iter: int = 100) -> float:
    """
    Estima ||A||_2^2, a constante de Lipschitz do gradiente de 0.5||Ax-b||^2.
    """
    n = A.shape[1]
    v = np.random.randn(n)
    v /= np.linalg.norm(v) + 1e-12
    for _ in range(n_iter):
        v = A.T @ (A @ v)
        nv = np.linalg.norm(v) + 1e-12
        v /= nv
    Av = A @ v
    return float(np.dot(Av, Av)) + 1e-12


def fista_lasso(
    A: np.ndarray,
    b: np.ndarray,
    lam: float,
    weights: Optional[np.ndarray] = None,
    max_iter: int = 500,
    tol: float = 1e-6,
    x0: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Resolve:
        min_x 0.5 ||A x - b||_2^2 + lam * sum_i weights_i |x_i|
    Se weights=None, usa weights_i = 1.
    """
    n = A.shape[1]
    if weights is None:
        weights = np.ones(n)
    weights = np.asarray(weights).reshape(-1)

    if x0 is None:
        x = np.zeros(n)
    else:
        x = x0.copy()

    y = x.copy()
    t = 1.0
    L = power_iteration_lipschitz(A)
    step = 1.0 / L

    for _ in range(max_iter):
        grad = A.T @ (A @ y - b)
        x_new = soft_threshold(y - step * grad, step * lam * weights)
        t_new = 0.5 * (1.0 + math.sqrt(1.0 + 4.0 * t * t))
        y = x_new + ((t - 1.0) / t_new) * (x_new - x)

        rel = np.linalg.norm(x_new - x) / (np.linalg.norm(x) + 1e-10)
        x, t = x_new, t_new
        if rel < tol:
            break
    return x


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def relative_l2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    num = np.linalg.norm(y_true - y_pred)
    den = np.linalg.norm(y_true) + 1e-12
    return float(num / den)


def support_f1(alpha_true: np.ndarray, alpha_hat: np.ndarray, threshold: float = 1e-6) -> float:
    true_support = np.abs(alpha_true) > threshold
    pred_support = np.abs(alpha_hat) > threshold
    tp = np.sum(true_support & pred_support)
    fp = np.sum(~true_support & pred_support)
    fn = np.sum(true_support & ~pred_support)
    if tp == 0:
        return 0.0
    precision = tp / (tp + fp + 1e-12)
    recall = tp / (tp + fn + 1e-12)
    return float(2 * precision * recall / (precision + recall + 1e-12))


def aggregate_ci95(x: np.ndarray) -> Tuple[float, float]:
    mean = float(np.mean(x))
    se = float(np.std(x, ddof=1) / math.sqrt(max(len(x), 1))) if len(x) > 1 else 0.0
    return mean, 1.96 * se


# ============================================================
# 3) Geração sintética
# ============================================================

def random_feature_background(X: np.ndarray, n_output: int, rng: np.random.Generator) -> np.ndarray:
    """
    Gera componente global suave e não linear:
        y_bg = B2 tanh(B1 x + c) + termos senoidais leves
    """
    p = X.shape[1]
    h = 32
    W1 = rng.normal(scale=0.8, size=(p, h))
    c1 = rng.normal(scale=0.2, size=(h,))
    H = np.tanh(X @ W1 + c1)

    W2 = rng.normal(scale=0.7 / math.sqrt(h), size=(h, n_output))
    y_bg = H @ W2

    # componente baixa frequência na saída
    grid = np.linspace(0.0, 1.0, n_output)
    coeff1 = (X[:, 0:1] + 0.5 * X[:, 1:2])
    coeff2 = (0.7 * X[:, 2:3] - 0.3 * X[:, 3:4])
    y_bg += 0.4 * np.sin(2 * np.pi * coeff1 * grid[None, :])
    y_bg += 0.25 * np.cos(2 * np.pi * coeff2 * grid[None, :])
    return y_bg


def choose_support_from_u(
    u: np.ndarray,
    n_output: int,
    k: int,
    rng: np.random.Generator,
    mode: str
) -> np.ndarray:
    """
    Define suporte da inovação.
    - support_from_u: suporte parcialmente previsível a partir da entrada
    - random: suporte aleatório
    """
    if mode == "random":
        return rng.choice(n_output, size=k, replace=False)

    # suporte guiado por algumas features do input
    score = 0.9 * u[0] + 0.7 * u[1] - 0.4 * u[2] + 0.3 * u[3]
    center = int(((math.tanh(score) + 1.0) / 2.0) * (n_output - 1))
    offsets = np.array([-12, -6, -3, 0, 4, 9, 14, 18])
    candidates = np.clip(center + offsets, 0, n_output - 1)
    # mistura entre parte previsível e aleatória para evitar "oráculo"
    base = list(np.unique(candidates))
    rng.shuffle(base)
    picked = base[: min(k, len(base))]
    while len(picked) < k:
        cand = int(rng.integers(0, n_output))
        if cand not in picked:
            picked.append(cand)
    return np.array(picked, dtype=int)


def generate_sparse_alpha(
    X: np.ndarray,
    n_output: int,
    basis: str,
    k: int,
    amplitude: float,
    mode: str,
    rng: np.random.Generator,
) -> np.ndarray:
    n = X.shape[0]
    Alpha = np.zeros((n, n_output))
    for i in range(n):
        supp = choose_support_from_u(X[i], n_output, k, rng, mode=mode)

        # amplitudes dependentes de X para dar conteúdo regressivo real
        amps = amplitude * (
            0.6 * rng.choice([-1.0, 1.0], size=k)
            + 0.6 * np.tanh(X[i, 4:4 + min(k, X.shape[1] - 4)].mean() if X.shape[1] > 4 else X[i, 0])
        )

        # jitter
        amps = np.asarray(amps).reshape(-1)
        if amps.size < k:
            amps = np.pad(amps, (0, k - amps.size), mode="edge")
        amps = amps[:k] + 0.15 * rng.normal(size=k)

        Alpha[i, supp] = amps
    return Alpha


def make_dataset(cfg: Config, seed: int) -> Dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)

    n_total = cfg.n_train + cfg.n_val + cfg.n_test
    X = rng.normal(size=(n_total, cfg.p_input))

    y_bg = random_feature_background(X, cfg.n_output, rng)

    Alpha = generate_sparse_alpha(
        X=X,
        n_output=cfg.n_output,
        basis=cfg.residual_basis,
        k=cfg.residual_k,
        amplitude=cfg.residual_amplitude,
        mode=cfg.residual_mode,
        rng=rng,
    )

    Psi = get_basis(cfg.n_output, cfg.residual_basis)
    residual = Alpha @ Psi.T  # x = Psi alpha, linha a linha

    y = y_bg + residual + cfg.output_noise_std * rng.normal(size=y_bg.shape)

    out = {
        "X_train": X[:cfg.n_train],
        "X_val": X[cfg.n_train:cfg.n_train + cfg.n_val],
        "X_test": X[cfg.n_train + cfg.n_val:],
        "Y_train": y[:cfg.n_train],
        "Y_val": y[cfg.n_train:cfg.n_train + cfg.n_val],
        "Y_test": y[cfg.n_train + cfg.n_val:],
        "Ybg_train": y_bg[:cfg.n_train],
        "Ybg_val": y_bg[cfg.n_train:cfg.n_train + cfg.n_val],
        "Ybg_test": y_bg[cfg.n_train + cfg.n_val:],
        "Alpha_train": Alpha[:cfg.n_train],
        "Alpha_val": Alpha[cfg.n_train:cfg.n_train + cfg.n_val],
        "Alpha_test": Alpha[cfg.n_train + cfg.n_val:],
        "Psi": Psi,
    }
    return out


# ============================================================
# 4) Modelos de ML
# ============================================================

class MultiOutputMLP:
    def __init__(
        self,
        hidden_layer_sizes=(128, 128),
        max_iter=500,
        learning_rate_init=1e-3,
        alpha=1e-4,
        early_stopping=True,
        random_state=0,
    ):
        self.x_scaler = StandardScaler()
        self.y_scaler = StandardScaler()
        self.model = MLPRegressor(
            hidden_layer_sizes=hidden_layer_sizes,
            activation="relu",
            solver="adam",
            alpha=alpha,
            batch_size="auto",
            learning_rate_init=learning_rate_init,
            max_iter=max_iter,
            early_stopping=early_stopping,
            validation_fraction=0.1,
            n_iter_no_change=20,
            random_state=random_state,
            verbose=False,
        )

    def fit(self, X: np.ndarray, Y: np.ndarray) -> "MultiOutputMLP":
        Xs = self.x_scaler.fit_transform(X)
        Ys = self.y_scaler.fit_transform(Y)
        self.model.fit(Xs, Ys)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        Xs = self.x_scaler.transform(X)
        Ys = self.model.predict(Xs)
        return self.y_scaler.inverse_transform(Ys)


# ============================================================
# 5) Weighted prior a partir de um preditor auxiliar de alpha
# ============================================================

def build_weights_from_alpha_prediction(
    alpha_pred: np.ndarray,
    cfg: Config,
) -> np.ndarray:
    """
    Pesos pequenos onde o modelo prevê maior magnitude de alpha.
    """
    mag = np.abs(alpha_pred)
    if cfg.weight_mode == "inverse_magnitude":
        w = 1.0 / (mag + cfg.weight_eps) ** cfg.weight_power
        # normaliza para média ~1
        w = w / (np.mean(w) + 1e-12)
        w = np.clip(w, cfg.weight_clip_min, cfg.weight_clip_max)
        return w
    raise ValueError(f"weight_mode desconhecido: {cfg.weight_mode}")


# ============================================================
# 6) Seleção de lambda
# ============================================================

def evaluate_metric(y_true: np.ndarray, y_pred: np.ndarray, metric_name: str) -> float:
    if metric_name == "rmse":
        return rmse(y_true, y_pred)
    if metric_name == "mae":
        return float(mean_absolute_error(y_true, y_pred))
    raise ValueError(f"Métrica desconhecida: {metric_name}")


def select_lambda_for_method(
    method_name: str,
    cfg: Config,
    A: np.ndarray,
    M: np.ndarray,
    Psi: np.ndarray,
    X_val: np.ndarray,
    Y_val: np.ndarray,
    baseline_model: MultiOutputMLP,
    alpha_model: Optional[MultiOutputMLP] = None,
    rng: Optional[np.random.Generator] = None,
) -> float:
    best_lam = cfg.l1_lambda_grid[0]
    best_score = float("inf")

    Ybg_val = baseline_model.predict(X_val)
    if rng is None:
        rng = np.random.default_rng(0)

    for lam in cfg.l1_lambda_grid:
        preds = []
        for i in range(len(X_val)):
            b_i = M @ Y_val[i]
            if cfg.measurement_noise_std > 0.0:
                b_i = b_i + cfg.measurement_noise_std * rng.normal(size=M.shape[0])
            z_i = b_i - M @ Ybg_val[i]

            if method_name == "hybrid":
                alpha_hat = fista_lasso(
                    A=A,
                    b=z_i,
                    lam=lam,
                    weights=None,
                    max_iter=cfg.fista_max_iter,
                    tol=cfg.fista_tol,
                )
            elif method_name == "weighted_hybrid":
                if alpha_model is None:
                    raise ValueError("weighted_hybrid requer alpha_model.")
                alpha_pred_i = alpha_model.predict(X_val[i:i+1])[0]
                weights_i = build_weights_from_alpha_prediction(alpha_pred_i, cfg)
                alpha_hat = fista_lasso(
                    A=A,
                    b=z_i,
                    lam=lam,
                    weights=weights_i,
                    max_iter=cfg.fista_max_iter,
                    tol=cfg.fista_tol,
                )
            elif method_name == "cs_only":
                alpha_hat = fista_lasso(
                    A=A,
                    b=b_i,
                    lam=lam,
                    weights=None,
                    max_iter=cfg.fista_max_iter,
                    tol=cfg.fista_tol,
                )
            else:
                raise ValueError(f"Método desconhecido: {method_name}")

            if method_name == "cs_only":
                y_hat = Psi @ alpha_hat
            else:
                y_hat = Ybg_val[i] + Psi @ alpha_hat
            preds.append(y_hat)

        preds = np.array(preds)
        score = evaluate_metric(Y_val, preds, cfg.model_selection_metric)
        if score < best_score:
            best_score = score
            best_lam = lam

    return best_lam


# ============================================================
# 7) Avaliação
# ============================================================

def run_single_setting(
    cfg: Config,
    seed: int,
    measurement_ratio: float,
) -> Tuple[pd.DataFrame, Dict[str, np.ndarray]]:
    rng = np.random.default_rng(seed)

    data = make_dataset(cfg, seed=seed)
    X_train = data["X_train"]
    X_val = data["X_val"]
    X_test = data["X_test"]
    Y_train = data["Y_train"]
    Y_val = data["Y_val"]
    Y_test = data["Y_test"]
    Alpha_train = data["Alpha_train"]
    Alpha_test = data["Alpha_test"]
    Psi = data["Psi"]

    m = max(4, int(round(measurement_ratio * cfg.n_output)))
    M = build_measurement_matrix(m, cfg.n_output, cfg.measurement_kind, rng)
    A = M @ Psi

    # medições ruidosas no teste serão geradas sob demanda
    # baseline
    baseline = MultiOutputMLP(
        hidden_layer_sizes=cfg.baseline_hidden,
        max_iter=cfg.baseline_max_iter,
        learning_rate_init=cfg.baseline_learning_rate_init,
        alpha=cfg.baseline_alpha,
        early_stopping=cfg.baseline_early_stopping,
        random_state=seed,
    )
    baseline.fit(X_train, Y_train)

    # preditor auxiliar de alpha
    alpha_model = None
    if cfg.use_alpha_predictor:
        alpha_model = MultiOutputMLP(
            hidden_layer_sizes=cfg.alpha_hidden,
            max_iter=cfg.alpha_max_iter,
            learning_rate_init=cfg.alpha_learning_rate_init,
            alpha=cfg.alpha_alpha,
            early_stopping=cfg.alpha_early_stopping,
            random_state=seed + 999,
        )
        alpha_model.fit(X_train, Alpha_train)

    # seleção de lambda na validação
    lam_hybrid = select_lambda_for_method(
        method_name="hybrid",
        cfg=cfg,
        A=A,
        M=M,
        Psi=Psi,
        X_val=X_val,
        Y_val=Y_val,
        baseline_model=baseline,
        alpha_model=None,
        rng=rng,
    )

    lam_weighted = None
    if cfg.run_weighted_hybrid and cfg.use_alpha_predictor:
        lam_weighted = select_lambda_for_method(
            method_name="weighted_hybrid",
            cfg=cfg,
            A=A,
            M=M,
            Psi=Psi,
            X_val=X_val,
            Y_val=Y_val,
            baseline_model=baseline,
            alpha_model=alpha_model,
            rng=rng,
        )

    lam_cs_only = None
    if cfg.run_cs_only:
        lam_cs_only = select_lambda_for_method(
            method_name="cs_only",
            cfg=cfg,
            A=A,
            M=M,
            Psi=Psi,
            X_val=X_val,
            Y_val=Y_val,
            baseline_model=baseline,
            alpha_model=None,
            rng=rng,
        )

    Ybg_test = baseline.predict(X_test)
    if alpha_model is not None:
        Alpha_pred_test = alpha_model.predict(X_test)
    else:
        Alpha_pred_test = np.zeros_like(Alpha_test)

    rows = []
    stored_examples = {
        "Y_true": [],
        "Y_bg": [],
        "Y_hybrid": [],
        "Y_weighted": [],
        "Y_cs_only": [],
    }

    for i in range(len(X_test)):
        noise = cfg.measurement_noise_std * rng.normal(size=m)
        b_i = M @ Y_test[i] + noise

        # 1) ML-only
        y_ml = Ybg_test[i]
        rows.append({
            "seed": seed,
            "measurement_ratio": measurement_ratio,
            "method": "ml_only",
            "sample_id": i,
            "rmse": rmse(Y_test[i], y_ml),
            "mae": float(mean_absolute_error(Y_test[i], y_ml)),
            "relative_l2": relative_l2(Y_test[i], y_ml),
            "support_f1": np.nan,
            "lambda": np.nan,
            "m": m,
        })

        # 2) Hybrid
        z_i = b_i - M @ y_ml
        alpha_h = fista_lasso(
            A=A,
            b=z_i,
            lam=lam_hybrid,
            weights=None,
            max_iter=cfg.fista_max_iter,
            tol=cfg.fista_tol,
        )
        y_h = y_ml + Psi @ alpha_h
        rows.append({
            "seed": seed,
            "measurement_ratio": measurement_ratio,
            "method": "hybrid",
            "sample_id": i,
            "rmse": rmse(Y_test[i], y_h),
            "mae": float(mean_absolute_error(Y_test[i], y_h)),
            "relative_l2": relative_l2(Y_test[i], y_h),
            "support_f1": support_f1(Alpha_test[i], alpha_h),
            "lambda": lam_hybrid,
            "m": m,
        })

        # 3) Weighted Hybrid
        y_wh = None
        if lam_weighted is not None:
            weights_i = build_weights_from_alpha_prediction(Alpha_pred_test[i], cfg)
            alpha_wh = fista_lasso(
                A=A,
                b=z_i,
                lam=lam_weighted,
                weights=weights_i,
                max_iter=cfg.fista_max_iter,
                tol=cfg.fista_tol,
            )
            y_wh = y_ml + Psi @ alpha_wh
            rows.append({
                "seed": seed,
                "measurement_ratio": measurement_ratio,
                "method": "weighted_hybrid",
                "sample_id": i,
                "rmse": rmse(Y_test[i], y_wh),
                "mae": float(mean_absolute_error(Y_test[i], y_wh)),
                "relative_l2": relative_l2(Y_test[i], y_wh),
                "support_f1": support_f1(Alpha_test[i], alpha_wh),
                "lambda": lam_weighted,
                "m": m,
            })

        # 4) CS-only
        y_cs = None
        if lam_cs_only is not None:
            alpha_cs = fista_lasso(
                A=A,
                b=b_i,
                lam=lam_cs_only,
                weights=None,
                max_iter=cfg.fista_max_iter,
                tol=cfg.fista_tol,
            )
            y_cs = Psi @ alpha_cs
            rows.append({
                "seed": seed,
                "measurement_ratio": measurement_ratio,
                "method": "cs_only",
                "sample_id": i,
                "rmse": rmse(Y_test[i], y_cs),
                "mae": float(mean_absolute_error(Y_test[i], y_cs)),
                "relative_l2": relative_l2(Y_test[i], y_cs),
                "support_f1": support_f1(Alpha_test[i], alpha_cs),
                "lambda": lam_cs_only,
                "m": m,
            })

        if len(stored_examples["Y_true"]) < cfg.n_example_plots:
            stored_examples["Y_true"].append(Y_test[i].copy())
            stored_examples["Y_bg"].append(y_ml.copy())
            stored_examples["Y_hybrid"].append(y_h.copy())
            if y_wh is not None:
                stored_examples["Y_weighted"].append(y_wh.copy())
            if y_cs is not None:
                stored_examples["Y_cs_only"].append(y_cs.copy())

    df = pd.DataFrame(rows)
    for k in list(stored_examples.keys()):
        stored_examples[k] = np.array(stored_examples[k]) if len(stored_examples[k]) > 0 else None
    return df, stored_examples


# ============================================================
# 8) Relatórios e figuras
# ============================================================

def summarize_results(df: pd.DataFrame) -> pd.DataFrame:
    grouped = (
        df.groupby(["measurement_ratio", "method"])
        .agg(
            rmse_mean=("rmse", "mean"),
            rmse_std=("rmse", "std"),
            mae_mean=("mae", "mean"),
            mae_std=("mae", "std"),
            relative_l2_mean=("relative_l2", "mean"),
            relative_l2_std=("relative_l2", "std"),
            support_f1_mean=("support_f1", "mean"),
            support_f1_std=("support_f1", "std"),
            n=("rmse", "size"),
        )
        .reset_index()
    )
    return grouped


def plot_rmse_vs_measurement_ratio(summary: pd.DataFrame, save_path: str) -> None:
    plt.figure(figsize=(8, 5))
    methods = summary["method"].unique()
    for method in methods:
        sdf = summary[summary["method"] == method].sort_values("measurement_ratio")
        x = sdf["measurement_ratio"].values
        y = sdf["rmse_mean"].values
        yerr = sdf["rmse_std"].fillna(0.0).values
        plt.errorbar(x, y, yerr=yerr, marker="o", capsize=4, label=method)
    plt.xlabel("Measurement ratio m / N")
    plt.ylabel("RMSE médio no teste")
    plt.title("Viabilidade do híbrido SIR-CS vs razão de medição")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=180)
    plt.close()


def plot_examples(examples: Dict[str, np.ndarray], save_path: str) -> None:
    Y_true = examples["Y_true"]
    Y_bg = examples["Y_bg"]
    Y_h = examples["Y_hybrid"]
    Y_wh = examples.get("Y_weighted", None)
    Y_cs = examples.get("Y_cs_only", None)

    if Y_true is None or len(Y_true) == 0:
        return

    n_examples = len(Y_true)
    fig, axes = plt.subplots(n_examples, 1, figsize=(10, 3 * n_examples), squeeze=False)

    for i in range(n_examples):
        ax = axes[i, 0]
        ax.plot(Y_true[i], label="true")
        ax.plot(Y_bg[i], label="ml_only")
        ax.plot(Y_h[i], label="hybrid")
        if Y_wh is not None and len(Y_wh) > i:
            ax.plot(Y_wh[i], label="weighted_hybrid")
        if Y_cs is not None and len(Y_cs) > i:
            ax.plot(Y_cs[i], label="cs_only")
        ax.set_title(f"Exemplo {i}")
        ax.grid(True, alpha=0.3)
        ax.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=180)
    plt.close()


# ============================================================
# 9) Orquestração
# ============================================================

def print_stage_guidance(summary: pd.DataFrame) -> None:
    print("\n" + "=" * 72)
    print("LEITURA DOS RESULTADOS / DECISÃO METODOLÓGICA")
    print("=" * 72)

    pivot = summary.pivot_table(index="measurement_ratio", columns="method", values="rmse_mean")
    print(pivot.round(4).to_string())

    print("\nCritério mínimo de viabilidade:")
    print("1) hybrid deve superar ml_only em parte substancial das razões de medição;")
    print("2) o ganho do hybrid sobre ml_only deve persistir ao variar m/N (menos medidas tendem a prejudicar todos);")
    print("3) cs_only deve ser pior que hybrid quando a componente global não é esparsa;")
    print("4) weighted_hybrid pode superar hybrid quando o preditor de alpha for informativo (nem sempre).")

    print("\nSe esses 4 pontos ocorrerem, você já tem base para:")
    print("- análise de sensibilidade;")
    print("- estudo de ablação;")
    print("- versão publicável em dados sintéticos + caso aplicado.")


def main() -> None:
    cfg = Config()
    os.makedirs(cfg.save_dir, exist_ok=True)

    with open(os.path.join(cfg.save_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(asdict(cfg), f, indent=2)

    all_dfs = []
    first_examples = None

    t0 = time.time()
    for seed in cfg.seeds:
        for mr in cfg.measurement_ratios:
            print(f"[INFO] Rodando seed={seed} | measurement_ratio={mr:.2f}")
            df, examples = run_single_setting(cfg, seed=seed, measurement_ratio=mr)
            all_dfs.append(df)
            if first_examples is None:
                first_examples = examples

    detailed = pd.concat(all_dfs, ignore_index=True)
    summary = summarize_results(detailed)

    detailed_path = os.path.join(cfg.save_dir, "detailed_results.csv")
    summary_path = os.path.join(cfg.save_dir, "summary.csv")
    detailed.to_csv(detailed_path, index=False)
    summary.to_csv(summary_path, index=False)

    plot_rmse_vs_measurement_ratio(
        summary,
        save_path=os.path.join(cfg.save_dir, "plot_rmse_vs_measurement_ratio.png"),
    )
    if first_examples is not None:
        plot_examples(
            first_examples,
            save_path=os.path.join(cfg.save_dir, "example_reconstruction.png"),
        )

    elapsed = time.time() - t0

    print("\n" + "=" * 72)
    print("RESUMO")
    print("=" * 72)
    print(summary.round(4).to_string(index=False))
    print(f"\nArquivos salvos em: {os.path.abspath(cfg.save_dir)}")
    print(f"Tempo total: {elapsed:.2f} s")

    print_stage_guidance(summary)


if __name__ == "__main__":
    main()
