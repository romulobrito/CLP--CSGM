#!/usr/bin/env python3
"""
lfista_module.py

Importable LFISTA core: PyTorch MLP background, unrolled LFISTA block, hybrid model,
and training/evaluation used by sir_cs_lfista.py and sir_cs_pipeline_optimized.py.

ASCII-only identifiers and log strings.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def relative_l2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.linalg.norm(y_true - y_pred) / (np.linalg.norm(y_true) + 1e-12))


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


def build_measurement_matrix(m: int, n: int, kind: str, rng: np.random.Generator) -> np.ndarray:
    if kind == "gaussian":
        return rng.normal(0.0, 1.0 / math.sqrt(m), size=(m, n))
    if kind == "subsample":
        idx = rng.choice(n, size=m, replace=False)
        m_sub = np.zeros((m, n))
        m_sub[np.arange(m), idx] = 1.0
        return m_sub
    raise ValueError(f"measurement_kind unknown: {kind}")


def make_measurements(M: np.ndarray, Y: np.ndarray, noise_std: float, rng: np.random.Generator) -> np.ndarray:
    b = Y @ M.T
    if noise_std > 0.0:
        b = b + noise_std * rng.normal(size=b.shape)
    return b


@dataclass
class LFISTATrainConfig:
    """Training knobs for LFISTA; built from pipeline Config or LFISTA lab config."""

    device: str
    p_input: int
    n_output: int
    bg_hidden: Tuple[int, int]
    bg_dropout: float
    lfista_steps: int
    learn_step_sizes: bool
    learn_thresholds: bool
    init_step_scale: float
    init_threshold: float
    use_momentum: bool
    batch_size: int
    num_epochs_bg: int
    num_epochs_frozen: int
    num_epochs_joint: int
    lr_bg: float
    lr_frozen: float
    lr_joint: float
    weight_decay: float
    patience: int
    loss_alpha_weight: float
    loss_l1_alpha_weight: float
    measurement_noise_std: float
    bg_type: str = "mlp2"


class BackgroundMLP(nn.Module):
    """Background regressor u -> y with selectable capacity.

    bg_type:
      - "linear":  y = W u + c    (single Linear; OLS/ridge-like when trained).
      - "shallow": one hidden layer (GELU) with width = hidden[0].
      - "mlp2":    two hidden layers (GELU) -- legacy default.

    Lower-capacity bg models preserve residual sparsity, which the Psi-CS
    sparse block is designed to exploit.
    """

    def __init__(
        self,
        p_input: int,
        n_output: int,
        hidden: Tuple[int, int],
        dropout: float = 0.0,
        bg_type: str = "mlp2",
    ):
        super().__init__()
        bg_type_norm = str(bg_type).strip().lower()
        if bg_type_norm == "linear":
            self.net = nn.Linear(p_input, n_output)
        elif bg_type_norm == "shallow":
            h = int(hidden[0])
            self.net = nn.Sequential(
                nn.Linear(p_input, h),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(h, n_output),
            )
        elif bg_type_norm in ("mlp2", "deep", "default"):
            h1, h2 = hidden
            self.net = nn.Sequential(
                nn.Linear(p_input, h1),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(h1, h2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(h2, n_output),
            )
        else:
            raise ValueError(
                "Unknown bg_type: {} (expected linear|shallow|mlp2)".format(bg_type)
            )
        self.bg_type = bg_type_norm

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class LFISTAUnrolled(nn.Module):
    """Unrolled LFISTA with learnable step sizes and thresholds per layer."""

    def __init__(
        self,
        A: torch.Tensor,
        K: int,
        init_step_scale: float = 1.0,
        init_threshold: float = 1e-2,
        learn_step_sizes: bool = True,
        learn_thresholds: bool = True,
        use_momentum: bool = True,
    ):
        super().__init__()
        if A.dim() != 2:
            raise ValueError("A must be 2D")
        self.register_buffer("A", A)
        self.register_buffer("AT", A.transpose(0, 1))
        self.K = int(K)
        self.use_momentum = bool(use_momentum)

        with torch.no_grad():
            svals = torch.linalg.svdvals(A)
            lip = float((svals.max() ** 2).item()) + 1e-12
        self.register_buffer("L", torch.tensor(lip, dtype=A.dtype, device=A.device))

        init_eta = float(init_step_scale) / lip
        init_tau = float(init_threshold)

        eta0 = torch.full((self.K,), init_eta, dtype=A.dtype, device=A.device)
        tau0 = torch.full((self.K,), init_tau, dtype=A.dtype, device=A.device)

        if learn_step_sizes:
            self.log_eta = nn.Parameter(torch.log(torch.clamp(eta0, min=1e-8)))
        else:
            self.register_buffer("log_eta", torch.log(torch.clamp(eta0, min=1e-8)))

        if learn_thresholds:
            self.log_tau = nn.Parameter(torch.log(torch.clamp(tau0, min=1e-8)))
        else:
            self.register_buffer("log_tau", torch.log(torch.clamp(tau0, min=1e-8)))

    def _soft(self, x: torch.Tensor, thr: torch.Tensor) -> torch.Tensor:
        return torch.sign(x) * torch.relu(torch.abs(x) - thr)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        batch_size = z.shape[0]
        n_code = self.A.shape[1]
        alpha_prev = torch.zeros(batch_size, n_code, dtype=z.dtype, device=z.device)
        alpha = torch.zeros_like(alpha_prev)
        t_prev = torch.ones(batch_size, 1, dtype=z.dtype, device=z.device)

        for k in range(self.K):
            eta_k = torch.exp(self.log_eta[k]).clamp(min=1e-8, max=10.0)
            tau_k = torch.exp(self.log_tau[k]).clamp(min=1e-8, max=10.0)

            if self.use_momentum and k > 0:
                t_new = 0.5 * (1.0 + torch.sqrt(1.0 + 4.0 * t_prev * t_prev))
                beta = (t_prev - 1.0) / t_new
                v = alpha + beta * (alpha - alpha_prev)
            else:
                t_new = t_prev
                v = alpha

            Av = v @ self.A.transpose(0, 1)
            grad = (Av - z) @ self.A
            alpha_new = self._soft(v - eta_k * grad, eta_k * tau_k)

            alpha_prev = alpha
            alpha = alpha_new
            t_prev = t_new

        return alpha


class HybridLFISTA(nn.Module):
    def __init__(self, background: BackgroundMLP, sparse_block: LFISTAUnrolled, Psi: torch.Tensor, M: torch.Tensor):
        super().__init__()
        self.background = background
        self.sparse_block = sparse_block
        self.register_buffer("Psi", Psi)
        self.register_buffer("M", M)

    def forward(self, x: torch.Tensor, b: torch.Tensor) -> Dict[str, torch.Tensor]:
        y_bg = self.background(x)
        z = b - y_bg @ self.M.transpose(0, 1)
        alpha = self.sparse_block(z)
        y_hat = y_bg + alpha @ self.Psi.transpose(0, 1)
        return {"y_bg": y_bg, "z": z, "alpha": alpha, "y_hat": y_hat}


def make_torch_loader(X: np.ndarray, Y: np.ndarray, batch_size: int, shuffle: bool = True) -> DataLoader:
    ds = TensorDataset(torch.tensor(X, dtype=torch.float32), torch.tensor(Y, dtype=torch.float32))
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)


def evaluate_bg_model(model: BackgroundMLP, X: np.ndarray, Y: np.ndarray, device: str) -> Dict[str, float]:
    model.eval()
    with torch.no_grad():
        x = torch.tensor(X, dtype=torch.float32, device=device)
        y = torch.tensor(Y, dtype=torch.float32, device=device)
        y_hat = model(x)
    y_np = y.cpu().numpy()
    y_hat_np = y_hat.cpu().numpy()
    mae = float(np.mean(np.abs(y_np - y_hat_np)))
    return {
        "rmse": rmse(y_np, y_hat_np),
        "mae": mae,
        "relative_l2": relative_l2(y_np, y_hat_np),
    }


def train_background(
    cfg: LFISTATrainConfig,
    model: BackgroundMLP,
    X_train: np.ndarray,
    Y_train: np.ndarray,
    X_val: np.ndarray,
    Y_val: np.ndarray,
    log_fn: Optional[Callable[[str], None]],
) -> BackgroundMLP:
    device = cfg.device
    model.to(device)
    train_loader = make_torch_loader(X_train, Y_train, cfg.batch_size, shuffle=True)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr_bg, weight_decay=cfg.weight_decay)
    best_state = None
    best_val = float("inf")
    bad_epochs = 0

    def _log(msg: str) -> None:
        if log_fn:
            log_fn(msg)

    for epoch in range(1, cfg.num_epochs_bg + 1):
        model.train()
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            yhat = model(xb)
            loss = F.mse_loss(yhat, yb)
            opt.zero_grad()
            loss.backward()
            opt.step()

        val = evaluate_bg_model(model, X_val, Y_val, device)
        if val["rmse"] < best_val:
            best_val = val["rmse"]
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad_epochs = 0
        else:
            bad_epochs += 1
        if epoch == 1 or epoch % 10 == 0:
            _log(f"[bg] epoch={epoch:03d} val_rmse={val['rmse']:.6f}")
        if bad_epochs >= cfg.patience:
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model


def lfista_loss(
    out: Dict[str, torch.Tensor],
    y_true: torch.Tensor,
    alpha_true: Optional[torch.Tensor],
    cfg: LFISTATrainConfig,
) -> torch.Tensor:
    loss = F.mse_loss(out["y_hat"], y_true)
    if alpha_true is not None and cfg.loss_alpha_weight > 0.0:
        loss = loss + cfg.loss_alpha_weight * F.l1_loss(out["alpha"], alpha_true)
    if cfg.loss_l1_alpha_weight > 0.0:
        loss = loss + cfg.loss_l1_alpha_weight * torch.mean(torch.abs(out["alpha"]))
    return loss


def evaluate_hybrid(
    model: HybridLFISTA,
    X: np.ndarray,
    Y: np.ndarray,
    Alpha: np.ndarray,
    B: np.ndarray,
    cfg: LFISTATrainConfig,
) -> Dict[str, float]:
    device = cfg.device
    model.eval()
    with torch.no_grad():
        x = torch.tensor(X, dtype=torch.float32, device=device)
        b = torch.tensor(B, dtype=torch.float32, device=device)
        out = model(x, b)
    y_hat = out["y_hat"].cpu().numpy()
    alpha_hat = out["alpha"].cpu().numpy()
    n = len(Y)
    sf1 = np.mean([support_f1(Alpha[i], alpha_hat[i]) for i in range(n)])
    return {
        "rmse": rmse(Y, y_hat),
        "mae": float(np.mean(np.abs(Y - y_hat))),
        "relative_l2": relative_l2(Y, y_hat),
        "support_f1": float(sf1),
    }


def train_lfista_stage(
    cfg: LFISTATrainConfig,
    hybrid: HybridLFISTA,
    X_train: np.ndarray,
    Y_train: np.ndarray,
    Alpha_train: np.ndarray,
    B_train: np.ndarray,
    X_val: np.ndarray,
    Y_val: np.ndarray,
    Alpha_val: np.ndarray,
    B_val: np.ndarray,
    num_epochs: int,
    lr: float,
    train_background_flag: bool,
    log_fn: Optional[Callable[[str], None]],
) -> HybridLFISTA:
    device = cfg.device
    hybrid.to(device)

    for p in hybrid.background.parameters():
        p.requires_grad = bool(train_background_flag)

    params = [p for p in hybrid.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=cfg.weight_decay)

    ds = TensorDataset(
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(Y_train, dtype=torch.float32),
        torch.tensor(Alpha_train, dtype=torch.float32),
        torch.tensor(B_train, dtype=torch.float32),
    )
    loader = DataLoader(ds, batch_size=cfg.batch_size, shuffle=True)

    best_state = None
    best_val = float("inf")
    bad_epochs = 0

    stage_name = "joint" if train_background_flag else "frozen"

    def _log(msg: str) -> None:
        if log_fn:
            log_fn(msg)

    for epoch in range(1, num_epochs + 1):
        hybrid.train()
        for xb, yb, ab, bb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            ab = ab.to(device)
            bb = bb.to(device)
            out = hybrid(xb, bb)
            loss = lfista_loss(out, yb, ab, cfg)
            opt.zero_grad()
            loss.backward()
            opt.step()

        val = evaluate_hybrid(hybrid, X_val, Y_val, Alpha_val, B_val, cfg)
        if val["rmse"] < best_val:
            best_val = val["rmse"]
            best_state = {k: v.detach().cpu().clone() for k, v in hybrid.state_dict().items()}
            bad_epochs = 0
        else:
            bad_epochs += 1

        if epoch == 1 or epoch % 10 == 0:
            _log(f"[{stage_name}] epoch={epoch:03d} val_rmse={val['rmse']:.6f} val_f1={val['support_f1']:.4f}")
        if bad_epochs >= cfg.patience:
            break

    if best_state is not None:
        hybrid.load_state_dict(best_state)
    return hybrid


def run_lfista_experiment_dataframe(
    cfg_train: LFISTATrainConfig,
    seed: int,
    measurement_ratio: float,
    data: Dict[str, np.ndarray],
    M_np: np.ndarray,
    log_fn: Optional[Callable[[str], None]],
) -> Tuple[pd.DataFrame, Dict[str, np.ndarray]]:
    """
    One (seed, measurement_ratio) job: train LFISTA variants and return detailed rows
    plus flattened test predictions for optional GT/pred bundles.
    """
    rng = np.random.default_rng(seed)

    X_train = data["X_train"]
    X_val = data["X_val"]
    X_test = data["X_test"]
    Y_train = data["Y_train"]
    Y_val = data["Y_val"]
    Y_test = data["Y_test"]
    Alpha_train = data["Alpha_train"]
    Alpha_val = data["Alpha_val"]
    Alpha_test = data["Alpha_test"]
    Psi_np = data["Psi"]

    m = int(M_np.shape[0])
    B_train = make_measurements(M_np, Y_train, cfg_train.measurement_noise_std, rng)
    B_val = make_measurements(M_np, Y_val, cfg_train.measurement_noise_std, rng)
    B_test = make_measurements(M_np, Y_test, cfg_train.measurement_noise_std, rng)

    device = cfg_train.device
    Psi = torch.tensor(Psi_np, dtype=torch.float32, device=device)
    M_t = torch.tensor(M_np, dtype=torch.float32, device=device)
    A = M_t @ Psi

    def _log(msg: str) -> None:
        if log_fn:
            log_fn(msg)

    _log(f"--- [lfista] seed={seed} rho={measurement_ratio:.2f} ---")

    bg = BackgroundMLP(
        cfg_train.p_input,
        cfg_train.n_output,
        cfg_train.bg_hidden,
        cfg_train.bg_dropout,
        bg_type=cfg_train.bg_type,
    )
    bg = train_background(cfg_train, bg, X_train, Y_train, X_val, Y_val, log_fn)

    sparse_frozen = LFISTAUnrolled(
        A=A,
        K=cfg_train.lfista_steps,
        init_step_scale=cfg_train.init_step_scale,
        init_threshold=cfg_train.init_threshold,
        learn_step_sizes=cfg_train.learn_step_sizes,
        learn_thresholds=cfg_train.learn_thresholds,
        use_momentum=cfg_train.use_momentum,
    )
    hybrid_frozen = HybridLFISTA(bg, sparse_frozen, Psi, M_t)
    hybrid_frozen = train_lfista_stage(
        cfg_train,
        hybrid_frozen,
        X_train,
        Y_train,
        Alpha_train,
        B_train,
        X_val,
        Y_val,
        Alpha_val,
        B_val,
        num_epochs=cfg_train.num_epochs_frozen,
        lr=cfg_train.lr_frozen,
        train_background_flag=False,
        log_fn=log_fn,
    )

    sparse_joint = LFISTAUnrolled(
        A=A,
        K=cfg_train.lfista_steps,
        init_step_scale=cfg_train.init_step_scale,
        init_threshold=cfg_train.init_threshold,
        learn_step_sizes=cfg_train.learn_step_sizes,
        learn_thresholds=cfg_train.learn_thresholds,
        use_momentum=cfg_train.use_momentum,
    )
    hybrid_joint = HybridLFISTA(bg, sparse_joint, Psi, M_t)
    hybrid_joint.load_state_dict(hybrid_frozen.state_dict(), strict=False)
    hybrid_joint = train_lfista_stage(
        cfg_train,
        hybrid_joint,
        X_train,
        Y_train,
        Alpha_train,
        B_train,
        X_val,
        Y_val,
        Alpha_val,
        B_val,
        num_epochs=cfg_train.num_epochs_joint,
        lr=cfg_train.lr_joint,
        train_background_flag=True,
        log_fn=log_fn,
    )

    bg.eval()
    with torch.no_grad():
        y_bg_test = bg(torch.tensor(X_test, dtype=torch.float32, device=device)).cpu().numpy()

    frozen_eval = evaluate_hybrid(hybrid_frozen, X_test, Y_test, Alpha_test, B_test, cfg_train)
    joint_eval = evaluate_hybrid(hybrid_joint, X_test, Y_test, Alpha_test, B_test, cfg_train)
    ml_eval = {
        "rmse": rmse(Y_test, y_bg_test),
        "mae": float(np.mean(np.abs(Y_test - y_bg_test))),
        "relative_l2": relative_l2(Y_test, y_bg_test),
        "support_f1": np.nan,
    }

    with torch.no_grad():
        out_frozen = hybrid_frozen(
            torch.tensor(X_test, dtype=torch.float32, device=device),
            torch.tensor(B_test, dtype=torch.float32, device=device),
        )
        out_joint = hybrid_joint(
            torch.tensor(X_test, dtype=torch.float32, device=device),
            torch.tensor(B_test, dtype=torch.float32, device=device),
        )
    yh_frozen = out_frozen["y_hat"].cpu().numpy()
    ah_frozen = out_frozen["alpha"].cpu().numpy()
    yh_joint = out_joint["y_hat"].cpu().numpy()
    ah_joint = out_joint["alpha"].cpu().numpy()

    rows: List[Dict[str, Any]] = []
    for i in range(len(X_test)):
        rows.append(
            {
                "seed": seed,
                "measurement_ratio": measurement_ratio,
                "method": "ml_only_torch",
                "sample_id": i,
                "rmse": rmse(Y_test[i], y_bg_test[i]),
                "mae": float(np.mean(np.abs(Y_test[i] - y_bg_test[i]))),
                "relative_l2": relative_l2(Y_test[i], y_bg_test[i]),
                "support_f1": np.nan,
                "lambda": np.nan,
                "cs_engine": "none",
                "m": m,
            }
        )
        rows.append(
            {
                "seed": seed,
                "measurement_ratio": measurement_ratio,
                "method": "hybrid_lfista_frozen",
                "sample_id": i,
                "rmse": rmse(Y_test[i], yh_frozen[i]),
                "mae": float(np.mean(np.abs(Y_test[i] - yh_frozen[i]))),
                "relative_l2": relative_l2(Y_test[i], yh_frozen[i]),
                "support_f1": support_f1(Alpha_test[i], ah_frozen[i]),
                "lambda": np.nan,
                "cs_engine": "lfista",
                "m": m,
            }
        )
        rows.append(
            {
                "seed": seed,
                "measurement_ratio": measurement_ratio,
                "method": "hybrid_lfista_joint",
                "sample_id": i,
                "rmse": rmse(Y_test[i], yh_joint[i]),
                "mae": float(np.mean(np.abs(Y_test[i] - yh_joint[i]))),
                "relative_l2": relative_l2(Y_test[i], yh_joint[i]),
                "support_f1": support_f1(Alpha_test[i], ah_joint[i]),
                "lambda": np.nan,
                "cs_engine": "lfista",
                "m": m,
            }
        )

    _log(
        f"[lfista done] seed={seed} rho={measurement_ratio:.2f} | "
        f"ml_torch_rmse={ml_eval['rmse']:.4f} | frozen_rmse={frozen_eval['rmse']:.4f} | joint_rmse={joint_eval['rmse']:.4f}"
    )

    gt_extra: Dict[str, np.ndarray] = {
        "ml_only_torch": np.asarray(y_bg_test, dtype=float).ravel(),
        "hybrid_lfista_frozen": np.asarray(yh_frozen, dtype=float).ravel(),
        "hybrid_lfista_joint": np.asarray(yh_joint, dtype=float).ravel(),
    }
    return pd.DataFrame(rows), gt_extra
