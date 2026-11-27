#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
sir_cs_benchmark_direct_ub.py

Benchmark: direct [u,b] -> y baselines (MLP concat, PCA+MLP, AE+MLP) vs hybrid_fista
and optional LFISTA branch (ml_only_torch, hybrid_lfista_frozen, hybrid_lfista_joint)
on the same synthetic protocol as sir_cs_pipeline_optimized.py.

Artifacts per run:
    <base_dir>/runs/<run_id>/
        tables/           CSVs + summary_focus_direct_ub.csv + parity_pooled.npz (pooled y vs y_hat)
        figures/          01--07 PNGs + 08_example_ground_truth_vs_models.png + 09_parity_ground_truth_vs_prediction.png
        logs/run_console.log
        PROTOCOL.txt
        config.json
        RUN_MANIFEST.txt

External tensors (no make_dataset): run_direct_ub_from_data with data dict
(X_train, Y_train, Alpha_train, Psi, ...); see sir_cs_benchmark_real_well_direct_ub.py (F03-4).

Psi ablation (pilot): use --residual-basis identity|dct; same alpha rule, only Psi changes.
Organize with separate --base-dir per axis, e.g. outputs/direct_ub_psi_ablation/identity vs .../dct.
See docs/direct_ub_psi_ablation.txt.

M-axis (with Psi fixed, usually dct): use --measurement-kind gaussian|subsample.
Reference Gaussian + DCT run: see docs/direct_ub_m_ablation.txt.

Measurement-noise robustness (DCT + subsample, joint-only): see docs/direct_ub_robustness_measurement_noise.txt.
Use --measurement-noise-std FLOAT and optional --robustness-lite (three seeds, full rho grid).

Residual-k robustness (same protocol, vary generator / CS sparsity level k): see docs/direct_ub_robustness_residual_k.txt.
Use --residual-k INT (override cfg.residual_k after profile load).

ASCII-only source.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import warnings
from dataclasses import asdict
from typing import Dict, List, Optional, TextIO, Tuple

from sklearn.exceptions import ConvergenceWarning

import numpy as np
import pandas as pd

import csgm_m2_module as csgm
import direct_ub_baselines as dub
import external_benchmarks as extb
from sir_cs_pipeline_optimized import (
    METHOD_ORDER_DIRECT_UB,
    METHOD_ORDER_DIRECT_UB_JOINT_FOCUS,
    Config,
    MultiOutputMLP,
    apply_config_profile,
    build_lambda_selection_arrays,
    build_measurement_matrix,
    make_dataset,
    merge_gt_pred_bundles,
    method_display_name,
    plot_direct_ub_ground_truth_vs_models,
    plot_parity_ground_truth_vs_predictions,
    power_iteration_lipschitz,
    run_lfista_branch,
    save_all_comparison_plots,
    summarize_results_across_seeds,
    summarize_results_per_seed,
)


class _Tee:
    def __init__(self, real_stdout: TextIO, path: Optional[str]) -> None:
        self._real = real_stdout
        self._path = path
        self._f: Optional[TextIO] = None
        if path:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            self._f = open(path, "w", encoding="utf-8")

    def write(self, s: str) -> None:
        self._real.write(s)
        if self._f:
            self._f.write(s)
            self._f.flush()

    def flush(self) -> None:
        self._real.flush()
        if self._f:
            self._f.flush()

    def close(self) -> None:
        if self._f:
            self._f.close()
            self._f = None


def _log(tee: Optional[_Tee], msg: str) -> None:
    line = f"{msg}\n"
    if tee:
        tee.write(line)
    else:
        print(msg, flush=True)


def _parse_float_list(s: str) -> List[float]:
    out: List[float] = []
    for p in str(s).split(","):
        q = p.strip()
        if q:
            out.append(float(q))
    if not out:
        raise ValueError("empty float list")
    return out


def build_direct_ub_parity_fragment(
    Y_test: np.ndarray,
    Ybg_test: np.ndarray,
    pred_test_mlp: np.ndarray,
    pred_test_pca: np.ndarray,
    pred_test_ae: Optional[np.ndarray],
    run_ae: bool,
    y_hf_stack: Optional[np.ndarray],
    csgm_pred: Optional[Dict[str, np.ndarray]],
    lf_gt: Optional[Dict[str, np.ndarray]],
    joint_only: bool,
) -> Dict[str, np.ndarray]:
    """Flatten test tensors for parity scatter (pooled over samples and output coordinates)."""
    out: Dict[str, np.ndarray] = {
        "y_true": np.asarray(Y_test, dtype=np.float64).ravel(),
        "ml_only": np.asarray(Ybg_test, dtype=np.float64).ravel(),
        "mlp_concat_ub": np.asarray(pred_test_mlp, dtype=np.float64).ravel(),
        "pca_regression_ub": np.asarray(pred_test_pca, dtype=np.float64).ravel(),
    }
    if run_ae and pred_test_ae is not None:
        out["ae_regression_ub"] = np.asarray(pred_test_ae, dtype=np.float64).ravel()
    if y_hf_stack is not None and y_hf_stack.size > 0:
        out["hybrid_fista"] = np.asarray(y_hf_stack, dtype=np.float64).ravel()
    if csgm_pred:
        for k, v in csgm_pred.items():
            out[str(k)] = np.asarray(v, dtype=np.float64).ravel()
    if lf_gt:
        if joint_only:
            k = "hybrid_lfista_joint"
            if k in lf_gt:
                out[k] = np.asarray(lf_gt[k], dtype=np.float64).ravel()
        else:
            for k, v in lf_gt.items():
                out[str(k)] = np.asarray(v, dtype=np.float64).ravel()
    return out


def run_direct_ub_from_data(
    cfg: Config,
    dub_cfg: dub.DirectUBTrainConfig,
    data: Dict[str, np.ndarray],
    seed: int,
    measurement_ratio: float,
    include_hybrid_fista: bool,
    run_ae: bool,
    include_lfista: bool,
    joint_only: bool,
) -> Tuple[pd.DataFrame, Dict[str, np.ndarray], Optional[Dict[str, np.ndarray]]]:
    """
    Same as run_direct_ub_single_setting but with a pre-built data dict (synthetic or real).
    Required keys: X_train, X_val, X_test, Y_train, Y_val, Y_test, Alpha_train, Alpha_val,
    Alpha_test, Psi. Shapes must match cfg.n_train, cfg.n_val, cfg.n_test, cfg.p_input, cfg.n_output.
    Third return: optional dict of (n_ex, L) arrays for 08_example_ground_truth_vs_models.png.
    """
    # Offset 0 keeps the published stream: Omega and measurement noise
    # come from the model seed. A nonzero offset redraws only that stream.
    offset = int(getattr(cfg, "measurement_seed_offset", 0) or 0)
    measurement_seed = int(seed) + offset
    rng = np.random.default_rng(measurement_seed)
    X_train = data["X_train"]
    X_val = data["X_val"]
    X_test = data["X_test"]
    Y_train = data["Y_train"]
    Y_val = data["Y_val"]
    Y_test = data["Y_test"]
    Alpha_test = data["Alpha_test"]
    Psi = data["Psi"]

    m = max(4, int(round(measurement_ratio * cfg.n_output)))
    M = build_measurement_matrix(m, cfg.n_output, cfg.measurement_kind, rng)
    omega_records = getattr(cfg, "omega_records", None)
    if omega_records is not None and str(cfg.measurement_kind) == "subsample":
        coords = [int(np.flatnonzero(M[row])[0]) for row in range(int(M.shape[0]))]
        omega_records.append(
            {
                "model_seed": int(seed),
                "measurement_seed": int(measurement_seed),
                "measurement_seed_offset": int(offset),
                "measurement_ratio": float(measurement_ratio),
                "m": int(m),
                "omega_order": ",".join(str(idx) for idx in coords),
            }
        )
    A = M @ Psi
    L_A = power_iteration_lipschitz(A, n_iter=cfg.power_iteration_n_iter)

    B_train = dub.make_B(Y_train, M, cfg.measurement_noise_std, rng)
    B_val = dub.make_B(Y_val, M, cfg.measurement_noise_std, rng)
    B_test = dub.make_B(Y_test, M, cfg.measurement_noise_std, rng)

    Xb_train = dub.concat_ub(X_train, B_train)
    Xb_val = dub.concat_ub(X_val, B_val)
    Xb_test = dub.concat_ub(X_test, B_test)
    ub_scaler = dub.fit_scaler_ub(Xb_train)

    baseline = MultiOutputMLP(
        hidden_layer_sizes=cfg.baseline_hidden,
        max_iter=cfg.baseline_max_iter,
        learning_rate_init=cfg.baseline_learning_rate_init,
        alpha=cfg.baseline_alpha,
        early_stopping=cfg.baseline_early_stopping,
        random_state=seed,
    )
    baseline.fit(X_train, Y_train)

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
        alpha_model.fit(X_train, data["Alpha_train"])

    strict_paired_b = bool(getattr(cfg, "paper_strict_paired_b", False))
    if strict_paired_b:
        y_sel = Y_val
        ybg_sel = baseline.predict(X_val)
        b_sel = B_val
        z_sel = b_sel - ybg_sel @ M.T
        _ap = alpha_model.predict(X_val) if alpha_model is not None else None
    else:
        y_sel, ybg_sel, b_sel, z_sel, _ap = build_lambda_selection_arrays(
            cfg=cfg,
            M=M,
            X_val=X_val,
            Y_val=Y_val,
            baseline_model=baseline,
            alpha_model=alpha_model,
            rng=rng,
        )

    lam_hf = None
    if include_hybrid_fista:
        lam_hf = extb.hybrid_fista_best_lambda(
            cfg, A, Psi, y_sel, ybg_sel, b_sel, z_sel, L_A
        )

    pred_val_mlp, pred_test_mlp = dub.fit_predict_mlp_concat(
        cfg, seed, Xb_train, Y_train, Xb_val, Y_val, Xb_test, ub_scaler
    )
    pred_val_pca, pred_test_pca, best_r = dub.fit_predict_pca_regression_ub(
        cfg, seed, dub_cfg, Xb_train, Y_train, Xb_val, Y_val, Xb_test, ub_scaler
    )
    pred_test_ae: Optional[np.ndarray] = None
    if run_ae:
        pred_val_ae, pred_test_ae = dub.fit_predict_ae_regression_ub(
            cfg, seed, dub_cfg, Xb_train, Y_train, Xb_val, Y_val, Xb_test, ub_scaler
        )
    pred_test_ae_mask: Optional[np.ndarray] = None
    run_explicit_mask = bool(getattr(cfg, "run_ae_explicit_mask", False))
    if run_explicit_mask:
        if str(cfg.measurement_kind) != "subsample":
            raise ValueError("run_ae_explicit_mask requires measurement_kind=subsample.")
        mask_train = dub.concat_ub(X_train, dub.embed_sparse_observation(B_train, M))
        mask_val = dub.concat_ub(X_val, dub.embed_sparse_observation(B_val, M))
        mask_test = dub.concat_ub(X_test, dub.embed_sparse_observation(B_test, M))
        mask_scaler = dub.fit_scaler_ub(mask_train)
        _pred_val_mask, pred_test_ae_mask = dub.fit_predict_ae_regression_ub(
            cfg,
            seed + 100,
            dub_cfg,
            mask_train,
            Y_train,
            mask_val,
            Y_val,
            mask_test,
            mask_scaler,
        )

    Ybg_test = baseline.predict(X_test)
    nan_f = float("nan")
    rows: List[dict[str, float | int | str]] = []
    n_test = len(X_test)
    y_hf_rows: List[np.ndarray] = []

    for i in range(n_test):
        if strict_paired_b:
            b_i = B_test[i]
        else:
            noise = cfg.measurement_noise_std * rng.normal(size=m)
            b_i = M @ Y_test[i] + noise
        y_ml = Ybg_test[i]
        z_i = b_i - M @ y_ml

        rows.append(
            extb.per_sample_metrics_row(
                seed,
                measurement_ratio,
                "ml_only",
                i,
                Y_test[i],
                y_ml,
                Alpha_test[i],
                np.zeros_like(Alpha_test[i]),
                nan_f,
                "none",
                m,
                support_f1_override=nan_f,
            )
        )
        rows.append(
            extb.per_sample_metrics_row(
                seed,
                measurement_ratio,
                "mlp_concat_ub",
                i,
                Y_test[i],
                pred_test_mlp[i],
                Alpha_test[i],
                np.zeros_like(Alpha_test[i]),
                nan_f,
                "direct_ub",
                m,
                support_f1_override=nan_f,
            )
        )
        rows.append(
            extb.per_sample_metrics_row(
                seed,
                measurement_ratio,
                "pca_regression_ub",
                i,
                Y_test[i],
                pred_test_pca[i],
                Alpha_test[i],
                np.zeros_like(Alpha_test[i]),
                float(best_r),
                "pca_mlp",
                m,
                support_f1_override=nan_f,
            )
        )
        if run_ae:
            rows.append(
                extb.per_sample_metrics_row(
                    seed,
                    measurement_ratio,
                    "ae_regression_ub",
                    i,
                    Y_test[i],
                    pred_test_ae[i],
                    Alpha_test[i],
                    np.zeros_like(Alpha_test[i]),
                    float(dub_cfg.ae_latent_dim),
                    "ae_mlp",
                    m,
                    support_f1_override=nan_f,
                )
            )
        if pred_test_ae_mask is not None:
            rows.append(
                extb.per_sample_metrics_row(
                    seed,
                    measurement_ratio,
                    "ae_regression_ub_mask",
                    i,
                    Y_test[i],
                    pred_test_ae_mask[i],
                    Alpha_test[i],
                    np.zeros_like(Alpha_test[i]),
                    float(dub_cfg.ae_latent_dim),
                    "ae_mlp_mask",
                    m,
                    support_f1_override=nan_f,
                )
            )
        if include_hybrid_fista and lam_hf is not None:
            ah_hf, y_hf = extb.hybrid_fista_predict_one(
                cfg, A, Psi, y_ml, z_i, lam_hf, L_A
            )
            y_hf_rows.append(np.asarray(y_hf, dtype=np.float64))
            rows.append(
                extb.per_sample_metrics_row(
                    seed,
                    measurement_ratio,
                    "hybrid_fista",
                    i,
                    Y_test[i],
                    y_hf,
                    Alpha_test[i],
                    ah_hf,
                    lam_hf,
                    "fista",
                    m,
                )
            )

    df = pd.DataFrame(rows)
    y_hf_stack: Optional[np.ndarray]
    if y_hf_rows:
        y_hf_stack = np.stack(y_hf_rows, axis=0)
    else:
        y_hf_stack = None

    csgm_pred: Optional[Dict[str, np.ndarray]] = None
    csgm_result: Optional[csgm.CSGMM2Result] = None
    if bool(getattr(cfg, "run_csgm_m2", False)):
        csgm_result = csgm.run_csgm_m2_experiment_dataframe(
            cfg=cfg,
            seed=seed,
            measurement_ratio=measurement_ratio,
            X_train=X_train,
            X_val=X_val,
            X_test=X_test,
            Y_train=Y_train,
            Y_val=Y_val,
            Y_test=Y_test,
            Alpha_test=Alpha_test,
            M=M,
            B_val=B_val,
            B_test=B_test,
        )
        df = pd.concat([df, csgm_result.df], ignore_index=True)
        csgm_method = "{}_prior_csgm".format(str(cfg.csgm_prior_type).strip().lower())
        csgm_pred = {csgm_method: csgm_result.predictions}

    lf_gt: Optional[Dict[str, np.ndarray]] = None
    if include_lfista:

        def _lf_log(_msg: str) -> None:
            return None

        lf_df, lf_gt = run_lfista_branch(
            cfg,
            seed,
            measurement_ratio,
            data,
            M,
            _lf_log,
        )
        if joint_only:
            lf_df = lf_df[lf_df["method"] == "hybrid_lfista_joint"].copy()
        df = pd.concat([df, lf_df], ignore_index=True)

    parity_fragment = build_direct_ub_parity_fragment(
        Y_test,
        Ybg_test,
        pred_test_mlp,
        pred_test_pca,
        pred_test_ae,
        run_ae,
        y_hf_stack,
        csgm_pred,
        lf_gt,
        joint_only,
    )
    line_examples: Optional[Dict[str, np.ndarray]] = None
    if int(cfg.n_example_plots) > 0 and n_test > 0:
        n_ex = min(int(cfg.n_example_plots), n_test)
        line_examples = {
            "Y_true": np.asarray(Y_test[:n_ex], dtype=np.float64),
            "ml_only": np.asarray(Ybg_test[:n_ex], dtype=np.float64),
            "mlp_concat_ub": np.asarray(pred_test_mlp[:n_ex], dtype=np.float64),
            "pca_regression_ub": np.asarray(pred_test_pca[:n_ex], dtype=np.float64),
        }
        if run_ae and pred_test_ae is not None:
            line_examples["ae_regression_ub"] = np.asarray(pred_test_ae[:n_ex], dtype=np.float64)
        if y_hf_stack is not None and include_hybrid_fista:
            line_examples["hybrid_fista"] = np.asarray(y_hf_stack[:n_ex], dtype=np.float64)
        if csgm_pred:
            for k, v in csgm_pred.items():
                line_examples[str(k)] = np.asarray(v[:n_ex], dtype=np.float64)
        if lf_gt is not None and include_lfista:
            n_out = int(cfg.n_output)
            n_el = n_test * n_out
            for k in ("hybrid_lfista_joint", "hybrid_lfista_frozen"):
                if k in lf_gt:
                    v = np.asarray(lf_gt[k], dtype=np.float64).ravel()
                    if v.size == n_el:
                        yh = v.reshape(n_test, n_out)
                        line_examples[k] = yh[:n_ex].copy()
    return df, parity_fragment, line_examples


def run_direct_ub_single_setting(
    cfg: Config,
    dub_cfg: dub.DirectUBTrainConfig,
    seed: int,
    measurement_ratio: float,
    include_hybrid_fista: bool,
    run_ae: bool,
    include_lfista: bool,
    joint_only: bool,
) -> Tuple[pd.DataFrame, Dict[str, np.ndarray]]:
    data = make_dataset(cfg, seed=seed)
    df, pfrag, _ = run_direct_ub_from_data(
        cfg,
        dub_cfg,
        data,
        seed,
        measurement_ratio,
        include_hybrid_fista,
        run_ae,
        include_lfista,
        joint_only,
    )
    return df, pfrag


def save_focus_tables(
    run_root: str,
    summary: pd.DataFrame,
    per_seed: pd.DataFrame,
    include_hybrid_fista: bool,
    run_ae: bool,
    include_lfista: bool,
    joint_only: bool,
    include_csgm_m2: bool,
) -> List[str]:
    if joint_only:
        focus = list(METHOD_ORDER_DIRECT_UB_JOINT_FOCUS)
        if not run_ae:
            focus = [x for x in focus if x != "ae_regression_ub"]
        if not include_lfista:
            focus = [x for x in focus if x != "hybrid_lfista_joint"]
    else:
        focus = list(METHOD_ORDER_DIRECT_UB)
        if not include_hybrid_fista:
            focus = [x for x in focus if x != "hybrid_fista"]
        if not run_ae:
            focus = [x for x in focus if x != "ae_regression_ub"]
        if not include_lfista:
            focus = [
                x
                for x in focus
                if x not in ("ml_only_torch", "hybrid_lfista_frozen", "hybrid_lfista_joint")
            ]
    sub_s = summary[summary["method"].isin(focus)].sort_values(["measurement_ratio", "method"])
    sub_p = per_seed[per_seed["method"].isin(focus)].sort_values(
        ["seed", "measurement_ratio", "method"]
    )
    tables = os.path.join(run_root, "tables")
    if joint_only:
        if include_csgm_m2 and not include_lfista:
            p1 = os.path.join(tables, "summary_focus_clp_csgm_vs_ub.csv")
            p2 = os.path.join(tables, "summary_by_seed_focus_clp_csgm_vs_ub.csv")
        else:
            p1 = os.path.join(tables, "summary_focus_lfista_joint_vs_ub.csv")
            p2 = os.path.join(tables, "summary_by_seed_focus_lfista_joint_vs_ub.csv")
    else:
        p1 = os.path.join(tables, "summary_focus_direct_ub.csv")
        p2 = os.path.join(tables, "summary_by_seed_focus_direct_ub.csv")
    sub_s.to_csv(p1, index=False)
    sub_p.to_csv(p2, index=False)
    return [p1, p2]


def write_protocol(
    run_root: str,
    joint_only: bool,
    residual_basis: str,
    measurement_kind: str,
    measurement_noise_std: float,
    residual_k: int,
    include_csgm_m2: bool = False,
    include_lfista: bool = True,
) -> str:
    common = [
        "Direct [u,b] -> y benchmark (methodological protocol)",
        "",
        f"0) residual_basis={residual_basis} (Psi = get_basis(N, residual_basis); y_res = Psi @ alpha per row).",
        f"0a) residual_k={residual_k} (synthetic innovation sparsity level in make_dataset; OMP oracle k in hybrid CS when used).",
        "   Alpha support/amplitudes use the same generator rule as identity runs; only Psi differs in Psi-axis studies.",
        f"0b) measurement_kind={measurement_kind} (M = build_measurement_matrix: gaussian = i.i.d. N(0,1/sqrt(m)) rows;",
        "    subsample = m distinct coordinate indicators, one per row). M-axis studies change only this, keeping other knobs fixed when configured.",
        f"0c) measurement_noise_std={measurement_noise_std} (eta in b = M y + eta on train/val/test and per-sample test rows).",
        "",
        "1) Same synthetic generator and splits as sir_cs_pipeline_optimized (make_dataset).",
        "2) Same M, noise on b, per (seed, measurement_ratio) as hybrid CS evaluation.",
        "3) Train features: [u, b] with b = M y + eta on train/val/test (independent noise per row).",
        "4) StandardScaler fit on train [u,b] only; transform val/test.",
        "5) mlp_concat_ub: sklearn MLPRegressor, same hidden/max_iter/alpha as Config baseline MLP.",
        "6) pca_regression_ub: PCA on Y_train only; r chosen by val RMSE/MAE (cfg.model_selection_metric);",
        "   MLP maps scaled [u,b] to PCA scores; inverse transform to y.",
        "7) ae_regression_ub: PyTorch AE on standardized Y_train; MLP maps [u,b] to latent; decode.",
        "8) ml_only: sklearn MLP on u only (reference).",
    ]
    if joint_only and include_lfista:
        extra = [
            "9) Proposed method only: hybrid_lfista_joint (run_lfista_branch, keep joint rows only).",
            "   No hybrid_fista, ml_only_torch, or hybrid_lfista_frozen in this run.",
            "",
        ]
    elif joint_only:
        extra = [
            "9) CSGM-focused run: hybrid_lfista_joint was intentionally excluded.",
            "   Focus methods are ml_only, mlp_concat_ub, pca_regression_ub, ae_regression_ub, and CLP-CSGM.",
            "",
        ]
    else:
        extra = [
            "8b) hybrid_fista: same validation lambda selection and per-test residual CS as pipeline.",
            "9) LFISTA branch optional: ml_only_torch, hybrid_lfista_frozen, hybrid_lfista_joint from run_lfista_branch.",
            "",
        ]
    if include_csgm_m2:
        extra.extend(
            [
                "10) Conditional Latent-Prior CSGM (CLP-CSGM) branch: ridge_prior_csgm or mlp_prior_csgm.",
                "    AE decoder G(z) is trained on standardized Y_train only.",
                "    Prior h(u) maps context u to z0.",
                "    For validation/test rows, solve:",
                "      z_hat = argmin_z ||M G(z) - b||_2^2 + lambda ||z - z0(u)||_2^2.",
                "    Lambda is selected on validation for the same seed and measurement_ratio.",
                "",
            ]
        )
    text = "\n".join(common + extra)
    path = os.path.join(run_root, "PROTOCOL.txt")
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)
    return path


def write_run_manifest(
    run_root: str,
    run_id: str,
    elapsed_s: float,
    plot_paths: List[str],
    focus_paths: List[str],
    joint_only: bool,
    residual_basis: str,
    measurement_kind: str,
    measurement_noise_std: float,
    residual_k: int,
    include_csgm_m2: bool = False,
    include_lfista: bool = True,
) -> str:
    if joint_only and include_csgm_m2 and not include_lfista:
        title = "Direct [u,b] benchmark: CLP-CSGM vs direct UB baselines."
    elif joint_only:
        title = "Direct [u,b] benchmark: hybrid_lfista_joint vs mlp_concat / PCA / AE (joint-only mode)."
    else:
        title = "Direct [u,b] benchmark (+ optional full LFISTA branch)."
    lines = [
        title,
        f"run_id: {run_id}",
        f"elapsed_seconds: {elapsed_s:.1f}",
        f"residual_basis: {residual_basis}",
        f"measurement_kind: {measurement_kind}",
        f"measurement_noise_std: {measurement_noise_std}",
        f"residual_k: {residual_k}",
        "",
        "tables/",
        "  detailed_results.csv",
        "  summary_by_seed.csv",
        "  summary.csv",
    ]
    if joint_only and include_csgm_m2 and not include_lfista:
        lines.extend(
            [
                "  summary_focus_clp_csgm_vs_ub.csv",
                "  summary_by_seed_focus_clp_csgm_vs_ub.csv",
            ]
        )
    elif joint_only:
        lines.extend(
            [
                "  summary_focus_lfista_joint_vs_ub.csv",
                "  summary_by_seed_focus_lfista_joint_vs_ub.csv",
            ]
        )
    if include_csgm_m2:
        lines.append("  includes CLP-CSGM rows (ridge_prior_csgm/mlp_prior_csgm when enabled)")
    else:
        lines.extend(
            [
                "  summary_focus_direct_ub.csv",
                "  summary_by_seed_focus_direct_ub.csv",
            ]
        )
    lines.extend(
        [
            "  PROTOCOL.txt",
            "",
            "figures/",
        ]
    )
    for p in sorted(plot_paths):
        lines.append(f"  {os.path.relpath(p, run_root)}")
    lines.append("")
    lines.append("focus_tables:")
    for p in focus_paths:
        lines.append(f"  {os.path.relpath(p, run_root)}")
    path = os.path.join(run_root, "RUN_MANIFEST.txt")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    return path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Direct [u,b]->y baselines vs hybrid_fista.")
    p.add_argument("--explore", action="store_true", help="Fast profile (one seed, two ratios).")
    p.add_argument(
        "--base-dir",
        type=str,
        default="outputs/direct_ub_benchmark",
        help="Base directory; runs go to base_dir/runs/<run_id>/.",
    )
    p.add_argument("--run-id", type=str, default="", help="Run folder name (default: timestamp).")
    p.add_argument("--no-hybrid-fista", action="store_true", help="Skip hybrid_fista reference.")
    p.add_argument("--no-ae", action="store_true", help="Skip AE baseline (faster).")
    p.add_argument("--no-plots", action="store_true", help="Skip figure generation.")
    p.add_argument(
        "--no-parity",
        action="store_true",
        help="Skip parity scatter (09_parity_...) and tables/parity_pooled.npz export.",
    )
    p.add_argument(
        "--no-lfista",
        action="store_true",
        help="Skip PyTorch LFISTA branch (ml_only_torch, hybrid_lfista_frozen, hybrid_lfista_joint).",
    )
    p.add_argument(
        "--run-csgm-m2",
        action="store_true",
        help="Enable optional conditional CSGM M2 branch (ridge/MLP prior in AE latent space).",
    )
    p.add_argument(
        "--csgm-prior-type",
        type=str,
        default="ridge",
        choices=("ridge", "mlp"),
        help="CSGM M2 prior h(u)->z0.",
    )
    p.add_argument("--csgm-latent-dim", type=int, default=16)
    p.add_argument("--csgm-ae-epochs", type=int, default=200)
    p.add_argument("--csgm-iters", type=int, default=400)
    p.add_argument("--csgm-restarts", type=int, default=3)
    p.add_argument("--csgm-opt-lr", type=float, default=0.05)
    p.add_argument(
        "--csgm-lambda-grid",
        type=str,
        default="0.0001,0.0003,0.001,0.003,0.01,0.03,0.1",
        help="Comma-separated lambda grid for validation selection.",
    )
    p.add_argument(
        "--joint-only",
        action="store_true",
        help=(
            "Compare only hybrid_lfista_joint to direct [u,b] baselines (mlp_concat, PCA, AE); "
            "omit hybrid_fista, ml_only_torch, hybrid_lfista_frozen. Implies LFISTA on."
        ),
    )
    p.add_argument(
        "--residual-basis",
        type=str,
        default="identity",
        choices=("identity", "dct"),
        help=(
            "Residual operator Psi for make_dataset and CS (identity vs DCT). "
            "Default identity. For Psi-axis pilot, run twice with separate --base-dir (see docs/direct_ub_psi_ablation.txt)."
        ),
    )
    p.add_argument(
        "--measurement-kind",
        type=str,
        default="gaussian",
        choices=("gaussian", "subsample"),
        help=(
            "How M is built: gaussian (dense) vs subsample (coordinate selection). "
            "Default gaussian. For M-axis with DCT Psi, see docs/direct_ub_m_ablation.txt."
        ),
    )
    p.add_argument(
        "--measurement-noise-std",
        type=float,
        default=None,
        help="Override cfg.measurement_noise_std after profile load (default: keep profile value, usually 0.02).",
    )
    p.add_argument(
        "--residual-k",
        type=int,
        default=None,
        help="Override cfg.residual_k after profile load (synthetic residual sparsity / CS k; default: profile value, often 6).",
    )
    p.add_argument(
        "--robustness-lite",
        action="store_true",
        help=(
            "Joint-only profile with three seeds and full rho grid (direct_ub_lfista_joint_robustness_lite). "
            "Mutually exclusive with --explore; requires --joint-only."
        ),
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if bool(args.joint_only) and bool(args.no_lfista):
        print("--joint-only requires LFISTA; remove --no-lfista.", file=sys.stderr)
        sys.exit(2)
    if bool(args.explore) and bool(args.robustness_lite):
        print("Use only one of --explore or --robustness-lite.", file=sys.stderr)
        sys.exit(2)
    if bool(args.robustness_lite) and not bool(args.joint_only):
        print("--robustness-lite requires --joint-only.", file=sys.stderr)
        sys.exit(2)
    warnings.filterwarnings("ignore", category=ConvergenceWarning)
    cfg = Config()
    cfg.log_progress = False
    joint_only = bool(args.joint_only)
    if joint_only:
        if bool(args.robustness_lite):
            cfg.config_profile = "direct_ub_lfista_joint_robustness_lite"
        elif args.explore:
            cfg.config_profile = "direct_ub_lfista_joint_only_explore"
        else:
            cfg.config_profile = "direct_ub_lfista_joint_only"
    elif args.explore:
        cfg.config_profile = "direct_ub_benchmark_explore"
    else:
        cfg.config_profile = "direct_ub_benchmark"
    apply_config_profile(cfg)
    if args.measurement_noise_std is not None:
        cfg.measurement_noise_std = float(args.measurement_noise_std)
    if args.residual_k is not None:
        k_override = int(args.residual_k)
        if k_override < 1:
            print("--residual-k must be >= 1.", file=sys.stderr)
            sys.exit(2)
        cfg.residual_k = k_override
    rb = str(args.residual_basis).strip().lower()
    if rb not in ("identity", "dct"):
        print("--residual-basis must be identity or dct.", file=sys.stderr)
        sys.exit(2)
    cfg.residual_basis = rb  # type: ignore[assignment]
    mk = str(args.measurement_kind).strip().lower()
    if mk not in ("gaussian", "subsample"):
        print("--measurement-kind must be gaussian or subsample.", file=sys.stderr)
        sys.exit(2)
    cfg.measurement_kind = mk  # type: ignore[assignment]

    include_lfista = joint_only or (not bool(args.no_lfista))
    if include_lfista:
        cfg.run_lfista = True
    cfg.run_csgm_m2 = bool(args.run_csgm_m2)
    cfg.csgm_prior_type = str(args.csgm_prior_type).strip().lower()
    cfg.csgm_latent_dim = int(args.csgm_latent_dim)
    cfg.csgm_ae_epochs = int(args.csgm_ae_epochs)
    cfg.csgm_iters = int(args.csgm_iters)
    cfg.csgm_restarts = int(args.csgm_restarts)
    cfg.csgm_opt_lr = float(args.csgm_opt_lr)
    cfg.csgm_lambda_grid = _parse_float_list(str(args.csgm_lambda_grid))

    include_hf = False if joint_only else (not bool(args.no_hybrid_fista))

    dub_cfg = dub.DirectUBTrainConfig()

    run_id = args.run_id.strip() or time.strftime("%Y%m%d_%H%M%S")
    base_dir = os.path.abspath(args.base_dir)
    run_root = os.path.join(base_dir, "runs", run_id)
    tables_dir = os.path.join(run_root, "tables")
    logs_dir = os.path.join(run_root, "logs")
    os.makedirs(tables_dir, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)

    cfg.save_dir = run_root
    cfg.plots_subdir = "figures"

    log_path = os.path.join(logs_dir, "run_console.log")
    old_stdout = sys.stdout
    tee = _Tee(old_stdout, log_path)
    sys.stdout = tee  # type: ignore[assignment]

    run_ae = not bool(args.no_ae)
    all_dfs: List[pd.DataFrame] = []
    parity_bundles: List[Dict[str, np.ndarray]] = []
    first_line_examples: Optional[Dict[str, np.ndarray]] = None
    t0 = time.time()
    job_idx = 0
    total = len(cfg.seeds) * len(cfg.measurement_ratios)
    try:
        _log(tee, f"Run root: {run_root}")
        _log(
            tee,
            f"Profile: {cfg.config_profile} | jobs: {total} | joint_only: {joint_only} | "
            f"hybrid_fista: {include_hf} | ae: {run_ae} | lfista: {include_lfista} | "
            f"csgm_m2: {cfg.run_csgm_m2} | csgm_prior: {cfg.csgm_prior_type} | "
            f"residual_basis: {cfg.residual_basis} | measurement_kind: {cfg.measurement_kind} | "
            f"measurement_noise_std: {cfg.measurement_noise_std} | residual_k: {cfg.residual_k}",
        )
        for seed in cfg.seeds:
            for mr in cfg.measurement_ratios:
                job_idx += 1
                _log(tee, f"--- job {job_idx}/{total} seed={seed} measurement_ratio={mr:.2f} ---")
                data_job = make_dataset(cfg, seed=seed)
                df, pfrag, line_ex = run_direct_ub_from_data(
                    cfg,
                    dub_cfg,
                    data_job,
                    seed,
                    float(mr),
                    include_hf,
                    run_ae,
                    include_lfista,
                    joint_only,
                )
                all_dfs.append(df)
                parity_bundles.append(pfrag)
                if first_line_examples is None and line_ex is not None:
                    first_line_examples = line_ex

        detailed = pd.concat(all_dfs, ignore_index=True)
        per_seed = summarize_results_per_seed(detailed)
        summary = summarize_results_across_seeds(per_seed)
        detailed["method_label"] = detailed["method"].map(method_display_name)
        per_seed["method_label"] = per_seed["method"].map(method_display_name)
        summary["method_label"] = summary["method"].map(method_display_name)

        detailed.to_csv(os.path.join(tables_dir, "detailed_results.csv"), index=False)
        per_seed.to_csv(os.path.join(tables_dir, "summary_by_seed.csv"), index=False)
        summary.to_csv(os.path.join(tables_dir, "summary.csv"), index=False)

        focus_paths = save_focus_tables(
            run_root,
            summary,
            per_seed,
            include_hf,
            run_ae,
            include_lfista,
            joint_only,
            bool(cfg.run_csgm_m2),
        )
        proto_path = write_protocol(
            run_root,
            joint_only,
            str(cfg.residual_basis),
            str(cfg.measurement_kind),
            float(cfg.measurement_noise_std),
            int(cfg.residual_k),
            bool(cfg.run_csgm_m2),
            include_lfista,
        )

        plot_paths: List[str] = []
        if not args.no_plots:
            plot_paths = save_all_comparison_plots(cfg, summary, per_seed)
            _log(tee, f"Figures: {len(plot_paths)} files under {os.path.join(run_root, 'figures')}")
        if (not bool(args.no_plots)) and first_line_examples is not None:
            p08 = os.path.join(run_root, cfg.plots_subdir, "08_example_ground_truth_vs_models.png")
            plot_direct_ub_ground_truth_vs_models(first_line_examples, p08)
            plot_paths.append(p08)
            _log(tee, "Figure: " + p08)
        if not bool(args.no_plots) and not bool(args.no_parity) and parity_bundles:
            merged_parity = merge_gt_pred_bundles(parity_bundles)
            npz_path = os.path.join(tables_dir, "parity_pooled.npz")
            np.savez_compressed(npz_path, **merged_parity)
            parity_png = os.path.join(run_root, cfg.plots_subdir, "09_parity_ground_truth_vs_prediction.png")
            plot_parity_ground_truth_vs_predictions(cfg, merged_parity, parity_png)
            plot_paths.append(npz_path)
            plot_paths.append(parity_png)
            _log(tee, f"Parity: {parity_png} | arrays: {npz_path}")

        cfg_dump = asdict(cfg)
        dub_dump = asdict(dub_cfg)
        dub_dump["pca_r_grid"] = list(dub_dump["pca_r_grid"])
        cfg_dump["direct_ub_benchmark"] = {
            "run_id": run_id,
            "joint_only_lfista_joint": joint_only,
            "include_hybrid_fista": include_hf,
            "include_lfista": include_lfista,
            "include_ae": run_ae,
            "include_csgm_m2": bool(cfg.run_csgm_m2),
            "residual_basis": str(cfg.residual_basis),
            "measurement_kind": str(cfg.measurement_kind),
            "dub_cfg": dub_dump,
            "protocol_txt": proto_path,
            "tables_dir": tables_dir,
            "figures_dir": os.path.join(run_root, "figures"),
        }
        with open(os.path.join(run_root, "config.json"), "w", encoding="utf-8") as f:
            json.dump(cfg_dump, f, indent=2)

        elapsed = time.time() - t0
        manifest_path = write_run_manifest(
            run_root,
            run_id,
            elapsed,
            plot_paths,
            focus_paths + [proto_path],
            joint_only,
            str(cfg.residual_basis),
            str(cfg.measurement_kind),
            float(cfg.measurement_noise_std),
            int(cfg.residual_k),
            bool(cfg.run_csgm_m2),
            include_lfista,
        )
        _log(tee, f"Done in {elapsed:.1f}s | manifest: {manifest_path}")
    finally:
        sys.stdout = old_stdout
        tee.close()

    print(f"Artifacts: {run_root}", flush=True)


if __name__ == "__main__":
    main()
