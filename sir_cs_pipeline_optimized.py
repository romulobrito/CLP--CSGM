
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
sir_cs_pipeline_optimized.py

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
    python sir_cs_pipeline_optimized.py
    python sir_cs_pipeline_optimized.py --profile phase0_baseline
    python sir_cs_pipeline_optimized.py --profile solver_comparison
    python sir_cs_pipeline_optimized.py --profile paper
    python sir_cs_pipeline_optimized.py --profile explore

Perfis:
    paper               defaults na classe Config (grade lambda refinada)
    phase0_baseline     roadmap Fase 0: 10 seeds, rho em 0.2..0.6, outputs/phase0_baseline/
    solver_comparison   Etapa 1: FISTA vs SPGL1; artefactos em outputs/solver_comparison/runs/<id>/
    explore             poucos dados, iteracao rapida

Saídas:
    outputs/
        detailed_results.csv   (uma linha por amostra de teste)
        summary_by_seed.csv    (media por seed; base para IC entre seeds)
        summary.csv            (media e desvio entre seeds; ver colunas)
        config.json
        ../paper/figures/synthetic/01_...10_... (metricas, exemplos, paridade, residuais)
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from dataclasses import asdict, dataclass, field
from typing import TYPE_CHECKING, Callable, Dict, List, Literal, Optional, Tuple, cast

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from lfista_module import LFISTATrainConfig
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
    # direct-UB benchmarks: same sparse measurement vector b across CSGM and [u,b] baselines on val/test.
    paper_strict_paired_b: bool = False

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
    power_iteration_n_iter: int = 100
    # Refined grid (roadmap Fase 0): extends toward 1e-4 to avoid optimum stuck at lower boundary.
    l1_lambda_grid: List[float] = field(
        default_factory=lambda: [1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2]
    )
    use_warm_starts: bool = True
    lambda_selection_max_samples: Optional[int] = None  # use None para usar toda a validação
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
    # Relative to save_dir: figures go under paper/figures/ for LaTeX (add e.g. real/ later).
    plots_subdir: str = "../paper/figures/synthetic"
    n_example_plots: int = 3
    # max pontos no scatter paridade GT vs pred (subamostra para performance)
    max_gt_scatter_points: int = 50000

    # métrica de seleção de lambda
    model_selection_metric: str = "rmse"  # "rmse" ou "mae"

    # logs de andamento (stdout)
    log_progress: bool = True
    # imprimir a cada N amostras no loop de teste; 0 = apenas inicio e fim do teste
    test_log_interval: int = 50

    # "paper": defaults above; "explore": fast iteration; "phase0_baseline": roadmap Fase 0;
    # "solver_comparison": Etapa 1 roadmap — hybrid e cs_only com FISTA e SPGL1 (sem weighted).
    # "lfista_integrated*": mesmo protocolo sintetico + ramo PyTorch LFISTA (artefactos dedicados).
    # "lfista_vs_classical*": Phase 0 + dual_cs_solver (hybrid_fista, ...) + LFISTA; comparacao directa.
    config_profile: Literal[
        "paper",
        "explore",
        "phase0_baseline",
        "solver_comparison",
        "lfista_integrated",
        "lfista_integrated_explore",
        "lfista_vs_classical",
        "lfista_vs_classical_explore",
        "robustness_phase3",
        "robustness_phase3_explore",
        "external_benchmark_stage1",
        "external_benchmark_stage1_explore",
        "direct_ub_benchmark",
        "direct_ub_benchmark_explore",
        "direct_ub_lfista_joint_only",
        "direct_ub_lfista_joint_only_explore",
        "direct_ub_lfista_joint_robustness_lite",
        "real_well_f03_direct_ub",
        "cross_well_vc_direct_ub",
    ] = "paper"
    # Filled when profile is robustness_phase3* (CLI --robustness-axis / --robustness-value).
    robustness_axis: str = ""
    robustness_value_raw: str = ""
    # se True, zera warm-start a cada novo lambda na grade (mais limpo, mais lento)
    reset_warm_start_each_lambda: bool = False

    # Opcional: se definido, append de cada log_line (perfil solver_comparison preenche)
    artifact_log_path: Optional[str] = None

    # Etapa 1: comparar nucleo sparse FISTA vs SPGL1 (PyLops) nos mesmos dados
    dual_cs_solver: bool = False
    spgl1_iter_lim: int = 2000
    spgl1_opt_tol: float = 1e-4
    spgl1_bp_tol: float = 1e-4
    spgl1_ls_tol: float = 1e-4
    # Grade de tau para modo LASSO do SPGL1 (tau>0, sigma=0); afinar se necessario
    spgl1_tau_grid: List[float] = field(
        default_factory=lambda: [1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 0.1]
    )

    # LFISTA (PyTorch): so actived when run_lfista=True; default off preserves classic runs.
    run_lfista: bool = False
    lfista_device: str = ""
    lfista_bg_hidden: Tuple[int, int] = (128, 128)
    lfista_bg_dropout: float = 0.0
    # Background regressor architecture family used inside LFISTA.
    # Supported: "mlp2" (default, two hidden layers), "shallow" (one hidden),
    # "linear" (single Linear). Lower capacity preserves residual sparsity.
    lfista_bg_type: str = "mlp2"
    lfista_steps: int = 8
    lfista_learn_step_sizes: bool = True
    lfista_learn_thresholds: bool = True
    lfista_init_step_scale: float = 1.0
    lfista_init_threshold: float = 1e-2
    lfista_use_momentum: bool = True
    lfista_batch_size: int = 128
    lfista_num_epochs_bg: int = 80
    lfista_num_epochs_frozen: int = 60
    lfista_num_epochs_joint: int = 80
    lfista_lr_bg: float = 1e-3
    lfista_lr_frozen: float = 1e-3
    lfista_lr_joint: float = 5e-4
    lfista_weight_decay: float = 1e-5
    lfista_patience: int = 12
    lfista_loss_alpha_weight: float = 0.0
    lfista_loss_l1_alpha_weight: float = 0.0

    # CSGM M2 (conditional generative-prior CS): optional direct-UB branch.
    run_csgm_m2: bool = False
    run_csgm_ablations: bool = False
    run_ae_explicit_mask: bool = False
    csgm_device: str = ""
    csgm_prior_type: str = "ridge"  # "ridge" or "mlp"
    csgm_latent_dim: int = 16
    csgm_hidden_dim: int = 128
    csgm_ae_epochs: int = 200
    csgm_batch_size: int = 64
    csgm_ae_lr: float = 1e-3
    csgm_weight_decay: float = 1e-5
    csgm_iters: int = 400
    csgm_opt_lr: float = 0.05
    csgm_restarts: int = 3
    csgm_lambda_grid: List[float] = field(
        default_factory=lambda: [1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1]
    )
    csgm_ridge_alpha: float = 1.0
    csgm_prior_hidden: Tuple[int, int] = (128, 128)
    csgm_prior_max_iter: int = 500
    csgm_prior_learning_rate_init: float = 1e-3
    csgm_prior_alpha: float = 1e-4
    csgm_prior_early_stopping: bool = True


# ============================================================
# 2) Utilidades matemáticas
# ============================================================


def apply_config_profile(cfg: Config) -> None:
    """Adjust hyperparameters per profile (paper = defaults in Config dataclass)."""
    if cfg.config_profile == "lfista_integrated":
        cfg.seeds = [7, 13, 23, 29, 31, 37, 41, 43, 47, 53]
        cfg.measurement_ratios = [0.2, 0.3, 0.4, 0.5, 0.6]
        cfg.l1_lambda_grid = [1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2]
        cfg.save_dir = "outputs/lfista_integrated"
        cfg.plots_subdir = "../paper/figures/lfista_integrated"
        cfg.run_lfista = True
        return
    if cfg.config_profile == "lfista_integrated_explore":
        cfg.seeds = [7]
        cfg.measurement_ratios = [0.3, 0.5]
        cfg.n_train = 600
        cfg.n_val = 80
        cfg.n_test = 100
        cfg.fista_max_iter = 150
        cfg.l1_lambda_grid = [1e-3, 1e-2, 1e-1]
        cfg.lambda_selection_max_samples = 80
        cfg.power_iteration_n_iter = 80
        cfg.save_dir = "outputs/lfista_integrated"
        cfg.plots_subdir = "../paper/figures/lfista_integrated"
        cfg.run_lfista = True
        cfg.lfista_num_epochs_bg = 25
        cfg.lfista_num_epochs_frozen = 20
        cfg.lfista_num_epochs_joint = 25
        cfg.lfista_steps = 5
        return
    if cfg.config_profile == "lfista_vs_classical":
        # Etapa 2 vs hibrido classico: mesmo protocolo Phase 0 + hybrid_fista (FISTA) + LFISTA PyTorch.
        cfg.seeds = [7, 13, 23, 29, 31, 37, 41, 43, 47, 53]
        cfg.measurement_ratios = [0.2, 0.3, 0.4, 0.5, 0.6]
        cfg.l1_lambda_grid = [1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2]
        cfg.save_dir = "outputs/lfista_vs_classical"
        cfg.plots_subdir = "../paper/figures/lfista_vs_classical"
        cfg.dual_cs_solver = True
        cfg.run_weighted_hybrid = False
        cfg.run_cs_only = True
        cfg.run_lfista = True
        return
    if cfg.config_profile == "lfista_vs_classical_explore":
        cfg.seeds = [7]
        cfg.measurement_ratios = [0.3, 0.5]
        cfg.n_train = 600
        cfg.n_val = 80
        cfg.n_test = 100
        cfg.fista_max_iter = 150
        cfg.l1_lambda_grid = [1e-3, 1e-2, 1e-1]
        cfg.lambda_selection_max_samples = 80
        cfg.power_iteration_n_iter = 80
        cfg.save_dir = "outputs/lfista_vs_classical"
        cfg.plots_subdir = "../paper/figures/lfista_vs_classical"
        cfg.dual_cs_solver = True
        cfg.run_weighted_hybrid = False
        cfg.run_cs_only = True
        cfg.run_lfista = True
        cfg.lfista_num_epochs_bg = 25
        cfg.lfista_num_epochs_frozen = 20
        cfg.lfista_num_epochs_joint = 25
        cfg.lfista_steps = 5
        return
    if cfg.config_profile == "phase0_baseline":
        # Roadmap Fase 0: frozen baseline, 10 seeds, rho up to 0.6, separate output dir.
        cfg.seeds = [7, 13, 23, 29, 31, 37, 41, 43, 47, 53]
        cfg.measurement_ratios = [0.2, 0.3, 0.4, 0.5, 0.6]
        cfg.l1_lambda_grid = [1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2]
        cfg.save_dir = "outputs/phase0_baseline"
        cfg.plots_subdir = "../paper/figures/phase0_baseline"
        return
    if cfg.config_profile == "solver_comparison":
        # Roadmap Etapa 1: mesmo protocolo Phase 0, hybrid e cs_only com FISTA + SPGL1 (sem weighted).
        cfg.seeds = [7, 13, 23, 29, 31, 37, 41, 43, 47, 53]
        cfg.measurement_ratios = [0.2, 0.3, 0.4, 0.5, 0.6]
        cfg.l1_lambda_grid = [1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2]
        cfg.save_dir = "outputs/solver_comparison"
        cfg.plots_subdir = "../paper/figures/solver_comparison"
        cfg.dual_cs_solver = True
        cfg.run_weighted_hybrid = False
        cfg.run_cs_only = True
        return
    if cfg.config_profile == "explore":
        cfg.seeds = [7]
        cfg.measurement_ratios = [0.3, 0.5]
        cfg.n_train = 600
        cfg.n_val = 80
        cfg.n_test = 100
        cfg.fista_max_iter = 150
        cfg.l1_lambda_grid = [1e-3, 1e-2, 1e-1]
        cfg.lambda_selection_max_samples = 80
        cfg.power_iteration_n_iter = 80
        return
    if cfg.config_profile == "robustness_phase3":
        # Roadmap Etapa 3A: same protocol as lfista_vs_classical; save_dir set in main() per axis/value.
        cfg.seeds = [7, 13, 23, 29, 31, 37, 41, 43, 47, 53]
        cfg.measurement_ratios = [0.2, 0.3, 0.4, 0.5, 0.6]
        cfg.l1_lambda_grid = [1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2]
        cfg.dual_cs_solver = True
        cfg.run_weighted_hybrid = False
        cfg.run_cs_only = True
        cfg.run_lfista = True
        return
    if cfg.config_profile == "robustness_phase3_explore":
        cfg.seeds = [7]
        cfg.measurement_ratios = [0.3, 0.5]
        cfg.n_train = 600
        cfg.n_val = 80
        cfg.n_test = 100
        cfg.fista_max_iter = 150
        cfg.l1_lambda_grid = [1e-3, 1e-2, 1e-1]
        cfg.lambda_selection_max_samples = 80
        cfg.power_iteration_n_iter = 80
        cfg.dual_cs_solver = True
        cfg.run_weighted_hybrid = False
        cfg.run_cs_only = True
        cfg.run_lfista = True
        cfg.lfista_num_epochs_bg = 25
        cfg.lfista_num_epochs_frozen = 20
        cfg.lfista_num_epochs_joint = 25
        cfg.lfista_steps = 5
        return
    if cfg.config_profile == "external_benchmark_stage1":
        cfg.seeds = [7, 13, 23, 29, 31, 37, 41, 43, 47, 53]
        cfg.measurement_ratios = [0.2, 0.3, 0.4, 0.5, 0.6]
        cfg.l1_lambda_grid = [1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2]
        cfg.plots_subdir = "figures"
        return
    if cfg.config_profile == "external_benchmark_stage1_explore":
        cfg.seeds = [7]
        cfg.measurement_ratios = [0.3, 0.5]
        cfg.n_train = 600
        cfg.n_val = 80
        cfg.n_test = 100
        cfg.fista_max_iter = 150
        cfg.l1_lambda_grid = [1e-3, 1e-2, 1e-1]
        cfg.lambda_selection_max_samples = 80
        cfg.power_iteration_n_iter = 80
        cfg.plots_subdir = "figures"
        return
    if cfg.config_profile == "direct_ub_benchmark":
        cfg.seeds = [7, 13, 23, 29, 31, 37, 41, 43, 47, 53]
        cfg.measurement_ratios = [0.2, 0.3, 0.4, 0.5, 0.6]
        cfg.l1_lambda_grid = [1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2]
        cfg.plots_subdir = "figures"
        return
    if cfg.config_profile == "direct_ub_benchmark_explore":
        cfg.seeds = [7]
        cfg.measurement_ratios = [0.3, 0.5]
        cfg.n_train = 600
        cfg.n_val = 80
        cfg.n_test = 100
        cfg.fista_max_iter = 150
        cfg.l1_lambda_grid = [1e-3, 1e-2, 1e-1]
        cfg.lambda_selection_max_samples = 80
        cfg.power_iteration_n_iter = 80
        cfg.lfista_num_epochs_bg = 25
        cfg.lfista_num_epochs_frozen = 20
        cfg.lfista_num_epochs_joint = 25
        cfg.lfista_steps = 5
        cfg.plots_subdir = "figures"
        return
    if cfg.config_profile == "direct_ub_lfista_joint_only":
        cfg.seeds = [7, 13, 23, 29, 31, 37, 41, 43, 47, 53]
        cfg.measurement_ratios = [0.2, 0.3, 0.4, 0.5, 0.6]
        cfg.l1_lambda_grid = [1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2]
        cfg.plots_subdir = "figures"
        return
    if cfg.config_profile == "direct_ub_lfista_joint_only_explore":
        cfg.seeds = [7]
        cfg.measurement_ratios = [0.3, 0.5]
        cfg.n_train = 600
        cfg.n_val = 80
        cfg.n_test = 100
        cfg.fista_max_iter = 150
        cfg.l1_lambda_grid = [1e-3, 1e-2, 1e-1]
        cfg.lambda_selection_max_samples = 80
        cfg.power_iteration_n_iter = 80
        cfg.lfista_num_epochs_bg = 25
        cfg.lfista_num_epochs_frozen = 20
        cfg.lfista_num_epochs_joint = 25
        cfg.lfista_steps = 5
        cfg.plots_subdir = "figures"
        return
    # Subset of seeds, full rho grid: faster robustness sweeps vs full ten-seed paper runs.
    if cfg.config_profile == "direct_ub_lfista_joint_robustness_lite":
        cfg.seeds = [7, 23, 41]
        cfg.measurement_ratios = [0.2, 0.3, 0.4, 0.5, 0.6]
        cfg.l1_lambda_grid = [1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2]
        cfg.plots_subdir = "figures"
        return
    if cfg.config_profile == "real_well_f03_direct_ub":
        cfg.seeds = [7, 23, 41]
        cfg.measurement_ratios = [0.2, 0.3, 0.4, 0.5, 0.6]
        cfg.l1_lambda_grid = [1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2]
        cfg.lfista_num_epochs_bg = 25
        cfg.lfista_num_epochs_frozen = 20
        cfg.lfista_num_epochs_joint = 25
        cfg.lfista_steps = 5
        cfg.log_progress = False
        cfg.plots_subdir = "figures"
        cfg.run_lfista = True
        return
    if cfg.config_profile == "cross_well_vc_direct_ub":
        cfg.seeds = [7, 23, 41]
        cfg.measurement_ratios = [0.05, 0.1, 0.15, 0.2, 0.3]
        cfg.l1_lambda_grid = [1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2]
        cfg.lfista_num_epochs_bg = 150
        cfg.lfista_num_epochs_frozen = 40
        cfg.lfista_num_epochs_joint = 50
        cfg.lfista_steps = 6
        cfg.log_progress = False
        cfg.plots_subdir = "figures"
        cfg.run_lfista = True
        return


def robustness_value_slug(raw: str) -> str:
    """Filesystem-safe token from user value string (e.g. 0.02 -> 0p02)."""
    t = raw.strip()
    return t.replace(".", "p").replace("-", "m")


def apply_robustness_param_override(cfg: Config, axis: str, val_str: str) -> None:
    """Set one synthetic knob; other fields stay at baseline profile defaults."""
    key = axis.strip()
    if key == "residual_k":
        cfg.residual_k = int(round(float(val_str)))
        return
    if key == "measurement_noise_std":
        cfg.measurement_noise_std = float(val_str)
        return
    if key == "residual_amplitude":
        cfg.residual_amplitude = float(val_str)
        return
    if key == "output_noise_std":
        cfg.output_noise_std = float(val_str)
        return
    if key == "measurement_ratio":
        rho = float(val_str)
        cfg.measurement_ratios = [rho]
        return
    raise ValueError(
        "Unknown robustness axis: "
        + key
        + " (expected residual_k, measurement_noise_std, residual_amplitude, "
        + "output_noise_std, measurement_ratio)"
    )


def layout_robustness_phase3_run(cfg: Config, run_id: str, axis: str, value_slug: str) -> None:
    """
    outputs/robustness_phase3/<axis>/v_<slug>/runs/<run_id>/
    paper/figures/robustness_phase3/<axis>/v_<slug>/runs/<run_id>/
    Symlink outputs/robustness_phase3/<axis>/LATEST -> v_<slug>/runs/<run_id>
    """
    axis_safe = axis.replace(os.sep, "").replace(".", "")
    if ".." in axis_safe or axis_safe == "":
        raise ValueError("Invalid robustness axis")
    slug_safe = value_slug.replace(os.sep, "").replace("..", "")
    value_dir = "v_" + slug_safe
    base_axis = os.path.join("outputs", "robustness_phase3", axis_safe)
    base_save = os.path.join(base_axis, value_dir)
    cfg.save_dir = os.path.join(base_save, "runs", run_id)
    cwd = os.getcwd()
    fig_abs = os.path.abspath(
        os.path.join(
            cwd, "paper", "figures", "robustness_phase3", axis_safe, value_dir, "runs", run_id
        )
    )
    save_abs = os.path.abspath(os.path.join(cwd, cfg.save_dir))
    cfg.plots_subdir = os.path.relpath(fig_abs, save_abs)
    os.makedirs(fig_abs, exist_ok=True)
    os.makedirs(save_abs, exist_ok=True)
    latest = os.path.join(cwd, base_axis, "LATEST")
    try:
        os.remove(latest)
    except FileNotFoundError:
        pass
    rel_target = os.path.join(value_dir, "runs", run_id)
    os.symlink(rel_target, latest, target_is_directory=False)

    readme = os.path.join(save_abs, "README_RUN.txt")
    started = time.strftime("%Y-%m-%dT%H:%M:%S")
    lines = [
        "SIR-CS robustness_phase3 (Etapa 3 roadmap): one axis value per run.",
        "run_id: " + run_id,
        "robustness_axis: " + axis_safe,
        "value_slug: " + slug_safe,
        "started_local: " + started,
        "cwd: " + cwd,
        "argv: " + json.dumps(sys.argv),
        "",
        "Artifacts: detailed_results.csv, summary*.csv, config.json, PROTOCOL.txt, run_console.log",
        "Figures (repo): paper/figures/robustness_phase3/"
        + axis_safe
        + "/"
        + value_dir
        + "/runs/"
        + run_id
        + "/",
        "",
        "Symlink: outputs/robustness_phase3/" + axis_safe + "/LATEST -> " + rel_target,
        "",
    ]
    with open(readme, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def log_line(cfg: Config, msg: str) -> None:
    """Print com flush para aparecer em tempo real em pipelines longos."""
    if cfg.log_progress:
        print(msg, flush=True)
    if cfg.artifact_log_path:
        with open(cfg.artifact_log_path, "a", encoding="utf-8") as fp:
            fp.write(msg + "\n")


def layout_solver_comparison_run(cfg: Config, run_id: str) -> None:
    """
    Organiza artefactos da Etapa 1 sob outputs/solver_comparison/runs/<run_id>/
    e figuras em paper/figures/solver_comparison/runs/<run_id>/.
    Atualiza cfg.save_dir, cfg.plots_subdir e cria symlink outputs/solver_comparison/LATEST.
    """
    base_save = cfg.save_dir
    cfg.save_dir = os.path.join(base_save, "runs", run_id)
    cwd = os.getcwd()
    fig_abs = os.path.abspath(
        os.path.join(cwd, "paper", "figures", "solver_comparison", "runs", run_id)
    )
    save_abs = os.path.abspath(os.path.join(cwd, cfg.save_dir))
    cfg.plots_subdir = os.path.relpath(fig_abs, save_abs)
    os.makedirs(fig_abs, exist_ok=True)
    os.makedirs(save_abs, exist_ok=True)

    latest = os.path.join(cwd, base_save, "LATEST")
    try:
        os.remove(latest)
    except FileNotFoundError:
        pass
    os.symlink(os.path.join("runs", run_id), latest, target_is_directory=False)

    readme = os.path.join(save_abs, "README_RUN.txt")
    started = time.strftime("%Y-%m-%dT%H:%M:%S")
    lines = [
        "SIR-CS solver_comparison run",
        "run_id: " + run_id,
        "started_local: " + started,
        "cwd: " + cwd,
        "argv: " + json.dumps(sys.argv),
        "",
        "Artifacts (this folder):",
        "  detailed_results.csv, summary_by_seed.csv, summary.csv",
        "  config.json, PROTOCOL.txt, run_console.log, README_RUN.txt",
        "",
        "Figures (relative to repo root):",
        "  paper/figures/solver_comparison/runs/" + run_id + "/",
        "",
        "Symlink: outputs/solver_comparison/LATEST -> runs/" + run_id,
        "",
    ]
    with open(readme, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def layout_lfista_integrated_run(cfg: Config, run_id: str) -> None:
    """
    Artefactos em outputs/lfista_integrated/runs/<run_id>/ e figuras em
    paper/figures/lfista_integrated/runs/<run_id>/.
    """
    base_save = cfg.save_dir
    cfg.save_dir = os.path.join(base_save, "runs", run_id)
    cwd = os.getcwd()
    fig_abs = os.path.abspath(
        os.path.join(cwd, "paper", "figures", "lfista_integrated", "runs", run_id)
    )
    save_abs = os.path.abspath(os.path.join(cwd, cfg.save_dir))
    cfg.plots_subdir = os.path.relpath(fig_abs, save_abs)
    os.makedirs(fig_abs, exist_ok=True)
    os.makedirs(save_abs, exist_ok=True)

    latest = os.path.join(cwd, base_save, "LATEST")
    try:
        os.remove(latest)
    except FileNotFoundError:
        pass
    os.symlink(os.path.join("runs", run_id), latest, target_is_directory=False)

    readme = os.path.join(save_abs, "README_RUN.txt")
    started = time.strftime("%Y-%m-%dT%H:%M:%S")
    lines = [
        "SIR-CS lfista_integrated run (PyTorch LFISTA + classic pipeline).",
        "run_id: " + run_id,
        "started_local: " + started,
        "cwd: " + cwd,
        "argv: " + json.dumps(sys.argv),
        "",
        "Artifacts (this folder):",
        "  detailed_results.csv, summary_by_seed.csv, summary.csv",
        "  config.json, PROTOCOL.txt, run_console.log, README_RUN.txt",
        "",
        "Figures (repo): paper/figures/lfista_integrated/runs/" + run_id + "/",
        "",
        "Symlink: outputs/lfista_integrated/LATEST -> runs/" + run_id,
        "",
    ]
    with open(readme, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def layout_lfista_vs_classical_run(cfg: Config, run_id: str) -> None:
    """
    Artefactos em outputs/lfista_vs_classical/runs/<run_id>/ e figuras em
    paper/figures/lfista_vs_classical/runs/<run_id>/.
    Comparacao: hybrid_fista (FISTA classico no residual) vs LFISTA + ml_only / ml_only_torch.
    """
    base_save = cfg.save_dir
    cfg.save_dir = os.path.join(base_save, "runs", run_id)
    cwd = os.getcwd()
    fig_abs = os.path.abspath(
        os.path.join(cwd, "paper", "figures", "lfista_vs_classical", "runs", run_id)
    )
    save_abs = os.path.abspath(os.path.join(cwd, cfg.save_dir))
    cfg.plots_subdir = os.path.relpath(fig_abs, save_abs)
    os.makedirs(fig_abs, exist_ok=True)
    os.makedirs(save_abs, exist_ok=True)

    latest = os.path.join(cwd, base_save, "LATEST")
    try:
        os.remove(latest)
    except FileNotFoundError:
        pass
    os.symlink(os.path.join("runs", run_id), latest, target_is_directory=False)

    readme = os.path.join(save_abs, "README_RUN.txt")
    started = time.strftime("%Y-%m-%dT%H:%M:%S")
    lines = [
        "SIR-CS lfista_vs_classical run (hybrid_fista + LFISTA + SPGL1 baselines, same protocol).",
        "run_id: " + run_id,
        "started_local: " + started,
        "cwd: " + cwd,
        "argv: " + json.dumps(sys.argv),
        "",
        "Artifacts (this folder):",
        "  detailed_results.csv, summary_by_seed.csv, summary.csv",
        "  summary_focus_*.csv (subset for ml_only, ml_only_torch, hybrid_fista, LFISTA)",
        "  FOCUS_COMPARISON.txt",
        "  config.json, PROTOCOL.txt, run_console.log, README_RUN.txt",
        "",
        "Figures (repo): paper/figures/lfista_vs_classical/runs/" + run_id + "/",
        "",
        "Symlink: outputs/lfista_vs_classical/LATEST -> runs/" + run_id,
        "",
    ]
    with open(readme, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


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
    """
    Orthonormal basis Psi such that y = Psi @ alpha, alpha = Psi.T @ y.

    Supported: identity, dct, haar, db4, sym4, fd1.
    Wavelet bases (haar, db4, sym4) require n = power of 2 and use pywt
    periodization. fd1 is a QR-orthogonalized DC+forward-difference basis.
    """
    key = basis_name.strip().lower()
    if key == "identity":
        return np.eye(n)
    if key == "dct":
        return orthonormal_dct_matrix(n)
    if key in ("haar", "db4", "sym4"):
        from bases_extra import build_wavelet_basis

        return build_wavelet_basis(n, key)
    if key == "fd1":
        from bases_extra import build_fd1_basis

        return build_fd1_basis(n)
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


def power_iteration_lipschitz(A: np.ndarray, n_iter: int = 100, v0: Optional[np.ndarray] = None) -> float:
    """
    Estima ||A||_2^2, a constante de Lipschitz do gradiente de 0.5||Ax-b||^2.
    """
    n = A.shape[1]
    if v0 is None:
        v = np.random.randn(n)
    else:
        v = np.asarray(v0, dtype=float).reshape(-1).copy()
        if v.size != n:
            raise ValueError("v0 deve ter dimensão compatível com o número de colunas de A.")
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
    L: Optional[float] = None,
) -> np.ndarray:
    """
    Resolve:
        min_x 0.5 ||A x - b||_2^2 + lam * sum_i weights_i |x_i|
    Se weights=None, usa weights_i = 1.
    """
    n = A.shape[1]
    if weights is None:
        weights = np.ones(n)
    weights = np.asarray(weights, dtype=float).reshape(-1)
    if weights.size != n:
        raise ValueError(
            f"weights length {weights.size} != A columns {n}",
        )

    if x0 is None:
        x = np.zeros(n)
    else:
        x = x0.copy()

    y = x.copy()
    t = 1.0
    if L is None:
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


def spgl1_lasso_alpha(A: np.ndarray, b: np.ndarray, tau: float, cfg: Config) -> np.ndarray:
    """
    SPGL1 (PyLops) em modo LASSO: tau > 0, sigma = 0.
    Operador A denso via MatrixMult (adequado para measurement_kind gaussian).
    """
    import pylops
    from pylops.optimization.sparsity import spgl1

    Op = pylops.MatrixMult(np.asarray(A, dtype=np.float64))
    y = np.asarray(b, dtype=np.float64).reshape(-1)
    xinv, _, _info = spgl1(
        Op,
        y,
        tau=float(tau),
        sigma=0.0,
        show=False,
        iter_lim=int(cfg.spgl1_iter_lim),
        opt_tol=float(cfg.spgl1_opt_tol),
        bp_tol=float(cfg.spgl1_bp_tol),
        ls_tol=float(cfg.spgl1_ls_tol),
    )
    return np.asarray(xinv, dtype=float).reshape(-1)


def solve_sparse_alpha(
    A: np.ndarray,
    b: np.ndarray,
    cs_engine: str,
    regularization: float,
    weights: Optional[np.ndarray],
    L_A: Optional[float],
    cfg: Config,
    x0: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Dispatcher: fista (penalized L1) ou spgl1 (LASSO via PyLops)."""
    if cs_engine == "fista":
        return fista_lasso(
            A=A,
            b=b,
            lam=float(regularization),
            weights=weights,
            max_iter=cfg.fista_max_iter,
            tol=cfg.fista_tol,
            x0=x0,
            L=L_A,
        )
    if cs_engine == "spgl1":
        if weights is not None:
            raise ValueError("spgl1: weighted l1 ainda nao suportado neste pipeline.")
        return spgl1_lasso_alpha(A, b, tau=float(regularization), cfg=cfg)
    raise ValueError(f"cs_engine desconhecido: {cs_engine}")


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


def build_lambda_selection_arrays(
    cfg: Config,
    M: np.ndarray,
    X_val: np.ndarray,
    Y_val: np.ndarray,
    baseline_model: MultiOutputMLP,
    alpha_model: Optional[MultiOutputMLP],
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """
    Um unico subsample de validacao (se configurado) e medicoes b = M y + eta
    compartilhadas por hybrid, weighted_hybrid e cs_only na escolha de lambda.
    z = b - M y_bg usa o mesmo b ruidoso do teste quando measurement_noise_std > 0.
    """
    n_val = len(X_val)
    if cfg.lambda_selection_max_samples is not None and cfg.lambda_selection_max_samples < n_val:
        idx = rng.choice(n_val, size=cfg.lambda_selection_max_samples, replace=False)
        idx.sort()
        X_sel = X_val[idx]
        Y_sel = Y_val[idx]
    else:
        X_sel = X_val
        Y_sel = Y_val

    Ybg_sel = baseline_model.predict(X_sel)
    b_clean = Y_sel @ M.T
    if cfg.measurement_noise_std > 0.0:
        noise = cfg.measurement_noise_std * rng.normal(size=b_clean.shape)
        b_sel = b_clean + noise
    else:
        b_sel = b_clean
    z_sel = b_sel - Ybg_sel @ M.T
    alpha_pred_sel = alpha_model.predict(X_sel) if alpha_model is not None else None
    return Y_sel, Ybg_sel, b_sel, z_sel, alpha_pred_sel


def select_regularization_for_cs_method(
    log_label: str,
    method_kind: str,
    cs_engine: str,
    cfg: Config,
    A: np.ndarray,
    Psi: np.ndarray,
    y_sel: np.ndarray,
    ybg_sel: np.ndarray,
    b_sel: np.ndarray,
    z_sel: np.ndarray,
    alpha_pred_sel: Optional[np.ndarray],
    L_A: Optional[float],
) -> float:
    """
    Seleciona lambda (FISTA) ou tau (SPGL1 LASSO) em validacao para hybrid / weighted_hybrid / cs_only.
    log_label: texto no log (ex.: hybrid_fista); method_kind: hybrid | weighted_hybrid | cs_only.
    """
    if cs_engine == "fista":
        grid: List[float] = list(cfg.l1_lambda_grid)
        reg_short = "lam"
    elif cs_engine == "spgl1":
        grid = list(cfg.spgl1_tau_grid)
        reg_short = "tau"
    else:
        raise ValueError(f"cs_engine invalido: {cs_engine}")

    if len(grid) == 0:
        raise ValueError("grade de regularizacao vazia")

    best_reg = float(grid[0])
    best_score = float("inf")

    if L_A is None:
        L_A = power_iteration_lipschitz(A, n_iter=cfg.power_iteration_n_iter)

    n_code = A.shape[1]
    n_samples = len(y_sel)
    use_warm = cfg.use_warm_starts and cs_engine == "fista"
    warm_starts = np.zeros((n_samples, n_code)) if use_warm else None

    t_method0 = time.perf_counter()
    grid_n = len(grid)
    log_line(
        cfg,
        f"    [cs] {log_label} ({cs_engine}) start | n_sel={n_samples} | grid={grid_n} {reg_short}",
    )

    for j, reg in enumerate(grid):
        if cfg.reset_warm_start_each_lambda and warm_starts is not None:
            warm_starts.fill(0.0)
        t_lam0 = time.perf_counter()
        preds = np.zeros_like(y_sel)
        for i in range(n_samples):
            if method_kind == "hybrid":
                rhs = z_sel[i]
                weights_i = None
            elif method_kind == "weighted_hybrid":
                if alpha_pred_sel is None:
                    raise ValueError("weighted_hybrid requer alpha_pred_sel.")
                rhs = z_sel[i]
                weights_i = build_weights_from_alpha_prediction(alpha_pred_sel[i], cfg)
            elif method_kind == "cs_only":
                rhs = b_sel[i]
                weights_i = None
            else:
                raise ValueError(f"method_kind desconhecido: {method_kind}")

            alpha_hat = solve_sparse_alpha(
                A=A,
                b=rhs,
                cs_engine=cs_engine,
                regularization=float(reg),
                weights=weights_i,
                L_A=L_A,
                cfg=cfg,
                x0=warm_starts[i] if warm_starts is not None else None,
            )
            if method_kind in ("hybrid", "weighted_hybrid"):
                y_hat = ybg_sel[i] + Psi @ alpha_hat
            else:
                y_hat = Psi @ alpha_hat

            if warm_starts is not None:
                warm_starts[i] = alpha_hat
            preds[i] = y_hat

        score = evaluate_metric(y_sel, preds, cfg.model_selection_metric)
        dt_lam = time.perf_counter() - t_lam0
        log_line(
            cfg,
            f"    [cs] {log_label} grid {j + 1}/{grid_n} {reg_short}={reg:g} "
            f"{cfg.model_selection_metric}={score:.6f} ({dt_lam:.2f}s)",
        )
        if score < best_score:
            best_score = score
            best_reg = float(reg)

    dt_method = time.perf_counter() - t_method0
    log_line(
        cfg,
        f"    [cs] {log_label} ({cs_engine}) best_{reg_short}={best_reg:g} ({dt_method:.1f}s total)",
    )
    return best_reg


def select_lambda_for_method(
    method_name: str,
    cfg: Config,
    A: np.ndarray,
    Psi: np.ndarray,
    y_sel: np.ndarray,
    ybg_sel: np.ndarray,
    b_sel: np.ndarray,
    z_sel: np.ndarray,
    alpha_pred_sel: Optional[np.ndarray],
    L_A: Optional[float],
) -> float:
    """Compat: FISTA apenas, nomes de metodo legacy (hybrid, ...)."""
    return select_regularization_for_cs_method(
        log_label=method_name,
        method_kind=method_name,
        cs_engine="fista",
        cfg=cfg,
        A=A,
        Psi=Psi,
        y_sel=y_sel,
        ybg_sel=ybg_sel,
        b_sel=b_sel,
        z_sel=z_sel,
        alpha_pred_sel=alpha_pred_sel,
        L_A=L_A,
    )


def lfista_train_config_from_pipeline(cfg: Config) -> "LFISTATrainConfig":
    import torch

    from lfista_module import LFISTATrainConfig

    dev = cfg.lfista_device.strip() if cfg.lfista_device else (
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    return LFISTATrainConfig(
        device=dev,
        p_input=cfg.p_input,
        n_output=cfg.n_output,
        bg_hidden=cfg.lfista_bg_hidden,
        bg_dropout=cfg.lfista_bg_dropout,
        bg_type=cfg.lfista_bg_type,
        lfista_steps=cfg.lfista_steps,
        learn_step_sizes=cfg.lfista_learn_step_sizes,
        learn_thresholds=cfg.lfista_learn_thresholds,
        init_step_scale=cfg.lfista_init_step_scale,
        init_threshold=cfg.lfista_init_threshold,
        use_momentum=cfg.lfista_use_momentum,
        batch_size=cfg.lfista_batch_size,
        num_epochs_bg=cfg.lfista_num_epochs_bg,
        num_epochs_frozen=cfg.lfista_num_epochs_frozen,
        num_epochs_joint=cfg.lfista_num_epochs_joint,
        lr_bg=cfg.lfista_lr_bg,
        lr_frozen=cfg.lfista_lr_frozen,
        lr_joint=cfg.lfista_lr_joint,
        weight_decay=cfg.lfista_weight_decay,
        patience=cfg.lfista_patience,
        loss_alpha_weight=cfg.lfista_loss_alpha_weight,
        loss_l1_alpha_weight=cfg.lfista_loss_l1_alpha_weight,
        measurement_noise_std=cfg.measurement_noise_std,
    )


def run_lfista_branch(
    cfg: Config,
    seed: int,
    measurement_ratio: float,
    data: Dict[str, np.ndarray],
    M_np: np.ndarray,
    log_fn: Callable[[str], None],
) -> Tuple[pd.DataFrame, Dict[str, np.ndarray]]:
    from lfista_module import run_lfista_experiment_dataframe

    tcfg = lfista_train_config_from_pipeline(cfg)
    return run_lfista_experiment_dataframe(tcfg, seed, measurement_ratio, data, M_np, log_fn)


# ============================================================
# 7) Avaliação
# ============================================================

def run_single_setting(
    cfg: Config,
    seed: int,
    measurement_ratio: float,
) -> Tuple[pd.DataFrame, Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    rng = np.random.default_rng(seed)
    t_block0 = time.perf_counter()

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

    log_line(
        cfg,
        f"  [data] train={cfg.n_train} val={cfg.n_val} test={cfg.n_test} N={cfg.n_output}",
    )

    m = max(4, int(round(measurement_ratio * cfg.n_output)))
    M = build_measurement_matrix(m, cfg.n_output, cfg.measurement_kind, rng)
    A = M @ Psi
    L_A = power_iteration_lipschitz(A, n_iter=cfg.power_iteration_n_iter)

    log_line(
        cfg,
        f"  [setup] m={m} measurement_kind={cfg.measurement_kind} | A.shape={A.shape}",
    )

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
    log_line(cfg, "  [train] baseline MLP done")

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
        log_line(cfg, "  [train] alpha predictor MLP done")

    # Um subsample + um ruido eta em b compartilhados por todos os metodos na escolha de lambda
    y_sel, ybg_sel, b_sel, z_sel, alpha_pred_sel = build_lambda_selection_arrays(
        cfg=cfg,
        M=M,
        X_val=X_val,
        Y_val=Y_val,
        baseline_model=baseline,
        alpha_model=alpha_model,
        rng=rng,
    )
    log_line(
        cfg,
        f"  [val] arrays for lambda selection | n_sel={len(y_sel)} | noise_std={cfg.measurement_noise_std}",
    )

    lam_hf: Optional[float] = None
    tau_hs: Optional[float] = None
    lam_cf: Optional[float] = None
    tau_cs: Optional[float] = None
    lam_hybrid: float = 0.0
    lam_weighted: Optional[float] = None
    lam_cs_only: Optional[float] = None

    if cfg.dual_cs_solver:
        lam_hf = select_regularization_for_cs_method(
            "hybrid_fista",
            "hybrid",
            "fista",
            cfg,
            A,
            Psi,
            y_sel,
            ybg_sel,
            b_sel,
            z_sel,
            None,
            L_A,
        )
        tau_hs = select_regularization_for_cs_method(
            "hybrid_spgl1",
            "hybrid",
            "spgl1",
            cfg,
            A,
            Psi,
            y_sel,
            ybg_sel,
            b_sel,
            z_sel,
            None,
            L_A,
        )
        if cfg.run_cs_only:
            lam_cf = select_regularization_for_cs_method(
                "cs_only_fista",
                "cs_only",
                "fista",
                cfg,
                A,
                Psi,
                y_sel,
                ybg_sel,
                b_sel,
                z_sel,
                None,
                L_A,
            )
            tau_cs = select_regularization_for_cs_method(
                "cs_only_spgl1",
                "cs_only",
                "spgl1",
                cfg,
                A,
                Psi,
                y_sel,
                ybg_sel,
                b_sel,
                z_sel,
                None,
                L_A,
            )
        lam_hybrid = float(lam_hf)
        lam_cs_only = float(lam_cf) if lam_cf is not None else None
    else:
        lam_hybrid = select_lambda_for_method(
            method_name="hybrid",
            cfg=cfg,
            A=A,
            Psi=Psi,
            y_sel=y_sel,
            ybg_sel=ybg_sel,
            b_sel=b_sel,
            z_sel=z_sel,
            alpha_pred_sel=None,
            L_A=L_A,
        )

        if cfg.run_weighted_hybrid and cfg.use_alpha_predictor:
            lam_weighted = select_lambda_for_method(
                method_name="weighted_hybrid",
                cfg=cfg,
                A=A,
                Psi=Psi,
                y_sel=y_sel,
                ybg_sel=ybg_sel,
                b_sel=b_sel,
                z_sel=z_sel,
                alpha_pred_sel=alpha_pred_sel,
                L_A=L_A,
            )

        if cfg.run_cs_only:
            lam_cs_only = select_lambda_for_method(
                method_name="cs_only",
                cfg=cfg,
                A=A,
                Psi=Psi,
                y_sel=y_sel,
                ybg_sel=ybg_sel,
                b_sel=b_sel,
                z_sel=z_sel,
                alpha_pred_sel=None,
                L_A=L_A,
            )

    log_line(cfg, "  [val] sparse regularization grid search finished for enabled methods")

    Ybg_test = baseline.predict(X_test)
    if alpha_model is not None:
        Alpha_pred_test = alpha_model.predict(X_test)
    else:
        Alpha_pred_test = np.zeros_like(Alpha_test)

    rows = []
    if cfg.dual_cs_solver:
        stored_examples = {
            "Y_true": [],
            "Y_bg": [],
            "Y_hybrid_fista": [],
            "Y_hybrid_spgl1": [],
            "Y_cs_only_fista": [],
            "Y_cs_only_spgl1": [],
        }
    else:
        stored_examples = {
            "Y_true": [],
            "Y_bg": [],
            "Y_hybrid": [],
            "Y_weighted": [],
            "Y_cs_only": [],
        }

    flat_true: List[np.ndarray] = []
    flat_ml: List[np.ndarray] = []
    flat_hybrid: List[np.ndarray] = []
    flat_weighted: List[np.ndarray] = []
    flat_cs: List[np.ndarray] = []
    flat_hf: List[np.ndarray] = []
    flat_hs: List[np.ndarray] = []
    flat_cf: List[np.ndarray] = []
    flat_csg: List[np.ndarray] = []

    n_test_samples = len(X_test)
    log_line(
        cfg,
        f"  [test] evaluating {n_test_samples} test samples (sparse recovery per method)...",
    )

    for i in range(len(X_test)):
        noise = cfg.measurement_noise_std * rng.normal(size=m)
        b_i = M @ Y_test[i] + noise

        y_ml = Ybg_test[i]
        rows.append(
            {
                "seed": seed,
                "measurement_ratio": measurement_ratio,
                "method": "ml_only",
                "sample_id": i,
                "rmse": rmse(Y_test[i], y_ml),
                "mae": float(mean_absolute_error(Y_test[i], y_ml)),
                "relative_l2": relative_l2(Y_test[i], y_ml),
                "support_f1": np.nan,
                "lambda": np.nan,
                "cs_engine": "none",
                "m": m,
            }
        )

        z_i = b_i - M @ y_ml

        if cfg.dual_cs_solver:
            assert lam_hf is not None and tau_hs is not None
            specs = [
                ("hybrid_fista", "fista", float(lam_hf), "hybrid"),
                ("hybrid_spgl1", "spgl1", float(tau_hs), "hybrid"),
            ]
            if cfg.run_cs_only and lam_cf is not None and tau_cs is not None:
                specs.extend(
                    [
                        ("cs_only_fista", "fista", float(lam_cf), "cs_only"),
                        ("cs_only_spgl1", "spgl1", float(tau_cs), "cs_only"),
                    ]
                )
            preds_ex: Dict[str, np.ndarray] = {}
            for method_tag, engine, reg, kind in specs:
                rhs = z_i if kind == "hybrid" else b_i
                alpha_hat = solve_sparse_alpha(
                    A=A,
                    b=rhs,
                    cs_engine=engine,
                    regularization=reg,
                    weights=None,
                    L_A=L_A,
                    cfg=cfg,
                    x0=None,
                )
                if kind == "hybrid":
                    y_hat = y_ml + Psi @ alpha_hat
                else:
                    y_hat = Psi @ alpha_hat
                preds_ex[method_tag] = y_hat
                rows.append(
                    {
                        "seed": seed,
                        "measurement_ratio": measurement_ratio,
                        "method": method_tag,
                        "sample_id": i,
                        "rmse": rmse(Y_test[i], y_hat),
                        "mae": float(mean_absolute_error(Y_test[i], y_hat)),
                        "relative_l2": relative_l2(Y_test[i], y_hat),
                        "support_f1": support_f1(Alpha_test[i], alpha_hat),
                        "lambda": reg,
                        "cs_engine": engine,
                        "m": m,
                    }
                )

            flat_true.append(np.asarray(Y_test[i], dtype=float).ravel())
            flat_ml.append(np.asarray(y_ml, dtype=float).ravel())
            flat_hf.append(np.asarray(preds_ex["hybrid_fista"], dtype=float).ravel())
            flat_hs.append(np.asarray(preds_ex["hybrid_spgl1"], dtype=float).ravel())
            if "cs_only_fista" in preds_ex:
                flat_cf.append(np.asarray(preds_ex["cs_only_fista"], dtype=float).ravel())
                flat_csg.append(np.asarray(preds_ex["cs_only_spgl1"], dtype=float).ravel())

            if len(stored_examples["Y_true"]) < cfg.n_example_plots:
                stored_examples["Y_true"].append(Y_test[i].copy())
                stored_examples["Y_bg"].append(y_ml.copy())
                stored_examples["Y_hybrid_fista"].append(preds_ex["hybrid_fista"].copy())
                stored_examples["Y_hybrid_spgl1"].append(preds_ex["hybrid_spgl1"].copy())
                if "cs_only_fista" in preds_ex:
                    stored_examples["Y_cs_only_fista"].append(preds_ex["cs_only_fista"].copy())
                    stored_examples["Y_cs_only_spgl1"].append(preds_ex["cs_only_spgl1"].copy())
        else:
            alpha_h = solve_sparse_alpha(
                A=A,
                b=z_i,
                cs_engine="fista",
                regularization=float(lam_hybrid),
                weights=None,
                L_A=L_A,
                cfg=cfg,
                x0=None,
            )
            y_h = y_ml + Psi @ alpha_h
            flat_true.append(np.asarray(Y_test[i], dtype=float).ravel())
            flat_ml.append(np.asarray(y_ml, dtype=float).ravel())
            flat_hybrid.append(np.asarray(y_h, dtype=float).ravel())

            rows.append(
                {
                    "seed": seed,
                    "measurement_ratio": measurement_ratio,
                    "method": "hybrid",
                    "sample_id": i,
                    "rmse": rmse(Y_test[i], y_h),
                    "mae": float(mean_absolute_error(Y_test[i], y_h)),
                    "relative_l2": relative_l2(Y_test[i], y_h),
                    "support_f1": support_f1(Alpha_test[i], alpha_h),
                    "lambda": lam_hybrid,
                    "cs_engine": "fista",
                    "m": m,
                }
            )

            y_wh = None
            if lam_weighted is not None:
                weights_i = build_weights_from_alpha_prediction(Alpha_pred_test[i], cfg)
                alpha_wh = solve_sparse_alpha(
                    A=A,
                    b=z_i,
                    cs_engine="fista",
                    regularization=float(lam_weighted),
                    weights=weights_i,
                    L_A=L_A,
                    cfg=cfg,
                    x0=None,
                )
                y_wh = y_ml + Psi @ alpha_wh
                flat_weighted.append(np.asarray(y_wh, dtype=float).ravel())
                rows.append(
                    {
                        "seed": seed,
                        "measurement_ratio": measurement_ratio,
                        "method": "weighted_hybrid",
                        "sample_id": i,
                        "rmse": rmse(Y_test[i], y_wh),
                        "mae": float(mean_absolute_error(Y_test[i], y_wh)),
                        "relative_l2": relative_l2(Y_test[i], y_wh),
                        "support_f1": support_f1(Alpha_test[i], alpha_wh),
                        "lambda": lam_weighted,
                        "cs_engine": "fista",
                        "m": m,
                    }
                )

            y_cs = None
            if lam_cs_only is not None:
                alpha_cs = solve_sparse_alpha(
                    A=A,
                    b=b_i,
                    cs_engine="fista",
                    regularization=float(lam_cs_only),
                    weights=None,
                    L_A=L_A,
                    cfg=cfg,
                    x0=None,
                )
                y_cs = Psi @ alpha_cs
                flat_cs.append(np.asarray(y_cs, dtype=float).ravel())
                rows.append(
                    {
                        "seed": seed,
                        "measurement_ratio": measurement_ratio,
                        "method": "cs_only",
                        "sample_id": i,
                        "rmse": rmse(Y_test[i], y_cs),
                        "mae": float(mean_absolute_error(Y_test[i], y_cs)),
                        "relative_l2": relative_l2(Y_test[i], y_cs),
                        "support_f1": support_f1(Alpha_test[i], alpha_cs),
                        "lambda": lam_cs_only,
                        "cs_engine": "fista",
                        "m": m,
                    }
                )

            if len(stored_examples["Y_true"]) < cfg.n_example_plots:
                stored_examples["Y_true"].append(Y_test[i].copy())
                stored_examples["Y_bg"].append(y_ml.copy())
                stored_examples["Y_hybrid"].append(y_h.copy())
                if y_wh is not None:
                    stored_examples["Y_weighted"].append(y_wh.copy())
                if y_cs is not None:
                    stored_examples["Y_cs_only"].append(y_cs.copy())

        if cfg.log_progress and cfg.test_log_interval > 0:
            step = cfg.test_log_interval
            if (i + 1) == 1 or (i + 1) % step == 0 or (i + 1) == n_test_samples:
                log_line(cfg, f"  [test] progress {i + 1}/{n_test_samples}")
        elif cfg.log_progress and cfg.test_log_interval == 0 and (i + 1) == n_test_samples:
            log_line(cfg, f"  [test] finished {n_test_samples} samples")

    dt_block = time.perf_counter() - t_block0
    log_line(cfg, f"  [done] seed={seed} m/N={measurement_ratio:.2f} block_time={dt_block:.1f}s")

    df = pd.DataFrame(rows)
    for k in list(stored_examples.keys()):
        stored_examples[k] = np.array(stored_examples[k]) if len(stored_examples[k]) > 0 else None

    if cfg.dual_cs_solver:
        gt_pred_bundle = {
            "y_true": np.concatenate(flat_true),
            "ml_only": np.concatenate(flat_ml),
            "hybrid_fista": np.concatenate(flat_hf),
            "hybrid_spgl1": np.concatenate(flat_hs),
        }
        if len(flat_cf) == len(flat_true):
            gt_pred_bundle["cs_only_fista"] = np.concatenate(flat_cf)
            gt_pred_bundle["cs_only_spgl1"] = np.concatenate(flat_csg)
    else:
        gt_pred_bundle = {
            "y_true": np.concatenate(flat_true),
            "ml_only": np.concatenate(flat_ml),
            "hybrid": np.concatenate(flat_hybrid),
        }
        if len(flat_weighted) == len(flat_true):
            gt_pred_bundle["weighted_hybrid"] = np.concatenate(flat_weighted)
        if len(flat_cs) == len(flat_true):
            gt_pred_bundle["cs_only"] = np.concatenate(flat_cs)

    if cfg.run_lfista:

        def _lf_log(msg: str) -> None:
            log_line(cfg, msg)

        lf_df, lf_gt = run_lfista_branch(
            cfg,
            seed,
            measurement_ratio,
            data,
            M,
            _lf_log,
        )
        df = pd.concat([df, lf_df], ignore_index=True)
        for k, v in lf_gt.items():
            gt_pred_bundle[k] = v

    return df, stored_examples, gt_pred_bundle


# ============================================================
# 8) Relatórios e figuras
# ============================================================

def summarize_results_per_seed(df: pd.DataFrame) -> pd.DataFrame:
    """
    Agrega primeiro sobre amostras de teste dentro de cada seed.
    Uma linha por (seed, measurement_ratio, method).
    """
    g = (
        df.groupby(["seed", "measurement_ratio", "method"], dropna=False)
        .agg(
            rmse_mean=("rmse", "mean"),
            mae_mean=("mae", "mean"),
            relative_l2_mean=("relative_l2", "mean"),
            support_f1_mean=("support_f1", "mean"),
            n_test_samples=("rmse", "size"),
        )
        .reset_index()
    )
    return g


def _sem_series(s: pd.Series) -> float:
    n = len(s)
    if n <= 1:
        return 0.0
    return float(s.std(ddof=1) / math.sqrt(n))


def summarize_results_across_seeds(per_seed: pd.DataFrame) -> pd.DataFrame:
    """
    Resume entre seeds independentes. Barras de erro devem usar std_across_seeds ou SEM.
    """
    g = (
        per_seed.groupby(["measurement_ratio", "method"], dropna=False)
        .agg(
            rmse_mean=("rmse_mean", "mean"),
            rmse_std_across_seeds=("rmse_mean", lambda s: float(s.std(ddof=1)) if len(s) > 1 else 0.0),
            rmse_sem=("rmse_mean", _sem_series),
            mae_mean=("mae_mean", "mean"),
            mae_std_across_seeds=("mae_mean", lambda s: float(s.std(ddof=1)) if len(s) > 1 else 0.0),
            relative_l2_mean=("relative_l2_mean", "mean"),
            relative_l2_std_across_seeds=(
                "relative_l2_mean",
                lambda s: float(s.std(ddof=1)) if len(s) > 1 else 0.0,
            ),
            support_f1_mean=("support_f1_mean", "mean"),
            support_f1_std_across_seeds=(
                "support_f1_mean",
                lambda s: float(s.std(ddof=1)) if len(s) > 1 else 0.0,
            ),
            n_seeds=("seed", "nunique"),
            n_test_samples_per_run=("n_test_samples", "first"),
        )
        .reset_index()
    )
    # Student-t 95% half-width with df = n_seeds - 1 (not the normal z=1.96 factor).
    # For n_seeds=3 this uses t_{2,0.975} ~= 4.303.
    def _t95_half(row: pd.Series) -> float:
        n = int(row["n_seeds"])
        if n <= 1:
            return 0.0
        # Critical values for common small df; asymptotic 1.96 for large n.
        tcrit = {
            1: 12.706204736432095,
            2: 4.302652729911275,
            3: 3.182446305284263,
            4: 2.7764451051977987,
            5: 2.57058183563631,
            6: 2.4469118511449682,
            7: 2.3646242515927846,
            8: 2.306004135204166,
            9: 2.2621571627982046,
        }.get(n - 1, 1.96)
        return float(tcrit) * float(row["rmse_sem"])

    g["rmse_ci95_half"] = g.apply(_t95_half, axis=1)
    return g.sort_values(["measurement_ratio", "method"]).reset_index(drop=True)


# Cores consistentes entre figuras (metodo -> cor)
METHOD_COLORS: Dict[str, str] = {
    "ml_only": "#1f77b4",
    "hybrid": "#ff7f0e",
    "weighted_hybrid": "#2ca02c",
    "cs_only": "#d62728",
    "hybrid_fista": "#ff7f0e",
    "hybrid_spgl1": "#bcbd22",
    "cs_only_fista": "#d62728",
    "cs_only_spgl1": "#9467bd",
    "ml_only_torch": "#17becf",
    "hybrid_lfista_frozen": "#e377c2",
    "hybrid_lfista_joint": "#e6550d",
    "ext_sklearn_lasso_S1_hybrid": "#8c564b",
    "ext_sklearn_lasso_S2_cs_only": "#bc82bd",
    "ext_sklearn_omp_S3_hybrid_oracle_k": "#7f7f7f",
    "mlp_concat_ub": "#2ca02c",
    "pca_regression_ub": "#9467bd",
    "ae_regression_ub": "#17becf",
    "ae_regression_ub_mask": "#1a9aa8",
    "ridge_prior_csgm": "#d62728",
    "mlp_prior_csgm": "#8c564b",
    "ridge_prior_only_decoder": "#ff9896",
    "mlp_prior_only_decoder": "#c49c94",
    "measurement_only_csgm": "#7f7f7f",
}
METHOD_DISPLAY_NAMES: Dict[str, str] = {
    "ml_only": "ML only",
    "ml_only_torch": "ML only torch",
    "mlp_concat_ub": "MLP [u,b]",
    "pca_regression_ub": "PCA [u,b]",
    "ae_regression_ub": "AE [u,b]",
    "ae_regression_ub_mask": "AE [u,b,mask]",
    "ridge_prior_csgm": "CLP-CSGM Ridge",
    "mlp_prior_csgm": "CLP-CSGM MLP",
    "ridge_prior_only_decoder": "Prior-only Ridge",
    "mlp_prior_only_decoder": "Prior-only MLP",
    "measurement_only_csgm": "Measurement-only CSGM",
    "hybrid_fista": "Hybrid FISTA",
    "hybrid_spgl1": "Hybrid SPGL1",
    "cs_only_fista": "CS only FISTA",
    "cs_only_spgl1": "CS only SPGL1",
    "hybrid_lfista_frozen": "Hybrid LFISTA frozen",
    "hybrid_lfista_joint": "Hybrid LFISTA joint",
    "hybrid": "Hybrid FISTA",
    "weighted_hybrid": "Weighted hybrid",
    "cs_only": "CS only",
}


def method_display_name(method: str) -> str:
    """Return the paper-facing method label while preserving raw method ids in tables."""
    return METHOD_DISPLAY_NAMES.get(method, method)


METHOD_ORDER_DIRECT_UB = [
    "ml_only",
    "ml_only_torch",
    "mlp_concat_ub",
    "pca_regression_ub",
    "ae_regression_ub",
    "ridge_prior_csgm",
    "mlp_prior_csgm",
    "ridge_prior_only_decoder",
    "mlp_prior_only_decoder",
    "measurement_only_csgm",
    "hybrid_fista",
    "hybrid_lfista_frozen",
    "hybrid_lfista_joint",
]
METHOD_ORDER_DIRECT_UB_JOINT_FOCUS = [
    "ml_only",
    "mlp_concat_ub",
    "pca_regression_ub",
    "ae_regression_ub",
    "ridge_prior_csgm",
    "mlp_prior_csgm",
    "ridge_prior_only_decoder",
    "mlp_prior_only_decoder",
    "measurement_only_csgm",
    "hybrid_lfista_joint",
]
METHOD_ORDER_STAGE1_EXTERNAL = [
    "ml_only",
    "ext_sklearn_lasso_S1_hybrid",
    "ext_sklearn_lasso_S2_cs_only",
    "ext_sklearn_omp_S3_hybrid_oracle_k",
    "hybrid_fista",
]
METHOD_ORDER_DEFAULT = ["ml_only", "hybrid", "weighted_hybrid", "cs_only"]
METHOD_ORDER_SOLVER_CMP = [
    "ml_only",
    "hybrid_fista",
    "hybrid_spgl1",
    "cs_only_fista",
    "cs_only_spgl1",
]
METHOD_ORDER_LFISTA_TAIL = [
    "ml_only_torch",
    "hybrid_lfista_frozen",
    "hybrid_lfista_joint",
]
# Legend order when dual_cs_solver and run_lfista: central comparison first, then SPGL1/cs_only.
METHOD_ORDER_LFISTA_VS_CLASSICAL = [
    "ml_only",
    "ml_only_torch",
    "hybrid_fista",
    "hybrid_lfista_frozen",
    "hybrid_lfista_joint",
    "hybrid_spgl1",
    "cs_only_fista",
    "cs_only_spgl1",
]

LFISTA_VS_CLASSICAL_FOCUS_METHODS = [
    "ml_only",
    "ml_only_torch",
    "hybrid_fista",
    "hybrid_lfista_frozen",
    "hybrid_lfista_joint",
]


def is_lfista_vs_classical_profile(cfg: Config) -> bool:
    return cfg.config_profile in (
        "lfista_vs_classical",
        "lfista_vs_classical_explore",
        "robustness_phase3",
        "robustness_phase3_explore",
    )


def save_lfista_vs_classical_focus_tables(
    cfg: Config,
    summary: pd.DataFrame,
    per_seed: pd.DataFrame,
) -> List[str]:
    """
    Escreve CSVs filtrados para os metodos centrais da Etapa 2 vs hybrid_fista classico.
    """
    if not is_lfista_vs_classical_profile(cfg):
        return []
    focus = LFISTA_VS_CLASSICAL_FOCUS_METHODS
    sub_s = summary[summary["method"].isin(focus)].sort_values(["measurement_ratio", "method"])
    sub_p = per_seed[per_seed["method"].isin(focus)].sort_values(
        ["seed", "measurement_ratio", "method"]
    )
    p1 = os.path.join(cfg.save_dir, "summary_focus_ml_hybrid_fista_lfista.csv")
    p2 = os.path.join(cfg.save_dir, "summary_by_seed_focus_ml_hybrid_fista_lfista.csv")
    sub_s.to_csv(p1, index=False)
    sub_p.to_csv(p2, index=False)
    focus_txt = os.path.join(cfg.save_dir, "FOCUS_COMPARISON.txt")
    lines = [
        "Etapa 2 focus comparison (same synthetic protocol as full run).",
        "Methods in focus files:",
        "  ml_only (sklearn MLP), ml_only_torch (PyTorch MLP),",
        "  hybrid_fista (FISTA on residual, classical),",
        "  hybrid_lfista_frozen, hybrid_lfista_joint (PyTorch + LFISTA).",
        "Full summary.csv still includes hybrid_spgl1, cs_only_* for Etapa 1 context.",
        "",
    ]
    with open(focus_txt, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    return [p1, p2, focus_txt]


def method_order_for_cfg(cfg: Config) -> List[str]:
    if cfg.config_profile in (
        "direct_ub_lfista_joint_only",
        "direct_ub_lfista_joint_only_explore",
        "direct_ub_lfista_joint_robustness_lite",
        "real_well_f03_direct_ub",
        "cross_well_vc_direct_ub",
    ):
        return list(METHOD_ORDER_DIRECT_UB_JOINT_FOCUS)
    if cfg.config_profile in ("direct_ub_benchmark", "direct_ub_benchmark_explore"):
        return list(METHOD_ORDER_DIRECT_UB)
    if cfg.config_profile in ("external_benchmark_stage1", "external_benchmark_stage1_explore"):
        return list(METHOD_ORDER_STAGE1_EXTERNAL)
    if cfg.run_lfista and cfg.dual_cs_solver:
        return list(METHOD_ORDER_LFISTA_VS_CLASSICAL)
    if cfg.run_lfista:
        base = list(METHOD_ORDER_DEFAULT)
        out = base + [m for m in METHOD_ORDER_LFISTA_TAIL if m not in base]
        return out
    if cfg.dual_cs_solver:
        return list(METHOD_ORDER_SOLVER_CMP)
    return list(METHOD_ORDER_DEFAULT)


def plots_directory(cfg: Config) -> str:
    p = os.path.join(cfg.save_dir, cfg.plots_subdir)
    os.makedirs(p, exist_ok=True)
    return p


def plot_metric_vs_measurement_ratio(
    summary: pd.DataFrame,
    mean_col: str,
    std_col: str,
    ylabel: str,
    title: str,
    save_path: str,
    cfg: Config,
) -> None:
    plt.figure(figsize=(9, 5))
    order = method_order_for_cfg(cfg)
    methods = [m for m in order if m in set(summary["method"].unique())]
    for method in methods:
        sdf = summary[summary["method"] == method].sort_values("measurement_ratio")
        x = sdf["measurement_ratio"].values
        y = sdf[mean_col].values
        yerr = sdf[std_col].fillna(0.0).values
        color = METHOD_COLORS.get(method, None)
        plt.errorbar(
            x,
            y,
            yerr=yerr,
            marker="o",
            capsize=4,
            label=method_display_name(method),
            color=color,
            linewidth=1.8,
        )
    plt.xlabel("Measurement ratio m / N")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(save_path, dpi=180)
    plt.close()


def plot_grouped_bars_metric(
    summary: pd.DataFrame,
    mean_col: str,
    ylabel: str,
    title: str,
    save_path: str,
    cfg: Config,
) -> None:
    ratios = sorted(summary["measurement_ratio"].unique())
    order = method_order_for_cfg(cfg)
    methods = [m for m in order if m in set(summary["method"].unique())]
    if not ratios or not methods:
        return
    n_r = len(ratios)
    n_m = len(methods)
    x = np.arange(n_r, dtype=float)
    width = min(0.85 / max(n_m, 1), 0.2)
    fig, ax = plt.subplots(figsize=(max(8, n_r * 1.2), 5))
    for i, method in enumerate(methods):
        heights = []
        for r in ratios:
            sub = summary[(summary["measurement_ratio"] == r) & (summary["method"] == method)]
            heights.append(float(sub[mean_col].values[0]) if len(sub) else 0.0)
        offset = (i - (n_m - 1) / 2.0) * width
        ax.bar(
            x + offset,
            heights,
            width,
            label=method_display_name(method),
            color=METHOD_COLORS.get(method),
            edgecolor="white",
            linewidth=0.5,
        )
    ax.set_xticks(x)
    ax.set_xticklabels([f"{r:.2f}" for r in ratios])
    ax.set_xlabel("Measurement ratio m / N")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=180)
    plt.close()


def build_gain_vs_baseline(
    per_seed: pd.DataFrame,
    baseline: str,
    metric_col: str,
) -> pd.DataFrame:
    """
    Para cada seed e razao: ganho = metrica(baseline) - metrica(metodo).
    RMSE/MAE: positivo => metodo melhor que baseline.
    """
    rows: List[Dict[str, object]] = []
    for (seed, mr), grp in per_seed.groupby(["seed", "measurement_ratio"]):
        base_rows = grp[grp["method"] == baseline]
        if len(base_rows) == 0:
            continue
        bval = float(base_rows[metric_col].values[0])
        for _, row in grp[grp["method"] != baseline].iterrows():
            mname = str(row["method"])
            rows.append(
                {
                    "seed": seed,
                    "measurement_ratio": mr,
                    "method": mname,
                    "gain": bval - float(row[metric_col]),
                }
            )
    return pd.DataFrame(rows)


def summarize_gain_across_seeds(gain_df: pd.DataFrame) -> pd.DataFrame:
    if len(gain_df) == 0:
        return pd.DataFrame()
    g = (
        gain_df.groupby(["measurement_ratio", "method"])
        .agg(
            gain_mean=("gain", "mean"),
            gain_std_across_seeds=("gain", lambda s: float(s.std(ddof=1)) if len(s) > 1 else 0.0),
            n_seeds=("seed", "nunique"),
        )
        .reset_index()
    )
    return g


def plot_gain_vs_ml_only(
    gain_summary: pd.DataFrame,
    metric_name: str,
    save_path: str,
    cfg: Config,
    baseline_method: str = "ml_only",
) -> None:
    if len(gain_summary) == 0:
        return
    plt.figure(figsize=(9, 5))
    order = method_order_for_cfg(cfg)
    methods = [m for m in order if m in set(gain_summary["method"].unique()) and m != baseline_method]
    for method in methods:
        sdf = gain_summary[gain_summary["method"] == method].sort_values("measurement_ratio")
        x = sdf["measurement_ratio"].values
        y = sdf["gain_mean"].values
        yerr = sdf["gain_std_across_seeds"].fillna(0.0).values
        color = METHOD_COLORS.get(method, None)
        plt.errorbar(
            x,
            y,
            yerr=yerr,
            marker="s",
            capsize=4,
            label=method_display_name(method),
            color=color,
            linewidth=1.8,
        )
    plt.axhline(0.0, color="gray", linestyle="--", linewidth=1)
    plt.xlabel("Measurement ratio m / N")
    plt.ylabel(f"Gain ({metric_name}): baseline - method (positive is better)")
    plt.title(
        "Improvement over {} vs {} (seed uncertainty)".format(
            metric_name, method_display_name(baseline_method)
        )
    )
    plt.grid(True, alpha=0.3)
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(save_path, dpi=180)
    plt.close()


def save_all_comparison_plots(
    cfg: Config,
    summary: pd.DataFrame,
    per_seed: pd.DataFrame,
) -> List[str]:
    """
    Salva conjunto de figuras comparativas em save_dir/plots_subdir (default: paper/figures/synthetic).
    Retorna lista de caminhos criados.
    """
    pdir = plots_directory(cfg)
    paths: List[str] = []

    def _save(name: str) -> str:
        path = os.path.join(pdir, name)
        paths.append(path)
        return path

    plot_metric_vs_measurement_ratio(
        summary,
        "rmse_mean",
        "rmse_std_across_seeds",
        "RMSE (mean over test; bars = std across seeds)",
        "RMSE vs measurement ratio",
        _save("01_rmse_vs_measurement_ratio.png"),
        cfg,
    )
    plot_metric_vs_measurement_ratio(
        summary,
        "mae_mean",
        "mae_std_across_seeds",
        "MAE (mean over test; bars = std across seeds)",
        "MAE vs measurement ratio",
        _save("02_mae_vs_measurement_ratio.png"),
        cfg,
    )
    plot_metric_vs_measurement_ratio(
        summary,
        "relative_l2_mean",
        "relative_l2_std_across_seeds",
        "Relative L2 error (mean over test; bars = std across seeds)",
        "Relative L2 vs measurement ratio",
        _save("03_relative_l2_vs_measurement_ratio.png"),
        cfg,
    )

    sub_f1 = summary.dropna(subset=["support_f1_mean"])
    if len(sub_f1) > 0:
        plot_metric_vs_measurement_ratio(
            sub_f1,
            "support_f1_mean",
            "support_f1_std_across_seeds",
            "Support F1 (mean over test; bars = std across seeds)",
            "Support F1 vs measurement ratio",
            _save("04_support_f1_vs_measurement_ratio.png"),
            cfg,
        )

    plot_grouped_bars_metric(
        summary,
        "rmse_mean",
        "RMSE",
        "RMSE grouped by measurement ratio (bars = methods)",
        _save("05_rmse_grouped_bars_by_ratio.png"),
        cfg,
    )

    g_rmse = build_gain_vs_baseline(per_seed, "ml_only", "rmse_mean")
    if len(g_rmse) > 0:
        gsum = summarize_gain_across_seeds(g_rmse)
        plot_gain_vs_ml_only(gsum, "RMSE", _save("06_gain_rmse_over_ml_only.png"), cfg)

    g_mae = build_gain_vs_baseline(per_seed, "ml_only", "mae_mean")
    if len(g_mae) > 0:
        gsum_m = summarize_gain_across_seeds(g_mae)
        plot_gain_vs_ml_only(gsum_m, "MAE", _save("07_gain_mae_over_ml_only.png"), cfg)

    if cfg.run_lfista and "ml_only_torch" in set(per_seed["method"].values):
        g_rmse_t = build_gain_vs_baseline(per_seed, "ml_only_torch", "rmse_mean")
        if len(g_rmse_t) > 0:
            gsum_rt = summarize_gain_across_seeds(g_rmse_t)
            plot_gain_vs_ml_only(
                gsum_rt,
                "RMSE",
                _save("11_gain_rmse_over_ml_only_torch.png"),
                cfg,
                baseline_method="ml_only_torch",
            )
        g_mae_t = build_gain_vs_baseline(per_seed, "ml_only_torch", "mae_mean")
        if len(g_mae_t) > 0:
            gsum_mt = summarize_gain_across_seeds(g_mae_t)
            plot_gain_vs_ml_only(
                gsum_mt,
                "MAE",
                _save("12_gain_mae_over_ml_only_torch.png"),
                cfg,
                baseline_method="ml_only_torch",
            )

    return paths


def merge_gt_pred_bundles(bundles: List[Dict[str, np.ndarray]]) -> Dict[str, np.ndarray]:
    """Concatena vetores y_true / y_pred de cada job (seed, measurement_ratio)."""
    if not bundles:
        return {}
    all_keys: set = set()
    for b in bundles:
        all_keys |= set(b.keys())
    out: Dict[str, np.ndarray] = {}
    for k in sorted(all_keys):
        parts = [b[k] for b in bundles if k in b]
        if parts:
            out[k] = np.concatenate(parts)
    return out


def _subsample_pair(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    max_n: int,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray]:
    n = int(y_true.size)
    if n <= max_n:
        return y_true, y_pred
    idx = rng.choice(n, size=max_n, replace=False)
    return y_true[idx], y_pred[idx]


def plot_parity_ground_truth_vs_predictions(
    cfg: Config,
    merged: Dict[str, np.ndarray],
    save_path: str,
) -> None:
    """
    Painel 2x2: eixo x = ground truth, eixo y = predicao (linha identidade).
    """
    if "y_true" not in merged:
        return
    y_true_full = merged["y_true"]
    rng = np.random.default_rng(12345)
    order = method_order_for_cfg(cfg)
    methods_plot = [m for m in order if m in merged and m != "y_true"]
    if not methods_plot:
        return
    n_m = len(methods_plot)
    ncols = 2
    nrows = int(math.ceil(n_m / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4.5 * nrows), squeeze=False)
    lo = float(np.min(y_true_full))
    hi = float(np.max(y_true_full))
    for ax_idx, method in enumerate(methods_plot):
        r, c = divmod(ax_idx, ncols)
        ax = axes[r][c]
        y_p = merged[method]
        if y_p.size != y_true_full.size:
            continue
        yt, yp = _subsample_pair(y_true_full, y_p, cfg.max_gt_scatter_points, rng)
        ax.scatter(yt, yp, s=4, alpha=0.2, color=METHOD_COLORS.get(method, None), rasterized=True)
        ax.plot([lo, hi], [lo, hi], "k--", linewidth=1.0, label="identity")
        if len(yt) > 2:
            cc = float(np.corrcoef(yt, yp)[0, 1])
            ax.set_title("{} (corr={:.4f})".format(method_display_name(method), cc))
        else:
            ax.set_title(method_display_name(method))
        ax.set_xlabel("Ground truth y")
        ax.set_ylabel("Prediction y_hat")
        ax.set_aspect("equal", adjustable="box")
        ax.grid(True, alpha=0.3)
    for ax_idx in range(n_m, nrows * ncols):
        r, c = divmod(ax_idx, ncols)
        axes[r][c].set_visible(False)
    fig.suptitle("Parity: ground truth vs model prediction (all test samples pooled)", fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=180)
    plt.close()


def plot_residual_distributions_gt_vs_models(
    cfg: Config,
    merged: Dict[str, np.ndarray],
    save_path: str,
) -> None:
    """Histogramas do residual (y_hat - y) por metodo."""
    if "y_true" not in merged:
        return
    y_t = merged["y_true"]
    order = method_order_for_cfg(cfg)
    methods_plot = [m for m in order if m in merged and m != "y_true"]
    if not methods_plot:
        return
    n_m = len(methods_plot)
    ncols = 2
    nrows = int(math.ceil(n_m / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows), squeeze=False)
    for ax_idx, method in enumerate(methods_plot):
        r, c = divmod(ax_idx, ncols)
        ax = axes[r][c]
        y_p = merged[method]
        if y_p.size != y_t.size:
            continue
        res = y_p - y_t
        ax.hist(res, bins=60, color=METHOD_COLORS.get(method, "#888888"), alpha=0.85, edgecolor="white")
        ax.axvline(0.0, color="black", linestyle="--", linewidth=1)
        ax.set_title("{} residual (y_hat - y_true)".format(method_display_name(method)))
        ax.set_xlabel("Residual")
        ax.set_ylabel("Count")
        ax.grid(True, axis="y", alpha=0.3)
    for ax_idx in range(n_m, nrows * ncols):
        r, c = divmod(ax_idx, ncols)
        axes[r][c].set_visible(False)
    fig.suptitle("Residual distributions vs ground truth", fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=180)
    plt.close()


def save_ground_truth_vs_model_plots(
    cfg: Config,
    merged_gt: Dict[str, np.ndarray],
) -> List[str]:
    pdir = plots_directory(cfg)
    paths: List[str] = []
    p1 = os.path.join(pdir, "09_parity_ground_truth_vs_prediction.png")
    p2 = os.path.join(pdir, "10_residual_distributions_gt_vs_models.png")
    plot_parity_ground_truth_vs_predictions(cfg, merged_gt, p1)
    plot_residual_distributions_gt_vs_models(cfg, merged_gt, p2)
    paths.extend([p1, p2])
    return paths


def plot_examples(cfg: Config, examples: Dict[str, np.ndarray], save_path: str) -> None:
    Y_true = examples["Y_true"]
    Y_bg = examples["Y_bg"]
    if Y_true is None or len(Y_true) == 0:
        return

    n_examples = len(Y_true)
    fig, axes = plt.subplots(n_examples, 1, figsize=(10, 3 * n_examples), squeeze=False)

    for i in range(n_examples):
        ax = axes[i, 0]
        ax.plot(Y_true[i], label="ground_truth")
        ax.plot(Y_bg[i], label="ml_only")
        if cfg.dual_cs_solver:
            y_hf = examples.get("Y_hybrid_fista")
            y_hs = examples.get("Y_hybrid_spgl1")
            y_cf = examples.get("Y_cs_only_fista")
            y_cg = examples.get("Y_cs_only_spgl1")
            if y_hf is not None and len(y_hf) > i:
                ax.plot(y_hf[i], label="hybrid_fista")
            if y_hs is not None and len(y_hs) > i:
                ax.plot(y_hs[i], label="hybrid_spgl1")
            if y_cf is not None and len(y_cf) > i:
                ax.plot(y_cf[i], label="cs_only_fista")
            if y_cg is not None and len(y_cg) > i:
                ax.plot(y_cg[i], label="cs_only_spgl1")
        else:
            Y_h = examples["Y_hybrid"]
            Y_wh = examples.get("Y_weighted", None)
            Y_cs = examples.get("Y_cs_only", None)
            ax.plot(Y_h[i], label="hybrid")
            if Y_wh is not None and len(Y_wh) > i:
                ax.plot(Y_wh[i], label="weighted_hybrid")
            if Y_cs is not None and len(Y_cs) > i:
                ax.plot(Y_cs[i], label="cs_only")
        ax.set_title(f"Example {i}")
        ax.grid(True, alpha=0.3)
        ax.legend()
    fig.suptitle("Ground truth vs models (output index)", fontsize=12, y=1.0)
    plt.tight_layout()
    plt.savefig(save_path, dpi=180)
    plt.close()


def plot_direct_ub_ground_truth_vs_models(
    examples: Dict[str, np.ndarray],
    save_path: str,
) -> None:
    """
    Line plot per test example: y(L) vs output index for direct [u,b] benchmark methods.
    Keys in examples: Y_true (required), plus any of ml_only, mlp_concat_ub, pca_regression_ub,
    ae_regression_ub, hybrid_fista, hybrid_lfista_frozen, hybrid_lfista_joint.
    """
    import matplotlib.pyplot as plt

    y_key = "Y_true"
    if y_key not in examples or examples[y_key] is None or len(examples[y_key]) == 0:
        return
    y_true = np.asarray(examples[y_key], dtype=np.float64)
    n_examples = int(y_true.shape[0])
    order = [
        ("Y_true", "ground_truth"),
        ("ml_only", "ml_only"),
        ("mlp_concat_ub", "mlp_concat_ub"),
        ("pca_regression_ub", "pca_regression_ub"),
        ("ae_regression_ub", "ae_regression_ub"),
        ("ridge_prior_csgm", "CLP-CSGM Ridge"),
        ("mlp_prior_csgm", "CLP-CSGM MLP"),
        ("hybrid_fista", "hybrid_fista"),
        ("hybrid_lfista_frozen", "hybrid_lfista_frozen"),
        ("hybrid_lfista_joint", "hybrid_lfista_joint"),
    ]
    color_gt = "#1f77b4"
    fig, axes = plt.subplots(n_examples, 1, figsize=(10, 3 * n_examples), squeeze=False)
    for i in range(n_examples):
        ax = axes[i, 0]
        ax.plot(y_true[i], color=color_gt, label="ground_truth", linewidth=1.4)
        for data_key, leg in order[1:]:
            if data_key not in examples:
                continue
            arr = np.asarray(examples[data_key], dtype=np.float64)
            if arr.shape[0] <= i:
                continue
            col = METHOD_COLORS.get(data_key, "#7f7f7f")
            ax.plot(arr[i], color=col, label=leg, linewidth=1.1, alpha=0.9)
        ax.set_title("Example {}".format(i))
        ax.set_xlabel("output index")
        ax.set_ylabel("value")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best", fontsize=8)
    fig.suptitle("Ground truth vs models (output index)", fontsize=12, y=1.0)
    for ax_row in axes:
        ax_row[0].margins(x=0.01)
    pdir = os.path.dirname(os.path.abspath(save_path))
    if pdir:
        os.makedirs(pdir, exist_ok=True)
    plt.tight_layout()
    fig.savefig(save_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_real_well_depth_profile(
    depth_axis: np.ndarray,
    profiles: Dict[str, np.ndarray],
    save_path: str,
    train_test_boundary: Optional[float] = None,
    title: str = "F03-4 porosity profile (test block reconstruction)",
    methods_to_plot: Optional[List[str]] = None,
    profile_x_label: str = "porosity",
) -> None:
    """
    Depth profile: target property (x-axis) vs depth (y-axis, increasing downward),
    matching the external-report convention. profiles must contain key 'observed'
    (required); any other keys are treated as model predictions (point-wise, aligned
    with depth_axis via overlapping-window averaging). NaN-safe: NaNs are left as gaps.
    """
    import matplotlib.pyplot as plt

    d = np.asarray(depth_axis, dtype=np.float64).ravel()
    if d.size == 0 or "observed" not in profiles:
        return
    obs = np.asarray(profiles["observed"], dtype=np.float64).ravel()
    if obs.shape[0] != d.shape[0]:
        raise ValueError("depth_axis and observed must have equal length.")

    mask_obs = np.isfinite(obs)
    if not bool(np.any(mask_obs)):
        return

    default_order = [
        "ml_only",
        "mlp_concat_ub",
        "pca_regression_ub",
        "ae_regression_ub",
        "ridge_prior_csgm",
        "mlp_prior_csgm",
        "hybrid_fista",
        "hybrid_lfista_frozen",
        "hybrid_lfista_joint",
    ]
    keys = methods_to_plot if methods_to_plot else default_order
    present = [k for k in keys if k in profiles]
    n_models = len(present)

    fig_w = 5.0
    fig_h = 7.5
    n_cols = max(1, 1 + n_models)
    fig, axes = plt.subplots(1, n_cols, figsize=(fig_w * n_cols * 0.6 + 2.5, fig_h), sharey=True)
    if n_cols == 1:
        axes = np.array([axes])
    obs_color = "#404040"

    ax0 = axes[0]
    ax0.plot(obs[mask_obs], d[mask_obs], color=obs_color, linewidth=1.1, label="observed")
    if train_test_boundary is not None and np.isfinite(float(train_test_boundary)):
        ax0.axhline(float(train_test_boundary), color="#d62728", linestyle="--", linewidth=1.0,
                    label="train/test limit")
    ax0.set_xlabel(str(profile_x_label))
    ax0.set_ylabel("depth (m)")
    ax0.grid(True, alpha=0.3)
    ax0.legend(loc="best", fontsize=10)
    ax0.set_title("observed", fontsize=11)

    for ax_i, k in enumerate(present, start=1):
        ax = axes[ax_i]
        pr = np.asarray(profiles[k], dtype=np.float64).ravel()
        if pr.shape[0] != d.shape[0]:
            continue
        m_pr = np.isfinite(pr) & mask_obs
        col = METHOD_COLORS.get(k, "#7f7f7f")
        ax.plot(obs[mask_obs], d[mask_obs], color=obs_color, linewidth=0.9, alpha=0.8, label="observed")
        label = method_display_name(k)
        ax.plot(pr[m_pr], d[m_pr], color=col, linewidth=1.2, label=label)
        if train_test_boundary is not None and np.isfinite(float(train_test_boundary)):
            ax.axhline(float(train_test_boundary), color="#d62728", linestyle="--", linewidth=1.0)
        ax.set_xlabel(str(profile_x_label))
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best", fontsize=10)
        ax.set_title(label, fontsize=11)

    for ax in axes:
        ax.invert_yaxis()

    fig.suptitle(title, fontsize=13, y=1.0)
    pdir = os.path.dirname(os.path.abspath(save_path))
    if pdir:
        os.makedirs(pdir, exist_ok=True)
    plt.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


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
    print("1) hybrid deve superar ml_only em parte substancial das razoes de medicao;")
    print("2) o ganho do hybrid sobre ml_only deve persistir ao variar m/N;")
    print("3) cs_only deve ser pior que hybrid quando a componente global nao e esparsa;")
    print("4) weighted_hybrid pode superar hybrid quando o preditor de alpha for informativo.")

    print("\nSe esses 4 pontos ocorrerem, você já tem base para:")
    print("- análise de sensibilidade;")
    print("- estudo de ablação;")
    print("- versão publicável em dados sintéticos + caso aplicado.")


def parse_cli_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SIR-CS synthetic pipeline (SIR-CS / hybrid CS).")
    parser.add_argument(
        "--profile",
        type=str,
        default="paper",
        choices=[
            "paper",
            "explore",
            "phase0_baseline",
            "solver_comparison",
            "lfista_integrated",
            "lfista_integrated_explore",
            "lfista_vs_classical",
            "lfista_vs_classical_explore",
            "robustness_phase3",
            "robustness_phase3_explore",
            "external_benchmark_stage1",
            "external_benchmark_stage1_explore",
            "direct_ub_benchmark",
            "direct_ub_benchmark_explore",
            "direct_ub_lfista_joint_only",
            "direct_ub_lfista_joint_only_explore",
        ],
        help="Profiles: paper/explore/phase0; solver_comparison; lfista_integrated; lfista_vs_classical; "
        "robustness_phase3 (Etapa 3 roadmap, one axis via --robustness-*); "
        "external_benchmark_stage1 (sklearn Etapa 1 benchmark; use sir_cs_benchmark_stage1.py).",
    )
    parser.add_argument(
        "--robustness-axis",
        type=str,
        default="",
        metavar="NAME",
        help="Etapa 3: residual_k | measurement_noise_std | residual_amplitude | output_noise_std | measurement_ratio",
    )
    parser.add_argument(
        "--robustness-value",
        type=str,
        default="",
        metavar="VAL",
        help="Single value for that axis (e.g. 6 or 0.02). Required for robustness_phase3* profiles.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_cli_args()
    cfg = Config()
    cfg.config_profile = cast(
        Literal[
            "paper",
            "explore",
            "phase0_baseline",
            "solver_comparison",
            "lfista_integrated",
            "lfista_integrated_explore",
            "lfista_vs_classical",
            "lfista_vs_classical_explore",
            "robustness_phase3",
            "robustness_phase3_explore",
            "external_benchmark_stage1",
            "external_benchmark_stage1_explore",
            "direct_ub_benchmark",
            "direct_ub_benchmark_explore",
            "direct_ub_lfista_joint_only",
            "direct_ub_lfista_joint_only_explore",
        ],
        args.profile,
    )
    apply_config_profile(cfg)
    run_artifact_id = ""
    if cfg.config_profile == "solver_comparison":
        run_artifact_id = time.strftime("%Y%m%d_%H%M%S")
        layout_solver_comparison_run(cfg, run_artifact_id)
        cfg.artifact_log_path = os.path.join(cfg.save_dir, "run_console.log")
    elif cfg.config_profile in ("lfista_integrated", "lfista_integrated_explore"):
        run_artifact_id = time.strftime("%Y%m%d_%H%M%S")
        layout_lfista_integrated_run(cfg, run_artifact_id)
        cfg.artifact_log_path = os.path.join(cfg.save_dir, "run_console.log")
    elif cfg.config_profile in ("lfista_vs_classical", "lfista_vs_classical_explore"):
        run_artifact_id = time.strftime("%Y%m%d_%H%M%S")
        layout_lfista_vs_classical_run(cfg, run_artifact_id)
        cfg.artifact_log_path = os.path.join(cfg.save_dir, "run_console.log")
    elif cfg.config_profile in ("robustness_phase3", "robustness_phase3_explore"):
        if not args.robustness_axis.strip() or not args.robustness_value.strip():
            print(
                "ERROR: robustness_phase3* requires --robustness-axis and --robustness-value",
                file=sys.stderr,
            )
            sys.exit(2)
        apply_robustness_param_override(cfg, args.robustness_axis.strip(), args.robustness_value.strip())
        cfg.robustness_axis = args.robustness_axis.strip()
        cfg.robustness_value_raw = args.robustness_value.strip()
        run_artifact_id = time.strftime("%Y%m%d_%H%M%S")
        layout_robustness_phase3_run(
            cfg, run_artifact_id, cfg.robustness_axis, robustness_value_slug(cfg.robustness_value_raw)
        )
        cfg.artifact_log_path = os.path.join(cfg.save_dir, "run_console.log")
    else:
        os.makedirs(cfg.save_dir, exist_ok=True)

    if cfg.config_profile == "solver_comparison":
        protocol = "\n".join(
            [
                "Solver comparison protocol (roadmap Etapa 1).",
                "dual_cs_solver=True: hybrid_fista, hybrid_spgl1, cs_only_fista, cs_only_spgl1 + ml_only.",
                "Requires: pylops and spgl1 (pip install pylops spgl1).",
                "",
                "run_id: " + run_artifact_id,
                "save_dir (relative): " + cfg.save_dir,
                "plots_subdir (relative to save_dir): " + cfg.plots_subdir,
                "",
                "seeds: " + str(cfg.seeds),
                "measurement_ratios: " + str(cfg.measurement_ratios),
                "l1_lambda_grid (FISTA): " + str(cfg.l1_lambda_grid),
                "spgl1_tau_grid (SPGL1 LASSO): " + str(cfg.spgl1_tau_grid),
                "",
                "Figures (repo): paper/figures/solver_comparison/runs/" + run_artifact_id + "/",
                "Symlink: outputs/solver_comparison/LATEST -> this run",
            ]
        )
        with open(os.path.join(cfg.save_dir, "PROTOCOL.txt"), "w", encoding="utf-8") as f:
            f.write(protocol + "\n")

    if cfg.config_profile in ("lfista_integrated", "lfista_integrated_explore"):
        protocol = "\n".join(
            [
                "lfista_integrated protocol: classic pipeline + optional PyTorch LFISTA branch.",
                "run_lfista=True appends ml_only_torch, hybrid_lfista_frozen, hybrid_lfista_joint.",
                "Requires: torch (pip install torch).",
                "",
                "run_id: " + run_artifact_id,
                "save_dir (relative): " + cfg.save_dir,
                "plots_subdir (relative to save_dir): " + cfg.plots_subdir,
                "",
                "seeds: " + str(cfg.seeds),
                "measurement_ratios: " + str(cfg.measurement_ratios),
                "lfista_steps (K): " + str(cfg.lfista_steps),
                "",
                "Figures (repo): paper/figures/lfista_integrated/runs/" + run_artifact_id + "/",
                "Symlink: outputs/lfista_integrated/LATEST -> this run",
            ]
        )
        with open(os.path.join(cfg.save_dir, "PROTOCOL.txt"), "w", encoding="utf-8") as f:
            f.write(protocol + "\n")

    if cfg.config_profile in ("lfista_vs_classical", "lfista_vs_classical_explore"):
        protocol = "\n".join(
            [
                "lfista_vs_classical protocol: Phase 0 seeds/rho + dual_cs_solver + LFISTA branch.",
                "Methods: ml_only, hybrid_fista, hybrid_spgl1, cs_only_fista, cs_only_spgl1,",
                "ml_only_torch, hybrid_lfista_frozen, hybrid_lfista_joint.",
                "Requires: torch, pylops, spgl1 (pip install -r requirements.txt).",
                "",
                "run_id: " + run_artifact_id,
                "save_dir (relative): " + cfg.save_dir,
                "plots_subdir (relative to save_dir): " + cfg.plots_subdir,
                "",
                "seeds: " + str(cfg.seeds),
                "measurement_ratios: " + str(cfg.measurement_ratios),
                "dual_cs_solver: True | run_lfista: True",
                "Focus CSVs: summary_focus_ml_hybrid_fista_lfista.csv, summary_by_seed_focus_*.csv",
                "",
                "Figures (repo): paper/figures/lfista_vs_classical/runs/" + run_artifact_id + "/",
                "Symlink: outputs/lfista_vs_classical/LATEST -> this run",
            ]
        )
        with open(os.path.join(cfg.save_dir, "PROTOCOL.txt"), "w", encoding="utf-8") as f:
            f.write(protocol + "\n")

    if cfg.config_profile == "phase0_baseline":
        protocol = "\n".join(
            [
                "Phase 0 baseline protocol (roadmap_proximos_passos.md Fase 0).",
                "ASCII log; UTF-8 optional for user-facing strings elsewhere.",
                "",
                "seeds (10, fixed list): " + str(cfg.seeds),
                "measurement_ratios: " + str(cfg.measurement_ratios),
                "l1_lambda_grid: " + str(cfg.l1_lambda_grid),
                "residual_basis: identity | residual_k: 6 | residual_mode: support_from_u",
                "measurement_kind: gaussian | noise stds: measurement 0.02, output 0.01",
                "n_train/n_val/n_test: 1200/300/300 | p_input: 12 | n_output: 128",
                "",
                "Artifacts: detailed_results.csv, summary_by_seed.csv, summary.csv, config.json",
                "Figures: ../paper/figures/phase0_baseline/ (relative to save_dir)",
            ]
        )
        with open(os.path.join(cfg.save_dir, "PROTOCOL.txt"), "w", encoding="utf-8") as f:
            f.write(protocol + "\n")

    if cfg.config_profile in ("robustness_phase3", "robustness_phase3_explore"):
        protocol = "\n".join(
            [
                "robustness_phase3 protocol (roadmap Etapa 3): vary one synthetic knob; rest matches profile baseline.",
                "Methods: same as lfista_vs_classical (dual_cs_solver + run_lfista).",
                "Requires: torch, pylops, spgl1.",
                "",
                "robustness_axis: " + cfg.robustness_axis,
                "robustness_value: " + cfg.robustness_value_raw,
                "run_id: " + run_artifact_id,
                "save_dir (relative): " + cfg.save_dir,
                "plots_subdir (relative to save_dir): " + cfg.plots_subdir,
                "",
                "seeds: " + str(cfg.seeds),
                "measurement_ratios: " + str(cfg.measurement_ratios),
                "residual_k: "
                + str(cfg.residual_k)
                + " | measurement_noise_std: "
                + str(cfg.measurement_noise_std)
                + " | output_noise_std: "
                + str(cfg.output_noise_std)
                + " | residual_amplitude: "
                + str(cfg.residual_amplitude),
                "",
                "Figures: paper/figures/robustness_phase3/<axis>/v_<slug>/runs/<run_id>/",
                "Symlink: outputs/robustness_phase3/<axis>/LATEST -> v_<slug>/runs/<run_id>",
            ]
        )
        with open(os.path.join(cfg.save_dir, "PROTOCOL.txt"), "w", encoding="utf-8") as f:
            f.write(protocol + "\n")

    with open(os.path.join(cfg.save_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(asdict(cfg), f, indent=2)

    total_jobs = len(cfg.seeds) * len(cfg.measurement_ratios)
    log_line(
        cfg,
        f"=== SIR-CS start | profile={cfg.config_profile} | jobs={total_jobs} | "
        f"seeds={cfg.seeds} | measurement_ratios={cfg.measurement_ratios} | "
        f"output_dir={cfg.save_dir} ===",
    )

    all_dfs = []
    first_examples = None
    all_gt_bundles: List[Dict[str, np.ndarray]] = []

    t0 = time.time()
    job_idx = 0
    for seed in cfg.seeds:
        for mr in cfg.measurement_ratios:
            job_idx += 1
            log_line(
                cfg,
                f"--- job {job_idx}/{total_jobs} seed={seed} measurement_ratio={mr:.2f} ---",
            )
            df, examples, gt_bundle = run_single_setting(cfg, seed=seed, measurement_ratio=mr)
            all_dfs.append(df)
            all_gt_bundles.append(gt_bundle)
            if first_examples is None:
                first_examples = examples

    detailed = pd.concat(all_dfs, ignore_index=True)
    per_seed = summarize_results_per_seed(detailed)
    summary = summarize_results_across_seeds(per_seed)

    detailed_path = os.path.join(cfg.save_dir, "detailed_results.csv")
    per_seed_path = os.path.join(cfg.save_dir, "summary_by_seed.csv")
    summary_path = os.path.join(cfg.save_dir, "summary.csv")
    detailed.to_csv(detailed_path, index=False)
    per_seed.to_csv(per_seed_path, index=False)
    summary.to_csv(summary_path, index=False)

    focus_paths = save_lfista_vs_classical_focus_tables(cfg, summary, per_seed)
    if focus_paths and cfg.log_progress:
        log_line(cfg, "  [artifacts] focus tables: " + ", ".join(os.path.basename(p) for p in focus_paths))

    plot_paths = save_all_comparison_plots(cfg, summary, per_seed)

    if first_examples is not None:
        ex_path = os.path.join(plots_directory(cfg), "08_example_ground_truth_vs_models.png")
        plot_examples(cfg, first_examples, save_path=ex_path)
        plot_paths.append(ex_path)

    merged_gt = merge_gt_pred_bundles(all_gt_bundles)
    if len(merged_gt) >= 2 and "y_true" in merged_gt:
        plot_paths.extend(save_ground_truth_vs_model_plots(cfg, merged_gt))

    elapsed = time.time() - t0

    if run_artifact_id:
        readme_p = os.path.join(cfg.save_dir, "README_RUN.txt")
        finished = time.strftime("%Y-%m-%dT%H:%M:%S")
        if cfg.config_profile == "solver_comparison":
            fig_run_hint = "paper/figures/solver_comparison/runs/" + run_artifact_id + "/"
        elif cfg.config_profile in ("lfista_integrated", "lfista_integrated_explore"):
            fig_run_hint = "paper/figures/lfista_integrated/runs/" + run_artifact_id + "/"
        elif cfg.config_profile in ("lfista_vs_classical", "lfista_vs_classical_explore"):
            fig_run_hint = "paper/figures/lfista_vs_classical/runs/" + run_artifact_id + "/"
        elif cfg.config_profile in ("robustness_phase3", "robustness_phase3_explore"):
            vdir = "v_" + robustness_value_slug(cfg.robustness_value_raw)
            fig_run_hint = (
                "paper/figures/robustness_phase3/"
                + cfg.robustness_axis
                + "/"
                + vdir
                + "/runs/"
                + run_artifact_id
                + "/"
            )
        else:
            fig_run_hint = ""
        extra = [
            "",
            "finished_local: " + finished,
            "elapsed_seconds: {:.1f}".format(elapsed),
            "",
            "CSV and logs in this directory; figures under " + fig_run_hint,
            "",
            "Plot files:",
        ]
        with open(readme_p, "a", encoding="utf-8") as f:
            f.write("\n".join(extra) + "\n")
            for pp in sorted(plot_paths):
                f.write("  " + os.path.basename(pp) + "\n")

    print("\n" + "=" * 72)
    print("RESUMO")
    print("=" * 72)
    print(summary.round(4).to_string(index=False))
    out_abs = os.path.abspath(cfg.save_dir)
    print(f"\nArquivos salvos em: {out_abs}")
    print(f"  detailed_results.csv, summary_by_seed.csv, summary.csv, config.json")
    print(f"  Figuras comparativas ({len(plot_paths)} arquivos) em:")
    print(f"  {os.path.join(out_abs, cfg.plots_subdir)}/")
    for pp in sorted(plot_paths):
        print(f"    - {os.path.basename(pp)}")
    print(f"Tempo total: {elapsed:.2f} s")

    print_stage_guidance(summary)


if __name__ == "__main__":
    main()
