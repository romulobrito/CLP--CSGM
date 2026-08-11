# CLP-CSGM

Conditional Latent-Prior Compressed Sensing with Generative Models (CLP-CSGM)
for petrophysical property estimation from dense logs and sparse target
measurements.

This repository contains the code needed to reproduce CLP-CSGM experiments.
It intentionally excludes private raw data and heavy experiment outputs.

## License

Source code in this repository is distributed under the MIT License.
See the `LICENSE` file for the full text.

## 1) What this repository contains

- Core CLP-CSGM implementation:
  - `csgm_m2_module.py`
- Shared benchmark infrastructure:
  - `sir_cs_benchmark_direct_ub.py`
  - `sir_cs_pipeline_optimized.py`
  - `direct_ub_baselines.py`
  - `external_benchmarks.py`
- Dataset loaders/builders:
  - `multi_well_vc.py`
  - `real_well_f03.py`
- Main benchmark launchers:
  - `sir_cs_benchmark_multi_well_vc.py`
  - `sir_cs_benchmark_real_well_direct_ub.py`
- Reproduction helpers:
  - `scripts/`

Note: some filenames keep historical `sir_cs_*` names because CLP-CSGM was
developed on top of a broader CS benchmark codebase. For CLP-CSGM runs, use
`--run-csgm-m2` and `--no-lfista`.

## 2) Method summary

CLP-CSGM learns:

1. a decoder `G(z)` via autoencoder on target windows;
2. a conditional latent prior `z0 = h(u)` from dense log windows.

At validation and test time, latent refinement solves:

`z_hat = argmin_z ||M G(z) - b||_2^2 + lambda ||z - z0(u)||_2^2`

and returns:

`y_hat = G(z_hat)`

where:

- `M` is the sparse measurement operator (typically coordinate subsampling),
- `b` is the sparse target observation vector,
- `lambda` is selected on validation split.

## 3) Data requirements

### 3.1 Repository policy

This repository ships **code plus a small synthetic smoke dataset**. It does
**not** redistribute the processed F3 paper tables or private field datasets.

- **Synthetic smoke (included):** `data/synthetic/` provides dummy well-log
  tables so reviewers can install dependencies and execute CLP-CSGM smoke tests
  with no external download. See Section 6.1 and `data/README.md`.
- **F3 / Zhang et al. (Sensors):** the study publishes scripts and data via an
  external archive and cites industry/academic data access (see their Data
  Availability Statement). Users should obtain the tabulated inputs from the
  sources linked in that paper and place them locally under `data/` for full
  paper reproduction (Section 5).
- **Lapa / Alvarez et al. (Journal of Applied Geophysics):** the article states
  that the authors **do not have permission to share the data**. Do **not**
  expect a public drop-in dataset for those supplementary scripts; keep any local
  copies private and compliant with your data-use agreement.

### 3.2 Where to look first (external pointers)

These are **starting points** cited in the literature; follow each provider's
current terms:

- Zhang et al. (2025) Data Availability (scripts + data bundle):  
  `https://gitee.com/zhangjixju/ml-code/tree/master`
- F3 Demo / OpendTect context (often referenced alongside F3 studies):  
  `https://terranubis.com/datainfo/F3-Demo-2020`

### 3.3 Local layout expected by the benchmarks

Synthetic tables already present for offline tests:

- `data/synthetic/SYN-A,SYN-B,SYN-C_6logs.txt`
- `data/synthetic/SYN-D_6logs.txt`
- `data/synthetic/SYN-D_AC+GR+Porosity.txt`

For the **main paper** benchmark commands (Section 5), also place:

- `data/F02-1,F03-2,F06-1_6logs_30dB.txt`
- `data/F03-4_6logs_30dB.txt`
- `data/F03-4_AC+GR+Porosity.txt`

Optional inputs for supplementary Lapa/Auddys-style scripts are **not** provided
here; organize private paths consistent with your agreements.

## 4) Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Main dependencies are listed in `requirements.txt`:
`numpy`, `pandas`, `matplotlib`, `scikit-learn`, `torch`, `PyWavelets`, `scipy`.

## 5) Main reproducibility runs

The main cross-well Vc benchmark excludes gamma ray from the dense input
`u`. Use `--channels sonic,rhob,ai,vp` (sonic, bulk density, acoustic
impedance, and Vp). This matches the manuscript protocol and the default of
`sir_cs_benchmark_multi_well_vc.py`. GR is omitted to avoid the trivial
GR -> Vc shortcut (Larionov-derived target). The GR-inclusive setting is
only a sensitivity check (Appendix B.7); see Section 8 below.

### 5.1 Cross-well Vc, main no-GR protocol (Ridge prior + CLP-CSGM ablations)

```bash
python sir_cs_benchmark_multi_well_vc.py \
  --train-path data/F02-1,F03-2,F06-1_6logs_30dB.txt \
  --test-path data/F03-4_6logs_30dB.txt \
  --channels sonic,rhob,ai,vp \
  --target vc \
  --step 8 \
  --rhos 0.05,0.10,0.20 \
  --seeds 7,23,41 \
  --run-csgm-m2 \
  --run-csgm-ablations \
  --csgm-prior-type ridge \
  --no-lfista
```

Repeat with `--step 16` and `--step 32`.

### 5.2 Cross-well Vc, main no-GR protocol (MLP prior sensitivity)

```bash
python sir_cs_benchmark_multi_well_vc.py \
  --train-path data/F02-1,F03-2,F06-1_6logs_30dB.txt \
  --test-path data/F03-4_6logs_30dB.txt \
  --channels sonic,rhob,ai,vp \
  --target vc \
  --step 8 \
  --rhos 0.05,0.10,0.20 \
  --seeds 7,23,41 \
  --run-csgm-m2 \
  --csgm-prior-type mlp \
  --no-lfista
```

Repeat with `--step 16` and `--step 32`.

### 5.3 Real-well F03-4 GR-only porosity (Ridge + ablations)

```bash
python sir_cs_benchmark_real_well_direct_ub.py \
  --data-path data/F03-4_AC+GR+Porosity.txt \
  --u-channels gr \
  --rhos 0.20,0.30,0.40,0.50,0.60 \
  --seeds 7,23,41 \
  --run-csgm-m2 \
  --run-csgm-ablations \
  --csgm-prior-type ridge \
  --no-lfista
```

### 5.4 Real-well F03-4 GR-only porosity (MLP prior sensitivity)

```bash
python sir_cs_benchmark_real_well_direct_ub.py \
  --data-path data/F03-4_AC+GR+Porosity.txt \
  --u-channels gr \
  --rhos 0.20,0.30,0.40,0.50,0.60 \
  --seeds 7,23,41 \
  --run-csgm-m2 \
  --csgm-prior-type mlp \
  --no-lfista
```

## 6) Smoke tests

### 6.1 Offline synthetic smoke (no external data; recommended first check)

Uses only the committed tables under `data/synthetic/`. This path is intended
for CAGEO/code evaluation when the paper F3 files are not yet available locally.
It exercises CLP-CSGM Ridge, structural ablations, and both benchmark launchers
with reduced epochs/iterations.

```bash
bash scripts/run_synthetic_smoke.sh
```

Equivalent explicit commands:

```bash
python scripts/make_synthetic_smoke_data.py   # optional regenerate

python sir_cs_benchmark_multi_well_vc.py \
  --train-path data/synthetic/SYN-A,SYN-B,SYN-C_6logs.txt \
  --test-path data/synthetic/SYN-D_6logs.txt \
  --channels sonic,rhob,ai,vp \
  --target vc \
  --window-len 64 \
  --step 16 \
  --fast \
  --run-csgm-m2 \
  --run-csgm-ablations \
  --csgm-prior-type ridge \
  --csgm-ae-epochs 8 \
  --csgm-iters 20 \
  --csgm-restarts 1 \
  --csgm-lambda-grid 0.01 \
  --no-lfista \
  --base-dir outputs/synthetic_smoke/cross_well_vc \
  --run-id synthetic_crosswell_smoke

python sir_cs_benchmark_real_well_direct_ub.py \
  --data-path data/synthetic/SYN-D_AC+GR+Porosity.txt \
  --u-channels gr \
  --window-len 64 \
  --step 8 \
  --fast \
  --run-csgm-m2 \
  --run-csgm-ablations \
  --csgm-prior-type ridge \
  --csgm-ae-epochs 8 \
  --csgm-iters 20 \
  --csgm-restarts 1 \
  --csgm-lambda-grid 0.01 \
  --no-lfista \
  --base-dir outputs/synthetic_smoke/real_well \
  --run-id synthetic_porosity_smoke
```

Synthetic RMSE values are **not** comparable to the manuscript tables.

### 6.2 Fast smoke on paper F3 files (requires local data)

Same main no-GR channels as Section 5.1, but needs the F3 tables listed in
Section 3.3.

```bash
python sir_cs_benchmark_multi_well_vc.py \
  --train-path data/F02-1,F03-2,F06-1_6logs_30dB.txt \
  --test-path data/F03-4_6logs_30dB.txt \
  --channels sonic,rhob,ai,vp \
  --target vc \
  --step 32 \
  --run-csgm-m2 \
  --run-csgm-ablations \
  --csgm-prior-type ridge \
  --no-lfista \
  --fast

python sir_cs_benchmark_real_well_direct_ub.py \
  --data-path data/F03-4_AC+GR+Porosity.txt \
  --u-channels gr \
  --run-csgm-m2 \
  --run-csgm-ablations \
  --csgm-prior-type ridge \
  --no-lfista \
  --fast
```

## 7) Robustness and supplementary scripts

Scripts commonly used for paper supplementary analyses:

- `scripts/make_synthetic_smoke_data.py`
- `scripts/run_synthetic_smoke.sh`
- `scripts/auddys_smoke_direct_ub.py`
- `scripts/auddys_clp_csgm_eda.py`
- `scripts/clp_csgm_diagnostic_assets.py`
- `scripts/clp_csgm_runtime_study.py`
- `scripts/clp_csgm_srec_diagnostics.py`
- `scripts/clp_csgm_ablation_assets.py`
- `scripts/clp_csgm_paper_assets.py`
- `scripts/clp_csgm_quick_figures.py`
- `scripts/clp_csgm_uncertainty_assets.py`

Student-t seed-level intervals and paired Delta-RMSE tables/figures for the manuscript are regenerated with `scripts/clp_csgm_uncertainty_assets.py` (requires local `outputs/` summaries from the main benchmarks).

## 8) Appendix B.7: GR-inclusive sensitivity and embargo boundary check

These commands are **not** the main cross-well benchmark. They reproduce the
Appendix B.7 channel-ablation / embargo-style checks: reintroducing GR into
`u` (stronger dense context) and optionally dropping overlap-prone
train/validation boundary windows.

GR-inclusive sensitivity (step 16 example; do not treat as the main result):

```bash
python sir_cs_benchmark_multi_well_vc.py \
  --train-path data/F02-1,F03-2,F06-1_6logs_30dB.txt \
  --test-path data/F03-4_6logs_30dB.txt \
  --channels sonic,rhob,gr,ai,vp \
  --target vc \
  --step 16 \
  --rhos 0.05,0.10,0.20 \
  --seeds 7,23,41 \
  --run-csgm-m2 \
  --run-csgm-ablations \
  --csgm-prior-type ridge \
  --no-lfista
```

Embargo-style boundary check (main no-GR channels + boundary-window removal):

```bash
python sir_cs_benchmark_multi_well_vc.py \
  --train-path data/F02-1,F03-2,F06-1_6logs_30dB.txt \
  --test-path data/F03-4_6logs_30dB.txt \
  --channels sonic,rhob,ai,vp \
  --target vc \
  --step 16 \
  --rhos 0.05,0.10,0.20 \
  --seeds 7,23,41 \
  --val-embargo-windows 4 \
  --run-csgm-m2 \
  --run-csgm-ablations \
  --csgm-prior-type ridge \
  --no-lfista
```

`--val-embargo-windows <k>` drops overlap-prone boundary windows between the
training and validation tails. Combine with `--channels sonic,rhob,gr,ai,vp`
only when reproducing the GR-inclusive embargo row of Appendix B.7.

## 9) Outputs and repository policy

This repository should remain lightweight:

- do not commit `outputs/**` heavy run artifacts;
- do not commit processed well-log tables or field datasets (see Section 3);
- keep only code, small configs/tables needed for reproducibility logic.

## 10) Reproducibility defaults

- Main seeds: `7,23,41`
- Cross-well low-data steps: `8,16,32`
- Cross-well rhos: `0.05,0.10,0.20`
- Cross-well dense channels (main): `sonic,rhob,ai,vp` (GR excluded)
- Cross-well GR-inclusive channels (Appendix B.7 only): `sonic,rhob,gr,ai,vp`
- F03-4 rhos: `0.20,0.30,0.40,0.50,0.60`
- Main method: CLP-CSGM Ridge
- Sensitivity method: CLP-CSGM MLP prior

## 11) Citation

If you use this codebase, cite the CLP-CSGM paper and the corresponding data
source papers referenced in the manuscript.
