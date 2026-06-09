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

This repository ships **code only**. It does **not** commit processed well-log
tables or field datasets. Reasons:

- **F3 / Zhang et al. (Sensors):** the study publishes scripts and data via an
  external archive and cites industry/academic data access (see their Data
  Availability Statement). Redistribution here would require checking **their**
  archive license and any **third-party** data terms (e.g., F3 demo projects).
  Users should obtain files from the sources linked in that paper and place them
  locally under `data/`.
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

Create `data/` at the repository root and place required files there after you
obtain them under the applicable licenses.

Required files for the **main paper** benchmark commands:

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
`numpy`, `pandas`, `matplotlib`, `scikit-learn`, `torch`, `PyWavelets`.

## 5) Main reproducibility runs

### 5.1 Cross-well Vc (Ridge prior + CLP-CSGM ablations)

```bash
python sir_cs_benchmark_multi_well_vc.py \
  --train-path data/F02-1,F03-2,F06-1_6logs_30dB.txt \
  --test-path data/F03-4_6logs_30dB.txt \
  --channels sonic,rhob,gr,ai,vp \
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

### 5.2 Cross-well Vc (MLP prior sensitivity)

```bash
python sir_cs_benchmark_multi_well_vc.py \
  --train-path data/F02-1,F03-2,F06-1_6logs_30dB.txt \
  --test-path data/F03-4_6logs_30dB.txt \
  --channels sonic,rhob,gr,ai,vp \
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

## 6) Fast smoke tests

```bash
python sir_cs_benchmark_multi_well_vc.py \
  --train-path data/F02-1,F03-2,F06-1_6logs_30dB.txt \
  --test-path data/F03-4_6logs_30dB.txt \
  --channels sonic,rhob,gr,ai,vp \
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

- `scripts/auddys_smoke_direct_ub.py`
- `scripts/auddys_clp_csgm_eda.py`
- `scripts/clp_csgm_diagnostic_assets.py`
- `scripts/clp_csgm_runtime_study.py`
- `scripts/clp_csgm_srec_diagnostics.py`
- `scripts/clp_csgm_ablation_assets.py`
- `scripts/clp_csgm_paper_assets.py`
- `scripts/clp_csgm_quick_figures.py`

## 8) Embargo boundary check (anti-leakage)

For cross-well train/validation overlap diagnostics, use:

- `--val-embargo-windows <k>` in `sir_cs_benchmark_multi_well_vc.py`

This drops overlap-prone boundary windows between train and validation tails.

## 9) Outputs and repository policy

This repository should remain lightweight:

- do not commit `outputs/**` heavy run artifacts;
- do not commit processed well-log tables or field datasets (see Section 3);
- keep only code, small configs/tables needed for reproducibility logic.

## 10) Reproducibility defaults

- Main seeds: `7,23,41`
- Cross-well low-data steps: `8,16,32`
- Cross-well rhos: `0.05,0.10,0.20`
- F03-4 rhos: `0.20,0.30,0.40,0.50,0.60`
- Main method: CLP-CSGM Ridge
- Sensitivity method: CLP-CSGM MLP prior

## 11) Citation

If you use this codebase, cite the CLP-CSGM paper and the corresponding data
source papers referenced in the manuscript.
