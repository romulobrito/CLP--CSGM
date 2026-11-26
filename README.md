# CLP-CSGM

Conditional Latent-Prior Compressed Sensing with Generative Models (CLP-CSGM) for reconstruction of petrophysical target logs from continuous auxiliary measurements and sparse target observations.

Code accompanying *Conditional Latent-Prior Generative Compressed Sensing for Petrophysical Target-Log Reconstruction*, revised manuscript for Computers & Geosciences. This repository is distributed under the [MIT License](LICENSE).

**Validation status:** revision code and standalone reproduction drivers are being prepared. The commands below describe the intended workflow; clean-environment execution and numerical agreement with the manuscript have not yet been verified for this revision. No field datasets or trained models are distributed here.

## Method

Train an autoencoder on complete target windows and a conditional latent map h from dense auxiliary windows u to an initial code z0 = h(u). Given sparse observations b and measurement operator M, refine the latent code by minimizing

`||M G(z) - b||² + lambda ||z - z0||²`

and reconstruct with the decoder G. Target standardization is accounted for in the measurement term; lambda is selected on validation data. The decoder is fixed during refinement. The central implementation is [csgm_m2_module.py](csgm_m2_module.py).

## Install and try without field data

The manuscript reports Python 3.10, NumPy 1.26, pandas 2.3, Matplotlib 3.7, scikit-learn 1.6, PyTorch 2.6 and PyWavelets 1.4. A multi-core CPU with at least 8 GB RAM is reported; CUDA is optional. The current requirements file specifies minimum dependencies, not a validated exact environment lock.

```bash
git clone https://github.com/romulobrito/CLP--CSGM.git
cd CLP--CSGM
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
bash scripts/run_synthetic_smoke.sh
```

Use the revision branch while preparation is in progress: `git checkout revision/cageo-d-26-01398-reproducibility` before installing. The included synthetic tables exercise both main launchers with reduced training and refinement budgets; their errors cannot be compared to paper results.

## Reproduce the paper experiments

Supply authorized F3 input tables as described in [Data requirements](docs/DATA_REQUIREMENTS.md). The main cross-well protocol uses sonic, RHOB, AI and Vp, without GR because the target Vc is derived from gamma ray. The held-out well is F03-4. The separate single-well porosity benchmark uses GR only.

```bash
# Main paired comparison: ten seeds, Table 4
bash scripts/run_r1m04_seed_panel.sh

# Main comparison, separate three-seed benchmarks, ablations,
# controlled prior comparisons, and main cross-well figures
bash scripts/run_reproduce_clp_csgm.sh
```

The primary comparison uses seeds 7,23,41,13,29,31,37,43,47,53; the wider method grid, structural ablations, F03-4 and most supplements retain seeds 7,23,41. Do not replace the ten-seed table with three-seed outputs. Run supplemental scripts separately as documented in [Experiments](docs/EXPERIMENTS.md).

## Repository guide

| Location | Purpose |
| --- | --- |
| csgm_m2_module.py | Autoencoder, conditional prior and latent refinement |
| direct_ub_baselines.py | Direct supervised comparators |
| multi_well_vc.py; real_well_f03.py | Data loading, windowing and partitioning |
| sir_cs_benchmark_multi_well_vc.py; sir_cs_benchmark_real_well_direct_ub.py | Benchmark entry points |
| sir_cs_benchmark_direct_ub.py; sir_cs_pipeline_optimized.py; sir_cs_pipeline.py | Shared experimental infrastructure |
| scripts/ | Experiment drivers, aggregation, figures and synthetic smoke |
| docs/ | Protocols, experiment map, data requirements and provenance |

Historical module names are retained for compatibility. For CLP-CSGM benchmarks use --run-csgm-m2 and --no-lfista. Generated outputs belong under outputs/ and are not committed.

## Documentation

- [Experiment-to-evidence map and commands](docs/EXPERIMENTS.md)
- [Data requirements and redistribution policy](docs/DATA_REQUIREMENTS.md)
- [Reproduction procedure and validation limits](docs/REPRODUCIBILITY.md)
- [Result dependency map](docs/RESULT_DEPENDENCIES.md)
- [Source transfer manifest](docs/SOURCE_MANIFEST.json)
- [Frozen experimental protocols](docs/protocols/)

Cite the manuscript and the corresponding source-data publications when using this code. Publication metadata and the immutable release reference will be added only when confirmed.
