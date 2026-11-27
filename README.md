# CLP-CSGM

Conditional Latent-Prior Compressed Sensing with Generative Models (CLP-CSGM) for reconstruction of petrophysical target logs from continuous auxiliary measurements and sparse target observations.

Code accompanying *Conditional Latent-Prior Generative Compressed Sensing for Petrophysical Target-Log Reconstruction*, revised manuscript for Computers & Geosciences. This repository is distributed under the [MIT License](LICENSE).

**Validation status:** all 21 experimental tables (04–24) were independently executed locally by the author, and all 26 numerical figures regenerated. Scientific conclusions were preserved with documented exceptions in numerical magnitudes and seed counts: 867/1124 metrics agree within the original published-precision tolerance, three tables fully match, and two targeted repeatability checks were exact in the tested environment. This is partial numerical agreement, not complete historical numerical equivalence. See the [field reproducibility report](docs/FIELD_REPRODUCIBILITY_REPORT.md). No field datasets or trained models are distributed here. The exact package-version snapshot is available; fresh-install validation passed in both the [technical](https://github.com/romulobrito/CLP--CSGM/actions/runs/46042013857) and [synthetic](https://github.com/romulobrito/CLP--CSGM/actions/runs/46042013860) CI jobs.

## Method

Train an autoencoder on complete target windows and a conditional latent map h from dense auxiliary windows u to an initial code z0 = h(u). Given sparse observations b and measurement operator M, refine the latent code by minimizing

`||M G(z) - b||² + lambda ||z - z0||²`

and reconstruct with the decoder G. Target standardization is accounted for in the measurement term; lambda is selected on validation data. The decoder is fixed during refinement. The central implementation is [csgm_m2_module.py](csgm_m2_module.py).

## Install and try without field data

The reference replay used CPython 3.10.22 on Linux x86_64 (Ubuntu 22.04.5 LTS), CPU PyTorch 2.6.0+cpu, and one numerical thread. The 33 exact installed distribution versions are recorded in [requirements-release-cpu.txt](requirements-release-cpu.txt); [environment documentation](docs/RELEASE_ENVIRONMENT.md) describes the observed hardware, numerical libraries and limitations. The minimum-version requirements.txt is retained for other installations, not exact replay.

```bash
git clone https://github.com/romulobrito/CLP--CSGM.git
cd CLP--CSGM
# For the reviewed version, check out the immutable release tag once published.
# Ensure "python" resolves to CPython 3.10.22.
bash scripts/install_release_cpu.sh
source .venv-release/bin/activate
export CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export MPLBACKEND=Agg
python scripts/check_release_environment.py
bash scripts/run_synthetic_smoke.sh
```

The installer creates a new virtualenv and refuses to modify an existing one. The included synthetic tables exercise both main launchers with reduced budgets; their errors cannot be compared to paper results. Version matching does not guarantee bitwise agreement across hardware or wheel builds.

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
- [Field reproducibility report](docs/FIELD_REPRODUCIBILITY_REPORT.md)
- [Recorded CPU environment and installation](docs/RELEASE_ENVIRONMENT.md)
- [Result dependency map](docs/RESULT_DEPENDENCIES.md)
- [Source transfer manifest](docs/SOURCE_MANIFEST.json)
- [Frozen experimental protocols](docs/protocols/)

Cite the manuscript and the corresponding source-data publications when using this code. Publication metadata and the immutable release reference will be added only when confirmed.
