# RELEASE-05e — recorded CPU environment

## Source and scope

Author supplied the complete post-execution snapshot of the unchanged local virtual environment used for Gates A–E on 27 November 2026. Raw record SHA-256 `e3946441f4f02519bcde27bd71c750ffda97b00d5dcab10cceceba5784eb8954`, 21,948 bytes, 467 lines, ASCII; hash, length and line count independently checked on the supplied file. The raw record is retained by the author; local usernames, hostnames and absolute paths are intentionally not published. The 33 package versions are transcribed from the complete pip freeze --all block in requirements-release-cpu.txt. No local or editable dependency is present in the reported freeze. All installed packages reported installer pip; direct_url.json did not preserve original wheel/index URLs. This is the observed replay environment, not the unrecovered original manuscript environment.

| Component | Recorded value |
|---|---|
| OS | Ubuntu 22.04.5 LTS, kernel 6.8.0-138-generic, x86_64, glibc 2.35 |
| CPU | Intel Core i5-10300H, 4 physical cores, 8 logical threads, AVX2; nproc available 1 at collection |
| Python | CPython 3.10.22, GCC 11.4.0 |
| Package tools | pip 26.3, setuptools 79.0.1; pip check passed |
| NumPy | 1.26.4; OpenBLAS 0.3.23.dev, runtime Haswell/pthreads, 1 thread |
| PyTorch | 2.6.0+cpu, git 2236df1770800ffea5697b11b0bb0d910b2e59e1, AVX2, MKL 2024.2 build 20240605, oneDNN 3.5.3, OpenMP 4.5 |
| Parallel settings | PyTorch intra-op 1, inter-op 4; OMP and MKL max 1 |
| GPU | CUDA disabled and unavailable; CPU build |
| Locale at collection | LC_ALL=C |

The reported execution variables were CUDA_VISIBLE_DEVICES empty, OMP_NUM_THREADS=1, MKL_NUM_THREADS=1, MPLBACKEND=Agg. OPENBLAS_NUM_THREADS, BLIS_NUM_THREADS, VECLIB_MAXIMUM_THREADS, NUMEXPR_NUM_THREADS and PYTHONHASHSEED were unset. CI explicitly sets OPENBLAS_NUM_THREADS=1 to control effective threads, rather than claiming the same process environment or hardware. Inter-op thread count is recorded, not forced by modifying scientific code. The snapshot was collected after the campaign, not instrumented at every optimizer step.

## Installation

On Linux x86_64 with CPython 3.10.22 available as python, from a fresh checkout:

```bash
bash scripts/install_release_cpu.sh
source .venv-release/bin/activate
export CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export MPLBACKEND=Agg
python scripts/check_release_environment.py
python -m pip check
bash scripts/run_synthetic_smoke.sh
```

The installer creates a new environment, refusing an existing .venv-release. It bootstraps the exact package tools, obtains the CPU wheel from the official PyTorch CPU index (chosen installation source, not a recovered historical URL), then resolves the full version-pinned list from PyPI using binary wheels only. It checks dependency consistency and exact distribution set/version match, including package tools, writes the fresh freeze and a machine-readable report under outputs/validation. No fallback versions or automatic loosening. CI runs the technical and synthetic checks with this installer. If an exact version or wheel cannot be installed, report the blocker instead of silently relaxing the snapshot.

This is an exact version snapshot, not a cryptographic wheel lock: original wheel hashes and index URLs are unavailable. Different compatible wheel builds, hardware, BLAS, interpreter builds and runtime threading may affect results even with equal package versions. Timing is hardware-dependent. Full fresh-install CI validation is pending at the time of this commit. No new field training is authorized or required.
