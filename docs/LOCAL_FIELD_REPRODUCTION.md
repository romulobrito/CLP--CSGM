# RELEASE-05d.2c — local field reproduction

Status: NOT_RUN_AWAITING_AUTHORIZED_LOCAL_INPUTS. The code and 21 adapters passed artificial fixtures at b4f05405a30566a33f650288d9345b8876725ee1. Neither that result nor the historical aggregate tables establishes independent field reproduction.

## Inputs and isolation

Create a fresh local checkout of the public preparation branch, separate from the original CS-ML worktree. Record the full checkout commit, interpreter, platform, CPU/GPU, package versions, and input SHA-256 hashes locally. Do not copy historical outputs, model weights, or predictions into the fresh checkout. Supply authorized inputs at their canonical names under data/ (or local symlinks if permitted), because some legacy utilities use these paths. Verify git status and git check-ignore before execution. Never force-add restricted data or publish raw/derived arrays.

Required roles:
- data/F02-1,F03-2,F06-1_6logs_30dB.txt
- data/F03-4_6logs_30dB.txt
- data/F03-4_AC+GR+Porosity.txt
- data/Auddys_table.xlsx (Lapa, Logs sheet)
- data/861_integrated_logs_enriched.xlsx (distinct enriched well-861 input, 87 rows; verify against source Logs sheet)

Use the historical authorized processed inputs, not merely files with matching names. Where an original hash is unavailable, report that limitation.

## Gate A: primary numerical comparison (Table 4)

Use the verified CPU environment: Python 3.10.22, NumPy 1.26.4, pandas 2.3.3, Matplotlib 3.7.5, scikit-learn 1.6.1, PyTorch 2.6.0+cpu, PyWavelets 1.4.1, SciPy 1.15.3, openpyxl 3.1.5. Record pip freeze and pip check. This is the CI-validated set, not a claim about the exact historical hardware or a complete transitive lock.

```bash
export CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export MPLBACKEND=Agg
python -m pip check
python scripts/check_release.py
bash scripts/run_synthetic_smoke.sh
SEEDS=7,23,41,13,29,31,37,43,47,53 bash scripts/run_r1m04_seed_panel.sh
python scripts/normalize_manuscript_results.py --tables 04 --out-dir outputs/observed_gate_a
python scripts/compare_manuscript_results.py --tables 04 --observed-dir outputs/observed_gate_a --report outputs/validation/gate_a_comparison.json
```

The primary driver already aggregates Table 4. Check the three actual run configs and split metadata against the protocol, including four channels in order sonic,rhob,ai,vp (no GR), target Vc, window length 64, steps 8/16/32, ratios .05/.10/.20, minimum four observations, noise .01, ten paired seeds, depth-tail validation .1, development/test well separation, 104/11/97,52/6/49,27/3/25 train/val/test windows. Check seed-level identities and actual observation sets before interpreting the comparison. Report failures without changing tolerances, training settings or references. If Table 4 does not match, stop the expansion and diagnose environment, inputs, protocol and code differences. Preserve all new evidence locally.

## Subsequent gates (only after Gate A review)

B: three-seed F3 benchmark, ablations, prior classes and F03 porosity. Follow EXPERIMENTS.md; the full main driver reruns the primary panel and invokes a dedicated figure regeneration, so avoid accidentally replacing Gate A evidence. Use a separate fresh checkout or approved output isolation for this next gate.

C: supplemental GR/embargo, development splits, explicit mask, observation placement, reference simple baselines, noise sweep, seven-candidate smooth audit, S-REC and runtime, then F03 stress tests. Respect upstream dependencies and explicitly select the stress source CSV. Timing is informational.

D: Lapa and well 861. Recover exact Lapa run config and input provenance from the local historical runs before executing Table 22. The enriched 861 workbook is not interchangeable with the general Lapa workbook; use the published within-window withheld score. Check run_r1m11_861_phi_lab.py and aggregate_861_tables.py with the local workbook. Compare Tables 23 and 24.

E: qualitative figures and all remaining protocol details, including historical profile overlap selection. No image or profile reproduction claim without checking the selected case.

## Return for audit

Return only authorized aggregate metrics (including normalized table04.csv), comparison report, source/destination hashes, config metadata, seed coverage, command/exit code, environment record and a list of deviations. Do not return raw logs containing field values, raw input files, depth curves, weights, predictions or restricted data. If disclosure of metadata or aggregates is restricted, return a redacted summary. No automatic merge or release tag. This runbook is a staged plan, not an executed field test.
