# RELEASE-05d.2c — local field reproduction

Status: Gate A independently executed with partial numerical agreement and preserved scientific conclusions; targeted repeatability confirmed. See FIELD_GATE_A_AUDIT.md. Gate B completed with partial numerical agreement and targeted MLP repeatability confirmed (see FIELD_GATE_B_AUDIT.md). Gate C completed with partial numeric agreement (Table 14 exact; see FIELD_GATE_C_AUDIT.md). Gate D is next; Gates D–E remain untested. The code and 21 adapters passed artificial fixtures at b4f05405a30566a33f650288d9345b8876725ee1.

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

B: three-seed F3 benchmark, ablations, prior classes and F03 porosity. Gate A diagnostic reviewed; progression permitted with its partial numerical status unchanged. Use the isolated commands below rather than rerunning the full driver.

C: supplemental GR/embargo, development splits, explicit mask, observation placement, reference simple baselines, noise sweep, seven-candidate smooth audit, S-REC and runtime, then F03 stress tests. Respect upstream dependencies and explicitly select the stress source CSV. Timing is informational.

D: Lapa and well 861. Recover exact Lapa run config and input provenance from the local historical runs before executing Table 22. The enriched 861 workbook is not interchangeable with the general Lapa workbook; use the published within-window withheld score. Check run_r1m11_861_phi_lab.py and aggregate_861_tables.py with the local workbook. Compare Tables 23 and 24.

E: qualitative figures and all remaining protocol details, including historical profile overlap selection. No image or profile reproduction claim without checking the selected case.

## Gate B: main three-seed benchmarks (Tables 05–10, 12)

Use the same clean, detached checkout and validated environment as Gate A, retaining its outputs. The commands below are the main driver's benchmark commands without Table 4 or figure regeneration. Before launching, confirm that all eight target run directories and the two ablation aggregate output families are absent; if not, stop and preserve existing evidence or use a separate checkout. Do not import historical outputs.

```bash
set -euo pipefail
export CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export MPLBACKEND=Agg
TRAIN="data/F02-1,F03-2,F06-1_6logs_30dB.txt"
TEST="data/F03-4_6logs_30dB.txt"
PORO="data/F03-4_AC+GR+Porosity.txt"
for step in 8 16 32; do
  python sir_cs_benchmark_multi_well_vc.py \
    --train-path "$TRAIN" --test-path "$TEST" --channels sonic,rhob,ai,vp \
    --target vc --step "$step" --rhos 0.05,0.10,0.20 --seeds 7,23,41 \
    --val-frac 0.1 --val-split-mode tail --run-csgm-m2 --run-csgm-ablations \
    --csgm-prior-type ridge --no-lfista \
    --run-id "crosswell_step$(printf '%02d' "$step")_clp_csgm_ablation_ridge"
  python sir_cs_benchmark_multi_well_vc.py \
    --train-path "$TRAIN" --test-path "$TEST" --channels sonic,rhob,ai,vp \
    --target vc --step "$step" --rhos 0.05,0.10,0.20 --seeds 7,23,41 \
    --val-frac 0.1 --val-split-mode tail --run-csgm-m2 \
    --csgm-prior-type mlp --no-lfista \
    --base-dir outputs/reviewer_revision/r1m06_prior_class_controlled/crosswell \
    --run-id "step$(printf '%02d' "$step")_mlp"
done
python sir_cs_benchmark_real_well_direct_ub.py \
  --data-path "$PORO" --u-channels gr --rhos 0.20,0.30,0.40,0.50,0.60 \
  --seeds 7,23,41 --run-csgm-m2 --run-csgm-ablations \
  --csgm-prior-type ridge --no-lfista --run-id f03_full_gr_only_clp_csgm_ridge
python sir_cs_benchmark_real_well_direct_ub.py \
  --data-path "$PORO" --u-channels gr --rhos 0.20,0.30,0.40,0.50,0.60 \
  --seeds 7,23,41 --run-csgm-m2 --csgm-prior-type mlp --no-lfista \
  --base-dir outputs/reviewer_revision/r1m06_prior_class_controlled/f03_gr_only \
  --run-id mlp_depth_first
python scripts/clp_csgm_ablation_assets.py
python scripts/clp_csgm_r1m06_controlled_assets.py
python scripts/normalize_manuscript_results.py --tables 05 06 07 08 09 10 12 \
  --out-dir outputs/observed_gate_b
python scripts/compare_manuscript_results.py --tables 05 06 07 08 09 10 12 \
  --observed-dir outputs/observed_gate_b --report outputs/validation/gate_b_comparison.json
```

Check actual configs before interpretation: F3 no-GR, window length 64, steps 8/16/32, three seeds, rho .05/.10/.20, noise .01; F03 GR-only, window 64, step 1, depth-first .6/.2/.2 split, rho .20–.60, noise .02. Check matching seed identities, observation masks and methods within each comparison. Confirm the controlled Ridge/MLP settings and ablation definitions. Return per-table comparison status, metric counts, max absolute/relative differences, and whether the direction of each scientific comparison is preserved. Do not change tolerances or rerun failed cells to obtain agreement. If any producer or normalizer fails, record the actual error; do not fabricate a result. Figure regeneration and Gates C–E are outside this gate. Preserve all evidence locally.

## Gate C: complementary diagnostics and robustness (Tables 11, 13–21)

Use the exact scientific checkout e3bc30027b6bbcb331a08cc620c397e183ae3106 and existing authorized inputs/environment. Preserve Gates A/B and both repeats. Read EXPERIMENTS.md, RESULT_NORMALIZATION.md and HISTORICAL_PROVENANCE.md before starting. Check every target output root is absent (including the stress bundle), never import historical result directories. Run each producer separately, recording exit codes and logs; if a dependency fails, stop that dependent chain and record NOT_RUN. Do not change scientific settings, source or tolerance to obtain agreement. These commands are based on inspected entry points and have not yet been run on field inputs in this gate.

```bash
export CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export MPLBACKEND=Agg
# C1: protocol sensitivity and observation placement
python scripts/run_b7_gr_embargo.py --dry-run
python scripts/run_b7_gr_embargo.py
bash scripts/run_r1m03_step32_dev_splits.sh
bash scripts/run_r1m08_explicit_mask.sh
bash scripts/run_r2m04_omega_placement.sh
# C2: dependent simple baselines and noise tests
python scripts/run_r3m03_simple_baselines.py
python scripts/run_r3m03_noise_sweep.py
python scripts/run_r3m03_smooth_audit.py
# C3: empirical separation, runtime and stress
python scripts/clp_csgm_runtime_study.py
python scripts/clp_csgm_srec_diagnostics.py
python scripts/f03_stress_benchmark_bundle.py --dry-run --bundle-root outputs/real_well_f03/gate_c_stress_dry_run
python scripts/f03_stress_benchmark_bundle.py --bundle-root outputs/real_well_f03/gate_c_stress
# Normalize fresh producer results and compare, retaining nonzero comparison exit status
python scripts/normalize_manuscript_results.py --tables 11 13 14 15 16 17 18 19 20 21 \
  --source 21=outputs/real_well_f03/gate_c_stress/comparison_summary_by_scenario.csv \
  --out-dir outputs/observed_gate_c
python scripts/compare_manuscript_results.py --tables 11 13 14 15 16 17 18 19 20 21 \
  --observed-dir outputs/observed_gate_c --report outputs/validation/gate_c_comparison.json
```

Confirm B7 four GR/embargo cases; ten dev split seeds with fixed model seed 7; explicit-mask comparator; five offsets 0/10007/20011/30013/40009 and three model seeds; offset-zero dependency for simple baselines; noise sweep fixed draws and seven lambda values 1e-4 through 100; smoothing seven alpha values 1e-6,1e-4,1e-2,1,1e2,1e4,1e6 (not extended audit). Runtime/S-REC use their separate historical GR-inclusive cross-well diagnostic (step 16, noise .02 for runtime), not the no-GR main benchmark. Timing is informational; report hardware and thread setting. S-REC is sampled empirical separation, not a global proof. F03 stress cases A–D should be verified against Table 21 configuration, including split, noise and step. Check actual producer schema and all output paths against normalizer; if mismatches occur return the exact error and producer fields. Do not silently replace inputs or select another run by numeric agreement.

Return all ten table statuses, matched/total metrics, signed differences by metric, max absolute and relative discrepancies with units, protocol/config and mask coverage, raw ranking conclusions (not solely aggregator assertions), and the noise-level win counts of CLP versus interpolation and smoothing. Record exceptions including missing diagnostics. If a table is missing, classify NOT_RUN instead of inventing values. Do not advance to Gate D/E or run extra figure scripts. Table 20 timing must not be counted as evidence of numerical reproducibility of historical wall-clock performance.

## Gate D: supplementary laboratory-related benchmarks (Tables 22–24)

Same scientific checkout and environment as Gates A–C. Preserve outputs, use new empty producer roots, and verify the actual authorized input hashes. The enriched 861 workbook is available locally according to Gate A (SHA-256 4c275a8dcf49042ea9db6f3abb76043a6b2ed622ae226391ffc1c56f7b7a4d7b). No raw field data or predictions are to be transferred.

D1, well 861 (Tables 23–24): run with the original enriched workbook, not a substitute. Confirm 87 rows, 29 per depth block, source target Phi_lab (pu), four auxiliary columns GR (API), Res_Deep, Res_Shallow, Lithotype, window 8, step 1, 16 validation rows, seeds 7/23/41, rho .25/.50, noise 0. The code evaluates withheld coordinates per window, not globally unseen physical depths. Run the index diagnostic separately; it must not be interpreted as the field performance.

```bash
export CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export MPLBACKEND=Agg
python scripts/audit_861_depth_indices.py --row-count 87
python scripts/run_r1m11_861_phi_lab.py --excel-path data/861_integrated_logs_enriched.xlsx
python scripts/aggregate_861_tables.py
python scripts/normalize_manuscript_results.py --tables 23 24 --out-dir outputs/observed_gate_d_861
python scripts/compare_manuscript_results.py --tables 23 24 --observed-dir outputs/observed_gate_d_861 --report outputs/validation/gate_d_861_comparison.json
```

D2, Lapa (Table 22): BEFORE training inspect both original local historical runs strict_b_20260510 and auddys_favorable_u7_step4_ridge_mlp_csgm_20260506. Locate their config.json, manifests and summary.csv; confirm both against the historical Table 22 full-precision source if present and the printed reference. Record exact ordered seven channels, workbook/sheet and hash (current Auddys_table.xlsx SHA-256 f82c3b6124ba8bb44878063d1a194c1d042b5cd0944d69a5bef66a293906c71f), log10 transform, split mode and fractions, counts, noise, measurement kind, both prior classes and their order, all CSGM and direct AE settings. The current launcher uses contiguous window splitting, direct AE epochs=80 and deduplicates common method rows keeping the first prior run. Verify those settings against historical evidence; do not substitute F03 depth-first split or defaults just because they are available. The historical run IDs are not numerically unique; if both configurations are equivalent and match published data, document lineage ambiguity rather than asserting a unique source. If any critical parameter cannot be recovered or current launcher cannot express the historical configuration, classify Table 22 NOT_RUN and report the blocker. Do not invent an exact command.

After preflight, use scripts/auddys_smoke_direct_ub.py with explicitly recovered options, new base directory outputs/reviewer_revision/gate_d_lapa and run ID lapa_replay. Include both prior types in recovered order and required ablations if historical config confirms. Verify complete method/seed/rho grid and actual config. Normalize and compare the fresh result only:

```bash
python scripts/normalize_manuscript_results.py --tables 22 \
  --source 22=outputs/reviewer_revision/gate_d_lapa/runs/lapa_replay/tables/summary.csv \
  --out-dir outputs/observed_gate_d_lapa
python scripts/compare_manuscript_results.py --tables 22 --observed-dir outputs/observed_gate_d_lapa --report outputs/validation/gate_d_lapa_comparison.json
```

Report D1/D2 separately: counts, exact/partial numeric status, signed discrepancies, method ranks, CLP vs AE pooled and per-block results, strictly positive paired wins, measurement-only comparator and whether manuscript conclusions are retained. No tuning against references. Do not rerun other gates or advance to figure audit. Complete remaining Gate C read-only checks in FIELD_GATE_C_AUDIT.md using existing outputs only.

## Return for audit

Return only authorized aggregate metrics (including normalized table04.csv), comparison report, source/destination hashes, config metadata, seed coverage, command/exit code, environment record and a list of deviations. Do not return raw logs containing field values, raw input files, depth curves, weights, predictions or restricted data. If disclosure of metadata or aggregates is restricted, return a redacted summary. No automatic merge or release tag. This runbook is a staged plan, not an executed field test.
