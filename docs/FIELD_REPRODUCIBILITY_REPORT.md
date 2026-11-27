# Field reproducibility report

## Scope and evidence

The independent local campaign used the scientific source checkout `e3bc30027b6bbcb331a08cc620c397e183ae3106`. The figure metadata correction at `324326d7c0d782cf13f22ec287b4dd3a0acae682` left the nine examined scientific modules unchanged. Field results below are reported by the author from independent local execution and audits; raw local outputs and restricted datasets were not retrieved by the remote audit. Technical and synthetic CI logs were independently inspected. No field data, model weights or raw predictions are included.

## Numerical comparison

| Gate | Tables | Matched | Numeric metrics | Outside tolerance |
|---|---|---:|---:|---:|
| A | 04 | 55 | 63 | 8 |
| B | 05–10, 12 | 215 | 361 | 146 |
| C | 11, 13–21 | 463 | 565 | 102 |
| D | 22–24 | 134 | 135 | 1 |
| Total | 04–24 | 867 | 1124 | 257 |

All 21 experimental tables executed and compared; Tables 14, 22 and 23 passed strict NUMERIC_MATCH. Twenty timing metrics excluded from numeric counts. The comparator and printed-precision tolerance remain unchanged. This count is descriptive, not an independent-sample statistical measure or scientific accuracy score. The campaign does not establish full historical numerical equivalence.

The main cross-well comparison preserved 7/9 mean wins (all six at steps 16/32), 68/90 individual wins and the nine confidence interval zero-exclusion decisions. Noise robustness counts versus interpolation (0,7,9,9) and smoothing (1,5,8,9) match at the four noise levels. The Table 15 step8/rho .20 explicit-mask seed count changed 3 to 2; the manuscript's 17/18 at steps16/32 was preserved. The F03 Ridge vs MLP gain at rho .50 decreased from 19.71895% to 12.88934% while preserving the ordering; uncertainty magnitudes in Tables 6 and 12 also changed substantially in some cells. Do not claim identical gains or uncertainty. Lapa Table 22 matched 105/105 values in full precision against the May 6 run; strict May 10 flag not supported by the public launcher, lineage remains partial. Well 861 retained pooled AE advantage despite individual CLP wins in 5/9 and 6/9 pairs; Table 24 had one rounding-tolerance exception. Main conclusions preserved according to author audits, with these explicit exceptions.

## Targeted repeatability

Cross-well Ridge/AE step32 seed53 rho .20: 50/50 individual RMSEs identical between current-environment runs. F03 MLP seed23 rho .50: 224/224 identical, lambda selected naturally at .0001 in both current runs (historical .0003). These are repeatability checks for two tested settings, not a guarantee across all configurations or platforms. Historical complete environment, observation vectors and some optimizer diagnostics were unavailable. No specific cause of historical differences proved.

## Figure campaign

All 29 figures inventoried: 3 conceptual, 26 numerical regenerated, none blocked. Author classification: 6 NUMERIC_INPUT_MATCH and 20 REGENERATED_WITH_DIFFERENCES; Figure 29 pooled pointwise identity is not established solely by aggregate agreement. Historical profile overlap/selection and qualitative examples recovered. Figure 6 full five-method rankings preserved 9/9; existing profile NPZ unchanged during infrastructure repair. No claim of historical pixel identity or exact numerical equivalence for every plotted curve.

## Environment and release status

Current local environment reported Python 3.10.22, NumPy 1.26.4, pandas 2.3.3, Matplotlib 3.7.5, scikit-learn 1.6.1, PyTorch 2.6.0+cpu, PyWavelets 1.4.1, SciPy 1.15.3, openpyxl 3.1.5, Linux x86_64, CPU, OMP_NUM_THREADS=1, MKL_NUM_THREADS=1. Complete local transitive lock still to be supplied and checked; not the recovered historical environment. Technical CI 46035970271 and synthetic CI 46035970154 passed at infrastructure commit. No new field training required on present evidence. Release merge/tag not yet performed.

Detailed evidence: [Gate A](FIELD_GATE_A_AUDIT.md), [Gate B](FIELD_GATE_B_AUDIT.md), [Gate C](FIELD_GATE_C_AUDIT.md), [Gate D](FIELD_GATE_D_AUDIT.md), [Gate E](FIELD_GATE_E_AUDIT.md), [machine-readable manifest](SOURCE_MANIFEST.json).
