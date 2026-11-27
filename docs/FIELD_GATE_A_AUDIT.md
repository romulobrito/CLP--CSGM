# Gate A field audit and targeted repeatability (26 November 2026)

Status: PARTIAL_NUMERIC_MATCH_CONCLUSIONS_PRESERVED; REPEATABILITY_CONFIRMED_FOR_TESTED_CELL.

## Evidence and scope

The author reported a fresh independent local checkout of the public preparation repository at e3bc30027b6bbcb331a08cc620c397e183ae3106, clean tracked files, no historical outputs, weights or predictions copied, and no commits or pushes from the local run. The reports, aggregate values and hashes below are author-reported evidence; the raw local files have not been independently retrieved or rehashed by the remote audit. No restricted data or raw predictions are included here.

The remote Git audit compared the nine scientific modules in CS-ML historical commit 53190576cd8321d927682c86881927b9d51aacb1, source checkpoint c4df022497f6b4a3438dd8372335df58178cfe89, and public execution commit e3bc30027b6bbcb331a08cc620c397e183ae3106: all nine Git blobs identical. Launcher/aggregator changes were paths/comments only. The local audit confirmed identical cross-well inputs and effective configurations. Historical package versions, thread settings, BLAS and actual observation/noise vectors were not recovered.

Environment reported for independent execution: Python 3.10.22, NumPy 1.26.4, pandas 2.3.3, Matplotlib 3.7.5, scikit-learn 1.6.1, PyTorch 2.6.0+cpu, PyWavelets 1.4.1, SciPy 1.15.3, openpyxl 3.1.5, CPU, OMP_NUM_THREADS=1, MKL_NUM_THREADS=1, Linux x86_64. This is not the verified historical environment.

## Primary Table 4

All nine step/rho cells found; 55/63 metrics within the existing printed-precision comparator tolerance, eight outside. Largest absolute difference to printed reference 1.28594e-5. Comparator remains INCOMPLETE_OR_DIFFERENT. Seven of nine mean rankings favor CLP, all six at steps 16/32; individual wins 68/90; all nine confidence intervals exclude zero, identical to the published table. No reference or tolerance modified.

The local audit identified step 32, rho .20, seed 53 as the dominant deviation in that cell (84.688% of summed absolute seed discrepancies), concentrated in two windows. Selected lambda 0.0001 in all ten seeds in that cell in both historical and reproduced runs. The exact cause of historical differences is unresolved. Historical omega and observed b not available for direct verification.

## Targeted current-environment repeatability

One new run of seed 53, step 32, rho .20 in the same current environment. Author reports 50/50 individual RMSE values identical (25 windows for each of CLP and AE), max absolute RMSE difference zero, identical reported scientific fields including RMSE, MAE, relative L2, lambda, validation score, generator AE training reconstruction RMSE, measurement seed and omega order. Canonical scientific SHA-256 d96267d92561cb992fdbf7c1a6332fc0cb6b215f0ef85c867a7a8140aff1400d. Raw CSVs differ in scope and metadata; no claim of raw-file byte identity. The result establishes repeatability for this tested setting only, not every seed or platform.

Decision: close the targeted diagnosis with explicit unresolved historical environment/observation limitations, retain the original strict comparator and partial Gate A status, and proceed to Gate B. No claim of exact field reproduction for Table 4 or any untested table. No scientific source changes.
