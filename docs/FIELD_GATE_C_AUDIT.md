# Gate C field reproduction — partial numerical match

Author-reported local results at scientific commit e3bc30027b6bbcb331a08cc620c397e183ae3106. All producers exited zero, all ten tables normalized, no NOT_RUN; strict comparison exited 1. Local report outputs/validation/gate_c_comparison.json and normalized outputs/observed_gate_c. Raw report, hashes and complete per-seed records were not independently retrieved remotely.

| Table | Matched | Numeric metrics | Reported result |
|---|---:|---:|---|
| 11 | 18 | 30 | Largest signed difference -0.0195303 (-3.0468%) |
| 13 | 57 | 72 | Largest difference -0.438298 percentage points; rankings 12/12 preserved |
| 14 | 27 | 27 | NUMERIC_MATCH |
| 15 | 42 | 63 | Mean signs 18/18 preserved; one explicit-mask seed-win count changed |
| 16 | 46 | 54 | Winners 9/9 preserved |
| 17 | 39 | 45 | Rankings 9/9 preserved |
| 18 | 104 | 108 | Signs 36/36 preserved |
| 19 | 104 | 108 | Rankings 36/36 preserved |
| 20 | 8 | 10 | RMSE ordering preserved; 20 timing metrics informational |
| 21 | 18 | 48 | Gain signs 12/12 preserved |
| Total | 463 | 565 | 102 outside tolerance; timing excluded |

## Noise sensitivity

CLP mean RMSE below interpolation in 0/9, 7/9, 9/9, 9/9 cells at noise .01, .05, .10, .20; below smoothing in 1/9, 5/9, 8/9, 9/9 cells, respectively. All eight counts match manuscript. These are cell-level results, not independent proof of seed-level claims. Appendix B.12 additionally reports all three seeds at noise .10 and .20 versus interpolation; verify the actual per-seed counts from existing outputs.

## Explicit mask sensitivity

Table 15 step 8 rho .20 clp_wins_mask changed 3 -> 2. Reported 18/18 mean-difference signs preserved. Manuscript Appendix B.10 separately states 7/9 mean wins versus mask-aware AE, all six mean wins at steps 16/32, and 17/18 paired wins at steps 16/32. The reported changed count is outside those latter steps; verify the 17/18 and all other counts from existing raw results, not solely successful aggregation. No new training warranted on evidence so far. Do not claim all seed-level win counts match.

## Decision

Retain the strict comparator and partial status. Gates A–C have 18/21 tables executed; one of these tables has full numerical agreement. Record outstanding read-only checks (full config/mask audit, complete report SHA-256, signed discrepancy context, high-noise seed wins and Table 15 win counts). Proceed to Gate D independently of those read-only checks; Lapa must pass its historical configuration preflight. No causal attribution of remaining numeric discrepancies, no claim of complete reproduction, no changes to scientific source.
