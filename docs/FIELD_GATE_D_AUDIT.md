# Gate D field reproduction — laboratory-related benchmarks

Author-reported local execution at scientific commit e3bc30027b6bbcb331a08cc620c397e183ae3106. No raw local artifacts, Canvas or historical configurations independently retrieved by the remote audit. No data published.

| Table | Matched | Total | Status |
|---|---:|---:|---|
| 22 | 105 | 105 | NUMERIC_MATCH |
| 23 | 12 | 12 | NUMERIC_MATCH |
| 24 | 17 | 18 | Partial |
| Total | 134 | 135 | Partial |

Lapa: all 105 scientific RMSE, MAE and relative L2 values reportedly match the historical May 6 run at full precision. Both CLP prior classes beat direct AE at all five ratios. May 10 strict_b_20260510 differs by paper_strict_paired_b=True, a flag not exposed by the public launcher; the historical summaries are reported identical for Table 22, with differences confined to CI95 fields outside that table. Strict lineage remains PARTIAL, not a certified replay of that flag. Retain exact recovered historical command, effective configs and workbook identity locally; supply authorized metadata for final documentation. Do not add an unverified flag or claim exact equivalence of observation vectors from summary agreement alone.

Well 861: author confirmed enriched workbook SHA-256 4c275a8dcf49042ea9db6f3abb76043a6b2ed622ae226391ffc1c56f7b7a4d7b, 87 rows, three blocks, four auxiliary channels, L=8, step=1, seeds 7/23/41, rho .25/.50, zero added noise. CLP beats AE in 5/9 and 6/9 pairs but AE wins pooled means; CLP wins blocks fold_id 0 and 1, AE fold_id 2. Score remains within-window held-out coordinates. Table 24 only deviation: fold_id 2 (third depth block), rho .25, AE reference .0691 vs observed .0690474994, difference -0.0000525006. Slightly outside unchanged rounding tolerance. Rankings reportedly preserved. No new training warranted for this discrepancy.

All 21 tables 04–24 have been executed and compared. 867/1124 numeric metrics matched strict tolerance, 257 outside; 20 timing metrics informational. Three exact tables (14,22,23). These counts are aggregate comparison outcomes, not independent statistical trials or a scientific accuracy score. Rankings and main conclusions reportedly preserved, with explicit individual-count, gain and uncertainty exceptions in prior audits. Proceed to figure provenance and regeneration, keeping strict tolerances and partial numerical status.
