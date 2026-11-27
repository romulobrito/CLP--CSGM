# Gate E — figure audit and metadata compatibility correction

Author-provided local audit at e3bc30027b6bbcb331a08cc620c397e183ae3106: 29 figures, 3 conceptual (1,2,5), 6 NUMERIC_INPUT_MATCH (3,4,26–29), 17 REGENERATED_WITH_DIFFERENCES (7,9–23,25), 3 BLOCKED (6,8,24). Classifications are reported by the author; for Fig. 29 pooled parity, identical Table 22 aggregates alone do not establish pointwise parity identity. No historical PNG byte identity inferred from matching numerical inputs.

Historical selections confirmed: Figs 8/24 step16 seed7 rho .20 uniform_mean, Fig10 F03 seed7 rho .20 step1 uniform_mean, Figs14–17 historical examples/order/pooling. Cross-well profile NPZ hash c4cb794a3665196e07281d5465c0a3eafe82efd3c0a8db22948560e8d8215ccf; 832 exact depth points and exact observed profile; predicted profile differences RMSE CLP 1.04100855143e-5, AE 2.54989601658e-9. Fig10 observed and depth exact; CLP difference RMSE 2.6248902385e-5, AE 7.12745945456e-4. Raw local files were not independently accessed remotely.

Lapa recovered: density,gr,res_deep,res_shallow,phi_neutron,phi_sonic,phi_nd; Auddys_table.xlsx Logs sheet SHA-256 f82c3b6124ba8bb44878063d1a194c1d042b5cd0944d69a5bef66a293906c71f; log10 k_lab, L16 step4 contiguous window split .6/.2/.2, counts 10/3/5, seeds 7/23/41, rho .20–.60, noise .01, Ridge/MLP. May 6 replay numerically matches Table 22. Strict May 10 flag not exposed, unique lineage PARTIAL.

## Root cause and correction

Cross-well launcher assigns cfg.measurement_seed_offset dynamically, records it in detailed_results.csv and omega_sets.csv, then serializes dataclass fields with asdict(cfg). The missing JSON field is not evidence of a zero offset. The corrected builder requires complete seed/rho coverage and zero offset in BOTH existing records, validates recorded ordered Omega against the actual generator rule, rejects any conflicting explicit config, and records hashes of measurement evidence and per-seed summary in schema 2. No original config edits, no new training, no change to the launcher or scientific algorithms. All other checks, including plotter tolerance 5e-5, remain unchanged. Regression tests and CI validation pending.

Local continuation: use the corrected builder and plotter with the existing three Gate B runs and dedicated profile, not run_regenerate_figure_inputs.sh. Preserve prior files and hashes. Run python scripts/test_figure_measurement_evidence.py, then python scripts/build_regenerated_figure_inputs.py and python scripts/plot_r3m06_manuscript_figures.py. If successful, retry the two paper/uncertainty exporters after preserving outputs. If another check fails, report exact failure; do not weaken checks. Verify all five method means/uncertainties and profile hashes, report final classifications and SHA-256. No raw restricted data, no retraining, no merge/tag.

## Gate E.1 — completed and validated

Author-provided local report confirms infrastructure commit 324326d7c0d782cf13f22ec287b4dd3a0acae682, scientific checkout e3bc30027b6bbcb331a08cc620c397e183ae3106, nine scientific blobs identical, regression test passed, complete zero-offset evidence for three steps and profile, schema 2 provenance SHA-256 0000da5ea02966587cab4fce83998c7a5181dec6dcc849b01e86e511b91034db. Existing NPZ hash unchanged. No new training or tolerance change. Builder, plotter and both formerly blocked exporters exited zero.

| Figure | Generated SHA-256 | Status |
|---|---|---|
| 6 | 6fa831ddc375892da2d2b9c46452f5d2845b98a1efb68a9c3ff403661b26b3e6 | REGENERATED_WITH_DIFFERENCES |
| 8 | 9cdc722bf10ca743d2ad3b16d96486981c2dece523f5e4199cb9b6ce2975a5bf | REGENERATED_WITH_DIFFERENCES |
| 24 | d7c86368b409db0b89fad4abf855839075644aaa948cfdd0dd245c1a2c82b68d | REGENERATED_WITH_DIFFERENCES |

Final reported inventory: 3 conceptual, 6 NUMERIC_INPUT_MATCH, 20 REGENERATED_WITH_DIFFERENCES, zero blocked. Fig. 6 full ranking and winner preserved 9/9; ML-only and direct MLP means exact; PCA max absolute mean difference 8.33e-17; AE 1.90613e-5 (0.0733%); CLP 1.20506e-5 (0.0351%). Profile ML-only/direct MLP/PCA difference RMSE near machine precision; CLP and AE differences as recorded above. Figure 29 classification is author-reported; aggregate equality does not by itself verify pooled pointwise parity.

Remote CI independently checked: [technical run 46035970271](https://github.com/romulobrito/CLP--CSGM/actions/runs/46035970271), 112 static and 172 total technical checks, 124 adapter fixture checks, figure measurement regression passed; [synthetic run 46035970154](https://github.com/romulobrito/CLP--CSGM/actions/runs/46035970154), 2 benchmarks and 28 required method cells. Both success, pip check passed. Field files were not independently accessed remotely. Gate E closed; no new training required. Full environment lock pending.
