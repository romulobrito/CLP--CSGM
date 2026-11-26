# Result dependency map (RELEASE-03b)

The historical checkpoint contains **five-channel** configurations (GR included, p_input=320) at the legacy Ridge ablation run paths. The historical R3-M06 provenance explicitly distinguishes those from four-channel local summaries (p_input=256). Never copy or silently reuse the legacy m2_grid outputs. The regenerated bundle is explicitly labeled local_regeneration_not_historical.

| Manuscript evidence | Producer | Consumer |
| --- | --- | --- |
| Table 4 / B.9 (ten paired seeds) | run_r1m04_seed_panel.sh | clp_csgm_r1m04_seed_panel_assets.py |
| Fig. 6 (three seeds, five methods) | run_reproduce_clp_csgm.sh: three Ridge ablation runs | build_regenerated_figure_inputs.py; plot_r3m06_manuscript_figures.py |
| Figs. 8, 24 (step 16, seed 7, rho 0.20) | run_regenerate_figure_inputs.sh: dedicated single-cell run | depth_profile_test_block.npz; plot_r3m06_manuscript_figures.py |
| Tables 7–10 | main driver: separate Ridge/MLP runs | ablation and controlled-prior aggregators |
| Appendix B.8/B.10/B.11 | corresponding four revision launchers | paired aggregators |
| Appendix B.12 | omega placement offset-zero runs; simple baselines; noise sweep; seven-candidate smoothing and affine audit | respective plotting scripts |
| Appendix B.7, Table 13 / Fig. 20 | run_b7_gr_embargo.py | local table13.csv and figure20.png |
| Appendix E depth-index overlap | audit_861_depth_indices.py | local index-only diagnostic (no laboratory values) |

From the repository root, with authorized inputs, run the main driver. It invokes the figure-input regeneration script. To regenerate only the profile and figures after the three canonical Ridge runs exist, run `bash scripts/run_regenerate_figure_inputs.sh`. The builder checks run configuration, exact channels, wells, counts, seeds, ratios and selected CLP parameters; it verifies summary cell coverage and records local file and configuration hashes. The final plotter retains its published CLP mean check (absolute tolerance 5e-5). The historical mode remains explicit; it requires the original local provenance, not shipped with the package.

The legacy paper/uncertainty utilities now consume checked three-seed summaries, with distinct filenames for the older cross-well figures and paired tables. They can be invoked separately from the final figure driver. Their outputs must not replace the ten-seed Table 4. The raw profile and all generated results remain ignored local artifacts. No historical field data or result files are copied into Git.

Status: implementation and static inspection only. Actual execution, complete appendix coverage, input dataset equivalence, full per-method numerical comparison, exact environment lock and historical profile equivalence remain unverified. The profile reconstruction uses uniform mean overlap, as the launcher default; historical dedicated run config was not available in the checkpoint. Confirm this against the author's local run before claiming exact reproduction.
