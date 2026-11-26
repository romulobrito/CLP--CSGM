# Experiment map

All commands are run from the repository root after installing dependencies and supplying authorized data. This is an implementation map, not evidence of completed reproduction. Full numerical verification is pending.

## Main results

| Evidence | Protocol | Producer / consumer |
| --- | --- | --- |
| Table 4; Appendix B.9 | Cross-well, ten paired seeds | run_r1m04_seed_panel.sh; clp_csgm_r1m04_seed_panel_assets.py |
| Figure 6 | Five methods, three seeds | run_reproduce_clp_csgm.sh; build_regenerated_figure_inputs.py; plot_r3m06_manuscript_figures.py |
| Figures 8 and 24 | Step 16, seed 7, rho 0.20 | run_regenerate_figure_inputs.sh |
| Tables 5–6, 12; B.2–B.3 | F03-4 GR-only | sir_cs_benchmark_real_well_direct_ub.py; clp_csgm_paper_assets.py; clp_csgm_uncertainty_assets.py |
| Tables 7–8 | Structural ablations, three seeds | main driver; clp_csgm_ablation_assets.py |
| Tables 9–10 | Controlled Ridge/MLP latent prior | main driver; clp_csgm_r1m06_controlled_assets.py |

Cross-well: L=64, steps 8/16/32, nominal rho=0.05/0.10/0.20, at least four observations per window, validation fraction 0.1 per development well (tail). Train/val/test window counts by step: 104/11/97, 52/6/49, 27/3/25. Reference measurement noise std=0.01. F03-4 porosity: L=64, step 1, ratios 0.20/0.30/0.40/0.50/0.60, depth-first split 0.60/0.20/0.20, reference measurement noise std=0.02. Three-seed experiments use 7,23,41; Table 4 also uses 13,29,31,37,43,47,53. The common refinement grid is 0.0001,0.0003,0.001,0.003,0.01,0.03,0.1 unless the experiment explicitly defines a different protocol.

```bash
bash scripts/run_reproduce_clp_csgm.sh
# Additional paper tables, EDA and seed-level uncertainty from local runs:
python scripts/clp_csgm_paper_assets.py
python scripts/clp_csgm_uncertainty_assets.py
```

The legacy three-seed cross-well paired tables and figures carry distinct filenames. They are not Table 4. The main figure script requires the regenerated local bundle and checks published CLP means; historical provenance mode is separate. See RESULT_DEPENDENCIES.md.

## Supplementary diagnostics and sensitivity

| Section | Entry point | Status / dependency |
| --- | --- | --- |
| B.1, Table 11 | python scripts/clp_csgm_srec_diagnostics.py | Sampled separation, not a proof of global S-REC |
| B.4 | scripts/paper_sparse_reconstruction_figures.py --mode crosswell or f03 --skip-lfista | Qualitative windows; confirm selected example against paper |
| B.5 | F03-4 launcher and local profile outputs | Exact figure selection pending |
| B.6 | python scripts/clp_csgm_diagnostic_assets.py | Requires main benchmark runs |
| B.7, Table 13 / Fig. 20 | python scripts/run_b7_gr_embargo.py | Four controlled step-16 scenarios; --dry-run or --aggregate-only available |
| B.8, Table 14 | bash scripts/run_r1m03_step32_dev_splits.sh | Ten split seeds, fixed model/measurement seed 7 |
| B.10, Table 15 | bash scripts/run_r1m08_explicit_mask.sh | Three paired seeds |
| B.11, Table 16 | bash scripts/run_r2m04_omega_placement.sh | Five observation/noise draws, model seeds fixed |
| B.12, Tables 17–19 | See sequence below | Separate reference-noise and noise-sensitivity experiments |
| B.13 | bash scripts/run_regenerate_figure_inputs.sh | Dedicated profile; historical overlap mode still unverified |
| B.14, Table 20 | python scripts/clp_csgm_runtime_study.py | Hardware-dependent timing |
| C, Table 21 | python scripts/f03_stress_benchmark_bundle.py | Inspect --dry-run first |
| D, Table 22 | scripts/auddys_smoke_direct_ub.py | Authorized Lapa data and exact historical config needed |
| E, Tables 23–24 | python scripts/run_r1m11_861_phi_lab.py --excel-path /path/to/authorized.xlsx | Authorized 87-row workbook; within-window withheld score |

For B.7, retain L=64, step 16, rho grid 0.05,0.10,0.20, seeds 7,23,41, tail validation 0.1, noise 0.01, Ridge and the same train/test wells. Run the cross-well launcher four times, with channels sonic,rhob,ai,vp or sonic,rhob,gr,ai,vp and embargo 0 or 4. Use distinct run IDs and a separate base directory; compare with Table 13. The new B.7 driver generates table13.csv and figure20.png; numerical agreement is pending.

B.12 ordered local workflow (do not mix earlier saved runs):

```bash
bash scripts/run_r2m04_omega_placement.sh
python scripts/run_r3m03_simple_baselines.py
python scripts/run_r3m03_noise_sweep.py
python scripts/run_r3m03_smooth_audit.py
python scripts/run_r3m03_affine_limit_audit.py
python scripts/plot_r3m03_depth_profiles.py
python scripts/plot_r3m03_noise_panels.py
python scripts/plot_r3m03_rmse_vs_noise.py
```

The simple baseline script requires the offset-zero reference runs. The smooth and affine audit stages require their upstream detail tables. The noise sweep also has a separate --methods interp,smooth --tag option; this does not rerun CLP. Table 19 and Fig. 21 use run_r3m03_smooth_audit.py with seven alpha candidates (1e-6, 1e-4, 1e-2, 1, 1e2, 1e4, 1e6), not the exploratory 17-candidate interp_smooth_extended bundle. Fig. 22 uses the affine-limit audit on that same seven-candidate result. Noise std grid 0.01,0.05,0.10,0.20. At fixed step/model seed, comparison changes the noise scale rather than the observation locations; lambda is selected for each noise level.

Lapa: use --help to inspect launcher options; manuscript uses L=16, step 4, log10 k_lab, seven auxiliary channels, ratios 0.20–0.60 and three seeds, with both latent prior variants. A final command is deliberately not certified because the exact seven-channel configuration has not been located.

Well 861: three depth-held-out blocks, length 8, step 1, 16 validation rows, three seeds, ratios 0.25/0.50, no added noise. Outputs heldout_rmse_detail.csv, heldout_rmse_summary.csv and protocol.json. The score is withheld coordinates within each window, not globally withheld physical depths. Run python scripts/audit_861_depth_indices.py --row-count 87 to check the index geometry without laboratory data, or use --excel-path with the authorized workbook.

## Known gaps

See COVERAGE_AUDIT.md for evidence-by-evidence status. The scripts exist for the listed experiments, but command coverage for all figures, exact historical profile/configuration equivalence, Lapa settings, and all table-level numerical agreement remain open. The theoretical proofs in Appendix A are not executable experiments.
