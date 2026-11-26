# RELEASE-05c — coverage audit

Basis: revised 89-page manuscript, 26 Nov 2026; public preparation branch at b0bc692cacd027d0d86549e24c4b6cd64348e046; original source checkpoint c4df022497f6b4a3438dd8372335df58178cfe89. The table maps producers, not completed reproduction. No authorized field-data execution was performed.

## Experimental evidence

| Evidence | Producer / status | Remaining verification |
| --- | --- | --- |
| Tables 1–2; Figs. 3–4 | clp_csgm_paper_assets.py: EDA, correlations, profiles | Input identity; exact formatting |
| Table 3 | Core Config, DirectUBTrainConfig and CLP module | Compare config fields to manuscript |
| Table 4; B.9 | run_r1m04_seed_panel.sh and seed-panel aggregator | Ten-seed paired comparison; numerical verification |
| Fig. 6 | Three-seed no-GR runs and checked regenerated bundle | All five method means and uncertainties |
| Fig. 7; Table 18 | run_r3m03_noise_sweep.py; plot_r3m03_rmse_vs_noise.py | Refit series, separate from Table 17 |
| Figs. 8,24 | run_regenerate_figure_inputs.sh; plot_r3m06_manuscript_figures.py | Historical profile overlap mode unconfirmed |
| Tables 5–6,12; Figs. 9,12–13 | F03 launcher; paper and uncertainty assets | Exact final figure/table export and values |
| Fig. 10 | F03 launcher depth-profile NPZ | Dedicated rho 0.20 profile and exact seed/figure selection |
| Tables 7–8 | clp_csgm_ablation_assets.py | Three-seed structural ablations |
| Tables 9–10 | clp_csgm_r1m06_controlled_assets.py | Controlled Ridge vs MLP, no uncontrolled historical MLP |
| Table 11; Fig. 11 | clp_csgm_srec_diagnostics.py | Shares runtime case builder; verify historical setup and percentiles |
| Figs. 14–15 | paper_sparse_reconstruction_figures.py | Use F03 rho 0.20 and cross-well step 16 rho 0.10, seed 7, example 0; validate selections |
| Figs. 16–17 | F03 launcher qualitative and parity artifacts | Exact windows, pooled correlations and plot selection |
| Figs. 18–19 | clp_csgm_diagnostic_assets.py | Compare plots/metrics to manuscript |
| Table 13; Fig. 20 | New run_b7_gr_embargo.py | Four scenarios, correct window counts, paired seeds; unexecuted with field data |
| Table 14 | run_r1m03_step32_dev_splits.sh | Ten development splits; fixed measurement/model seed |
| Table 15 | run_r1m08_explicit_mask.sh | Explicit-mask AE |
| Table 16 | run_r2m04_omega_placement.sh | Five observation/noise draws |
| Table 17 | run_r3m03_simple_baselines.py after offset-zero runs | Reference-grid CLP and AE plus OLS/ridge/interpolation |
| Table 19; Fig. 21 | run_r3m03_smooth_audit.py | Seven-candidate grid, validation-only, no CLP refit |
| Fig. 22 | run_r3m03_affine_limit_audit.py | Same seven-candidate smoother, ceiling 1e6 |
| Fig. 23 | plot_r3m03_noise_panels.py | Step 32 seed 7 rho 0.20, noise 0.01/0.05 |
| Table 20; Fig. 25 | clp_csgm_runtime_study.py | Historical GR-inclusive cross-well and noise 0.02 in code; confirm published provenance, hardware-specific timing |
| Table 21 | f03_stress_benchmark_bundle.py | Four scenarios, exact outputs and values |
| Table 22; Figs. 26–29 | auddys_smoke_direct_ub.py | BLOCKED for exact reproduction: seven-channel configuration and export lineage not located in source tree; authorized workbook required |
| Tables 23–24 | run_r1m11_861_phi_lab.py | Authorized 87-row workbook; detail/summary outputs available, paired/fold exporter and numerical comparison pending |
| Appendix E overlap diagnostic | New audit_861_depth_indices.py | Exact index formula; no field values required for row-count mode |

Figures 1, 2, 5 are conceptual diagrams; Appendix A is theoretical, not a numerical reproduction workflow.

## Specific findings

1. B.12: the source config for r3m03_smooth_audit confirms the exact seven alpha values printed in the manuscript (1e-6,1e-4,1e-2,1,1e2,1e4,1e6). The separate interp_smooth_extended config uses 17 alpha candidates and an affine limit. It is not Table 19 and must not be substituted. Figure 22 depends on the seven-value audit detail.
2. B.7: the original producer launcher can express all four settings, but no dedicated standalone table/figure producer was supplied. The new driver uses the same launcher, checks the manuscript train/val/test window counts, and aggregates means, sample standard deviations and relative gains; no scientific model is modified.
3. Appendix E: the published runner samples Omega via seed*1009 + fold*17 + round(100*rho). Independent calculation using 87 integer row indices reproduced the manuscript's 18 configurations, 21–27 shared indices and 84.09–100% of scored entries measured in another window. This is a geometry check, not reproduction of RMSE or laboratory depth values. New script reuses published runner helpers. The original source audit contains additional workbook/provenance checks; these are not claimed by the new narrow script. Global-depth-mask exploratory results are not substituted for the published within-window score.
4. The historical runtime code uses five cross-well channels and measurement noise 0.02; S-REC shares its case builder. Table 20 is a representative timing run, not the main no-GR grid. Preserve the historical code pending local run provenance rather than forcing main-grid defaults.
5. No exact historical Lapa seven-channel configuration or dedicated run manifest was located in the inspected tracked checkpoint. Do not guess it. Check the local original run, including channel order, depth split, transform, noise, seeds and output naming.
6. Historical figure reproduction requires comparing selected examples and regenerated output, not merely observing a plot file exists. Several general-purpose exporters exist but exact paper-level selection/formatting is pending.

## Execution evidence and gate

RELEASE-05a recheck: [run 45896373226](https://github.com/romulobrito/CLP--CSGM/actions/runs/45896373226), 98 static and 151 total checks, zero failures. RELEASE-05b: [run 45896373313](https://github.com/romulobrito/CLP--CSGM/actions/runs/45896373313), two synthetic benchmarks and 28 required method cells, zero failures. Both tested b0bc692. New coverage tools have not yet run on the GitHub runner. No field outputs were copied or generated. Coverage audit delivered with open items, not a declaration of full reproduction.
