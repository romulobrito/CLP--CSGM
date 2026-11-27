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
| Table 11; Fig. 11 | clp_csgm_srec_diagnostics.py | Historical source/table lineage recovered; numerical check pending |
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
| Table 20; Fig. 25 | clp_csgm_runtime_study.py | Historical source/table lineage recovered; GR-inclusive, noise 0.02; hardware-specific timing |
| Table 21 | f03_stress_benchmark_bundle.py | Four scenarios, exact outputs and values |
| Table 22; Figs. 26–29 | auddys_smoke_direct_ub.py | BLOCKED for exact reproduction: EDA identifies a candidate seven-channel list; exact run config not tracked; authorized workbook required |
| Tables 23–24 | run_r1m11_861_phi_lab.py | Authorized 87-row workbook; new aggregate_861_tables.py produces paired/fold CSVs; numerical comparison pending |
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

RELEASE-05a recheck: [run 45896373226](https://github.com/romulobrito/CLP--CSGM/actions/runs/45896373226), 98 static and 151 total checks, zero failures. RELEASE-05b: [run 45896373313](https://github.com/romulobrito/CLP--CSGM/actions/runs/45896373313), two synthetic benchmarks and 28 required method cells, zero failures. Both tested b0bc692. At that earlier checkpoint the new coverage tools had not yet run on the GitHub runner; see the final validation below. No field outputs were copied or generated. Coverage audit delivered with open items, not a declaration of full reproduction.

## Follow-up audit (05c.2)

See [HISTORICAL_PROVENANCE.md](HISTORICAL_PROVENANCE.md) for the matching source-table blob evidence, Lapa candidate channels, remaining local metadata requests and the 861 table exporter. No field-data equivalence is claimed.

## Final coverage validation (05c.3)

Tested commit `c7fd7348b126fd32ebaf2f68532fa502662ef58d`:
- [Technical run 45897849839](https://github.com/romulobrito/CLP--CSGM/actions/runs/45897849839): 104 static checks, 160 total technical checks (including static checks), zero failures; pip check passed. [Artifact](https://github.com/romulobrito/CLP--CSGM/actions/runs/45897849839/artifacts/15154452658).
- [Synthetic run 45897849853](https://github.com/romulobrito/CLP--CSGM/actions/runs/45897849853): both synthetic benchmarks, 28 required method cells, zero failures. [Artifact](https://github.com/romulobrito/CLP--CSGM/actions/runs/45897849853/artifacts/15154836378).
- Coverage checks passed: B.7 command dry-run (not field execution), 861 index-only audit (18 configurations, 21–27 shared indices, 0.8409090909090909–1.0 fraction), and 861 aggregation self-test on constructed metrics (not paper RMSE).

See HISTORICAL_PROVENANCE.md for the author-reported local audit supplement. The coverage audit is consolidated with explicit exceptions; the full exact-protocol coverage gate is not certified. The 05d.1 reference/comparison preparation can proceed independently, but field reproduction and release claims must respect these exceptions. No merge or tag is authorized by this audit.

Outstanding:
- Exact Lapa configuration (ordered channels, split, noise and other run settings) not supplied in the local audit complement; historical origin non-unique.
- Historical overlap aggregation for cross-well Figures 8 and 24, and original F03 Figure 10 run/seed, not supplied.
- Exact remaining qualitative figure selections and field numerical comparison; no authorized inputs executed here.

## Superseding final campaign evidence

The preceding coverage map and pending items are historical checkpoints, not the current release status. Gates A–D executed all 21 tables; Gate E/E.1 regenerated all 26 numerical figures. Historical Lapa May 6 configuration, profile overlap and qualitative selections were recovered locally; strict Lapa lineage and full numerical equivalence remain limited. See FIELD_REPRODUCIBILITY_REPORT.md and FIELD_GATE_E_AUDIT.md. Fresh-install CI at 19991fc6b4de320b2c7d745bdd0782fd58ab46d9 passed in both jobs: [technical 46042013857](https://github.com/romulobrito/CLP--CSGM/actions/runs/46042013857) (116 static, 177 total technical checks, 124 adapter checks and figure regression), [synthetic 46042013860](https://github.com/romulobrito/CLP--CSGM/actions/runs/46042013860) (2 benchmarks, 28 method cells). Both reported VERSION_MATCH (33/33 distributions) and pip check passed.
