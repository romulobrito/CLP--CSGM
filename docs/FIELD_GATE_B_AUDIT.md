# Gate B field reproduction — targeted diagnostic closed

The author reported eight independent benchmarks completed, seven normalized tables (05–10 and 12), 215/361 metrics within the unchanged printed-precision tolerance and 146 outside. No table passed NUMERIC_MATCH. The executed scientific commit remains e3bc30027b6bbcb331a08cc620c397e183ae3106. Local comparison report SHA-256: 12b1f78faf033fcc72cf63085f5c647a6a475b28b0086e096df60886444d939c. The report and per-seed files were not independently downloaded in this remote audit. No restricted data are included.

| Table | Matched | Total | Reported maximum absolute deviation | Reported interpretation |
| --- | ---: | ---: | --- | --- |
| 05 | 24 | 35 | 0.000103720 | Rankings 5/5 preserved |
| 06 | 11 | 25 | 0.000395464 | Signs, win counts and CI zero inclusion preserved; relative deviation amplified near zero |
| 07 | 33 | 54 | 0.311657 percentage points | Rankings 9/9 preserved |
| 08 | 14 | 30 | 0.147279 percentage points | Rankings 5/5 preserved |
| 09 | 17 | 27 | 0.066315 percentage points | Prior-class ordering 9/9 preserved |
| 10 | 6 | 15 | 6.829662 percentage points | Ridge favored 5/5; magnitude deviation at rho .50 audited below |
| 12 | 110 | 175 | 0.000387204 relative L2 | Rankings 5/5 preserved; relative deviation of CI half-width 73.3969% |

All eight effective configs reportedly passed audit. Cross-well masks match; F03 omega_sets.csv was not generated. Author reports 13,440 shared scientific rows identical across Ridge/MLP runs, indirect evidence of paired observations, not direct identity of all observed b vectors.

## Remote source audit and critical case

Historical source CS-ML c4df022497f6b4a3438dd8372335df58178cfe89, paper_clp_csgm/tables/ablation_prior_class_f03_wide.csv (Git blob af43b017a81560883d6e4d08053b25742389a8e1) records at rho .50 MLP RMSE 0.015951187272594, Ridge RMSE 0.0128057804782929, gain 19.718950950474248%. The printed reference is 19.7190%. The original printed-reference comparison reported 6.829662 points. The subsequent full-precision historical comparison found 6.829612487338272 points; see the local diagnostic below.

Remote comparison of F03 launcher, csgm_m2_module.py, direct_ub_baselines.py, sir_cs_benchmark_direct_ub.py and sir_cs_pipeline_optimized.py confirms identical Git blobs in source and execution commit. The conditional MLP prior uses scikit-learn MLPRegressor with explicit random_state; its inputs include the trained generator latents, so both prior fitting and latent inversion are candidates for diagnosis. No specific causal mechanism established.

The existing prior-class aggregator contains historical winner-count assertions (seven MLP/two Ridge cross-well cells and five positive Ridge F03 gains). Its successful execution is not an independent test of the ranking: a changed ranking would cause an error. Inspect raw results for the ranking assessment. The assertions are not a model selection or training rule; do not use them to force agreement.

## Diagnostic scope (completed locally)

Compare full precision historical and observed Table 10 rows, all five ratios, then local historical and observed summary_by_seed.csv for both prior classes at rho .50 (seeds 7,23,41). Verify source lineage and effective configuration including MLP hidden layers, max iterations, early stopping, learning rate, alpha, generator settings, lambda grid and selected lambda; inspect validation and training reconstruction scores and available observation metadata. Recompute gain from unrounded means. Report signed differences and whether concentrated in a seed or widespread, and whether the magnitude claim in manuscript text remains supported. Compare Table 6 and 12 intervals from full-precision source and observed values; distinguish near-zero denominators from substantive changes. If historical artifacts are unavailable, record NOT_FOUND. No numeric match claim for Gate B.

## Local Gate B.1 diagnosis and B.2 repeatability (author-reported)

At rho .50, Ridge mean RMSE 0.0128057804782929 -> 0.0128006985509839, MLP mean RMSE 0.015951187272594 -> 0.0146947552976243, Ridge gain 19.7189509505% -> 12.8893384631%. Full-precision gain difference 6.829612487338272 percentage points. MLP seed 23 accounts for 93.41% of summed absolute seed differences; selected lambda changed from 0.0003 to 0.0001. The other fourteen MLP seed/rho choices did not change lambda. The generator reconstruction RMSE remained nearly identical. Effective scikit-learn prior random_state is seed+7000, i.e. 7023, verified in source. Historical environment, exact F03 observation vectors, full candidate validation scores and trained weights were not recovered. The lambda change is observed; its causal origin is unresolved.

Table 6 SE and CI half-width deviations were 7.46–38.49%; Table 12 AE dispersion/CI at rho .50 decreased 73.56%. These are real uncertainty-magnitude differences, not exclusively near-zero relative-denominator artifacts. Local audit of raw results reports preserved rankings, signs, win counts and CI zero inclusion. Do not claim identical uncertainty magnitudes or gains.

A single isolated current-environment MLP seed 23, rho .50 rerun selected lambda naturally and reproduced all 224 individual RMSEs exactly. Historical RMSE 0.016783438032258509, Gate B and repeat 0.012728246345251986; historical lambda 0.0003, both current runs 0.0001; current val_score 0.00901529018426415 and ae_recon_train_rmse 0.001620883417264032 identical. RMSE vector hash for both current runs 0980618e08ceddba3e67a220dca4790c5c031311ef71c77f3a542a998ae2f4a1; historical 2a9ab34a4ac9fd1d0a4d09668ef15ae2550157d5a90b1a031935e2dcbfd6b296. ML-only, direct MLP and PCA also reportedly repeat exactly; historical differences up to 2.34e-16. These hashes and metrics were provided by the author, not independently rehashed remotely. No scientific change or relaxation of printed tolerance. Gate B partial numeric status unchanged; targeted diagnosis closed, Gate C may proceed.
