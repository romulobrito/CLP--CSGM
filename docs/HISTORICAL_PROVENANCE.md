# Historical provenance and remaining local recovery (RELEASE-05c.2)

## Verified Git evidence

Source repository CS-ML, commit 53190576cd8321d927682c86881927b9d51aacb1, contains the runtime and empirical separation scripts alongside the original numeric tables. The manuscript main commit fa96db0193a12841067de3928db41e1d97115e56 contains byte-identical tables: runtime_summary.csv Git blob f82700e987d712ba443386155af31c407b3830a4; srec_percentiles.csv Git blob 0dedde711400ffab83b482be12ff53a420f4bca0. The historical runtime case builder uses five cross-well channels (sonic,rhob,gr,ai,vp), step 16, validation 0.1, seed 7, and measurement noise 0.02. The S-REC script shares this builder and evaluates sampled decoder-window secants without adding observation noise to the ratio. Preserve this historical diagnostic configuration, distinct from the primary no-GR benchmark. Co-location of code and numeric tables is provenance evidence, not proof of exact invocation or numerical replay.

Lapa: source EDA explicitly screens density,gr,res_deep,res_shallow,phi_neutron,phi_sonic,phi_nd (feature-target screening). This is a plausible ordered seven-channel list, not proof of the historical benchmark invocation. Manuscript Appendix D confirms L=16, step 4, log10 k_lab, ratios 0.20,0.30,0.40,0.50,0.60 and three seeds, with contiguous splits. Do not certify an executable reproduction command before recovering the original config, including noise, split fractions, prior variants, measurement settings, preprocessing and workbook identity. The script default four channels is not the reported experiment.

Cross-well depth profile: source commit a8a6fec4492d826f186a6d4b2bc339ef034f68d9 explicitly states that the local depth_profile_test_block.npz is not in its tracked no-GR summary bundle. Step 16, seed 7 and rho 0.20 are confirmed by the manuscript; uniform_mean is the current launcher default, not confirmed historical selection. Regeneration now additionally exports the overlap-mode comparison for review without changing the primary default.

Well 861: manuscript Tables 23–24 use paired fold/seed averages, AE minus CLP, and strictly positive AE-minus-CLP counts. The new aggregate_861_tables.py requires all 54 rows (3 folds x 3 seeds x 2 ratios x 3 methods) and exports both pooled and per-fold tables. The index-overlap diagnostic is separate. No global depth holdout or field reproduction is implied.

## Local evidence still required

Read-only inspection of the original CS-ML worktree (including ignored outputs):
- Locate Lapa runs under outputs/auddys_smoke_direct_ub/runs/ or other outputs trees with config.json, DATASET_MANIFEST.txt, PROTOCOL.txt, RUN_MANIFEST.txt, tables/summary.csv. Match all seven methods and all five ratios to Table 22, including both priors, before reporting the configuration.
- Locate outputs/paper_regen/crosswell_depth_profile/runs/step16_seed7_rho0p2/config.json and the profile provenance; report the overlap rule, channels, window counts, noise, selected seed/ratio, and input identity. For F03 Fig. 10 recover the original run/seed and selected rho 0.20.
- Check historical runtime/S-REC run metadata if available; confirm exact invocation against the co-committed CSVs. Return only protocol metadata and aggregate comparison, not workbooks or arrays.

History inspected: CS-ML main, revision checkpoint, first runner commit 0f111b0ae9deefe8173b229bd9d0a05c6553630c, manuscript checkpoint 53190576cd8321d927682c86881927b9d51aacb1, no-GR provenance commit a8a6fec4492d826f186a6d4b2bc339ef034f68d9; tracked trees contained no original Lapa run config or dedicated profile config. Do not infer missing run metadata from a default or from EDA alone.

## Local audit complement (author-reported, 26 November 2026)

The author supplied the following local inspection results; the underlying local files were not independently accessible in this audit:

- Lapa run `auddys_favorable_u7_step4_ridge_mlp_csgm_20260506` has the same aggregates and figures as run `strict_b_20260510`. The latter is the more likely manuscript source on chronology, but the numerical results do not uniquely identify the run. Status: PARTIAL for unique run lineage. The complete configuration and input identity were not supplied in this complement.
- The original Figure 8 PNG retained in `paper_clp_csgm/figures/r3m06_originals/` is reported byte-identical to the raw run PNG. Current Figures 8 and 24 are redraws of the same NPZ. Status: CONFIRMED by local audit for shared source and PNG identity; the historical overlap aggregation itself was not supplied.
- Historical command confirmed by local audit:

```bash
python scripts/clp_csgm_runtime_study.py && \
python scripts/clp_csgm_srec_diagnostics.py
```

- Lapa historical `split_depth_summary.csv`: NOT_FOUND, although the current code produces it. Its absence alone does not prevent reproduction if original partition settings and inputs can be established.

The full local configuration files, workbook identities, and profile aggregation setting have not been received. Do not silently replace these missing details with defaults or candidate channels. No restricted data or arrays were transferred.
