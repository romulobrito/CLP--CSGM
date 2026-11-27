# Data requirements

## What is distributed

Only the synthetic smoke tables in data/synthetic/ are included. They follow the benchmark schemas but are not field measurements. The F3 processed tables, Lapa workbooks, well-861 workbook, fitted weights and local prediction arrays are not distributed. Obtain input data through authorized sources; do not upload private data or derived arrays to the repository.

## F3: Zhang et al. (2025)

The manuscript studies processed targets associated with Zhang et al. (2025). The original README identifies the following starting points, not guaranteed drop-in equivalents:

- https://gitee.com/zhangjixju/ml-code/tree/master
- https://terranubis.com/datainfo/F3-Demo-2020

The exact processed input tables and their preprocessing provenance must be checked locally. Access to an F3 seismic volume alone does not establish equivalence to these well tables.

| Local filename under data/ | Role |
| --- | --- |
| F02-1,F03-2,F06-1_6logs_30dB.txt | Cross-well development pool |
| F03-4_6logs_30dB.txt | Completely held-out cross-well test |
| F03-4_AC+GR+Porosity.txt | Separate single-well experiment |

Tab-separated cross-well columns resolve to depth, sonic, rhob, gr, ai, vp, vc. Well boundaries are inferred from depth discontinuities and filenames; keep canonical well identifiers when relocating files. Vc is a gamma-ray-derived processed target, not an independently measured core clay fraction. Use sonic,rhob,ai,vp in the main benchmark; GR is included only for designated sensitivity checks. The porosity table resolves Depth, AC, GR, Porosity; GR is the only continuous input for that benchmark. Its target is processed density-derived porosity, not independently verified laboratory porosity.

The main driver and four revision shell drivers accept CLP_CROSS_TRAIN and CLP_CROSS_TEST; the main driver accepts CLP_F03_POROSITY. Use absolute paths when invoking commands outside the repository. Some historical utilities still expect canonical filenames under data/. See data/README.md for the loader schemas.

## Lapa permeability (Appendix D)

The data source is Alvarez et al. (2025). The original README records restrictions on redistribution. The launcher accepts --excel-path and --sheet Logs; the default workbook name is data/Auddys_table.xlsx. Required source columns: Depth(m), Density (g/cc), GR (API), Res_Deep, Res_Shallow, Phi_Neutron (pu), Phi_Sonic (pu), Phi_ND (pu), Phi_lab (pu), k_lab (mD), RQI, FZI_lab, Lithotype, HFU. The loader maps these to canonical names. The manuscript uses seven auxiliary channels, length 16, step 4 and log10 permeability; the EDA screens density,gr,res_deep,res_shallow,phi_neutron,phi_sonic,phi_nd, the order recovered in the local Gate D/E audit. The May 6 configuration was replayed with 105/105 Table 22 metrics matching; the May 10 strict paired-observation flag is not exposed in the public launcher. See FIELD_GATE_D_AUDIT.md and FIELD_GATE_E_AUDIT.md.

## Well 861 (Appendix E)

Supply an authorized enriched workbook through --excel-path (default data/861_integrated_logs_enriched.xlsx). Required exact headers: Depth(m), Phi_lab (pu), GR (API), Res_Deep, Res_Shallow, Lithotype. The launcher reads the default worksheet, sorts by depth, and checks finite numeric values. The manuscript reports 87 laboratory-porosity entries over 5205.91–5233.72 m; verify this subset against the source Logs table. Original individual laboratory assay records were unavailable for independent verification. Do not substitute the processed F3 porosity or another workbook. No global-depth holdout claim is made: scoring is on withheld coordinates within each test window.

## Record input identity locally

Before full reproduction, record input filenames, byte hashes, schema, row counts, preprocessing history, authorized source and permitted use. Keep the record and results local unless disclosure is explicitly permitted. Dataset equivalence has not been verified for the prepared revision.
