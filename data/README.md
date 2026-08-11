# Data Files

## Public synthetic smoke tables (included)

This repository ships a small **synthetic** bundle for offline code checks:

```text
data/synthetic/
  SYN-A,SYN-B,SYN-C_6logs.txt
  SYN-D_6logs.txt
  SYN-D_AC+GR+Porosity.txt
```

Regenerate with:

```bash
python scripts/make_synthetic_smoke_data.py
```

Run the no-external-data smoke tests with:

```bash
bash scripts/run_synthetic_smoke.sh
```

These tables match the loader schemas used by the paper benchmarks, but they
are **not** calibrated field data and must not be treated as substitutes for
the F3 experiments reported in the manuscript.

## Paper F3 tables (not redistributed here)

Place the following files in `data/` before running the full paper
reproduction commands in the root README:

```text
data/
  F02-1,F03-2,F06-1_6logs_30dB.txt
  F03-4_6logs_30dB.txt
  F03-4_AC+GR+Porosity.txt
```

Obtain the tabulated inputs from the public sources cited in the manuscript
(Zhang et al., 2025 Data Availability and related F3 portals), subject to their
license terms.

## Cross-Well Vc Files

Used by `sir_cs_benchmark_multi_well_vc.py`:

- paper: `F02-1,F03-2,F06-1_6logs_30dB.txt`, `F03-4_6logs_30dB.txt`
- synthetic smoke: `synthetic/SYN-A,SYN-B,SYN-C_6logs.txt`, `synthetic/SYN-D_6logs.txt`

Expected content:

- Tabular text readable by `pandas.read_csv(..., sep="\t")`.
- Columns resolvable to:
  - `depth`
  - `sonic`
  - `rhob`
  - `gr`
  - `ai`
  - `vp`
  - `vc`

The main paper protocol uses `--channels sonic,rhob,ai,vp` (GR excluded).
Well boundaries in concatenated files are inferred from depth discontinuities
and filename well names.

## F03-4 Porosity File

Used by `sir_cs_benchmark_real_well_direct_ub.py`:

- paper: `F03-4_AC+GR+Porosity.txt`
- synthetic smoke: `synthetic/SYN-D_AC+GR+Porosity.txt`

Expected content:

- Tab-separated table.
- Columns resolvable to:
  - `Depth`
  - `AC`
  - `GR`
  - `Porosity`

The paper experiment uses `--u-channels gr`, so only GR is used as the dense
input even though AC is present in the file.

## Privacy

Do not commit private field tables or redistributed F3 dumps unless you have
verified redistribution rights. The repository `.gitignore` excludes
`data/*.txt` by default, while `data/synthetic/**` remains tracked for CAGEO
offline evaluation.
