# Published numerical reference (RELEASE-05d.1)

Tables 04–24 are transcribed from the displayed LaTeX cells of manuscript commit fa96db0193a12841067de3928db41e1d97115e56. Source and destination Git blobs, column schemas and row counts are in REFERENCE_MANIFEST.json. These are published aggregates, not regenerated results. No well data, arrays or model weights are included. Tables 1–3 and graphical evidence are outside this gate. Tables 18/19 are normalized to long rows; plus/minus cells are split; win counts use their published denominators (Table 15: 3; Table 23: 9). Table 24 fold IDs follow ascending depth intervals.

Software fixture: `python scripts/compare_manuscript_results.py --self-test`.

For actual comparisons independently execute the experiments and normalize the aggregate CSV columns to the corresponding reference schemas, preserving full precision. Write them under a separate local directory, e.g. outputs/observed_tables/. Record commands, input hashes, configuration, commit and seed coverage independently. Never generate observed values from the references.

`python scripts/compare_manuscript_results.py --observed-dir outputs/observed_tables`

Use --tables 04 13 23 24 for a subset. Missing tables are NOT_RUN; missing, duplicate or unexpected row keys and mismatched values fail. Tolerance is half the last printed decimal unit plus 1e-12; integer counts and categorical flags exact; timing columns informational. Table 04 CI flag uses yes/no. Numeric keys accept equivalent decimal notation. No rounding of observed values is needed.

Producer-specific normalization adapters remain to be built. This is not automatic protocol verification or field reproduction. Lapa exact historical settings and profile overlap remain unresolved. The fixture tests software behavior only.
