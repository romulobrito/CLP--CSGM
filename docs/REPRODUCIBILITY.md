# Reproduction and validation procedure

## Current status

The revision source is selected from CS-ML commit c4df022497f6b4a3438dd8372335df58178cfe89. The public base and per-file source/destination hashes are recorded in SOURCE_MANIFEST.json. Unchanged scientific modules were checked by Git blob identity; infrastructure adaptations are tracked separately. This is not a claim of successful reproduction.

## Levels of verification

1. Source integrity: verify the manifest and the exact revision commit; inspect all changes from source to destination.
2. Installation and execution: use a clean Python 3.10 environment, record interpreter, platform, CPU/GPU, package versions and logs. Test imports and shell syntax, then run scripts/run_synthetic_smoke.sh. No numerical manuscript comparison is implied by a synthetic pass.
3. Protocol consistency: use authorized original inputs, record their local hashes and metadata, inspect generated config.json, DATASET_MANIFEST.txt, PROTOCOL.txt and RUN_MANIFEST.txt where available. Check channels, well separation, counts, seeds, observation sets, noise, lambda selection and metric definitions.
4. Scientific reproduction: compare regenerated per-seed and aggregate values to the manuscript and authorized reference outputs; distinguish the ten-seed primary pair from three-seed studies. Set comparison tolerances before execution and report discrepancies. Timing is hardware-dependent. Check figures against the intended experiment and profile overlap rule.

The manuscript reports Python 3.10 and NumPy 1.26, pandas 2.3, Matplotlib 3.7, scikit-learn 1.6, PyTorch 2.6, PyWavelets 1.4. The current requirements.txt is a minimum-version list; a tested exact lock is not yet available. Record pip freeze after the clean-environment test, including scipy and openpyxl versions.

## Execution order

Run the synthetic smoke first. Obtain and verify field inputs per DATA_REQUIREMENTS.md. Run the main driver and supplemental commands in EXPERIMENTS.md. For the primary cross-well figures, the regenerated bundle records hashes of its local summary files and configurations and of the dedicated depth profile. Historical five-channel runs must not be used as no-GR evidence. The plotter checks CLP means against its published reference values at an absolute tolerance of 5e-5; this check alone does not establish full equivalence.

Do not commit field inputs, prediction profiles, model checkpoints, or other restricted artifacts. Preserve run logs and input identities locally. Report what was actually executed and the commit used.

## Release gate

Before tagging, review complete experiment coverage; resolve missing historical Lapa/profile settings or disclose the limitation; verify static checks and synthetic execution; record full field-data reproduction results where authorized and available. If field inputs cannot be obtained, explicitly state which results were not reproduced. Do not claim complete numerical reproducibility solely from source identity or smoke tests. The final immutable tag must point to the reviewed commit, with matching manuscript code availability and reviewer response.
