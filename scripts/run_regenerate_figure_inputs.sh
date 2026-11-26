#!/usr/bin/env bash
# Regenerate local Figure 6 summaries and Figure 8/24 depth profile.
# Requires the three canonical three-seed Ridge ablation runs.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
TRAIN="${CLP_CROSS_TRAIN:-data/F02-1,F03-2,F06-1_6logs_30dB.txt}"
TEST="${CLP_CROSS_TEST:-data/F03-4_6logs_30dB.txt}"
export CLP_CROSS_TRAIN="$TRAIN" CLP_CROSS_TEST="$TEST"
for path in "$TRAIN" "$TEST"; do
  if [[ ! -f "$path" ]]; then echo "Missing local input: $path" >&2; exit 2; fi
done
# A single setting is necessary: the launcher saves the first job's profile.
python sir_cs_benchmark_multi_well_vc.py \
  --train-path "$TRAIN" --test-path "$TEST" --channels sonic,rhob,ai,vp \
  --target vc --window-len 64 --step 16 --rhos 0.20 --seeds 7 \
  --val-frac 0.1 --val-split-mode tail --measurement-kind subsample \
  --measurement-noise-std 0.01 --run-csgm-m2 --csgm-prior-type ridge \
  --no-lfista --profile-overlap-agg uniform_mean \
  --base-dir outputs/paper_regen/crosswell_depth_profile --run-id step16_seed7_rho0p2
python scripts/build_regenerated_figure_inputs.py
python scripts/plot_r3m06_manuscript_figures.py
