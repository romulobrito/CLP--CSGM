#!/usr/bin/env bash
# Primary F3 comparison and separate three-seed benchmarks.
# Field inputs are supplied locally by the user; no data are downloaded.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
TRAIN="${CLP_CROSS_TRAIN:-data/F02-1,F03-2,F06-1_6logs_30dB.txt}"
TEST="${CLP_CROSS_TEST:-data/F03-4_6logs_30dB.txt}"
PORO="${CLP_F03_POROSITY:-data/F03-4_AC+GR+Porosity.txt}"
for path in "$TRAIN" "$TEST" "$PORO"; do
  if [[ ! -f "$path" ]]; then echo "Missing local input: $path" >&2; exit 2; fi
done
export CLP_CROSS_TRAIN="$TRAIN" CLP_CROSS_TEST="$TEST" CLP_F03_POROSITY="$PORO"
# Ten-seed paired comparison (Table 4), not the three-seed ablation grid.
SEEDS=7,23,41,13,29,31,37,43,47,53 bash scripts/run_r1m04_seed_panel.sh
for step in 8 16 32; do
  python sir_cs_benchmark_multi_well_vc.py \
    --train-path "$TRAIN" --test-path "$TEST" --channels sonic,rhob,ai,vp \
    --target vc --step "$step" --rhos 0.05,0.10,0.20 --seeds 7,23,41 \
    --val-frac 0.1 --val-split-mode tail --run-csgm-m2 --run-csgm-ablations \
    --csgm-prior-type ridge --no-lfista \
    --run-id "crosswell_step$(printf '%02d' "$step")_clp_csgm_ablation_ridge"
  python sir_cs_benchmark_multi_well_vc.py \
    --train-path "$TRAIN" --test-path "$TEST" --channels sonic,rhob,ai,vp \
    --target vc --step "$step" --rhos 0.05,0.10,0.20 --seeds 7,23,41 \
    --val-frac 0.1 --val-split-mode tail --run-csgm-m2 \
    --csgm-prior-type mlp --no-lfista \
    --base-dir outputs/reviewer_revision/r1m06_prior_class_controlled/crosswell \
    --run-id "step$(printf '%02d' "$step")_mlp"
done
python sir_cs_benchmark_real_well_direct_ub.py \
  --data-path "$PORO" --u-channels gr --rhos 0.20,0.30,0.40,0.50,0.60 \
  --seeds 7,23,41 --run-csgm-m2 --run-csgm-ablations \
  --csgm-prior-type ridge --no-lfista --run-id f03_full_gr_only_clp_csgm_ridge
python sir_cs_benchmark_real_well_direct_ub.py \
  --data-path "$PORO" --u-channels gr --rhos 0.20,0.30,0.40,0.50,0.60 \
  --seeds 7,23,41 --run-csgm-m2 --csgm-prior-type mlp --no-lfista \
  --base-dir outputs/reviewer_revision/r1m06_prior_class_controlled/f03_gr_only \
  --run-id mlp_depth_first
python scripts/clp_csgm_ablation_assets.py
python scripts/clp_csgm_r1m06_controlled_assets.py
# Legacy figure utilities require additional named result bundles; run separately.
echo "Primary comparison, three-seed benchmarks and controlled prior tables finished."
# Generate local intermediate results and final manuscript-style figures.
bash scripts/run_regenerate_figure_inputs.sh
