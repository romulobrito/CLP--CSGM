#!/usr/bin/env bash
# R1-M03 optional supplement: 10 development train/val splits at step 32.
# Does not overwrite outputs/cross_well_vc/direct_ub/runs/crosswell_step32_*.
# ASCII-only. Run from the repository root.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

export N_SPLITS="${N_SPLITS:-10}"
BASE_DIR="outputs/reviewer_revision/r1m03_step32_dev_splits"
TRAIN="data/F02-1,F03-2,F06-1_6logs_30dB.txt"
TEST="data/F03-4_6logs_30dB.txt"

if [[ ! -f "$TRAIN" || ! -f "$TEST" ]]; then
  echo "missing F3 files: $TRAIN or $TEST" >&2
  exit 2
fi

mkdir -p "$BASE_DIR"

i=0
while [[ "$i" -lt "$N_SPLITS" ]]; do
  run_id="$(printf "split%02d" "$i")"
  echo "=== $run_id val-split-seed=$i ==="
  python sir_cs_benchmark_multi_well_vc.py \
    --train-path "$TRAIN" \
    --test-path "$TEST" \
    --channels sonic,rhob,ai,vp \
    --target vc \
    --step 32 \
    --rhos 0.05,0.10,0.20 \
    --seeds 7 \
    --val-frac 0.1 \
    --val-split-mode random_per_well \
    --val-split-seed "$i" \
    --run-csgm-m2 \
    --csgm-prior-type ridge \
    --no-lfista \
    --no-plots \
    --no-parity \
    --base-dir "$BASE_DIR" \
    --run-id "$run_id"
  i=$((i + 1))
done

cp -f paper_clp_csgm/docs/revision/PROTOCOL_R1M03_step32_dev_splits.md \
  outputs/reviewer_revision/r1m03_step32_dev_splits/PROTOCOL.txt

python - <<'PY'
import json
import os
root = "outputs/reviewer_revision/r1m03_step32_dev_splits"
os.makedirs(root, exist_ok=True)
cfg = {
    "step": 32,
    "rhos": [0.05, 0.10, 0.20],
    "omega_seed": 7,
    "val_split_mode": "random_per_well",
    "n_splits": int(os.environ["N_SPLITS"]),
    "methods": ["ridge_prior_csgm", "ae_regression_ub"],
    "channels": ["sonic", "rhob", "ai", "vp"],
}
with open(os.path.join(root, "config_frozen.json"), "w", encoding="utf-8") as handle:
    json.dump(cfg, handle, indent=2)
print("wrote", os.path.join(root, "config_frozen.json"))
PY

python scripts/clp_csgm_r1m03_step32_split_assets.py
echo "R1-M03 split supplement finished."
