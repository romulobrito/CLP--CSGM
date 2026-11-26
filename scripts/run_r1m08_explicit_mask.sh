#!/usr/bin/env bash
# R1-M08 supplement: AE with explicit window mask, paper 3 seeds, 9 cells.
# Does not overwrite production cross-well runs. ASCII-only.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

export PYTHONUNBUFFERED=1
export SEEDS="${SEEDS:-7,23,41}"
BASE_DIR="outputs/reviewer_revision/r1m08_explicit_mask"
TRAIN="data/F02-1,F03-2,F06-1_6logs_30dB.txt"
TEST="data/F03-4_6logs_30dB.txt"

if [[ ! -f "$TRAIN" || ! -f "$TEST" ]]; then
  echo "missing F3 files: $TRAIN or $TEST" >&2
  exit 2
fi

mkdir -p "$BASE_DIR"

for STEP in 8 16 32; do
  run_id="$(printf "step%02d" "$STEP")"
  echo "=== $run_id seeds=${SEEDS} ==="
  python sir_cs_benchmark_multi_well_vc.py \
    --train-path "$TRAIN" \
    --test-path "$TEST" \
    --channels sonic,rhob,ai,vp \
    --target vc \
    --step "$STEP" \
    --rhos 0.05,0.10,0.20 \
    --seeds "$SEEDS" \
    --val-frac 0.1 \
    --val-split-mode tail \
    --run-csgm-m2 \
    --csgm-prior-type ridge \
    --run-ae-explicit-mask \
    --no-lfista \
    --no-plots \
    --no-parity \
    --base-dir "$BASE_DIR" \
    --run-id "$run_id"
done

cp -f paper_clp_csgm/docs/revision/PROTOCOL_R1M08_explicit_mask.md \
  outputs/reviewer_revision/r1m08_explicit_mask/PROTOCOL.txt

python - <<'PY'
import json
import os
root = "outputs/reviewer_revision/r1m08_explicit_mask"
os.makedirs(root, exist_ok=True)
seeds_raw = os.environ.get("SEEDS", "7,23,41")
cfg = {
    "steps": [8, 16, 32],
    "rhos": [0.05, 0.10, 0.20],
    "seeds": [int(item.strip()) for item in seeds_raw.split(",") if item.strip()],
    "val_split_mode": "tail",
    "methods": [
        "ridge_prior_csgm",
        "ae_regression_ub",
        "ae_regression_ub_mask",
    ],
    "mask_encoding": "b_placed_in_R_N plus binary Omega mask",
    "channels": ["sonic", "rhob", "ai", "vp"],
    "device": "cpu",
}
with open(os.path.join(root, "config_frozen.json"), "w", encoding="utf-8") as handle:
    json.dump(cfg, handle, indent=2)
print("wrote", os.path.join(root, "config_frozen.json"))
PY

python scripts/clp_csgm_r1m08_explicit_mask_assets.py
echo "R1-M08 explicit-mask supplement finished."
