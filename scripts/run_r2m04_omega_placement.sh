#!/usr/bin/env bash
# R2-04: redraw Omega on the cross-well Vc grid. CPU only.
# Does not overwrite production runs or the main tables.
# ASCII-only. Run from the repository root.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=""
BASE_DIR="outputs/reviewer_revision/r2m04_omega_placement"
TRAIN="data/F02-1,F03-2,F06-1_6logs_30dB.txt"
TEST="data/F03-4_6logs_30dB.txt"
OFFSETS=(0 10007 20011 30013 40009)

if [[ ! -f "$TRAIN" || ! -f "$TEST" ]]; then
  echo "missing F3 files: $TRAIN or $TEST" >&2
  exit 2
fi

mkdir -p "$BASE_DIR/runs"
cp -f paper_clp_csgm/docs/revision/PROTOCOL_R2M04_omega_placement.md "$BASE_DIR/PROTOCOL.txt"
python - <<'PY'
import json
import os
root = "outputs/reviewer_revision/r2m04_omega_placement"
cfg = {
    "steps": [8, 16, 32],
    "rhos": [0.05, 0.10, 0.20],
    "model_seeds": [7, 23, 41],
    "measurement_seed_offsets": [0, 10007, 20011, 30013, 40009],
    "offset_meaning": "measurement_seed = model_seed + offset; offset 0 is the published Omega stream",
    "channels": ["sonic", "rhob", "ai", "vp"],
    "target": "vc",
    "metric": "full-window RMSE",
    "device": "cpu",
}
with open(os.path.join(root, "config_frozen.json"), "w", encoding="utf-8") as handle:
    json.dump(cfg, handle, indent=2)
print("wrote", os.path.join(root, "config_frozen.json"))
PY

for OFFSET in "${OFFSETS[@]}"; do
  for STEP in 8 16 32; do
    run_id="$(printf "offset%05d_step%02d" "$OFFSET" "$STEP")"
    echo "=== $run_id ==="
    python sir_cs_benchmark_multi_well_vc.py \
      --train-path "$TRAIN" \
      --test-path "$TEST" \
      --channels sonic,rhob,ai,vp \
      --target vc \
      --step "$STEP" \
      --rhos 0.05,0.10,0.20 \
      --seeds 7,23,41 \
      --measurement-seed-offset "$OFFSET" \
      --val-frac 0.1 \
      --val-split-mode tail \
      --run-csgm-m2 \
      --csgm-prior-type ridge \
      --no-lfista \
      --no-plots \
      --no-parity \
      --base-dir "$BASE_DIR" \
      --run-id "$run_id"
  done
done

python scripts/clp_csgm_r2m04_omega_assets.py
echo "R2-M04 omega placement finished."
