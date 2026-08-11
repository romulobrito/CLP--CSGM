#!/usr/bin/env bash
# Offline smoke tests using only the synthetic tables under data/synthetic/.
# No F3 / restricted field files are required.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

SYN_DIR="data/synthetic"
TRAIN="${SYN_DIR}/SYN-A,SYN-B,SYN-C_6logs.txt"
TEST="${SYN_DIR}/SYN-D_6logs.txt"
PORO="${SYN_DIR}/SYN-D_AC+GR+Porosity.txt"

if [[ ! -f "${TRAIN}" || ! -f "${TEST}" || ! -f "${PORO}" ]]; then
  echo "Synthetic tables missing; generating under ${SYN_DIR} ..."
  python scripts/make_synthetic_smoke_data.py --out-dir "${SYN_DIR}"
fi

echo "== Cross-well Vc synthetic smoke (main no-GR channels) =="
python sir_cs_benchmark_multi_well_vc.py \
  --train-path "${TRAIN}" \
  --test-path "${TEST}" \
  --channels sonic,rhob,ai,vp \
  --target vc \
  --window-len 64 \
  --step 16 \
  --fast \
  --run-csgm-m2 \
  --run-csgm-ablations \
  --csgm-prior-type ridge \
  --csgm-ae-epochs 8 \
  --csgm-iters 20 \
  --csgm-restarts 1 \
  --csgm-lambda-grid 0.01 \
  --no-lfista \
  --base-dir outputs/synthetic_smoke/cross_well_vc \
  --run-id synthetic_crosswell_smoke

echo "== F03-style GR-only porosity synthetic smoke =="
python sir_cs_benchmark_real_well_direct_ub.py \
  --data-path "${PORO}" \
  --u-channels gr \
  --window-len 64 \
  --step 8 \
  --fast \
  --run-csgm-m2 \
  --run-csgm-ablations \
  --csgm-prior-type ridge \
  --csgm-ae-epochs 8 \
  --csgm-iters 20 \
  --csgm-restarts 1 \
  --csgm-lambda-grid 0.01 \
  --no-lfista \
  --base-dir outputs/synthetic_smoke/real_well \
  --run-id synthetic_porosity_smoke

echo "Synthetic smoke finished OK."
echo "Outputs under outputs/synthetic_smoke/"
