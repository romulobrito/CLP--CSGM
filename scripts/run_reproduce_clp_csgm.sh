#!/usr/bin/env bash
set -euo pipefail

if [[ ! -f "data/F02-1,F03-2,F06-1_6logs_30dB.txt" ]]; then
  echo "Missing data/F02-1,F03-2,F06-1_6logs_30dB.txt"
  exit 1
fi

if [[ ! -f "data/F03-4_6logs_30dB.txt" ]]; then
  echo "Missing data/F03-4_6logs_30dB.txt"
  exit 1
fi

if [[ ! -f "data/F03-4_AC+GR+Porosity.txt" ]]; then
  echo "Missing data/F03-4_AC+GR+Porosity.txt"
  exit 1
fi

for step in 8 16 32; do
  python sir_cs_benchmark_multi_well_vc.py \
    --train-path "data/F02-1,F03-2,F06-1_6logs_30dB.txt" \
    --test-path "data/F03-4_6logs_30dB.txt" \
    --channels sonic,rhob,ai,vp \
    --target vc \
    --step "${step}" \
    --measurement-ratios 0.05,0.10,0.20 \
    --seeds 7,23,41 \
    --run-csgm-m2 \
    --run-csgm-ablations \
    --csgm-prior-type ridge \
    --no-lfista \
    --run-id "crosswell_step$(printf '%02d' "${step}")_clp_csgm_ablation_ridge"

  python sir_cs_benchmark_multi_well_vc.py \
    --train-path "data/F02-1,F03-2,F06-1_6logs_30dB.txt" \
    --test-path "data/F03-4_6logs_30dB.txt" \
    --channels sonic,rhob,ai,vp \
    --target vc \
    --step "${step}" \
    --measurement-ratios 0.05,0.10,0.20 \
    --seeds 7,23,41 \
    --run-csgm-m2 \
    --csgm-prior-type mlp \
    --no-lfista \
    --run-id "crosswell_step$(printf '%02d' "${step}")_clp_csgm_mlp_prior_full"
done

python sir_cs_benchmark_real_well_direct_ub.py \
  --data-path "data/F03-4_AC+GR+Porosity.txt" \
  --u-channels gr \
  --measurement-ratios 0.20,0.30,0.40,0.50,0.60 \
  --seeds 7,23,41 \
  --run-csgm-m2 \
  --run-csgm-ablations \
  --csgm-prior-type ridge \
  --no-lfista \
  --run-id "f03_gr_only_clp_csgm_ablation_ridge"

python sir_cs_benchmark_real_well_direct_ub.py \
  --data-path "data/F03-4_AC+GR+Porosity.txt" \
  --u-channels gr \
  --measurement-ratios 0.20,0.30,0.40,0.50,0.60 \
  --seeds 7,23,41 \
  --run-csgm-m2 \
  --csgm-prior-type mlp \
  --no-lfista \
  --run-id "f03_gr_only_clp_csgm_mlp_prior_full"

python scripts/clp_csgm_ablation_assets.py
python scripts/clp_csgm_paper_assets.py
python scripts/clp_csgm_quick_figures.py
python scripts/clp_csgm_runtime_study.py

