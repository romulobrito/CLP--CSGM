#!/usr/bin/env python3
"""Audit physical-depth overlap in the published well-861 window protocol.

No training. With --row-count 87, only depth indices and mask geometry
are used; no workbook is read. Use --excel-path for the authorized input.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

import run_r1m11_861_phi_lab as protocol


def audit(n_rows: int) -> pd.DataFrame:
    blocks = protocol._depth_blocks(n_rows, protocol.N_BLOCKS)
    rows = []
    for fold, block in enumerate(blocks):
        starts = np.asarray(protocol._window_starts(block, protocol.WINDOW_LEN, protocol.STEP))
        if not starts.size:
            raise ValueError("Empty test fold")
        for seed in protocol.SEEDS:
            for rho in protocol.RHOS:
                rng = np.random.default_rng(int(seed) * 1009 + fold * 17 + int(round(rho * 100)))
                observed, _ = protocol._sample_omega(protocol.WINDOW_LEN, rho, rng)
                withheld = np.setdiff1d(np.arange(protocol.WINDOW_LEN), observed)
                measured = (starts[:, None] + observed).ravel()
                scored = (starts[:, None] + withheld).ravel()
                shared = np.intersect1d(measured, scored)
                rows.append({"fold_id": fold, "seed": seed, "rho": rho,
                             "shared_depth_indices": int(shared.size),
                             "scored_entries": int(scored.size),
                             "fraction_scored_ever_measured": float(np.isin(scored, measured).mean())})
    return pd.DataFrame(rows)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    inputs = p.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--row-count", type=int)
    inputs.add_argument("--excel-path", type=Path)
    p.add_argument("--out-dir", type=Path, default=protocol.REPO_ROOT / "outputs/validation/861_depth_mask")
    args = p.parse_args()
    n = len(protocol._load_frame(args.excel_path)) if args.excel_path else args.row_count
    if n != 87:
        raise ValueError("The published protocol uses exactly 87 rows.")
    frame = audit(n)
    result = {"scope": "row-index geometry, not laboratory data or RMSE reproduction",
              "n_rows": n, "configurations": len(frame),
              "shared_min": int(frame.shared_depth_indices.min()),
              "shared_max": int(frame.shared_depth_indices.max()),
              "fraction_min": float(frame.fraction_scored_ever_measured.min()),
              "fraction_max": float(frame.fraction_scored_ever_measured.max())}
    args.out_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.out_dir / "depth_mask_geometry.csv", index=False)
    (args.out_dir / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    if len(frame) != 18 or (result["shared_min"], result["shared_max"]) != (21, 27) or not np.isclose(result["fraction_min"], 0.8409090909090909) or result["fraction_max"] != 1.0:
        raise RuntimeError("Published index-overlap diagnostic did not match.")


if __name__ == "__main__":
    main()
