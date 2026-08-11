#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate small synthetic well-log tables for offline smoke tests.

The files match the column schemas expected by multi_well_vc.py and
real_well_f03.py. They are not geologically calibrated and must not be used
as substitutes for the paper F3 benchmarks.

ASCII-only.
"""

from __future__ import annotations

import argparse
import os
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd


REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DEFAULT_OUT_DIR = os.path.join(REPO_ROOT, "data", "synthetic")


def _smooth_noise(n: int, rng: np.random.Generator, scale: float) -> np.ndarray:
    """Low-pass Gaussian noise via cumulative sum of white noise."""
    raw = rng.normal(0.0, 1.0, size=n)
    smooth = np.cumsum(raw)
    smooth = smooth - float(np.mean(smooth))
    std = float(np.std(smooth))
    if std < 1.0e-12:
        return np.zeros(n, dtype=np.float64)
    return (smooth / std) * float(scale)


def _make_well_block(
    well_name: str,
    n_rows: int,
    depth0: float,
    dz: float,
    seed: int,
) -> pd.DataFrame:
    """One synthetic well segment with 6-log + Vc columns."""
    rng = np.random.default_rng(int(seed))
    depth = depth0 + dz * np.arange(n_rows, dtype=np.float64)
    t = np.linspace(0.0, 2.0 * np.pi, n_rows, dtype=np.float64)

    gr = 55.0 + 18.0 * np.sin(t) + _smooth_noise(n_rows, rng, 4.0)
    sonic = 120.0 + 12.0 * np.cos(0.7 * t) + _smooth_noise(n_rows, rng, 2.5)
    rhob = 2.25 + 0.08 * np.sin(1.3 * t) + _smooth_noise(n_rows, rng, 0.02)
    vp = 1.0e6 / np.maximum(sonic, 80.0) + _smooth_noise(n_rows, rng, 40.0)
    ai = rhob * vp
    # Soft clay proxy correlated with GR, clipped to (0, 1).
    vc = 0.04 + 0.004 * np.clip(gr - 40.0, 0.0, None) + _smooth_noise(n_rows, rng, 0.01)
    vc = np.clip(vc, 0.02, 0.45)

    return pd.DataFrame(
        {
            "Depth(m)": depth,
            "Sonic": sonic,
            "Density": rhob,
            "Gamma_Ray": gr,
            "P_Impedance": ai,
            "Vp": vp,
            "Vc": vc,
            "_well": well_name,
        }
    )


def _make_porosity_table(n_rows: int, depth0: float, dz: float, seed: int) -> pd.DataFrame:
    """Synthetic AC/GR/Porosity table for the single-well loader."""
    rng = np.random.default_rng(int(seed))
    depth = depth0 + dz * np.arange(n_rows, dtype=np.float64)
    t = np.linspace(0.0, 2.0 * np.pi, n_rows, dtype=np.float64)
    gr = 60.0 + 15.0 * np.sin(t) + _smooth_noise(n_rows, rng, 3.0)
    ac = 130.0 + 10.0 * np.cos(0.8 * t) + _smooth_noise(n_rows, rng, 2.0)
    # Soft density proxy from AC, then a mild porosity link.
    rhob_proxy = 2.3 - 0.002 * (ac - 120.0)
    porosity = 0.22 + 0.04 * np.sin(1.1 * t) - 0.03 * (rhob_proxy - 2.2)
    porosity = porosity + _smooth_noise(n_rows, rng, 0.008)
    porosity = np.clip(porosity, 0.05, 0.40)
    return pd.DataFrame(
        {
            "Depth(m)": depth,
            "AC(us/ft)": ac,
            "GR(API)": gr,
            "Porosity": porosity,
        }
    )


def write_synthetic_bundle(
    out_dir: str,
    n_rows_per_well: int = 180,
    n_rows_porosity: int = 360,
    dz: float = 0.15,
    seed: int = 7,
) -> Dict[str, str]:
    """Write the three synthetic tables used by the smoke tests.

    The porosity table needs more rows because the F03 loader splits depth
    contiguously before windowing (train/val/test each require >= window_len).
    """
    os.makedirs(out_dir, exist_ok=True)

    train_wells: Sequence[Tuple[str, float, int]] = (
        ("SYN-A", 1000.0, seed + 1),
        ("SYN-B", 2000.0, seed + 2),
        ("SYN-C", 3000.0, seed + 3),
    )
    frames: List[pd.DataFrame] = []
    for name, depth0, well_seed in train_wells:
        frames.append(
            _make_well_block(
                well_name=name,
                n_rows=int(n_rows_per_well),
                depth0=float(depth0),
                dz=float(dz),
                seed=int(well_seed),
            )
        )
    train_df = pd.concat(frames, axis=0, ignore_index=True)
    train_df = train_df.drop(columns=["_well"])

    test_df = _make_well_block(
        well_name="SYN-D",
        n_rows=int(n_rows_per_well),
        depth0=1500.0,
        dz=float(dz),
        seed=int(seed + 4),
    ).drop(columns=["_well"])

    poro_df = _make_porosity_table(
        n_rows=int(n_rows_porosity),
        depth0=900.0,
        dz=float(dz),
        seed=int(seed + 5),
    )

    train_name = "SYN-A,SYN-B,SYN-C_6logs.txt"
    test_name = "SYN-D_6logs.txt"
    poro_name = "SYN-D_AC+GR+Porosity.txt"

    paths = {
        "train_6logs": os.path.join(out_dir, train_name),
        "test_6logs": os.path.join(out_dir, test_name),
        "porosity": os.path.join(out_dir, poro_name),
    }
    train_df.to_csv(paths["train_6logs"], sep="\t", index=False, float_format="%.8f")
    test_df.to_csv(paths["test_6logs"], sep="\t", index=False, float_format="%.8f")
    poro_df.to_csv(paths["porosity"], sep="\t", index=False, float_format="%.8f")
    return paths


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--out-dir",
        type=str,
        default=DEFAULT_OUT_DIR,
        help="Directory for synthetic tables (default: data/synthetic).",
    )
    p.add_argument("--n-rows", type=int, default=180, help="Rows per synthetic cross-well.")
    p.add_argument(
        "--n-rows-porosity",
        type=int,
        default=360,
        help="Rows for the synthetic porosity table (needs train/val/test >= L).",
    )
    p.add_argument("--dz", type=float, default=0.15, help="Depth step in meters.")
    p.add_argument("--seed", type=int, default=7, help="Base RNG seed.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    paths = write_synthetic_bundle(
        out_dir=os.path.abspath(str(args.out_dir)),
        n_rows_per_well=int(args.n_rows),
        n_rows_porosity=int(args.n_rows_porosity),
        dz=float(args.dz),
        seed=int(args.seed),
    )
    print("Wrote synthetic smoke tables:")
    for key, path in paths.items():
        print("  {}: {}".format(key, path))


if __name__ == "__main__":
    main()
