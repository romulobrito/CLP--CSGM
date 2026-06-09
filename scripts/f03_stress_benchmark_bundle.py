#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Orchestrate multiple F03-4 GR-only direct-UB stress benchmarks.

Each scenario is one invocation of sir_cs_benchmark_real_well_direct_ub.py with its own
base-dir and run-id, producing the usual tables/, figures/, logs/, manifests under:

  <bundle_root>/<scenario_slug>/runs/<run_id>/

The bundle root also gets STRESS_BUNDLE_MANIFEST.txt with scenario descriptions.

ASCII-only (log lines and manifest).

Usage (from repo root):
  python scripts/f03_stress_benchmark_bundle.py
  python scripts/f03_stress_benchmark_bundle.py --dry-run
  python scripts/f03_stress_benchmark_bundle.py --scenarios s01_low_rho,s02_high_noise
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@dataclass(frozen=True)
class Scenario:
    """One stress scenario: slug, human description, extra CLI args (no paths)."""

    slug: str
    description: str
    argv_extra: Tuple[str, ...]


def _build_scenarios(
    low_rho_grid: str,
    noise_baseline: float,
    noise_high: float,
    train_frac_default: float,
    val_frac_default: float,
    train_frac_low: float,
    val_frac_low: float,
    step_default: int,
    step_coarse: int,
) -> List[Scenario]:
    """
    Default grid aligns F03 low-rho stress with cross-well style rho sweep.
    """
    return [
        Scenario(
            slug="s01_low_rho_crosswell_grid",
            description=(
                "Low measurement ratios rho in {" + low_rho_grid + "}, baseline eta on b, "
                "standard depth split (train_frac=" + str(train_frac_default) + "). "
                "Stress sparse observation conditioning for all methods."
            ),
            argv_extra=(
                "--rhos",
                low_rho_grid,
                "--measurement-noise-std",
                str(noise_baseline),
                "--train-frac",
                str(train_frac_default),
                "--val-frac",
                str(val_frac_default),
                "--step",
                str(step_default),
            ),
        ),
        Scenario(
            slug="s02_low_rho_high_measurement_noise",
            description=(
                "Same rho grid as s01 but larger measurement noise std on b = M y + eta "
                "(eta=" + str(noise_high) + "). Degrades SNR for everyone."
            ),
            argv_extra=(
                "--rhos",
                low_rho_grid,
                "--measurement-noise-std",
                str(noise_high),
                "--train-frac",
                str(train_frac_default),
                "--val-frac",
                str(val_frac_default),
                "--step",
                str(step_default),
            ),
        ),
        Scenario(
            slug="s03_smaller_train_fraction",
            description=(
                "Fewer training rows via train_frac=" + str(train_frac_low) + ", val_frac="
                + str(val_frac_low) + "; same low-rho grid. Reduces capacity for h(u) and baselines."
            ),
            argv_extra=(
                "--rhos",
                low_rho_grid,
                "--measurement-noise-std",
                str(noise_baseline),
                "--train-frac",
                str(train_frac_low),
                "--val-frac",
                str(val_frac_low),
                "--step",
                str(step_default),
            ),
        ),
        Scenario(
            slug="s04_coarse_sliding_step",
            description=(
                "Larger sliding-window step (step=" + str(step_coarse) + ") with standard split; "
                "fewer overlapping train windows (low-data within F03). Same low-rho grid."
            ),
            argv_extra=(
                "--rhos",
                low_rho_grid,
                "--measurement-noise-std",
                str(noise_baseline),
                "--train-frac",
                str(train_frac_default),
                "--val-frac",
                str(val_frac_default),
                "--step",
                str(step_coarse),
            ),
        ),
    ]


def _parse_scenario_filter(raw: str, all_slugs: Sequence[str]) -> List[str]:
    if not raw.strip():
        return list(all_slugs)
    want = {p.strip() for p in raw.split(",") if p.strip()}
    unknown = want.difference(set(all_slugs))
    if unknown:
        raise SystemExit("Unknown scenario slug(s): " + ", ".join(sorted(unknown)))
    return [s for s in all_slugs if s in want]


def main() -> None:
    p = argparse.ArgumentParser(description="F03 stress benchmark bundle (organized outputs).")
    p.add_argument(
        "--bundle-root",
        type=str,
        default="",
        help="Root folder for this bundle (default: outputs/real_well_f03/f03_stress_bundle_<timestamp>).",
    )
    p.add_argument(
        "--data-path",
        type=str,
        default=os.path.join("data", "F03-4_AC+GR+Porosity.txt"),
        help="F03-4 TSV path relative to repo root unless absolute.",
    )
    p.add_argument(
        "--low-rho-grid",
        type=str,
        default="0.05,0.10,0.20",
        help="Comma-separated rho list for stress scenarios.",
    )
    p.add_argument("--noise-baseline", type=float, default=0.01, help="Eta std for s01/s03/s04.")
    p.add_argument("--noise-high", type=float, default=0.05, help="Eta std for s02.")
    p.add_argument("--train-frac", type=float, default=0.6)
    p.add_argument("--val-frac", type=float, default=0.2)
    p.add_argument("--train-frac-low", type=float, default=0.45, help="For s03.")
    p.add_argument("--val-frac-low", type=float, default=0.25)
    p.add_argument("--step", type=int, default=1)
    p.add_argument("--step-coarse", type=int, default=4, help="For s04.")
    p.add_argument(
        "--scenarios",
        type=str,
        default="",
        help="Comma-separated scenario slugs (empty = all).",
    )
    p.add_argument(
        "--with-lfista",
        action="store_true",
        help="Keep hybrid LFISTA branch (much slower). Default skips via --no-lfista.",
    )
    p.add_argument("--dry-run", action="store_true", help="Print commands only.")
    args = p.parse_args()

    scenarios = _build_scenarios(
        low_rho_grid=str(args.low_rho_grid).strip(),
        noise_baseline=float(args.noise_baseline),
        noise_high=float(args.noise_high),
        train_frac_default=float(args.train_frac),
        val_frac_default=float(args.val_frac),
        train_frac_low=float(args.train_frac_low),
        val_frac_low=float(args.val_frac_low),
        step_default=int(args.step),
        step_coarse=int(args.step_coarse),
    )
    slugs = [s.slug for s in scenarios]
    selected = _parse_scenario_filter(str(args.scenarios), slugs)
    scenarios = [s for s in scenarios if s.slug in selected]

    ts = time.strftime("%Y%m%d_%H%M%S")
    bundle_root = str(args.bundle_root).strip()
    if not bundle_root:
        bundle_root = os.path.join("outputs", "real_well_f03", "f03_stress_bundle_" + ts)
    bundle_root = os.path.abspath(os.path.join(_REPO_ROOT, bundle_root))
    os.makedirs(bundle_root, exist_ok=True)

    data_path = args.data_path
    if not os.path.isabs(data_path):
        data_path = os.path.join(_REPO_ROOT, data_path)
    if not os.path.isfile(data_path):
        print("Missing data file: " + data_path, file=sys.stderr)
        raise SystemExit(2)

    runner = os.path.join(_REPO_ROOT, "sir_cs_benchmark_real_well_direct_ub.py")
    if not os.path.isfile(runner):
        print("Missing runner: " + runner, file=sys.stderr)
        raise SystemExit(2)

    manifest_lines: List[str] = [
        "F03 stress benchmark bundle.",
        "",
        "bundle_root: " + bundle_root,
        "data_path: " + data_path,
        "low_rho_grid (default): " + str(args.low_rho_grid),
        "seeds: 7,23,41 (runner default)",
        "u_channels: gr (GR-only stress test)",
        "lfista: " + ("enabled" if args.with_lfista else "disabled (--no-lfista)"),
        "",
        "Notes:",
        "  - Block or non-uniform subsampling patterns are not implemented in measurement_kind;",
        "    only gaussian or coordinate subsample M is available without further code changes.",
        "  - Extra-well generalization is not included here; use multi-well runners separately.",
        "",
        "Scenarios:",
    ]

    results_index: List[Dict[str, str]] = []

    for sc in scenarios:
        scenario_dir = os.path.join(bundle_root, sc.slug)
        run_id = sc.slug + "_" + ts
        cmd: List[str] = [
            sys.executable,
            runner,
            "--data-path",
            data_path,
            "--base-dir",
            scenario_dir,
            "--run-id",
            run_id,
            "--u-channels",
            "gr",
            "--window-len",
            "64",
            "--residual-basis",
            "dct",
            "--measurement-kind",
            "subsample",
            "--run-csgm-m2",
            "--run-csgm-ablations",
            "--csgm-prior-type",
            "ridge",
        ]
        if not args.with_lfista:
            cmd.append("--no-lfista")
        cmd.extend(list(sc.argv_extra))

        manifest_lines.append("")
        manifest_lines.append("[" + sc.slug + "]")
        manifest_lines.append(sc.description)
        manifest_lines.append("run_root: " + os.path.join(scenario_dir, "runs", run_id))
        manifest_lines.append("command: " + " ".join(cmd))

        results_index.append(
            {
                "slug": sc.slug,
                "run_id": run_id,
                "scenario_dir": scenario_dir,
                "run_root": os.path.join(scenario_dir, "runs", run_id),
            }
        )

        print("[bundle] " + sc.slug + " -> " + run_id, flush=True)
        if args.dry_run:
            continue
        os.makedirs(scenario_dir, exist_ok=True)
        rc = subprocess.call(cmd, cwd=_REPO_ROOT)
        if rc != 0:
            print("[bundle] FAILED slug=" + sc.slug + " rc=" + str(rc), file=sys.stderr)
            raise SystemExit(rc)

    if args.dry_run:
        print("[bundle] dry-run: no manifest written, no subprocess executed.", flush=True)
        return

    index_json = os.path.join(bundle_root, "STRESS_BUNDLE_INDEX.json")
    with open(index_json, "w", encoding="utf-8") as f:
        json.dump({"created_at": ts, "scenarios": results_index}, f, indent=2)
        f.write("\n")

    manifest_path = os.path.join(bundle_root, "STRESS_BUNDLE_MANIFEST.txt")
    with open(manifest_path, "w", encoding="utf-8") as f:
        f.write("\n".join(manifest_lines) + "\n")

    print("[bundle] wrote " + manifest_path, flush=True)
    print("[bundle] wrote " + index_json, flush=True)

    # Optional: pooled comparison table for Ridge CSGM vs AE across scenarios (mean RMSE by rho).
    try:
        import pandas as pd

        rows_out: List[Dict[str, object]] = []
        for entry in results_index:
            summ = os.path.join(str(entry["run_root"]), "tables", "summary.csv")
            if not os.path.isfile(summ):
                continue
            df = pd.read_csv(summ)
            sub = df[df["method"].isin(["ridge_prior_csgm", "ae_regression_ub"])].copy()
            sub.insert(0, "scenario_slug", entry["slug"])
            rows_out.append(sub)
        if rows_out:
            merged = pd.concat(rows_out, ignore_index=True)
            out_csv = os.path.join(bundle_root, "comparison_summary_by_scenario.csv")
            merged.to_csv(out_csv, index=False)
            print("[bundle] wrote " + out_csv, flush=True)
    except Exception as exc:
        print("[bundle] optional comparison_summary skipped: " + str(exc), file=sys.stderr)


if __name__ == "__main__":
    main()
