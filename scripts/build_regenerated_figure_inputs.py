#!/usr/bin/env python3
"""Build local figure inputs from the explicitly identified three-seed runs.

Never read the historic five-channel m2_grid or substitute ten-seed results.
The generated provenance describes this execution, not the original manuscript run.
"""
from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "outputs/cross_well_vc/direct_ub/runs"
PROFILE_RUN = ROOT / "outputs/paper_regen/crosswell_depth_profile/runs/step16_seed7_rho0p2"
BUNDLE = ROOT / "outputs/reviewer_revision/r3m06_regenerated"
CHANNELS = ["sonic", "rhob", "ai", "vp"]
COUNTS = {8: (104, 11, 97), 16: (52, 6, 49), 32: (27, 3, 25)}
RHOS = [0.05, 0.10, 0.20]
SEEDS = [7, 23, 41]
METHODS = ("ml_only", "mlp_concat_ub", "pca_regression_ub", "ae_regression_ub", "ridge_prior_csgm")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def checked_measurement_evidence(run: Path, cfg: dict, seeds: list[int], rhos: list[float]) -> dict:
    """Verify saved observation streams; never infer a missing offset from defaults."""
    omega_path = run / "tables/omega_sets.csv"
    detail_path = run / "tables/detailed_results.csv"
    omega = pd.read_csv(omega_path)
    expected = {(s, r) for s in seeds for r in rhos}
    keys = list(zip(omega["model_seed"], omega["measurement_ratio"]))
    require(len(keys) == len(expected) and set(keys) == expected,
            f"{run}: missing or duplicate Omega cells")
    require((omega["measurement_seed_offset"] == 0).all() and
            (omega["measurement_seed"] == omega["model_seed"]).all(),
            f"{run}: nonzero or inconsistent recorded measurement offset")
    if "measurement_seed_offset" in cfg:
        require(cfg["measurement_seed_offset"] == 0,
                f"{run}: nonzero configured measurement offset")
    for row in omega.itertuples(index=False):
        m = max(4, int(round(float(row.measurement_ratio) * 64)))
        coords = [int(x) for x in str(row.omega_order).split(",")]
        wanted = np.random.default_rng(int(row.model_seed)).choice(64, size=m, replace=False).tolist()
        require(row.m == m and coords == wanted,
                f"{run}: recorded Omega does not match the zero-offset stream")
    detail = pd.read_csv(detail_path, usecols=["seed", "measurement_ratio",
                                              "measurement_seed_offset", "measurement_seed"])
    require(set(zip(detail["seed"], detail["measurement_ratio"])) == expected and
            (detail["measurement_seed_offset"] == 0).all() and
            (detail["measurement_seed"] == detail["seed"]).all(),
            f"{run}: inconsistent detailed measurement evidence")
    return {"effective_measurement_seed_offset": 0,
            "config_offset_present": "measurement_seed_offset" in cfg,
            "omega_sha256": sha256(omega_path),
            "detailed_results_sha256": sha256(detail_path)}


def check_config(run: Path, step: int, seeds: list[int], rhos: list[float]) -> dict:
    cfg = json.loads((run / "config.json").read_text(encoding="utf-8"))
    meta = cfg["cross_well_vc"]
    require(meta["channels"] == CHANNELS and int(meta["p_input"]) == 256,
            f"{run}: expected four channels, without GR")
    require(meta["train_wells"] == ["F02-1", "F03-2", "F06-1"] and
            meta["test_wells"] == ["F03-4"] and meta["target"] == "vc",
            f"{run}: wrong wells or target")
    require(int(meta["window_len"]) == 64 and int(meta["step"]) == step and
            tuple(int(meta[k]) for k in ("n_train", "n_val", "n_test")) == COUNTS[step],
            f"{run}: wrong window protocol")
    require(meta["val_split_mode"] == "tail" and float(meta["val_frac"]) == 0.1 and
            int(meta["val_embargo_windows"]) == 0, f"{run}: wrong validation split")
    require(cfg["seeds"] == seeds and cfg["measurement_ratios"] == rhos and
            cfg["measurement_kind"] == "subsample" and float(cfg["measurement_noise_std"]) == 0.01,
            f"{run}: wrong observation protocol")
    checked_measurement_evidence(run, cfg, seeds, rhos)
    expected = {"csgm_prior_type": "ridge", "run_csgm_m2": True, "run_lfista": False,
                "csgm_latent_dim": 16, "csgm_hidden_dim": 128, "csgm_ae_epochs": 200,
                "csgm_iters": 400, "csgm_restarts": 3, "csgm_opt_lr": 0.05,
                "csgm_lambda_grid": [0.0001, 0.0003, 0.001, 0.003, 0.01, 0.03, 0.1]}
    require(all(cfg.get(k) == v for k, v in expected.items()), f"{run}: wrong CLP configuration")
    return cfg


def checked_step(step: int) -> tuple[Path, Path, pd.DataFrame]:
    run = RUNS / f"crosswell_step{step:02d}_clp_csgm_ablation_ridge"
    cfg = check_config(run, step, SEEDS, RHOS)
    require(bool(cfg["run_csgm_ablations"]), f"{run}: expected ablation run")
    path = run / "tables/summary_focus_clp_csgm_vs_ub.csv"
    frame = pd.read_csv(path)
    subset = frame[frame["method"].isin(METHODS)].copy()
    keys = list(zip(subset["measurement_ratio"].astype(float), subset["method"]))
    require(len(keys) == 15 and set(keys) == {(r, m) for r in RHOS for m in METHODS},
            f"{run}: missing or duplicated cells")
    require((subset["n_seeds"] == 3).all() and
            np.isfinite(subset[["rmse_mean", "rmse_sem"]].to_numpy(dtype=float)).all(),
            f"{run}: invalid three-seed summaries")
    per = pd.read_csv(run / "tables/summary_by_seed.csv")
    per = per[per["method"].isin(METHODS)]
    pkeys = list(zip(per["seed"].astype(int), per["measurement_ratio"].astype(float), per["method"]))
    require(len(pkeys) == 45 and set(pkeys) == {(s, r, m) for s in SEEDS for r in RHOS for m in METHODS},
            f"{run}: missing or duplicated seed cells")
    return run, path, subset


def checked_profile() -> Path:
    cfg = check_config(PROFILE_RUN, 16, [7], [0.20])
    require(cfg["cross_well_vc"]["profile_overlap_agg"] == "uniform_mean",
            "Unexpected profile overlap aggregation")
    path = PROFILE_RUN / "tables/depth_profile_test_block.npz"
    with np.load(path, allow_pickle=False) as archive:
        depth = archive["depth_axis"]
        require(depth.ndim == 1 and depth.size == 832 and np.isfinite(depth).all() and
                (np.diff(depth) >= -1e-9).all(), "Unexpected profile depth axis")
        for name in ("observed", *METHODS):
            require(archive["profile_" + name].shape == depth.shape, f"Profile shape: {name}")
    return path


def build() -> None:
    frames, records = [], []
    for step in COUNTS:
        run, path, frame = checked_step(step)
        frame.insert(0, "step", step)
        frames.append(frame)
        records.append((step, run, path))
    profile = checked_profile()
    BUNDLE.mkdir(parents=True, exist_ok=True)
    manifest = {"schema": 2, "kind": "local_regeneration_not_historical",
                "protocol": "crosswell_vc_main_no_gr", "files": []}
    for step, run, source in records:
        name = f"step{step:02d}_summary_focus_clp_csgm_vs_ub.csv"
        dest = BUNDLE / name
        shutil.copyfile(source, dest)
        manifest["files"].append({"step": step, "file": name, "sha256": sha256(dest),
            "source": str(source.relative_to(ROOT)), "config_sha256": sha256(run / "config.json"),
            "worktree_channels": CHANNELS, "worktree_p_input": 256, "seeds": SEEDS,
            "measurement_evidence": checked_measurement_evidence(run, json.loads((run / "config.json").read_text()), SEEDS, RHOS),
            "per_seed_sha256": sha256(run / "tables/summary_by_seed.csv")})
    combined = BUNDLE / "summary_focus_clp_csgm_vs_ub.csv"
    pd.concat(frames, ignore_index=True).to_csv(combined, index=False)
    manifest["combined_sha256"] = sha256(combined)
    manifest["profile"] = {"source": str(profile.relative_to(ROOT)), "sha256": sha256(profile),
                           "config_sha256": sha256(PROFILE_RUN / "config.json"),
                           "measurement_evidence": checked_measurement_evidence(PROFILE_RUN, json.loads((PROFILE_RUN / "config.json").read_text()), [7], [0.20])}
    (BUNDLE / "provenance.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print("Wrote local regeneration bundle:", BUNDLE)


def checked_regenerated_paths() -> list[tuple[int, Path]]:
    manifest = json.loads((BUNDLE / "provenance.json").read_text(encoding="utf-8"))
    require(manifest["kind"] == "local_regeneration_not_historical" and manifest["schema"] == 2, "Wrong provenance kind/schema; rebuild local bundle")
    require(len(manifest["files"]) == 3, "Expected three summary files")
    results = []
    for step in COUNTS:
        run, source, _ = checked_step(step)
        record, = [r for r in manifest["files"] if r["step"] == step]
        name = f"step{step:02d}_summary_focus_clp_csgm_vs_ub.csv"
        require(record["file"] == name and record["source"] == str(source.relative_to(ROOT)) and
                record["config_sha256"] == sha256(run / "config.json"), "Summary provenance mismatch")
        require(record["measurement_evidence"] == checked_measurement_evidence(run, json.loads((run / "config.json").read_text()), SEEDS, RHOS) and
                record["per_seed_sha256"] == sha256(run / "tables/summary_by_seed.csv"), "Measurement or seed-summary provenance mismatch")
        path = BUNDLE / name
        require(sha256(path) == record["sha256"] == sha256(source), "Summary hash mismatch")
        results.append((step, path))
    profile = checked_profile()
    record = manifest["profile"]
    require(record["source"] == str(profile.relative_to(ROOT)) and record["sha256"] == sha256(profile) and
            record["config_sha256"] == sha256(PROFILE_RUN / "config.json"), "Profile provenance mismatch")
    require(record["measurement_evidence"] == checked_measurement_evidence(PROFILE_RUN, json.loads((PROFILE_RUN / "config.json").read_text()), [7], [0.20]), "Profile measurement provenance mismatch")
    require(manifest["combined_sha256"] == sha256(BUNDLE / "summary_focus_clp_csgm_vs_ub.csv"),
            "Combined summary hash mismatch")
    return results


if __name__ == "__main__":
    build()
