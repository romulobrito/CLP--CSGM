#!/usr/bin/env python3
"""Normalize independently generated producer CSVs for Tables 04–24.

No reference metric is used to construct observed metrics. Reference row keys
are used solely to check coverage. No training or field-data reads.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import t

import compare_manuscript_results as cmp

ROOT = Path(__file__).resolve().parents[1]
R = "outputs/reviewer_revision/"
P = "outputs/paper_assets/tables/"
F = "outputs/real_well_f03/direct_ub/runs/f03_full_gr_only_clp_csgm_ridge/tables/"
DEFAULTS = {
    "04": R+"r1m04_seed_panel/aggregate/tables/summary_by_cell.csv",
    "05": F+"summary.csv", "06": F+"summary_by_seed.csv",
    "07": P+"ablation_crosswell_wide.csv", "08": P+"ablation_f03_wide.csv",
    "09": P+"ablation_prior_class_crosswell_wide.csv",
    "10": P+"ablation_prior_class_f03_wide.csv",
    "11": "outputs/srec_diagnostics/tables/srec_percentiles.csv",
    "12": F+"summary.csv", "13": R+"b7_gr_embargo/aggregate/table13.csv",
    "14": R+"r1m03_step32_dev_splits/aggregate/tables/summary_by_rho.csv",
    "15": R+"r1m08_explicit_mask/aggregate/tables/summary_by_cell.csv",
    "16": R+"r2m04_omega_placement/aggregate/tables/summary_by_cell.csv",
    "17": R+"r3m03_simple_baselines/aggregate/tables/summary_by_cell.csv",
    "18": R+"r3m03_noise_sweep/tables/summary_by_cell.csv",
    "19": R+"r3m03_smooth_audit/tables/summary_by_cell.csv",
    "20": "outputs/runtime_clp_csgm/tables/runtime_summary.csv",
    "23": R+"r1m11_861_phi_lab/table23_paired.csv",
    "24": R+"r1m11_861_phi_lab/table24_by_block.csv",
}
LABELS = {"ml_only":"ML only", "mlp_concat_ub":"MLP [u,b]",
          "pca_regression_ub":"PCA [u,b]", "ae_regression_ub":"AE [u,b]",
          "ridge_prior_csgm":"CLP-CSGM Ridge", "mlp_prior_csgm":"CLP-CSGM MLP",
          "ridge_prior_only_decoder":"Prior-only Ridge",
          "measurement_only_csgm":"Measurement-only CSGM"}
F03_METHODS = set(LABELS) - {"mlp_prior_csgm"}
LAPA_METHODS = set(LABELS) - {"ridge_prior_only_decoder"}
CORE_METHODS = {"ml_only","mlp_concat_ub","pca_regression_ub","ae_regression_ub","ridge_prior_csgm"}
DATASETS = {"crosswell_vc_step16":"Cross-well Vc", "f03_gr_only":"F03-4 GR-only"}
DIAG_RHOS = {"crosswell_vc_step16":{0.05,0.10,0.20}, "f03_gr_only":{0.20,0.40,0.60}}
STRESS = {"s01_low_rho_crosswell_grid":"A", "s02_low_rho_high_measurement_noise":"B",
          "s03_smaller_train_fraction":"C", "s04_coarse_sliding_step":"D"}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def unique(df, keys):
    require(not df[keys].isna().any().any() and not df.duplicated(keys).any(),
            f"Missing or duplicate producer keys: {keys}")


def count(df, name, n):
    require(name in df and len(df)>0 and (pd.to_numeric(df[name],errors="raise")==n).all(),
            f"Expected {name}={n} in every producer row")


def grid(df, group, field, wanted):
    for g, part in df.groupby(group, dropna=False):
        require(set(part[field]) == set(wanted), f"Unexpected {field} grid: {g}")


def summary(df, methods):
    df = df[df.method.isin(methods)].copy()
    require(set(df.method)==methods, "Missing methods")
    unique(df, ["measurement_ratio","method"])
    count(df,"n_seeds",3)
    df["method"] = df.method.map(LABELS)
    return df.rename(columns={"measurement_ratio":"rho","rmse_mean":"rmse",
        "rmse_std_across_seeds":"rmse_std","mae_mean":"mae","relative_l2_mean":"relative_l2"})


def adapt(table, source):
    d=source.copy()
    if table=="04":
        count(d,"n_seeds",10)
        flags=d.ci_includes_zero.astype(str).str.lower()
        require(flags.isin(["true","false","yes","no"]).all(),"Invalid CI flag")
        d["ci_includes_zero"]=flags.map({"true":"yes","false":"no","yes":"yes","no":"no"})
        return d.rename(columns={"mean_delta_rmse":"delta_mean"})
    if table in {"05","12","22"}:
        d=summary(d,LAPA_METHODS if table=="22" else F03_METHODS)
        if table=="05":
            wide=d.pivot(index="method",columns="rho",values="rmse").reset_index()
            return wide.rename(columns={rho:f"rmse_rho{int(round(rho*100)):03d}" for rho in (0.2,0.3,0.4,0.5,0.6)})
        if table=="12":
            d["rmse_t95_half"]=d["rmse_sem"]*t.ppf(0.975,2)
        return d
    if table=="06":
        d=d[d.method.isin({"ae_regression_ub","ridge_prior_csgm"})].copy()
        unique(d,["measurement_ratio","seed","method"])
        grid(d,["measurement_ratio","method"],"seed",{7,23,41})
        p=d.pivot(index=["measurement_ratio","seed"],columns="method",values="rmse_mean").reset_index()
        delta=p["ae_regression_ub"]-p["ridge_prior_csgm"]
        p=p.assign(delta=delta)
        rows=[]
        for rho,g in p.groupby("measurement_ratio"):
            require(len(g)==3 and np.isfinite(g.delta).all(),"Incomplete paired seed panel")
            se=g.delta.std(ddof=1)/np.sqrt(3)
            rows.append(dict(rho=rho,delta_mean=g.delta.mean(),delta_se=se,
                delta_t95_half=se*t.ppf(0.975,2),clp_wins=(g.delta>0).sum(),ae_wins=(g.delta<0).sum()))
        return pd.DataFrame(rows)
    if table in {"07","08"}:
        return d.rename(columns={"measurement_ratio":"rho","ridge_prior_csgm":"clp_rmse",
            "ridge_prior_only_decoder":"prior_rmse","measurement_only_csgm":"measurement_only_rmse",
            "ae_regression_ub":"ae_rmse","gain_vs_prior_only_pct":"gain_vs_prior_pct",
            "gain_vs_measurement_only_pct":"gain_vs_measurement_pct"})
    if table in {"09","10"}:
        return d.rename(columns={"measurement_ratio":"rho","mlp_prior_csgm":"mlp_prior_rmse",
            "ridge_prior_csgm":"ridge_prior_rmse","ridge_gain_vs_mlp_pct":"ridge_gain_pct"})
    if table=="11":
        d=d[(d.ratio_type=="energy_normalized") & d.percentile.isin([1,5,10,50])].copy()
        count(d,"seed",7)
        unique(d,["dataset","measurement_ratio","percentile"])
        require(set(d.dataset)==set(DATASETS),"Wrong diagnostic datasets")
        grid(d,["dataset","measurement_ratio"],"percentile",{1,5,10,50})
        require((d.groupby(["dataset","measurement_ratio"]).m.nunique()==1).all(),"Inconsistent measurement counts")
        p=d.pivot(index=["dataset","measurement_ratio","m"],columns="percentile",values="value").reset_index()
        p["dataset"]=p.dataset.map(DATASETS)
        return p.rename(columns={"measurement_ratio":"rho",1:"p1",5:"p5",10:"p10",50:"median"})
    if table in {"13","23","24"}:
        return d
    if table=="14":
        count(d,"n_splits",10)
        return d.rename(columns={"delta_rmse_mean":"delta_mean"})
    if table=="15":
        count(d,"n_seeds",3)
        return d.rename(columns={"clp_rmse_mean":"clp_rmse","ae_rmse_mean":"ae_rmse",
            "ae_mask_rmse_mean":"ae_mask_rmse","delta_ae_minus_clp":"delta_ae",
            "delta_mask_minus_clp":"delta_mask","clp_wins_vs_ae":"clp_wins_ae",
            "clp_wins_vs_mask":"clp_wins_mask"})
    if table=="16":
        count(d,"n",15)
        return d.rename(columns={"clp_rmse_mean":"clp_rmse","ae_rmse_mean":"ae_rmse"})
    if table=="17":
        count(d,"n_seeds",3)
        return d.rename(columns={"linear_ub":"linear_rmse","ridge_direct_ub":"ridge_direct_rmse",
            "interp_sparse":"interpolation_rmse","ridge_prior_csgm":"clp_rmse","ae_regression_ub":"ae_rmse"})
    if table in {"18","19"}:
        count(d,"n_seeds",3)
        return d.rename(columns={"measurement_ratio":"rho","interp_rmse":"interpolation_rmse"})
    if table=="20":
        require(set(d.dataset)==set(DATASETS) and set(d.method)==CORE_METHODS,"Wrong runtime grid")
        unique(d,["dataset","method","measurement_ratio"])
        count(d,"n_runs",1)
        for dataset,rhos in DIAG_RHOS.items():
            grid(d[d.dataset==dataset],["method"],"measurement_ratio",rhos)
        p=d.groupby(["dataset","method"],as_index=False).agg(
            time_per_window_sec=("time_per_test_window_sec_mean","mean"),
            relative_time_vs_ae=("relative_time_vs_ae","mean"),rmse=("rmse_mean","mean"))
        p["dataset"]=p.dataset.map(DATASETS)
        p["method"]=p.method.map(LABELS)
        return p
    if table=="21":
        require(set(d.scenario_slug)==set(STRESS),"Wrong stress scenarios")
        require(set(d.method)=={"ae_regression_ub","ridge_prior_csgm"},"Wrong stress methods")
        count(d,"n_seeds",3)
        unique(d,["scenario_slug","measurement_ratio","method"])
        grid(d,["scenario_slug","measurement_ratio"],"method",{"ae_regression_ub","ridge_prior_csgm"})
        require((d.groupby(["scenario_slug","measurement_ratio"]).n_test_samples_per_run.nunique()==1).all(),"Mismatched test counts")
        p=d.pivot(index=["scenario_slug","measurement_ratio","n_test_samples_per_run"],columns="method",values="rmse_mean").reset_index()
        p=p.rename(columns={"measurement_ratio":"rho","n_test_samples_per_run":"n_test",
                "ae_regression_ub":"ae_rmse","ridge_prior_csgm":"clp_rmse"})
        p["case"]=p.scenario_slug.map(STRESS)
        p["relative_gain_pct"]=100*(p.ae_rmse-p.clp_rmse)/p.ae_rmse
        return p
    raise ValueError(f"No adapter: {table}")


def file_hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def normalize(spec, source, destination):
    table=spec["table"]
    path=source.resolve()
    require(not path.is_relative_to(cmp.REF.resolve()),"Reference directory cannot be a producer")
    h=file_hash(path)
    d=adapt(table,pd.read_csv(path))
    columns=spec["keys"]+spec["metrics"]
    d=d[columns].copy()
    unique(d,spec["keys"])
    for m in spec["metrics"]:
        if m not in spec["categorical_metrics"]:
            d[m]=pd.to_numeric(d[m],errors="raise")
            require(np.isfinite(d[m]).all(),f"Nonfinite metric {m}")
    require(file_hash(path)==h,"Producer changed during read")
    d.to_csv(destination,index=False)
    observed=cmp.read(destination,spec)
    expected=cmp.read(ROOT/spec["file"],spec)
    require(set(observed)==set(expected),"Missing/unexpected published row keys")
    return {"table":table,"status":"NORMALIZED","producer":str(path),"producer_sha256":h,
            "observed_sha256":file_hash(destination),"rows":len(d),
            "protocol_verification":"NOT_PERFORMED"}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out-dir",type=Path,required=True)
    p.add_argument("--source",action="append",default=[],metavar="TABLE=CSV",
        help="Override source path; mandatory for Tables 21 and 22")
    p.add_argument("--tables",nargs="+",choices=[f"{i:02d}" for i in range(4,25)])
    args=p.parse_args()
    specs=cmp.load()
    sources={k:ROOT/v for k,v in DEFAULTS.items()}
    overrides=set()
    for item in args.source:
        k,sep,v=item.partition("=")
        require(sep and k in {s["table"] for s in specs} and v and k not in overrides,"Invalid or repeated --source")
        overrides.add(k)
        sources[k]=Path(v)
    dest=args.out_dir.resolve()
    require(dest.is_relative_to((ROOT/"outputs").resolve()),"Use a new local directory under outputs/")
    require(not dest.exists() or (dest.is_dir() and not any(dest.iterdir())),"Output directory must be empty")
    dest.mkdir(parents=True,exist_ok=True)
    results=[]
    for s in specs:
        table=s["table"]
        if args.tables and table not in args.tables:
            continue
        source=sources.get(table)
        if source is None or not source.is_file():
            results.append({"table":table,"status":"NOT_RUN","reason":"Producer CSV absent or not selected",
                            "producer":str(source) if source else None})
            continue
        target=dest/f"table{table}.csv"
        try:
            results.append(normalize(s,source,target))
        except Exception as exc:
            target.unlink(missing_ok=True)
            results.append({"table":table,"status":"FAIL","reason":str(exc)})
    report={"scope":"Normalization only; not proof of protocol, data identity or reproduction.",
            "tables":results,"full_table_scope":len(results)==len(specs),
            "status":"NORMALIZED" if results and all(r["status"]=="NORMALIZED" for r in results) else "INCOMPLETE"}
    (dest/"NORMALIZATION_MANIFEST.json").write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps(report,indent=2))
    return int(report["status"]!="NORMALIZED")


if __name__=="__main__":
    raise SystemExit(main())
