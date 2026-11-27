#!/usr/bin/env python3
"""Independent artificial producer fixtures for all 21 manuscript adapters.

Published metrics are never inputs to these fixtures. Reference keys are used
by the normalizer only to enforce table coverage.
"""
from __future__ import annotations

import argparse
import itertools
import json
import math
from pathlib import Path
import subprocess
import sys
import tempfile

import numpy as np
import pandas as pd

import compare_manuscript_results as cmp
import normalize_manuscript_results as norm

ROOT=Path(__file__).resolve().parents[1]
CROSS=(0.05,0.10,0.20)
F03=(0.2,0.3,0.4,0.5,0.6)
STEPS=(8,16,32)
NOISE=(0.01,0.05,0.10,0.20)
SEEDS=(7,23,41)
METHODS={"ml_only":"ML only","mlp_concat_ub":"MLP [u,b]",
    "pca_regression_ub":"PCA [u,b]","ae_regression_ub":"AE [u,b]",
    "ridge_prior_csgm":"CLP-CSGM Ridge","mlp_prior_csgm":"CLP-CSGM MLP",
    "ridge_prior_only_decoder":"Prior-only Ridge","measurement_only_csgm":"Measurement-only CSGM"}
DATASETS={"crosswell_vc_step16":("Cross-well Vc",CROSS),
          "f03_gr_only":("F03-4 GR-only",(0.2,0.4,0.6))}
STRESS={"s01_low_rho_crosswell_grid":("A",224),
        "s02_low_rho_high_measurement_noise":("B",224),
        "s03_smaller_train_fraction":("C",369),
        "s04_coarse_sliding_step":("D",56)}
SCENARIOS=("Emb4+GR","Emb4-noGR","NoEmb+GR","NoEmb-noGR")
T2=4.302652729696142


def records(keys,values,fields):
    return [dict(zip(keys,k),**fields) for k in values]


def make_fixtures():
    fixtures={}
    cross=list(itertools.product(STEPS,CROSS))
    single=[(r,) for r in F03]
    def direct(table,keys,vals,fields,aliases=None,extra=None):
        expected=pd.DataFrame(records(keys,vals,fields))
        producer=expected.rename(columns=aliases or {}).assign(**(extra or {}))
        fixtures[table]=(producer,expected)
    direct("04",["step","rho"],cross,
        dict(n_seeds=10,delta_mean=-0.125,delta_se=0.031,delta_t95_half=0.070,
             clp_wins=2,ae_wins=8,ci_includes_zero="no"),
        {"delta_mean":"mean_delta_rmse"})
    fixtures["04"][0]["ci_includes_zero"]=False
    for table,exclude in (("12","mlp_prior_csgm"),("22","ridge_prior_only_decoder")):
        raw=[]; exp=[]
        for i,(method,label) in enumerate(METHODS.items()):
            if method==exclude:
                continue
            for rho in F03:
                score=0.7+i*0.071+rho*0.11
                raw.append(dict(measurement_ratio=rho,method=method,rmse_mean=score,
                    rmse_std_across_seeds=0.12,rmse_sem=0.05,
                    mae_mean=score/2,relative_l2_mean=score/3,n_seeds=3))
                row=dict(rho=rho,method=label,rmse=score,mae=score/2,relative_l2=score/3)
                if table=="12":
                    row.update(rmse_std=0.12,rmse_t95_half=0.05*T2)
                exp.append(row)
        fixtures[table]=(pd.DataFrame(raw),pd.DataFrame(exp))
    raw,full=fixtures["12"]
    wide=[]
    for method,g in full.groupby("method"):
        wide.append(dict(method=method,**{f"rmse_rho{int(round(r.rho*100)):03d}":r.rmse for r in g.itertuples()}))
    fixtures["05"]=(raw.copy(),pd.DataFrame(wide))
    raw=[];exp=[]
    for rho in F03:
        for seed,delta in zip(SEEDS,(-0.12,0.06,0.24)):
            raw.extend([dict(measurement_ratio=rho,seed=seed,method="ridge_prior_csgm",rmse_mean=0.8+rho),
                        dict(measurement_ratio=rho,seed=seed,method="ae_regression_ub",rmse_mean=0.8+rho+delta)])
        se=0.18/math.sqrt(3)
        exp.append(dict(rho=rho,delta_mean=0.06,delta_se=se,delta_t95_half=se*T2,clp_wins=2,ae_wins=1))
    fixtures["06"]=(pd.DataFrame(raw),pd.DataFrame(exp))
    ab=dict(clp_rmse=0.6,prior_rmse=0.8,measurement_only_rmse=1.2,ae_rmse=0.75,
        gain_vs_prior_pct=25.,gain_vs_measurement_pct=50.)
    alias=dict(rho="measurement_ratio",clp_rmse="ridge_prior_csgm",prior_rmse="ridge_prior_only_decoder",
        measurement_only_rmse="measurement_only_csgm",ae_rmse="ae_regression_ub",
        gain_vs_prior_pct="gain_vs_prior_only_pct",gain_vs_measurement_pct="gain_vs_measurement_only_pct")
    direct("07",["step","rho"],cross,ab,alias)
    direct("08",["rho"],single,ab,alias)
    prior=dict(mlp_prior_rmse=0.8,ridge_prior_rmse=0.6,ridge_gain_pct=25.)
    aliases=dict(rho="measurement_ratio",mlp_prior_rmse="mlp_prior_csgm",
                 ridge_prior_rmse="ridge_prior_csgm",ridge_gain_pct="ridge_gain_vs_mlp_pct")
    direct("09",["step","rho"],cross,prior,aliases)
    direct("10",["rho"],single,prior,aliases)
    raw=[];exp=[]
    for dataset,(label,rhos) in DATASETS.items():
        for rho in rhos:
            m=max(4,round(64*rho))
            row=dict(dataset=label,rho=rho,m=m)
            for pct,name,v in ((1,"p1",0.14),(5,"p5",0.35),(10,"p10",0.57),(50,"median",0.93)):
                raw.append(dict(dataset=dataset,measurement_ratio=rho,m=m,seed=7,
                                percentile=pct,ratio_type="energy_normalized",value=v))
                row[name]=v
            # Deliberately different unused diagnostic; must be filtered.
            raw.append(dict(dataset=dataset,measurement_ratio=rho,m=m,seed=7,
                            percentile=1,ratio_type="raw",value=9.99))
            exp.append(row)
    fixtures["11"]=(pd.DataFrame(raw),pd.DataFrame(exp))
    direct("13",["scenario","rho"],itertools.product(SCENARIOS,CROSS),
        dict(ridge_rmse_mean=0.6,ridge_rmse_std=0.05,ae_rmse=0.8,
             measurement_only_rmse=1.2,ridge_gain_vs_ae_pct=25.,ridge_gain_vs_meas_only_pct=50.))
    direct("14",["rho"],[(r,) for r in CROSS],
        dict(n_splits=10,n_train=27,n_val=3,clp_rmse_mean=0.6,clp_rmse_std=0.1,
        ae_rmse_mean=0.8,ae_rmse_std=0.2,delta_mean=0.2,clp_win_frac=0.8),
        {"delta_mean":"delta_rmse_mean"})
    direct("15",["step","rho"],cross,
        dict(clp_rmse=0.6,ae_rmse=0.8,ae_mask_rmse=0.7,delta_ae=0.2,delta_mask=0.1,
             clp_wins_ae=3,clp_wins_mask=2),
        dict(clp_rmse="clp_rmse_mean",ae_rmse="ae_rmse_mean",ae_mask_rmse="ae_mask_rmse_mean",
             delta_ae="delta_ae_minus_clp",delta_mask="delta_mask_minus_clp",
             clp_wins_ae="clp_wins_vs_ae",clp_wins_mask="clp_wins_vs_mask"),{"n_seeds":3})
    direct("16",["step","rho"],cross,
        dict(n=15,clp_rmse=0.6,ae_rmse=0.8,delta_mean=0.2,clp_wins=12,ae_wins=3),
        dict(clp_rmse="clp_rmse_mean",ae_rmse="ae_rmse_mean"))
    direct("17",["step","rho"],cross,
        dict(linear_rmse=0.9,ridge_direct_rmse=0.85,interpolation_rmse=0.5,clp_rmse=0.6,ae_rmse=0.8),
        dict(linear_rmse="linear_ub",ridge_direct_rmse="ridge_direct_ub",interpolation_rmse="interp_sparse",
             clp_rmse="ridge_prior_csgm",ae_rmse="ae_regression_ub"),{"n_seeds":3})
    for table,fields in (("18",dict(interpolation_rmse=0.8,clp_rmse=0.6,clp_wins=3)),
                         ("19",dict(interpolation_rmse=0.8,smooth_rmse=0.7,clp_rmse=0.6))):
        direct(table,["step","rho","noise_std"],itertools.product(STEPS,CROSS,NOISE),fields,
               dict(rho="measurement_ratio",interpolation_rmse="interp_rmse"),{"n_seeds":3})
    raw=[];exp=[]
    for ds,(label,rhos) in DATASETS.items():
        for i,method in enumerate(("ml_only","mlp_concat_ub","pca_regression_ub","ae_regression_ub","ridge_prior_csgm")):
            for j,rho in enumerate(rhos):
                raw.append(dict(dataset=ds,method=method,measurement_ratio=rho,n_runs=1,
                    time_per_test_window_sec_mean=(i+1)*(j+1),relative_time_vs_ae=(i+1)/4,rmse_mean=0.4+i*0.1+j*0.03))
            exp.append(dict(dataset=label,method=METHODS[method],time_per_window_sec=2*(i+1),
                       relative_time_vs_ae=(i+1)/4,rmse=0.43+i*0.1))
    fixtures["20"]=(pd.DataFrame(raw),pd.DataFrame(exp))
    raw=[];exp=[]
    for slug,(case,n) in STRESS.items():
        for rho in CROSS:
            a=0.9+rho;c=a*0.75
            for method,v in (("ae_regression_ub",a),("ridge_prior_csgm",c)):
                raw.append(dict(scenario_slug=slug,measurement_ratio=rho,method=method,
                               n_seeds=3,n_test_samples_per_run=n,rmse_mean=v))
            exp.append(dict(case=case,rho=rho,n_test=n,ae_rmse=a,clp_rmse=c,relative_gain_pct=25.))
    fixtures["21"]=(pd.DataFrame(raw),pd.DataFrame(exp))
    direct("23",["rho"],[(0.25,),(0.5,)],dict(clp=0.6,ae=0.8,prior_only=1.2,
        delta_ae_minus_clp=0.2,clp_wins=6,n_pairs=9))
    direct("24",["fold_id","rho"],itertools.product(range(3),(0.25,0.5)),
           dict(clp=0.6,ae=0.8,prior_only=1.2))
    return fixtures


def run(report):
    fixtures=make_fixtures()
    specs=cmp.load()
    assert set(fixtures)=={s["table"] for s in specs}
    base=ROOT/"outputs"
    base.mkdir(exist_ok=True)
    results=[]
    def rejected(s, raw, source, target, label):
        raw.to_csv(source,index=False)
        try:
            norm.normalize(s,source,target)
        except (ValueError,KeyError,AttributeError,TypeError):
            assert not target.exists(),f"Invalid output retained: {label}"
        else:
            raise AssertionError(f"Invalid input accepted: {label}")
        results.append({"table":s["table"],"test":label,"status":"PASS"})
    with tempfile.TemporaryDirectory(prefix="adapter-fixtures-",dir=base) as tmp:
        work=Path(tmp)
        sources=[]
        for s in specs:
            table=s["table"]
            raw,expected=fixtures[table]
            source=work/f"producer{table}.csv"
            target=work/f"normalized{table}.csv"
            raw.to_csv(source,index=False)
            h=norm.file_hash(source)
            result=norm.normalize(s,source,target)
            assert result["producer_sha256"]==h==norm.file_hash(source)
            assert result["observed_sha256"]==norm.file_hash(target)
            keys=s["keys"];cols=keys+s["metrics"]
            actual=pd.read_csv(target).sort_values(keys).reset_index(drop=True)
            exp=expected[cols].sort_values(keys).reset_index(drop=True)
            pd.testing.assert_frame_equal(actual,exp,check_dtype=False,rtol=1e-9,atol=1e-12)
            assert cmp.compare(s,ROOT/s["file"],target)["status"]=="FAIL", "Artificial metrics must not pass paper comparison"
            results.append({"table":table,"test":"valid_independent_fixture","status":"PASS"})
            target.unlink()
            # First row is relevant for all producer fixtures, including S-REC.
            rejected(s,raw.iloc[1:],source,target,"missing_row")
            rejected(s,pd.concat([raw,raw.iloc[:1]],ignore_index=True),source,target,"duplicate_row")
            metric={"04":"mean_delta_rmse","05":"rmse_mean","06":"rmse_mean","07":"ridge_prior_csgm",
                    "08":"ridge_prior_csgm","09":"mlp_prior_csgm","10":"mlp_prior_csgm",
                    "11":"value","12":"rmse_mean","13":"ridge_rmse_mean","14":"clp_rmse_mean",
                    "15":"clp_rmse_mean","16":"clp_rmse_mean","17":"linear_ub","18":"interp_rmse",
                    "19":"smooth_rmse","20":"rmse_mean","21":"rmse_mean","22":"rmse_mean",
                    "23":"clp","24":"clp"}[table]
            bad=raw.copy();bad[metric]=float("inf")
            rejected(s,bad,source,target,"nonfinite_metric")
            rejected(s,raw.drop(columns=[metric]),source,target,"missing_column")
            counts={"04":"n_seeds","05":"n_seeds","11":"seed","12":"n_seeds","14":"n_splits",
                    "15":"n_seeds","16":"n","17":"n_seeds","18":"n_seeds","19":"n_seeds",
                    "20":"n_runs","21":"n_seeds","22":"n_seeds","23":"n_pairs"}
            if table in counts:
                bad=raw.copy()
                bad[counts[table]]=-1
                rejected(s,bad,source,target,"invalid_count")
            if table=="05":
                extra=raw.iloc[:1].copy();extra["measurement_ratio"]=0.7
                rejected(s,pd.concat([raw,extra],ignore_index=True),source,target,"unexpected_ratio")
            raw.to_csv(source,index=False)
            sources.extend(["--source",f"{table}={source}"])
        out=work/"full_normalized"
        cli=[sys.executable,str(ROOT/"scripts/normalize_manuscript_results.py")]
        cmd=cli+["--out-dir",str(out)]+sources
        completed=subprocess.run(cmd,capture_output=True,text=True)
        assert completed.returncode==0,completed.stdout+completed.stderr
        manifest=json.loads((out/"NORMALIZATION_MANIFEST.json").read_text())
        assert manifest["status"]=="NORMALIZED" and len(manifest["tables"])==21
        results.append({"test":"full_cli_normalization","status":"PASS"})
        assert subprocess.run(cmd,capture_output=True).returncode!=0
        results.append({"test":"nonempty_destination_rejected","status":"PASS"})
        absent=work/"absent_output"
        completed=subprocess.run(cli+["--tables","22","--source",f"22={work/'missing.csv'}","--out-dir",str(absent)],capture_output=True)
        assert completed.returncode!=0
        assert json.loads((absent/"NORMALIZATION_MANIFEST.json").read_text())["tables"][0]["status"]=="NOT_RUN"
        results.append({"test":"absent_producer_not_run","status":"PASS"})
        blocked=work/"blocked_output"
        completed=subprocess.run(cli+["--tables","23","--source",f"23={ROOT/'reference_results/table23.csv'}","--out-dir",str(blocked)],capture_output=True)
        assert completed.returncode!=0 and not (blocked/"table23.csv").exists()
        results.append({"test":"reference_source_rejected","status":"PASS"})
    payload={"status":"PASS","scope":"Artificial fixtures only; no field reproduction or producer protocol certification.",
             "tables":len(specs),"tests":len(results),"checks":results}
    report.parent.mkdir(parents=True,exist_ok=True)
    report.write_text(json.dumps(payload,indent=2)+"\n")
    print(f"Adapter fixtures passed: {len(specs)} tables, {len(results)} checks.")


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--report",type=Path,default=ROOT/"outputs/validation/adapter_fixtures.json")
    run(p.parse_args().report)
