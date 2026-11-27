#!/usr/bin/env python3
"""Compare independently produced normalized aggregate CSVs against published tables.
Numerical comparison only: this does not certify data identity or protocol.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import tempfile
from decimal import Decimal, InvalidOperation
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REF = ROOT / "reference_results"
NUMERIC_KEYS = {"step", "rho", "noise_std", "fold_id"}


def number(value):
    x = Decimal(str(value).strip())
    if not x.is_finite():
        raise ValueError("Nonfinite number")
    return x


def read(path, spec):
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        required = set(spec["keys"] + spec["metrics"])
        if not reader.fieldnames or len(set(reader.fieldnames)) != len(reader.fieldnames) or not required.issubset(reader.fieldnames):
            raise ValueError(f"Missing or duplicate columns: {path}")
        out = {}
        for row in reader:
            if None in row or any(not row.get(k) or not row[k].strip() for k in required):
                raise ValueError(f"Malformed row: {path}")
            k = tuple(number(row[x]) if x in NUMERIC_KEYS else row[x].strip() for x in spec["keys"])
            if k in out:
                raise ValueError(f"Duplicate key {k}: {path}")
            out[k] = row
        return out


def blob(path):
    b = path.read_bytes()
    return hashlib.sha1(b"blob " + str(len(b)).encode() + b"\0" + b).hexdigest()


def compare(spec, ref, obs):
    a, b = read(ref, spec), read(obs, spec)
    errors = []
    matched = informational = 0
    for k in sorted(a.keys() - b.keys()):
        errors.append({"key": list(map(str, k)), "reason": "missing_row"})
    for k in sorted(b.keys() - a.keys()):
        errors.append({"key": list(map(str, k)), "reason": "unexpected_row"})
    for k in sorted(a.keys() & b.keys()):
        for m in spec["metrics"]:
            expected, actual = a[k][m], b[k][m]
            tol = None
            try:
                if m in spec["categorical_metrics"]:
                    ok = expected.strip().lower() == actual.strip().lower()
                else:
                    e, v = number(expected), number(actual)
                    if m in spec["informational_metrics"]:
                        informational += 1
                        continue
                    tol = Decimal(0) if m in spec["exact_metrics"] else Decimal("0.5") * Decimal(10) ** e.as_tuple().exponent + Decimal("1e-12")
                    ok = abs(e-v) <= tol
            except (ValueError, InvalidOperation):
                ok = False
            if ok:
                matched += 1
            else:
                errors.append({"key": list(map(str,k)), "metric": m, "expected": expected,
                               "actual": actual, "tolerance": str(tol), "reason": "value_mismatch"})
    return {"table":spec["table"], "status":"FAIL" if errors else "NUMERIC_MATCH",
            "reference_rows":len(a), "observed_rows":len(b), "matched":matched,
            "informational":informational, "failures":errors}


def load():
    specs = json.loads((REF/"REFERENCE_MANIFEST.json").read_text())["tables"]
    if [s["table"] for s in specs] != [f"{i:02d}" for i in range(4,25)]:
        raise ValueError("Reference table coverage mismatch")
    for s in specs:
        p = ROOT/s["file"]
        rows = read(p,s)
        if blob(p) != s["reference_blob"] or len(rows) != s["rows"]:
            raise ValueError(f"Reference integrity failure: {p}")
        for row in rows.values():
            for m in s["metrics"]:
                if m not in s["categorical_metrics"]:
                    number(row[m])
    return specs


def self_test(specs):
    # Identity is a software fixture only, never evidence of reproduction.
    for s in specs:
        p=ROOT/s["file"]
        assert compare(s,p,p)["status"]=="NUMERIC_MATCH"
    s=specs[0]
    p=ROOT/s["file"]
    with tempfile.TemporaryDirectory() as tmp:
        q=Path(tmp)/"observed.csv"
        with p.open() as f:
            rows=list(csv.DictReader(f))
        fields=list(rows[0])
        def write(rs):
            with q.open("w",newline="") as f:
                w=csv.DictWriter(f,fieldnames=fields)
                w.writeheader()
                w.writerows(rs)
        rows[0]["delta_mean"]=str(number(rows[0]["delta_mean"])+Decimal("0.001"))
        write(rows)
        assert compare(s,p,q)["status"]=="FAIL"
        write(rows[1:])
        assert any(x["reason"]=="missing_row" for x in compare(s,p,q)["failures"])
        write(rows+[rows[0]])
        try:
            compare(s,p,q)
        except ValueError:
            pass
        else:
            raise AssertionError("Duplicate row accepted")
    print("Reference integrity and comparator self-test passed: 21 tables.")


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--observed-dir",type=Path)
    p.add_argument("--tables",nargs="+",choices=[f"{i:02d}" for i in range(4,25)])
    p.add_argument("--report",type=Path,default=ROOT/"outputs/validation/manuscript_comparison.json")
    p.add_argument("--self-test",action="store_true")
    args=p.parse_args()
    specs=load()
    if args.self_test:
        self_test(specs)
        return 0
    if args.observed_dir is None or args.observed_dir.resolve()==REF.resolve():
        p.error("Supply a separate --observed-dir")
    results=[]
    for s in specs:
        if args.tables and s["table"] not in args.tables:
            continue
        ref=ROOT/s["file"]
        obs=args.observed_dir/ref.name
        if not obs.is_file():
            results.append({"table":s["table"],"status":"NOT_RUN","reason":"observed CSV absent"})
            continue
        try:
            if obs.resolve()==ref.resolve():
                raise ValueError("Reference cannot be observed input")
            r=compare(s,ref,obs)
            r["observed_sha256"]=hashlib.sha256(obs.read_bytes()).hexdigest()
            results.append(r)
        except (ValueError,InvalidOperation,OSError) as exc:
            results.append({"table":s["table"],"status":"FAIL","reason":str(exc)})
    status="NUMERIC_MATCH" if results and all(r["status"]=="NUMERIC_MATCH" for r in results) else "INCOMPLETE_OR_DIFFERENT"
    report={"status":status,"scope":"Numerical comparison only; protocol and provenance require independent verification.",
            "full_table_scope":len(results)==len(specs),"tables":results}
    args.report.parent.mkdir(parents=True,exist_ok=True)
    args.report.write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps({k:sum(r["status"]==k for r in results) for k in ("NUMERIC_MATCH","FAIL","NOT_RUN")}))
    return int(status!="NUMERIC_MATCH")


if __name__=="__main__":
    raise SystemExit(main())
