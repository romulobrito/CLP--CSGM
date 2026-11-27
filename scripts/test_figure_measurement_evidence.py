#!/usr/bin/env python3
"""Regression checks for missing and inconsistent figure measurement metadata."""
from pathlib import Path
import tempfile
import numpy as np
import pandas as pd
from build_regenerated_figure_inputs import checked_measurement_evidence


def main():
    with tempfile.TemporaryDirectory() as tmp:
        run = Path(tmp)
        tables = run / "tables"
        tables.mkdir()
        seeds, rhos = [7, 23], [.05, .20]
        rows = []
        for s in seeds:
            for r in rhos:
                m = max(4, round(r * 64))
                rows.append(dict(model_seed=s, measurement_ratio=r, measurement_seed=s,
                                 measurement_seed_offset=0, m=m,
                                 omega_order=",".join(map(str, np.random.default_rng(s).choice(64, size=m, replace=False)))))
        omega = pd.DataFrame(rows)
        detail = omega.rename(columns={"model_seed": "seed"})
        op, dp = tables / "omega_sets.csv", tables / "detailed_results.csv"

        def save(o=omega, d=detail):
            o.to_csv(op, index=False)
            d.to_csv(dp, index=False)

        def reject(cfg=None):
            try:
                checked_measurement_evidence(run, cfg or {}, seeds, rhos)
            except (RuntimeError, FileNotFoundError, KeyError, ValueError):
                return
            raise AssertionError("Invalid measurement evidence was accepted")

        save()
        assert not checked_measurement_evidence(run, {}, seeds, rhos)["config_offset_present"]
        assert checked_measurement_evidence(run, {"measurement_seed_offset": 0}, seeds, rhos)["config_offset_present"]
        reject({"measurement_seed_offset": 1})
        save(o=omega.iloc[:-1]); reject()
        save(o=pd.concat([omega, omega.iloc[:1]])); reject()
        bad = omega.copy(); bad.loc[0, "measurement_seed_offset"] = 1
        save(o=bad); reject()
        bad = omega.copy(); bad.loc[0, "measurement_seed"] = 8
        save(o=bad); reject()
        bad = omega.copy(); bad.loc[0, "omega_order"] = "0,1,2,3"
        save(o=bad); reject()
        bad = detail.copy(); bad.loc[0, "measurement_seed"] = 8
        save(d=bad); reject()
        save(d=detail.iloc[:-1]); reject()
        save(); op.unlink(); reject()
        save(); dp.unlink(); reject()
    print("Figure measurement evidence regression checks passed.")


if __name__ == "__main__":
    main()
