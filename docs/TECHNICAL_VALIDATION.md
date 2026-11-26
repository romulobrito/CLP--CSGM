# RELEASE-05a: technical validation

Source inspection of the RELEASE-04 commit covered 40 Python files and seven shell scripts. Explicit import names resolve to local modules, the standard library, listed requirements or optional legacy dependencies. The inspected literal launcher invocations (13 matched commands) use declared options. These observations are not Python compilation, shell parsing or runtime import tests.

Run on a checkout of the preparation branch:

```bash
python scripts/check_release.py --static-only --report outputs/validation/static.json
python scripts/check_release.py --report outputs/validation/technical.json
```

The checker compiles the source without running its main entry points, checks bash syntax, inventories AST imports and literal shell launcher options, loads each module in an isolated subprocess, records installed package versions, and runs selected CLI help commands. Optional legacy SPGL1 dependencies are not required for the main CLP-CSGM path. It performs no training and does not establish numerical reproduction.

## Executed result: PASS

GitHub Actions run [45893012576](https://github.com/romulobrito/CLP--CSGM/actions/runs/45893012576), tested commit `38da23de0fae4eaec37654be3fc2b2894539d948`, completed successfully on 26 November 2026. Static: 96/96; full technical checks: 148/148 (includes static checks), zero failures. pip check: no broken requirements. Python 3.10.22; NumPy 1.26.4, pandas 2.3.3, Matplotlib 3.7.5, scikit-learn 1.6.1, PyTorch 2.6.0+cpu, PyWavelets 1.4.1, SciPy 1.15.3, openpyxl 3.1.5. [Technical artifact](https://github.com/romulobrito/CLP--CSGM/actions/runs/45893012576/artifacts/15152142739). GitHub Actions infrastructure emitted Node.js deprecation warnings only. This does not test training or paper numerical agreement.

## Final coverage validation (05c.3)

Tested commit `c7fd7348b126fd32ebaf2f68532fa502662ef58d`:
- [Technical run 45897849839](https://github.com/romulobrito/CLP--CSGM/actions/runs/45897849839): 104 static checks, 160 total technical checks (including static checks), zero failures; pip check passed. [Artifact](https://github.com/romulobrito/CLP--CSGM/actions/runs/45897849839/artifacts/15154452658).
- [Synthetic run 45897849853](https://github.com/romulobrito/CLP--CSGM/actions/runs/45897849853): both synthetic benchmarks, 28 required method cells, zero failures. [Artifact](https://github.com/romulobrito/CLP--CSGM/actions/runs/45897849853/artifacts/15154836378).
- Coverage checks passed: B.7 command dry-run (not field execution), 861 index-only audit (18 configurations, 21–27 shared indices, 0.8409090909090909–1.0 fraction), and 861 aggregation self-test on constructed metrics (not paper RMSE).

See HISTORICAL_PROVENANCE.md for the author-reported local audit supplement. The coverage audit is consolidated with explicit exceptions; the full exact-protocol coverage gate is not certified. The 05d.1 reference/comparison preparation can proceed independently, but field reproduction and release claims must respect these exceptions. No merge or tag is authorized by this audit.

Outstanding:
- Exact Lapa configuration (ordered channels, split, noise and other run settings) not supplied in the local audit complement; historical origin non-unique.
- Historical overlap aggregation for cross-well Figures 8 and 24, and original F03 Figure 10 run/seed, not supplied.
- Exact remaining qualitative figure selections and field numerical comparison; no authorized inputs executed here.
