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
