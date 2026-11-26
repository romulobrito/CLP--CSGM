# RELEASE-05a: technical validation

Source inspection of the RELEASE-04 commit covered 40 Python files and seven shell scripts. Explicit import names resolve to local modules, the standard library, listed requirements or optional legacy dependencies. The inspected literal launcher invocations (13 matched commands) use declared options. These observations are not Python compilation, shell parsing or runtime import tests.

Run on a checkout of the preparation branch:

```bash
python scripts/check_release.py --static-only --report outputs/validation/static.json
python scripts/check_release.py --report outputs/validation/technical.json
```

The checker compiles the source without running its main entry points, checks bash syntax, inventories AST imports and literal shell launcher options, loads each module in an isolated subprocess, records installed package versions, and runs selected CLI help commands. Optional legacy SPGL1 dependencies are not required for the main CLP-CSGM path. It performs no training and does not establish numerical reproduction.

The local execution environment could not clone GitHub (DNS resolution failure) and uses Python 3.13.5, not the reported 3.10 environment. The checks have not yet been executed on the repository. Do not mark RELEASE-05a passed until an actual report is available. Do not publish field data or local prediction outputs.
