#!/usr/bin/env python3
"""Technical release checks; no training, field data access or output generation.

Run from any directory using Python 3.10+:
  python scripts/check_release.py --static-only
  python scripts/check_release.py
"""
from __future__ import annotations

import argparse
import ast
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
PACKAGES = ("numpy", "pandas", "matplotlib", "scikit-learn", "torch",
            "PyWavelets", "scipy", "openpyxl")
IMPORTS = {"numpy", "pandas", "matplotlib", "sklearn", "torch", "pywt",
           "scipy", "openpyxl"}
OPTIONAL = {"pylops", "spgl1"}


def run_command(args: list[str], timeout: int = 120) -> str:
    result = subprocess.run(args, cwd=ROOT, text=True, capture_output=True,
                            timeout=timeout, env={**os.environ, "MPLBACKEND": "Agg"})
    if result.returncode:
        raise RuntimeError(f"{args}: exit {result.returncode}\n{result.stdout}\n{result.stderr}")
    return result.stdout


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--static-only", action="store_true")
    parser.add_argument("--report", type=Path, default=None)
    args = parser.parse_args()
    py_files = sorted(ROOT.glob("*.py")) + sorted(SCRIPTS.glob("*.py"))
    sh_files = sorted(SCRIPTS.glob("*.sh"))
    failures: list[dict] = []
    checks: list[dict] = []
    trees: dict[Path, ast.AST] = {}

    def check(category, name, operation):
        try:
            operation()
            checks.append({"category": category, "name": str(name), "status": "pass"})
        except Exception as exc:
            failures.append({"category": category, "name": str(name), "error": str(exc)})
            print(f"FAIL {category} {name}: {exc}", file=sys.stderr)

    for path in py_files:
        def parse(path=path):
            source = path.read_text(encoding="utf-8")
            compile(source, str(path), "exec")
            trees[path] = ast.parse(source, filename=str(path))
        check("python_syntax", path.relative_to(ROOT), parse)
    for path in sh_files:
        check("bash_syntax", path.relative_to(ROOT),
              lambda path=path: run_command(["bash", "-n", str(path)]))

    def module_exists(name):
        top = name.split(".")[0]
        if top in sys.stdlib_module_names or top in IMPORTS or top in OPTIONAL:
            return True
        for base in (ROOT, SCRIPTS):
            candidate = base.joinpath(*name.split("."))
            if candidate.with_suffix(".py").is_file() or candidate.is_dir():
                return True
        return False

    for path, tree in trees.items():
        def imports(tree=tree):
            names = []
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    names.extend(alias.name for alias in node.names)
                elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                    names.append(node.module)
            missing = sorted({name for name in names if not module_exists(name)})
            if missing:
                raise RuntimeError(f"Unresolved explicit imports: {missing}")
        check("import_inventory", path.relative_to(ROOT), imports)

    def declared_flags(tree):
        return {arg.value for node in ast.walk(tree)
                if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                and node.func.attr == "add_argument" for arg in node.args
                if isinstance(arg, ast.Constant) and isinstance(arg.value, str)
                and arg.value.startswith("--")}

    for path in sh_files:
        def launcher_flags(path=path):
            source = path.read_text(encoding="utf-8").replace("\\\n", " ")
            for line in source.splitlines():
                if line.lstrip().startswith("#"):
                    continue
                match = re.search(r"(?:^|\s)python\s+([\w/]+\.py)\b(.*)", line)
                if not match:
                    continue
                target = ROOT / match.group(1)
                if target not in trees:
                    raise RuntimeError(f"Unresolved launcher: {target}")
                flags = set(re.findall(r"--[\w-]+", match.group(2)))
                unknown = flags - declared_flags(trees[target])
                if unknown:
                    raise RuntimeError(f"{target.name}: unknown flags {sorted(unknown)}")
        check("launcher_options", path.relative_to(ROOT), launcher_flags)

    versions = {}
    if not args.static_only:
        for package in PACKAGES:
            def package_version(package=package):
                versions[package] = importlib.metadata.version(package)
            check("dependency", package, package_version)
        # Isolate module-level execution, without invoking __main__ or training.
        for path in py_files:
            if path.resolve() == Path(__file__).resolve():
                continue
            code = ("import runpy,sys; from pathlib import Path; "
                    "p=Path(sys.argv[1]); sys.path.insert(0,str(p.parent)); "
                    "sys.path.insert(0,str(Path.cwd())); "
                    "runpy.run_path(str(p),run_name='__release_import_check__')")
            check("module_load", path.relative_to(ROOT),
                  lambda path=path: run_command([sys.executable, "-c", code, str(path)]))
        for name in ("sir_cs_benchmark_multi_well_vc.py", "sir_cs_benchmark_real_well_direct_ub.py",
                     "scripts/run_r1m11_861_phi_lab.py", "scripts/auddys_smoke_direct_ub.py"):
            check("cli_help", name, lambda name=name: run_command([sys.executable, name, "--help"]))

    report = {"mode": "static-only" if args.static_only else "static-and-imports",
              "python": sys.version, "platform": platform.platform(), "packages": versions,
              "checks": checks, "failures": failures,
              "status": "fail" if failures else "pass",
              "scope": "No model fitting or manuscript numerical reproduction."}
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"{len(checks)} checks passed; {len(failures)} failed; mode={report['mode']}")
    return int(bool(failures))


if __name__ == "__main__":
    raise SystemExit(main())
