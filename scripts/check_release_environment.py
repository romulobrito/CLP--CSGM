#!/usr/bin/env python3
"""Verify exact installed distribution versions against the local field snapshot."""
from __future__ import annotations
import importlib.metadata
import json
import platform
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def canonical(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def main() -> None:
    expected = {}
    for line in (ROOT / "requirements-release-cpu.txt").read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        name, version = line.split("==")
        key = canonical(name)
        if key in expected:
            raise ValueError("Duplicate lock entry: " + key)
        expected[key] = version
    actual = {}
    for dist in importlib.metadata.distributions():
        key = canonical(dist.metadata["Name"])
        if key in actual:
            raise ValueError("Duplicate installed distribution: " + key)
        actual[key] = dist.version
    missing = sorted(set(expected) - set(actual))
    extra = sorted(set(actual) - set(expected))
    different = {k: {"expected": expected[k], "actual": actual[k]}
                 for k in sorted(set(actual) & set(expected)) if actual[k] != expected[k]}
    platform_ok = (sys.version_info[:3] == (3, 10, 22)
                   and platform.python_implementation() == "CPython"
                   and platform.system() == "Linux" and platform.machine() == "x86_64")
    ok = platform_ok and len(expected) == 33 and not (missing or extra or different)
    report = {"status": "VERSION_MATCH" if ok else "VERSION_MISMATCH",
              "python": platform.python_version(), "platform": platform.platform(),
              "expected_count": len(expected), "actual_count": len(actual),
              "missing": missing, "extra": extra, "different": different,
              "installed": actual,
              "scope": "Package versions only; wheel hashes and hardware equivalence not certified."}
    out = ROOT / "outputs/validation/release-environment-check.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    if not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
