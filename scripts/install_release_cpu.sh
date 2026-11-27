#!/usr/bin/env bash
# Install the version-frozen CPU environment into a NEW local virtualenv.
set -euo pipefail
cd "$(dirname "$0")/.."
if [[ -e .venv-release || -L .venv-release ]]; then
  echo "Refusing to modify existing .venv-release; use a fresh checkout or move it first." >&2
  exit 2
fi
python - <<'PY'
import platform, sys
if sys.version_info[:3] != (3, 10, 22) or platform.system() != "Linux" or platform.machine() != "x86_64":
    raise SystemExit("The reference lock requires CPython 3.10.22 on Linux x86_64.")
if platform.python_implementation() != "CPython":
    raise SystemExit("CPython required.")
PY
python -m venv .venv-release
source .venv-release/bin/activate
python -m pip install --index-url https://pypi.org/simple --only-binary=:all: pip==26.3 setuptools==79.0.1
python -m pip install --index-url https://download.pytorch.org/whl/cpu --only-binary=:all: --no-deps 'torch==2.6.0+cpu'
python -m pip install --index-url https://pypi.org/simple --only-binary=:all: -r requirements-release-cpu.txt
python -m pip check
mkdir -p outputs/validation
python -m pip freeze --all > outputs/validation/pip-freeze-release.txt
python scripts/check_release_environment.py
