#!/bin/bash
# Set up a Python environment for CCM runs on the UIUC Campus Cluster.
# Run this on a login node. Heavy CCM execution should still be submitted with sbatch.

set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="${CCM_VENV_DIR:-${PROJECT_DIR}/.venv_ccm}"

cd "${PROJECT_DIR}"

module purge
module load python/3.11.11

python -m venv "${VENV_DIR}"
source "${VENV_DIR}/bin/activate"

python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements.txt

python - <<'PY'
import importlib
mods = ["numpy", "pandas", "scipy", "matplotlib", "seaborn", "statsmodels", "sklearn", "pywt", "pyEDM", "tqdm", "openpyxl"]
missing = [m for m in mods if importlib.util.find_spec(m) is None]
if missing:
    raise SystemExit(f"Missing packages after install: {missing}")
print("CCM cluster environment is ready.")
PY
