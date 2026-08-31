#!/bin/bash
#SBATCH --job-name=ccm_biwave
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=08:00:00
#SBATCH --array=0-7%1
#SBATCH --output=logs/ccm_biwave_%A_%a.out
#SBATCH --error=logs/ccm_biwave_%A_%a.err
#
# Submit example:
#   sbatch --account=<account_name> --partition=<partition_name> cluster/run_biwave_array.sbatch
#
# This should be run after the CCM scan stage has produced ccm_scan_* or selected_ccm_lags_* files.

set -euo pipefail

PROJECT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "${PROJECT_DIR}"
mkdir -p logs data/processed

module purge
module load python/3.11.11

VENV_DIR="${CCM_VENV_DIR:-${PROJECT_DIR}/.venv_ccm}"
source "${VENV_DIR}/bin/activate"

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export NUMEXPR_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export PYTHONUNBUFFERED=1

TASKS=(
  "spei 1m"
  "spei 3m"
  "spei 6m"
  "spei 12m"
  "spi 1m"
  "spi 3m"
  "spi 6m"
  "spi 12m"
)

read -r TARGET AGGREGATION <<< "${TASKS[$SLURM_ARRAY_TASK_ID]}"

echo "SLURM_JOB_ID=${SLURM_JOB_ID}"
echo "SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID}"
echo "Target=${TARGET}"
echo "Aggregation=${AGGREGATION}"
echo "Started: $(date)"

python scripts/run_multiscale.py --target "${TARGET}" --aggregation "${AGGREGATION}"

echo "Finished: $(date)"
