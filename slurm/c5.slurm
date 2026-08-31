#!/bin/bash
#SBATCH --job-name=sgl_c5_v2
#SBATCH --output=logs/c5v2_%j.out
#SBATCH --account=iharp
#SBATCH --cluster=chip-cpu
#SBATCH --partition=general
#SBATCH --qos=medium
#SBATCH --nodes=1 --ntasks-per-node=1 --cpus-per-task=8 --mem=32G --time=2:00:00
set -euo pipefail
ROOT="${SGL_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}"
cd "$ROOT"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
echo "start=$(date -Is) host=$(hostname)"
"${PYTHON:-python}" -u scripts/run_c5.py
echo "end=$(date -Is)"
