#!/bin/bash
# Cross-year validation: does the 2019 lead-time curve hold in 2018?
#
# 350 fits (14 horizons x 5 seeds x 5 folds). Each trains on 800 expert lakes
# and scores three test sets: the 200 held-out expert lakes, all 5146 2019
# machine-only lakes, and all 3846 2018 lakes.
#
# The earlier version subsampled the two external sets to 500 lakes per class
# to make accuracy comparable across their opposite skews. That discarded 4992
# lakes to estimate a quantity with a closed form: balanced accuracy is the
# mean of the per-class recalls, so macro-recall on the full set measures the
# same thing with every lake and less variance. Both are now reported.
#
# The external sets are 2.25x larger than before and there are 14 horizons
# rather than 10, so budget ~30 min. Wall 6 h is ample.

#SBATCH --job-name=sgl_crossyear
#SBATCH --output=logs/xy_%j.out
#SBATCH --account=iharp
#SBATCH --cluster=chip-cpu
#SBATCH --partition=general
#SBATCH --qos=medium
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=6:00:00

set -euo pipefail
ROOT="${SGL_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}"
cd "$ROOT"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
echo "start=$(date -Is) host=$(hostname) cpus=${SLURM_CPUS_PER_TASK}"
"${PYTHON:-python}" -u scripts/crossyear.py 14
echo "end=$(date -Is)"
