#!/bin/bash
# The truncated sweep: 14 calendar-aligned cutoffs, confusion matrices for
# every cell.
# v2 puts every horizon on the 1st or 15th of a month from 1 May, keeping only
# 31 Oct (melt-season end) and 31 Dec (full-year reference) off that pattern.
#
# v1 also stored per-class recall but no precision, which quietly forced every
# per-class analysis to be recall-based. Measured consequence: `buried` reads
# 80.3% recall at 1 June but only 64.7% precision, so a headline claim was
# one-sided. `run_one` now emits the full confusion matrix, from which
# precision, recall and F1 all derive -- and it does so for all eight cells,
# which retires the separate confusion job.
#
# 14 horizons x 2 varsets x 2 pipelines x 2 splits x 5 seeds x (5|6) folds
#   = 3080 fits. v1 ran 2200 in 189 min on 12 workers; the new May horizons are
# cheaper (shorter series), so budget ~4 h. Wall 8 h leaves slack.
# The sweep writes incrementally and resumes, so a timeout costs nothing.

#SBATCH --job-name=sgl_c1_v2
#SBATCH --output=logs/c1v2_%j.out
#SBATCH --account=iharp
#SBATCH --cluster=chip-cpu
#SBATCH --partition=general
#SBATCH --qos=medium
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=8:00:00

set -euo pipefail
ROOT="${SGL_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}"
cd "$ROOT"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
echo "start=$(date -Is) host=$(hostname) cpus=${SLURM_CPUS_PER_TASK}"
"${PYTHON:-python}" -u -c "
from sgl import config, classify
cfg = config.load()
classify.sweep(cfg, 'results/sweep_c1.parquet', n_jobs=14)
"
echo "end=$(date -Is)"
