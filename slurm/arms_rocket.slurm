#!/bin/bash
# ROCKET as a full arm, on exactly the cells MiniROCKET was run on.
#
# Not a baseline row: same 14 horizons, same 5 seeds, same 11 folds (5 k-fold +
# 6 held-out basins), same satellite variables, same causal pipeline. 770 fits.
# The question is whether the four T* dates are transform-invariant. If ROCKET
# reproduces them, the lead-time finding is a property of the record; if it
# moves them, the paper's central claim depends on an untested transform
# choice.
#
# One array task per horizon: 55 fits each. ROCKET is ~10x MiniROCKET per fit,
# so a single-node sequential run would be ~9 h; sharding across 14 tasks puts
# wall clock near 2 h. Shards merge afterwards.
#
# CHIP: --time must be set explicitly (DEFAULTTIME is 15 min) and qos=medium
# allows 1 day, comfortably over the 3.5 h request.

#SBATCH --job-name=sgl_rocket
#SBATCH --output=logs/rocket_%A_%a.out
#SBATCH --account=iharp
#SBATCH --cluster=chip-cpu
#SBATCH --partition=general
#SBATCH --qos=medium
#SBATCH --array=0-13
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=6:00:00

set -euo pipefail
ROOT="${SGL_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}"
cd "$ROOT"
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
echo "start=$(date -Is) host=$(hostname) task=${SLURM_ARRAY_TASK_ID}"
"${PYTHON:-python}" -u -c "
import os, warnings; warnings.filterwarnings('ignore')
from sgl import arms, config
cfg = config.load()
h = cfg['horizons'][int(os.environ['SLURM_ARRAY_TASK_ID'])]
print('horizon', h, flush=True)
df = arms.run_horizon('rocket', h, cfg, splits=('kfold','basin'),
                      seeds=cfg['seeds'], n_jobs=16)
df.to_parquet(f'results/shards/rocket_{h}.parquet', index=False)
print('wrote shard', h, len(df), 'rows')
"
echo "end=$(date -Is)"
