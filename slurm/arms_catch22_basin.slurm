#!/bin/bash
# catch22 baseline: the canonical cheap feature set, 22 features per channel.
#
# A reference point, not a candidate model, so k-fold only -- spatial transfer
# is a separate claim and does not need a baseline. Deterministic, so one seed.
# 5 fits per horizon, 70 in total.

#SBATCH --job-name=sgl_c22b
#SBATCH --output=logs/c22b_%A_%a.out
#SBATCH --account=iharp
#SBATCH --cluster=chip-cpu
#SBATCH --partition=general
#SBATCH --qos=normal
#SBATCH --array=0-13
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=2:00:00

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
df = arms.run_horizon('catch22', h, cfg, splits=('basin',), seeds=(42,), n_jobs=16)
df.to_parquet(f'results/shards/catch22basin_{h}.parquet', index=False)
print('wrote shard', h, len(df), 'rows')
"
echo "end=$(date -Is)"
