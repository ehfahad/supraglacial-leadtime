#!/bin/bash
# TEASER, the early-classification comparison: 5 seeds x 11 folds = 55 runs,
# one task each, producing the 10,000 per-lake decisions the paper reports.
#
# num_kernels is 2,000 rather than the 10,000 used elsewhere. TEASER fits both a
# base classifier and a one-class classifier with a parameter grid at each of
# the 14 decision points, so the full kernel count costs roughly five times as
# much without changing the comparison. It is recorded in the output.

#SBATCH --job-name=sgl_teaser
#SBATCH --output=logs/teaser_%A_%a.out
#SBATCH --account=iharp
#SBATCH --cluster=chip-cpu
#SBATCH --partition=general
#SBATCH --qos=medium
#SBATCH --array=0-54
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=8:00:00

set -euo pipefail
ROOT="${SGL_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}"
cd "$ROOT"
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
echo "start=$(date -Is) host=$(hostname) task=${SLURM_ARRAY_TASK_ID}"
"${PYTHON:-python}" -u -c "
import os, warnings; warnings.filterwarnings('ignore')
from pathlib import Path
from sgl import arms, config

JOBS = [(s, sp, f) for s in (42, 43, 44, 45, 46)
        for sp, folds in (('kfold', [str(i) for i in range(5)]),
                          ('basin', ['CW', 'NE', 'NO', 'NW', 'SE', 'SW']))
        for f in folds]

seed, split, fold = JOBS[int(os.environ['SLURM_ARRAY_TASK_ID'])]
out = Path(f'results/shards/teaser_{split}_{fold}_s{seed}.parquet')
if out.exists():
    print('shard already present:', out); raise SystemExit(0)
out.parent.mkdir(parents=True, exist_ok=True)

cfg = config.load()
df = arms.run_teaser(cfg, fold=fold, split=split, num_kernels=2000,
                     seed=seed, n_jobs=16)
df.to_parquet(out, index=False)
print('wrote', out, len(df), 'rows')
"
echo "end=$(date -Is)"
