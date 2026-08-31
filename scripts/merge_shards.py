"""Collect the array-job shards into the result tables the figures read.

The cluster scripts write one shard per task under ``results/shards/``. This
merges each arm into the single file the paper's tables are computed from.
Running it twice is harmless.
"""
from sgl import arms

ARMS = [
    ("results/shards/rocket_*.parquet", "results/arm_rocket.parquet"),
    ("results/shards/catch22_*.parquet", "results/baseline_catch22.parquet"),
    ("results/shards/catch22basin_*.parquet", "results/baseline_catch22_basin.parquet"),
    ("results/shards/teaser_*.parquet", "results/arm_teaser_all.parquet"),
]


def main() -> None:
    for pattern, out in ARMS:
        arms.merge(pattern, out)


if __name__ == "__main__":
    main()
