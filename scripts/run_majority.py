"""T1, the majority-class floor.

No fitting: predict the most frequent training class for every test lake. The
prediction ignores the input series entirely, so the result does not depend on
the cutoff and one row per (split, fold) covers all 14 horizons.

``arms.majority_floor`` gives accuracy and the predicted class; we add the
confusion matrix in the same ``cm_true__pred`` schema the other arms use, so
macro-recall is computable the same way for every learner.
"""
import numpy as np
import pandas as pd

from sgl import arms, classify, config

CLASSES = ["buried", "drain", "drain_slow", "refreeze"]


def main() -> None:
    cfg = config.load()
    exact = arms.majority_floor(cfg)

    tensor, y, folds, regions = classify.load_expert(cfg, "causal")
    rows = []
    for split in ("kfold", "basin"):
        for fold, mask in classify.split_folds(split, folds, regions):
            top = pd.Series(y[~mask]).value_counts().idxmax()
            row = {"transform": "majority", "split": split, "fold": fold,
                   "predicted": top, "n_train": int((~mask).sum()),
                   "n_test": int(mask.sum()),
                   "accuracy": float((y[mask] == top).mean())}
            for t in CLASSES:
                for p in CLASSES:
                    row[f"cm_{t}__{p}"] = int(((y[mask] == t) & (p == top)).sum())
            rows.append(row)
    out = pd.DataFrame(rows)

    merged = exact.merge(out[["split", "fold", "accuracy"]],
                         on=["split", "fold"], suffixes=("_arms", "_here"))
    delta = (merged.accuracy_arms - merged.accuracy_here).abs().max()
    print(f"max accuracy disagreement with arms.majority_floor: {delta:.2e}")

    out.to_parquet("results/arm_majority.parquet", index=False)
    print(f"wrote results/arm_majority.parquet: {len(out)} rows")

    for split, g in out.groupby("split"):
        rec = []
        for c in CLASSES:
            num = g[f"cm_{c}__{c}"].sum()
            den = sum(g[f"cm_{c}__{p}"].sum() for p in CLASSES)
            rec.append(num / den if den else np.nan)
        pooled_acc = (g.accuracy * g.n_test).sum() / g.n_test.sum()
        print(f"{split:6s} macro-recall {100 * np.nanmean(rec):.1f}%  "
              f"accuracy {100 * pooled_acc:.1f}%  "
              f"predicted {sorted(set(g.predicted))}")


if __name__ == "__main__":
    main()
