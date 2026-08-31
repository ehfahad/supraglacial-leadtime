"""Does MiniROCKET earn its place?

The lead-time claim is only interesting if it reflects information in the data
rather than the capacity of the classifier. The cheapest way to show that is a
baseline a reviewer cannot argue with: per-channel summary statistics of the
truncated series, fed to the same ridge classifier on the same folds.

Six statistics per channel (mean, sd, min, max, last value, OLS slope) over
days <= T. The nine satellite channels give 54 features, against MiniROCKET's
9,996. If the lead-time ordering survives here, it is a property of the record.
If MiniROCKET beats it, the gap is what the convolutional features buy.

Deterministic, so no seed axis: the statistics are fixed given the horizon.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeClassifierCV
from sklearn.metrics import accuracy_score, confusion_matrix

from sgl import classify, config
from sgl.store import ALL_VARS

CLASSES = ["buried", "drain", "drain_slow", "refreeze"]


def summarise(X: np.ndarray) -> np.ndarray:
    """(lakes, days, channels) -> (lakes, channels*6)."""
    n, d, c = X.shape
    t = np.arange(d, dtype=float)
    t = (t - t.mean()) / (t.std() or 1.0)
    # OLS slope against centred time, vectorised over lakes and channels
    slope = (X * t[None, :, None]).mean(axis=1) / (t ** 2).mean()
    return np.concatenate([
        X.mean(axis=1), X.std(axis=1), X.min(axis=1), X.max(axis=1),
        X[:, -1, :], slope,
    ], axis=1)


def main(out="results/simple_baseline.parquet"):
    cfg = config.load()
    tensor, y, folds, regions = classify.load_expert(cfg, "causal")
    idx = [ALL_VARS.index(v) for v in classify.VARIABLE_SETS["sat"]]

    rows = []
    for split in ("kfold", "basin"):
        for horizon in cfg["horizons"]:
            F = summarise(tensor[:, :horizon, idx].astype(np.float64))
            for fold, mask in classify.split_folds(split, folds, regions):
                tr = ~mask
                clf = RidgeClassifierCV(alphas=np.logspace(-3, 3, 10)).fit(F[tr], y[tr])
                pred = clf.predict(F[mask])
                cm = confusion_matrix(y[mask], pred, labels=CLASSES)
                rows.append({
                    "split": split, "horizon": horizon, "fold": fold,
                    "accuracy": float(accuracy_score(y[mask], pred)),
                    **{f"cm_{a}__{b}": int(cm[i, j])
                       for i, a in enumerate(CLASSES) for j, b in enumerate(CLASSES)},
                })
    df = pd.DataFrame(rows)
    df.to_parquet(out, index=False)
    print(f"wrote {out}: {len(df)} fits")
    return df


if __name__ == "__main__":
    main()
