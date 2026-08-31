"""Alternative transforms, run on exactly the cells MiniROCKET was run on.

Two different questions live here, and they are kept apart on purpose.

**Arms** are candidate headline models. ROCKET gets the same cutoffs, folds and
seeds as MiniROCKET, so the two are directly comparable and either could carry
the paper. ROCKET is MiniROCKET's predecessor; if the lead-time dates are
unchanged between them, the finding is not an artifact of one transform.

**Baselines** are reference points, not candidates. They answer "would something
simpler do this?". catch22 is the canonical cheap feature set; the
majority-class floor is exact and needs no fitting at all.

TEASER is neither: it is the early-classification comparison, and it commits
once per lake at a date of its own choosing rather than at a fixed cutoff.
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd

from . import classify
from .store import ALL_VARS

CLASSES = ["buried", "drain", "drain_slow", "refreeze"]
ALPHAS = np.logspace(-3, 3, 10)


# -------------------------------------------------------------------- transforms


def _rocket(X_train, X_test, seed, n_jobs):
    from sktime.transformations.panel.rocket import Rocket

    r = Rocket(num_kernels=10000, random_state=seed, n_jobs=n_jobs)
    return np.asarray(r.fit_transform(X_train)), np.asarray(r.transform(X_test))


def _catch22(X_train, X_test, seed, n_jobs):
    """22 canonical features per channel. Deterministic; `seed` is ignored.

    NaNs appear when a feature is undefined for a constant series, which happens
    on the earliest horizons before a lake has been observed. They are zeroed
    rather than dropped, so the feature matrix keeps a fixed width across folds.
    """
    from sktime.transformations.panel.catch22 import Catch22

    c = Catch22()
    a = np.nan_to_num(np.asarray(c.fit_transform(X_train), dtype=float))
    b = np.nan_to_num(np.asarray(c.transform(X_test), dtype=float))
    return a, b


TRANSFORMS = {"rocket": _rocket, "catch22": _catch22}


# -------------------------------------------------------------------- one fit


def run_one(transform, tensor, y, test_mask, variables, horizon, seed, n_jobs) -> dict:
    from sklearn.linear_model import RidgeClassifierCV
    from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

    idx = [ALL_VARS.index(v) for v in variables]
    X = np.ascontiguousarray(tensor[:, :horizon, :][:, :, idx].transpose(0, 2, 1))
    train = ~test_mask

    t0 = time.time()
    Xtr, Xte = TRANSFORMS[transform](X[train], X[test_mask], seed, n_jobs)
    clf = RidgeClassifierCV(alphas=ALPHAS).fit(Xtr, y[train])
    pred = clf.predict(Xte)
    truth = y[test_mask]
    cm = confusion_matrix(truth, pred, labels=CLASSES)
    return {
        "accuracy": float(accuracy_score(truth, pred)),
        "macro_f1": float(f1_score(truth, pred, average="macro", zero_division=0)),
        **{f"cm_{t}__{p}": int(cm[i, j])
           for i, t in enumerate(CLASSES) for j, p in enumerate(CLASSES)},
        "n_train": int(train.sum()),
        "n_test": int(test_mask.sum()),
        "n_features": int(Xtr.shape[1]),
        "alpha": float(clf.alpha_),
        "seconds": round(time.time() - t0, 2),
    }


# -------------------------------------------------------------------- drivers


def run_horizon(transform, horizon, cfg, splits, seeds, n_jobs=16) -> pd.DataFrame:
    """Every (split, fold, seed) for one horizon. One SLURM array task."""
    tensor, y, folds, regions = classify.load_expert(cfg, "causal")
    variables = classify.VARIABLE_SETS["sat"]

    rows = []
    for split in splits:
        for fold, mask in classify.split_folds(split, folds, regions):
            for seed in seeds:
                job = {"transform": transform, "varset": "sat", "pipeline": "causal",
                       "split": split, "horizon": horizon, "seed": seed, "fold": fold}
                try:
                    rows.append({**job,
                                 **run_one(transform, tensor, y, mask, variables,
                                           horizon, seed, n_jobs),
                                 "error": ""})
                except Exception as exc:  # noqa: BLE001 -- one bad fit must not lose the task
                    rows.append({**job, "error": f"{type(exc).__name__}: {exc}"})
                r = rows[-1]
                print(f"  [{horizon}] {split}/{fold}/s{seed} "
                      f"acc={r.get('accuracy', float('nan')):.3f} "
                      f"{r.get('seconds', 0):.0f}s {r['error']}", flush=True)
    return pd.DataFrame(rows)


def majority_floor(cfg) -> pd.DataFrame:
    """Exact, no fitting: predict the most common training class every time."""
    tensor, y, folds, regions = classify.load_expert(cfg, "causal")
    rows = []
    for split in ("kfold", "basin"):
        for fold, mask in classify.split_folds(split, folds, regions):
            train_y = y[~mask]
            top = pd.Series(train_y).value_counts().idxmax()
            rows.append({"transform": "majority", "split": split, "fold": fold,
                         "accuracy": float((y[mask] == top).mean()),
                         "predicted": top, "n_test": int(mask.sum())})
    return pd.DataFrame(rows)


def merge(pattern: str, out: str) -> pd.DataFrame:
    """Collect array-task shards into one file."""
    shards = sorted(Path().glob(pattern))
    df = pd.concat([pd.read_parquet(p) for p in shards], ignore_index=True)
    df.to_parquet(out, index=False)
    print(f"merged {len(shards)} shards -> {out}: {len(df)} rows, "
          f"{df.error.ne('').sum() if 'error' in df else 0} errors")
    return df


# -------------------------------------------------------------------- TEASER

# TEASER (Schaefer and Leser 2020) is the early-classification comparison, not
# another baseline. It answers a different question: we fix a date and ask what
# is knowable, it watches a lake and decides per lake when it has seen enough
# to commit.
#
# Decision dates are recovered by replaying the season: at each classification
# point the model is shown the series up to that point, and a lake's decision
# date is the first point at which it reports a decision. Lakes never decided
# fall back to the last point, which is how TEASER is scored in its own paper.


def run_teaser(cfg, fold, split="kfold", num_kernels=2000, seed=42, n_jobs=16) -> pd.DataFrame:
    from sktime.classification.early_classification import TEASER
    from sktime.classification.kernel_based import RocketClassifier

    tensor, y, folds, regions = classify.load_expert(cfg, "causal")
    idx = [ALL_VARS.index(v) for v in classify.VARIABLE_SETS["sat"]]
    horizons = [h for h in cfg["horizons"] if h <= 365]
    points = [h - 1 for h in horizons]                       # 0-based indices

    mask = dict(classify.split_folds(split, folds, regions))[fold]
    X = np.ascontiguousarray(tensor[:, :max(horizons), idx].transpose(0, 2, 1))
    train, test = ~mask, mask

    t0 = time.time()
    clf = TEASER(estimator=RocketClassifier(num_kernels=num_kernels),
                 classification_points=points, random_state=seed, n_jobs=n_jobs)
    clf.fit(X[train], y[train])
    print(f"  fitted in {time.time() - t0:.0f}s", flush=True)

    n = int(test.sum())
    decided_at = np.full(n, np.nan)
    final = np.array([None] * n, dtype=object)
    Xtest = X[test]
    # TEASER's streaming contract: once an instance is decided it must be
    # dropped from the array passed to the next update, with row order kept.
    # `remaining` maps surviving rows back to their original test-set position.
    remaining = np.arange(n)
    first = True
    for h, p_ in zip(horizons, points):
        if remaining.size == 0:
            break
        Xs = Xtest[remaining][:, :, :p_ + 1]
        pred, dec = clf.predict(Xs) if first else clf.update_predict(Xs)
        first = False
        pred = np.asarray(pred, object)
        dec = np.asarray(dec, bool)
        decided_at[remaining[dec]] = h
        final[remaining[dec]] = pred[dec]
        last_pred, last_idx = pred, remaining.copy()
        remaining = remaining[~dec]
        print(f"  point {h}: {int(dec.sum())} decided, {remaining.size} still open",
              flush=True)

    # Lakes TEASER never commits on are scored at the final point using its
    # last prediction for them, which is how its own paper reports them.
    undecided = np.isnan(decided_at)
    if undecided.any():
        pos = {o: i for i, o in enumerate(last_idx)}
        final[undecided] = [last_pred[pos[o]] for o in np.flatnonzero(undecided)]
        decided_at[undecided] = horizons[-1]
    return pd.DataFrame({
        "split": split, "fold": fold, "seed": seed, "num_kernels": num_kernels,
        "truth": y[test], "pred": final.astype(str),
        "decided_doy": decided_at.astype(int),
        "was_decided": ~undecided, "seconds": round(time.time() - t0, 1),
    })
