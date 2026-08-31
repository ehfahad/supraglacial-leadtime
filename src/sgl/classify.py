"""The truncated sweep: how early can each outcome be known?

For each cutoff day T, train a classifier on ``tensor[:, :T]`` and score it.
Sweeping T gives the lead-time curve, and the day each class plateaus is its
``T*``.

The sweep is a four-way cross product, and each axis answers something:

``horizon``   the question itself -- what is knowable by day T
``family``    a 2x2: variable set (satellite / satellite+CARRA) crossed with
              pipeline (causal / published). The first tells us whether the
              reanalysis forcing buys lead time; the second is the leakage
              ablation, since the published pipeline can see past T. The paper
              calls the two chains leakage-free and conventional.
``split``     kfold uses the source record's own 5 folds, which keeps the
              comparison against the released model on the same lakes; basin
              is leave-one-basin-out, which asks whether the model generalises
              to a region it has never seen.
``seed``      MiniROCKET's random dilations and biases. Without this axis a
              two-point accuracy difference between adjacent horizons cannot be
              distinguished from kernel noise.

One rule governs the whole file: **the transform is fitted on training lakes
only**. MiniROCKET's biases are quantiles of the training data, so fitting it on
all 1000 lakes would leak test-set distribution into training -- a small effect,
and a fatal one to concede in a paper about leakage.
"""

from __future__ import annotations

import time
from pathlib import Path
from itertools import product

import numpy as np
import pandas as pd

from .store import ALL_VARS
from .preprocess import DERIVED

VARIABLE_SETS = {"sat": DERIVED, "sat_carra": ALL_VARS}
PIPELINES = ["causal", "published"]


def families() -> list[tuple[str, str]]:
    """The 2x2: (variable set, pipeline)."""
    return list(product(VARIABLE_SETS, PIPELINES))


def load_expert(cfg, pipeline: str):
    """Expert-labelled lakes only, as (tensor, labels, folds, regions).

    Expert labels exist for 2019 alone, so the sweep is a single-season analysis. The
    tensor keeps all 14 variables; ``VARIABLE_SETS`` slices it per family.
    """
    from . import store

    ds = store.load(cfg, 2019, pipeline)
    keep = ds.expert_label.values != "N/A"
    tensor = store.to_tensor(ds, ALL_VARS)[keep]
    return (
        tensor,
        ds.expert_label.values[keep],
        ds.fold.values[keep].astype(int),
        ds.region.values[keep],
    )


def split_folds(split: str, folds: np.ndarray, regions: np.ndarray):
    """Yield ``(fold_name, test_mask)`` for one splitting scheme."""
    if split == "kfold":
        for f in sorted(np.unique(folds)):
            yield str(f), folds == f
    elif split == "basin":
        for r in sorted(np.unique(regions)):
            yield r, regions == r
    else:
        raise ValueError(f"unknown split {split!r}")


def _transform(X_train, X_test, num_kernels: int, seed: int):
    """MiniROCKET, fitted on training lakes only."""
    from sktime.transformations.panel.rocket import MiniRocketMultivariate

    mr = MiniRocketMultivariate(num_kernels=num_kernels, random_state=seed)
    return np.asarray(mr.fit_transform(X_train)), np.asarray(mr.transform(X_test))


def run_one(tensor, y, test_mask, variables, horizon, seed, cfg) -> dict:
    """One fit: slice at the horizon, transform, classify, score."""
    from sklearn.linear_model import RidgeClassifierCV
    from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                                 recall_score)

    idx = [ALL_VARS.index(v) for v in variables]
    # (n_lakes, channels, time) is what sktime's panel transformers expect.
    X = np.ascontiguousarray(tensor[:, :horizon, :][:, :, idx].transpose(0, 2, 1))
    train = ~test_mask

    conf = cfg["classify"]
    lo, hi, n = conf["alphas_log10"]
    t0 = time.time()
    Xtr, Xte = _transform(X[train], X[test_mask], conf["num_kernels"], seed)
    clf = RidgeClassifierCV(alphas=np.logspace(lo, hi, n)).fit(Xtr, y[train])
    pred = clf.predict(Xte)
    truth = y[test_mask]

    classes = sorted(np.unique(y))
    recalls = recall_score(truth, pred, labels=classes, average=None, zero_division=0)
    # Full confusion, flattened. Recall alone says a class was missed; it does not
    # say what the classifier said instead, and "what instead" is what turns the
    # lead-time curve into an explanation. One row per (true, predicted) pair,
    # counts not rates, so folds of unequal size can be summed.
    cm = confusion_matrix(truth, pred, labels=classes)
    return {
        "accuracy": float(accuracy_score(truth, pred)),
        "macro_f1": float(f1_score(truth, pred, average="macro", zero_division=0)),
        **{f"recall_{c}": float(r) for c, r in zip(classes, recalls)},
        **{f"cm_{t}__{p}": int(cm[i, j])
           for i, t in enumerate(classes) for j, p in enumerate(classes)},
        "n_train": int(train.sum()),
        "n_test": int(test_mask.sum()),
        "alpha": float(clf.alpha_),
        "seconds": round(time.time() - t0, 2),
    }


def sweep_jobs(cfg) -> list[dict]:
    """Every (horizon, family, split, fold, seed) combination, in a stable order."""
    jobs = []
    for varset, pipeline in families():
        for split in cfg["classify"]["splits"]:
            for horizon in cfg["horizons"]:
                for seed in cfg["seeds"]:
                    jobs.append(
                        {
                            "varset": varset,
                            "pipeline": pipeline,
                            "split": split,
                            "horizon": horizon,
                            "seed": seed,
                        }
                    )
    return jobs


KEY = ["varset", "pipeline", "split", "horizon", "seed", "fold"]


def completed(path) -> set:
    """Keys already present in a partial sweep, so a rerun resumes."""
    path = Path(path)
    if not path.exists():
        return set()
    done = pd.read_parquet(path)
    return set(map(tuple, done[KEY].astype(str).to_numpy()))


# --------------------------------------------------------------------------
# sweep driver
# --------------------------------------------------------------------------


def _one(job: dict, tensor, y, test_mask, cfg) -> dict:
    """Run one fit and return its result row, or record the failure.

    A crashed fit must not abort the sweep: 2200 fits over several hours will
    hit something, and losing the completed work to one traceback is the
    expensive failure mode. The row is written either way, with the error text,
    so a gap is visible rather than silent.
    """
    variables = VARIABLE_SETS[job["varset"]]
    try:
        metrics = run_one(tensor, y, test_mask, variables, job["horizon"], job["seed"], cfg)
        return {**job, **metrics, "error": ""}
    except Exception as exc:  # noqa: BLE001 -- deliberately broad, see docstring
        return {**job, "error": f"{type(exc).__name__}: {exc}"}


def sweep(cfg, out_path, n_jobs: int | None = None, chunk: int = 48, verbose: bool = True):
    """Run the full sweep, resuming from a partial result file.

    Results are written after every chunk rather than at the end, so a crash at
    fit 2000 costs one chunk instead of the run.

    ``out_path`` may be a string or a Path; it is coerced here so a caller that
    passes a plain string does not fail four lines later inside ``completed``.
    """
    from joblib import Parallel, delayed

    out_path = Path(out_path)
    n_jobs = n_jobs or cfg["classify"]["n_jobs"]
    loaded = {p: load_expert(cfg, p) for p in PIPELINES}

    # Expand each job group over its folds.
    expanded = []
    for job in sweep_jobs(cfg):
        _, _, folds, regions = loaded[job["pipeline"]]
        for fold_name, mask in split_folds(job["split"], folds, regions):
            expanded.append(({**job, "fold": fold_name}, mask))

    done = completed(out_path)
    todo = [(j, m) for j, m in expanded if tuple(str(j[k]) for k in KEY) not in done]
    if verbose:
        print(f"{len(expanded)} fits total, {len(done)} already done, {len(todo)} to run", flush=True)

    rows = pd.read_parquet(out_path).to_dict("records") if out_path.exists() else []
    out_path.parent.mkdir(parents=True, exist_ok=True)

    for start in range(0, len(todo), chunk):
        batch = todo[start : start + chunk]
        t0 = time.time()
        results = Parallel(n_jobs=n_jobs, backend="loky")(
            delayed(_one)(job, loaded[job["pipeline"]][0], loaded[job["pipeline"]][1], mask, cfg)
            for job, mask in batch
        )
        rows.extend(results)
        pd.DataFrame(rows).to_parquet(out_path, index=False)
        if verbose:
            failed = sum(1 for r in results if r.get("error"))
            print(
                f"  {min(start + chunk, len(todo)):5d}/{len(todo)}  "
                f"[{time.time() - t0:5.0f}s]  failures {failed}",
                flush=True,
            )
    return pd.DataFrame(rows)
