"""Does the lead-time curve survive a different melt season?

Everything else in the paper is 2019. 2019 was an extreme melt year; 2018 was
not, and its lakes drained about 18 days later (9 Jul +/- 15 versus 21 Jun +/-
20). So 2018 is a genuinely different regime, not a replicate.

2018 has no expert labels, only the published classifier's. Scoring against
those confounds two things -- the model failing to transfer, and the labels
being noisier than expert ones. Three runs separate them, all sharing the *same
trained models* so nothing differs but the test set:

    A  2019 expert, held out      the main sweep (anchor)
    B  2019 machine-only lakes    same year, noisier labels  -> label-noise cost
    C  2018 machine lakes         different year             -> B->C is the year

    B - A  = cost of imperfect labels
    C - B  = cost of a different year        <- the number a reviewer wants

Both external sets are badly imbalanced and in opposite directions (2019
machine-only is 43% drain_slow; 2018 is 35% refreeze), so raw accuracy is not
comparable across them. Macro-recall is the mean of the per-class recalls, so
computing it on the full imbalanced set estimates the balanced quantity using
every lake. Both are reported, from the same predictions over the full
populations:

    macro-recall     the balanced quantity, for the A/B/C decomposition
    raw accuracy     operational performance under each year's true class mix
"""
from __future__ import annotations

import sys
import time

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from sgl import classify, config, store

CLASSES = ["buried", "drain", "drain_slow", "refreeze"]


def external_sets(cfg, variables):
    """The two held-out populations in full. No subsampling: see the docstring."""
    out = {}

    ds19 = store.load(cfg, 2019, "causal")
    e, m = ds19.expert_label.values, ds19.machine_label.values
    sel = np.flatnonzero((e == "N/A") & (m != "N/A"))   # never seen in training
    out["B_2019_machine"] = (store.to_tensor(ds19, classify.ALL_VARS)[sel], m[sel])
    ds19.close()

    ds18 = store.load(cfg, 2018, "causal")
    m18 = ds18.machine_label.values
    sel18 = np.flatnonzero(m18 != "N/A")
    out["C_2018_machine"] = (store.to_tensor(ds18, classify.ALL_VARS)[sel18], m18[sel18])
    ds18.close()
    return out


def one(tensor, y, test_mask, ext, variables, horizon, seed, fold, cfg):
    """One trained model, scored on the held-out fold and both external sets."""
    from sklearn.linear_model import RidgeClassifierCV
    from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                                 precision_score, recall_score)
    from sktime.transformations.panel.rocket import MiniRocketMultivariate

    idx = [classify.ALL_VARS.index(v) for v in variables]

    def panel(t):
        return np.ascontiguousarray(t[:, :horizon, :][:, :, idx].transpose(0, 2, 1))

    conf = cfg["classify"]
    t0 = time.time()
    train = ~test_mask
    mr = MiniRocketMultivariate(num_kernels=conf["num_kernels"], random_state=seed)
    Xtr = np.asarray(mr.fit_transform(panel(tensor[train])))
    clf = RidgeClassifierCV(alphas=np.logspace(*conf["alphas_log10"][:2],
                                               conf["alphas_log10"][2])).fit(Xtr, y[train])

    rows = []
    targets = {"A_2019_expert": (tensor[test_mask], y[test_mask])}
    targets.update(ext)
    for name, (t_ext, y_ext) in targets.items():
        pred = clf.predict(np.asarray(mr.transform(panel(t_ext))))
        cm = confusion_matrix(y_ext, pred, labels=CLASSES)
        rec = recall_score(y_ext, pred, labels=CLASSES, average=None, zero_division=0)
        pre = precision_score(y_ext, pred, labels=CLASSES, average=None, zero_division=0)
        rows.append({
            "run": name, "horizon": horizon, "seed": seed, "fold": fold,
            # raw accuracy is operational (true class mix); macro-recall is the
            # balanced quantity a class-balanced test set would have estimated.
            "accuracy": float(accuracy_score(y_ext, pred)),
            "macro_recall": float(np.mean(rec)),
            "macro_f1": float(f1_score(y_ext, pred, average="macro", zero_division=0)),
            **{f"recall_{c}": float(r) for c, r in zip(CLASSES, rec)},
            **{f"precision_{c}": float(p) for c, p in zip(CLASSES, pre)},
            **{f"cm_{a}__{b}": int(cm[i, j])
               for i, a in enumerate(CLASSES) for j, b in enumerate(CLASSES)},
            "n_train": int(train.sum()), "n_test": int(len(y_ext)),
            "seconds": round(time.time() - t0, 1),
        })
    return rows


def main(n_jobs=14):
    cfg = config.load()
    variables = classify.VARIABLE_SETS["sat"]
    tensor, y, folds, regions = classify.load_expert(cfg, "causal")
    ext = external_sets(cfg, variables)
    for k, (t, l) in ext.items():
        vals, cnt = np.unique(l, return_counts=True)
        print(f"{k}: {t.shape[0]} lakes, {dict(zip(vals.tolist(), cnt.tolist()))}", flush=True)

    jobs = [(h, s, f, m)
            for h in cfg["horizons"] for s in cfg["seeds"]
            for f, m in classify.split_folds("kfold", folds, regions)]
    sizes = " + ".join(str(t.shape[0]) for t, _ in ext.values())
    print(f"{len(jobs)} fits on {n_jobs} workers "
          f"(each: train 800, score 200 + {sizes})", flush=True)

    out = Parallel(n_jobs=n_jobs, verbose=10)(
        delayed(one)(tensor, y, m, ext, variables, h, s, f, cfg)
        for h, s, f, m in jobs)
    df = pd.DataFrame([r for rows in out for r in rows])
    df.to_parquet("results/crossyear.parquet")
    print(f"wrote results/crossyear.parquet rows={len(df)}", flush=True)


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 14)
