"""Does the released full-season model survive truncation?

The upstream system has three parts:

* a ROCKET classifier on the radar anomaly series  (4 classes)
* a ROCKET classifier on the optical water series  (3 classes; burial excluded)
* a stacked XGBoost that combines their outputs with scalar features

Only the first two consume time series, so only they can be *truncated*. The
stacked model is where the interesting failure lives, and it is not a failure of
degree: three of its inputs cannot be computed before the season is over.

Two distinct claims, and they need separating
---------------------------------------------
1. **Degradation** -- what accuracy do the two ROCKET classifiers lose when we
   hide everything after day T? Measurable, and reported as a curve.
2. **Undefinedness** -- before mid-October the stacked model's ``S1_diff``
   feature does not exist, so the model cannot be evaluated at all. This is not
   an accuracy number; it is a statement about what is computable.

Conflating the two would understate the result: a reader seeing only a
degradation curve concludes the published model is merely worse early, when in
fact it cannot be run.

How truncation is done
----------------------
ROCKET's kernels are fitted to a fixed series length, so a horizon cannot simply
shorten the input. Instead the series is held constant after day T -- last
observation carried forward to the end of the window. That keeps the input shape
the models expect while encoding exactly what an operational system would know
on day T: nothing after it. Padding with anything data-dependent (a mean, a
climatology) would reintroduce the future through the back door.
"""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pandas as pd

# Upstream's window: 1 May to 15 Oct inclusive.
WINDOW_START, WINDOW_END = 121, 288

# From upstream ``prep_labels``.
LABEL_TO_NUM = {"refreeze": 0, "drain": 1, "drain_slow": 2, "buried": 3}
NUM_TO_LABEL = {v: k for k, v in LABEL_TO_NUM.items()}

# The upstream models were trained on folds 1-4 and tested on fold 0, so fold 0
# is the only honest evaluation set: every other lake is their training data.
HELDOUT_FOLD = 0

# Stacked-model inputs, grouped by when they become computable. Read from
# `_b_train_model.ipynb`; the grouping is what makes the truncation test meaningful.
FEATURE_AVAILABILITY = {
    "always": [
        "elevation", "area",
        "s1_0_conf", "s1_1_conf", "s1_2_conf", "s1_3_conf",
        "s2_0_conf", "s2_1_conf", "s2_2_conf",
        "s1_pred", "s2_pred",
    ],
    "needs_full_season": ["max_S2"],                                  # season maximum
    "needs_event": ["drain_time", "S1_res_drain", "S2_res_drain"],    # event must have occurred
    "needs_october": ["S1_diff"],                                     # 15 Oct - 1 Nov radar
}

# When each retrospective group first becomes computable, as a day of year.
AVAILABLE_FROM = {
    "always": 0,
    "needs_full_season": 288,   # 15 Oct, end of the upstream window
    "needs_event": None,        # lake-specific: the event date
    "needs_october": 288,       # the 15 Oct - 1 Nov window opens
}


def load_models(cfg) -> dict:
    """Load whichever upstream artifacts are importable.

    ``discrepency_model.pkl`` needs pycaret and may be absent; the two ROCKET
    stacks load with sktime alone. Missing entries are reported rather than
    raised, so the evaluation runs on what is available.
    """
    d = cfg.zenodo / "OUTPUT" / "models"
    out, missing = {}, {}
    for name in ("S1_rocket", "S1_classfier", "S2_rocket", "S2_classfier", "discrepency_model"):
        try:
            with open(d / f"{name}.pkl", "rb") as fh:
                out[name] = pickle.load(fh)
        except Exception as exc:  # noqa: BLE001
            missing[name] = f"{type(exc).__name__}: {exc}"
    return {"models": out, "missing": missing}


def truncate(series: np.ndarray, horizon: int) -> np.ndarray:
    """Hold every value after ``horizon`` at its day-``horizon`` value.

    ``series`` is ``(n_lakes, n_days)`` over the upstream window, indexed from
    WINDOW_START. A horizon at or beyond the window end leaves it untouched.
    """
    cut = horizon - WINDOW_START + 1
    if cut >= series.shape[1]:
        return series
    out = series.copy()
    out[:, cut:] = out[:, cut - 1][:, None]
    return out


def _panel(series: np.ndarray) -> pd.DataFrame:
    """sktime 1.1 nested-DataFrame format: one column of per-instance Series."""
    return pd.DataFrame({"data": [pd.Series(row) for row in series]})


def score_rocket(rocket, classifier, series: np.ndarray, y_true: np.ndarray) -> dict:
    """Run one upstream ROCKET stack on a (possibly truncated) series."""
    from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

    features = rocket.transform(_panel(series))
    pred = classifier.predict(np.asarray(features))
    # Per-class precision, recall and F1, not just the macro summary. We claim
    # the radar stack "collapses onto one or two classes" early; the per-class
    # numbers show that directly instead of leaving it inferred from macro-F1.
    # `labels` is taken from the truth so S1 (4 classes) and S2 (3, no buried)
    # each get the right matrix.
    labels = sorted(set(np.asarray(y_true).tolist()))
    cm = confusion_matrix(y_true, pred, labels=labels)
    per = {}
    for i, lab in enumerate(labels):
        name = NUM_TO_LABEL.get(lab, str(lab))
        rec = cm[i, i] / cm[i].sum() if cm[i].sum() else 0.0
        pre = cm[i, i] / cm[:, i].sum() if cm[:, i].sum() else 0.0
        per[f"recall_{name}"] = float(rec)
        per[f"precision_{name}"] = float(pre)
        per[f"f1_{name}"] = float(2 * rec * pre / (rec + pre)) if rec + pre else 0.0
    return {
        "accuracy": float(accuracy_score(y_true, pred)),
        "macro_f1": float(f1_score(y_true, pred, average="macro", zero_division=0)),
        "n": int(len(y_true)),
        **per,
        **{f"cm_{NUM_TO_LABEL.get(a, a)}__{NUM_TO_LABEL.get(b, b)}": int(cm[i, j])
           for i, a in enumerate(labels) for j, b in enumerate(labels)},
        # Kept so the score can be resampled without re-running the model.
        "_pred": np.asarray(pred),
        "_truth": np.asarray(y_true),
    }


def bootstrap_ci(y_true, pred, n_boot: int = 1000, seed: int = 0, level: float = 95):
    """Percentile bootstrap over the held-out lakes.

    This scores a *fixed* released model on one held-out fold, so there is no
    seed to vary and no error bar comes for free. What can still be quantified
    is how much the number depends on *which lakes* happened to land in that
    fold: resample the fold with replacement, rescore, repeat, and read off the
    percentiles.

    This is uncertainty from test-set sampling only. It does not capture what
    would change if the upstream model had been retrained -- we do not retrain
    it, by design, because the claim is about the released artefact as
    published.
    """
    from sklearn.metrics import accuracy_score, f1_score

    y_true, pred = np.asarray(y_true), np.asarray(pred)
    rng = np.random.default_rng(seed)
    n = len(y_true)
    acc, f1, degenerate = [], [], 0
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        t, p = y_true[idx], pred[idx]
        if len(np.unique(t)) < len(np.unique(y_true)):
            degenerate += 1          # a class vanished from the resample
        acc.append(accuracy_score(t, p))
        f1.append(f1_score(t, p, average="macro", zero_division=0))
    lo, hi = (100 - level) / 2, 100 - (100 - level) / 2
    return {
        "accuracy_lo": float(np.percentile(acc, lo)),
        "accuracy_hi": float(np.percentile(acc, hi)),
        "accuracy_se": float(np.std(acc, ddof=1)),
        "macro_f1_lo": float(np.percentile(f1, lo)),
        "macro_f1_hi": float(np.percentile(f1, hi)),
        "macro_f1_se": float(np.std(f1, ddof=1)),
        "n_boot": n_boot,
        "n_degenerate_resamples": degenerate,
    }


def load_heldout(cfg):
    """The upstream held-out fold, with the two series their models consume.

    Both come from the **published** pipeline, because that is what the models
    were trained on. Feeding them our causal features would measure a
    distribution shift rather than the effect of truncation.
    """
    from . import store

    ds = store.load(cfg, 2019, "published")
    expert = ds.expert_label.values != "N/A"
    fold = ds.fold.values
    keep = expert & (fold == HELDOUT_FOLD)

    window = slice(WINDOW_START, WINDOW_END + 1)
    return {
        "labels": ds.expert_label.values[keep],
        "hv_anom": ds.HV_anom.values[keep, window.start : window.stop],
        "pwater": ds.pwater.values[keep, window.start : window.stop],
        "event_doy": ds.event_doy.values[keep],
        "lake": ds.lake.values[keep],
    }


def truncation_table(cfg, horizons=None) -> pd.DataFrame:
    """Accuracy of both upstream ROCKET stacks at each horizon."""
    loaded = load_models(cfg)
    models, missing = loaded["models"], loaded["missing"]
    data = load_heldout(cfg)
    horizons = horizons or cfg["horizons"]

    y_all = np.array([LABEL_TO_NUM[l] for l in data["labels"]])
    not_buried = data["labels"] != "buried"

    rows = []
    for horizon in horizons:
        if {"S1_rocket", "S1_classfier"} <= models.keys():
            rows.append({
                "horizon": horizon, "model": "S1_rocket", "classes": 4,
                **score_rocket(models["S1_rocket"], models["S1_classfier"],
                               truncate(data["hv_anom"], horizon), y_all),
            })
        if {"S2_rocket", "S2_classfier"} <= models.keys():
            rows.append({
                "horizon": horizon, "model": "S2_rocket", "classes": 3,
                **score_rocket(models["S2_rocket"], models["S2_classfier"],
                               truncate(data["pwater"][not_buried], horizon),
                               y_all[not_buried]),
            })
    for r in rows:
        r.update(bootstrap_ci(r.pop("_truth"), r.pop("_pred"),
                              n_boot=cfg.get("baseline", {}).get("n_bootstrap", 1000)))
    table = pd.DataFrame(rows)
    table.attrs["missing_models"] = missing
    return table


def availability_table(cfg, horizons=None) -> pd.DataFrame:
    """Which stacked-model features exist at each horizon, and for how many lakes.

    ``needs_event`` is lake-specific -- a feature about
    the drainage exists only once that lake has drained -- so it is reported as
    a fraction rather than a yes/no.
    """
    data = load_heldout(cfg)
    horizons = horizons or cfg["horizons"]
    event = data["event_doy"]

    rows = []
    for horizon in horizons:
        row = {"horizon": horizon}
        for group, from_doy in AVAILABLE_FROM.items():
            if group == "needs_event":
                with np.errstate(invalid="ignore"):
                    row[group] = float(np.nanmean(event <= horizon))
            else:
                row[group] = float(horizon >= from_doy)
        row["stacked_model_runnable"] = bool(
            row["needs_full_season"] and row["needs_october"] and row["needs_event"] == 1.0
        )
        rows.append(row)
    return pd.DataFrame(rows)
