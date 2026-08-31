"""Two-way variance decomposition of every input channel over the melt window.

This is the evidence behind the CARRA ablation: two of the five reanalysis
channels are almost static within a lake across the season, so they act as lake
descriptors rather than as temporal signal.

Each channel's variance is split three ways over DOY 121-288:

    day       the seasonal cycle every lake shares. Carries NO discriminative
              signal: it is identical for every lake on a given date.
    lake      a static per-lake offset. Discriminative, but constant in time,
              so it behaves like elevation rather than like a time series.
    residual  lake-specific deviation from both. The genuinely dynamic,
              genuinely discriminative part.

Also reports how much of each CARRA channel a linear model can reconstruct from
the nine satellite channels, which tests the redundancy explanation.
"""
from __future__ import annotations

import numpy as np

from sgl import config, store
from sgl.preprocess import DERIVED
from sgl.store import ALL_VARS

MELT = (121, 288)


def decompose(x: np.ndarray) -> tuple[float, float, float]:
    """(day, lake, residual) shares of total variance for one (lakes, days) panel."""
    n, d = x.shape
    g = x.mean()
    day = x.mean(axis=0) - g
    lake = x.mean(axis=1) - g
    res = x - g - day[None, :] - lake[:, None]
    v = np.array([(day ** 2).sum() * n, (lake ** 2).sum() * d, (res ** 2).sum()])
    return tuple(v / v.sum())


def main():
    cfg = config.load()
    ds = store.load(cfg, 2019, "causal")
    keep = ds.expert_label.values != "N/A"
    lo, hi = MELT
    X = store.to_tensor(ds, ALL_VARS)[keep][:, lo - 1:hi, :].astype(np.float64)
    carra = set(cfg["carra_vars"])

    print(f"{X.shape[0]} expert lakes x {X.shape[1]} days (DOY {lo}-{hi}) x {X.shape[2]} channels\n")
    print(f"{'channel':<12}{'day':>9}{'lake':>9}{'resid':>9}{'discriminative':>17}")
    print("-" * 56)
    for k, v in enumerate(ALL_VARS):
        day, lake, res = decompose(X[:, :, k])
        tag = "  <- CARRA" if v in carra else ""
        print(f"{v:<12}{100*day:>8.1f}%{100*lake:>8.1f}%{100*res:>8.1f}%"
              f"{100*(lake+res):>16.1f}%{tag}")

    print("\nLinear reconstruction of each CARRA channel from the 9 satellite channels:")
    sat = [ALL_VARS.index(v) for v in DERIVED]
    A = np.column_stack([X[:, :, sat].reshape(-1, len(sat)),
                         np.ones(X.shape[0] * X.shape[1])])
    for v in cfg["carra_vars"]:
        y = X[:, :, ALL_VARS.index(v)].reshape(-1)
        beta, *_ = np.linalg.lstsq(A, y, rcond=None)
        r2 = 1 - ((y - A @ beta) ** 2).sum() / ((y - y.mean()) ** 2).sum()
        print(f"  {v:<10} R^2 = {r2:.2f}")


if __name__ == "__main__":
    main()
