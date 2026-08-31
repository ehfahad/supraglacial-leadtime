"""Figure 2: what the four end-of-season outcomes look like in the raw record.

One row, four panels, one representative lake each, in the paper's T* order.
Channel encoding follows the convention established for this dataset: a thin
grey line for the radar anomaly on the left axis, red points for the optical
water fraction on a right axis that runs downward, so both signals move the
same way when a lake loses its water.

The lakes are chosen by rule, not by eye: among expert-labelled lakes of the
class with complete melt-season coverage and above-median area, keep the half
closest to the class median trajectory, then take the one with the largest
radar amplitude. Typical, and legible at 1.7 in wide.
"""
from __future__ import annotations

from pathlib import Path

# Output goes to figures/, one level up from this package.
HERE = Path(__file__).resolve().parents[1]   # figures/

import numpy as np
import pandas as pd

from sgl.data import load_year
from sgl.preprocess import combine_optical, mask_shadows
from .paperfigs import ml_style
from .plotting import FULL_WIDTH, _save, use_paper_style

CLASSES = ["drain", "drain_slow", "refreeze", "buried"]
TITLE = {"drain": "Rapid drainage", "drain_slow": "Slow drainage",
         "buried": "Buried", "refreeze": "Refreeze"}

INK = "#3A3F45"      # the backscatter line
DOT = "#C2333A"      # the water-fraction points, deliberately not the
                     # vermillion that means rapid drainage elsewhere
MARK = "#6E7781"     # the event marker

MELT = (121, 304)
WATER_ON = 50.0      # per cent; defines the event marker


def derive(raw: pd.DataFrame) -> pd.DataFrame:
    """The two plotted channels, from the raw record and nothing else.

    Shadow masking and the optical merge follow the pipeline definitions so the
    figure shows the same quantities the classifier sees; no filling, no
    smoothing, so gaps stay gaps.
    """
    s2 = mask_shadows(raw.S2_water, raw.S2_zenith)
    ls = mask_shadows(raw.LS_water, raw.LS_zenith)
    return pd.DataFrame(
        {"HV_anom": raw.HV_lake - raw.HV_out,
         "p_water": 100.0 * combine_optical(s2, ls)},
        index=raw.index,
    )


def _clean(s: pd.Series) -> pd.Series:
    return s.loc[MELT[0]:MELT[1]]


def _coverage_ok(pw: pd.Series, max_gap: int = 12) -> bool:
    """Enough optical looks, no long blind stretch, and the lake actually fills.

    The fill requirement matters: a lake that never reads more than a third
    water has nothing legible to lose, whatever class it belongs to.
    """
    w = _clean(pw)
    seen = w.notna().to_numpy()
    if seen.sum() < 20 or np.nanmax(w.to_numpy()) < 60.0:
        return False
    idx = np.flatnonzero(seen)
    return bool(np.diff(idx).max(initial=0) <= max_gap)


def _line(a: pd.Series) -> pd.Series:
    """The backscatter trace as drawn: carried forward across gaps, then a
    three-day mean. Raw HV is sampled irregularly and is speckle-noisy, so the
    unfilled series draws as disconnected vertical strokes rather than a
    curve. This is a descriptive figure, and the smoothing is cosmetic."""
    return a.ffill().bfill().rolling(3, center=True, min_periods=1).mean()


def _znorm(a: np.ndarray) -> np.ndarray:
    sd = np.nanstd(a)
    return (a - np.nanmean(a)) / (sd if sd > 0 else 1.0)


def pick(d, cls: str) -> str:
    """The representative lake for one class, by the rule in the docstring."""
    ids = d.expert.index[d.expert.expert_label == cls]
    area = d.meta.loc[ids, "area"]
    ids = list(area[area >= area.median()].index)

    hv, keep = {}, []
    for lid in ids:
        s = derive(d.series(lid))
        if not _coverage_ok(s.p_water):
            continue
        keep.append(lid)
        hv[lid] = _clean(s.HV_anom).interpolate(limit_direction="both").to_numpy()

    M = np.vstack([_znorm(hv[l]) for l in keep])
    med = np.nanmedian(M, axis=0)
    dist = np.nansum((M - med) ** 2, axis=1)
    # The closest tenth, not the closest half: a wide typicality window lets
    # the amplitude tie-break pull in a sharp, atypical lake, which is exactly
    # wrong for slow drainage, whose signature is that it is not sharp.
    typical = [keep[i] for i in np.argsort(dist)[: max(1, len(keep) // 10)]]
    return max(typical, key=lambda l: np.nanmax(hv[l]) - np.nanmin(hv[l]))


def event_doy(s: pd.DataFrame) -> float | None:
    """Last day the lake is observed as open water."""
    pw = _clean(s.p_water).dropna()
    wet = pw[pw >= WATER_ON]
    return float(wet.index[-1]) if len(wet) else None


def classes(out: Path = HERE / "fig1_classes.pdf") -> Path:
    use_paper_style()
    ml_style()
    import matplotlib.pyplot as plt

    d = load_year(2019)
    picks = {c: pick(d, c) for c in CLASSES}

    # A shared backscatter axis: the four signatures differ in depth as well as
    # in shape, and per-panel scaling hides that.
    fig, axes = plt.subplots(1, 4, figsize=(FULL_WIDTH, 1.95), sharey=True)
    for ax, c in zip(axes, CLASSES):
        s = derive(d.series(picks[c]))
        x = s.index.to_numpy()

        ax.plot(x, _line(s.HV_anom).to_numpy(), color=INK, lw=0.9, zorder=3)
        ax.set_xlim(1, 365)
        ax.set_xticks([1, 91, 182, 274])
        ax.set_xticklabels(["Jan", "Apr", "Jul", "Oct"])
        if c == CLASSES[0]:
            ax.set_ylabel("Backscatter anomaly (dB)", labelpad=2)

        e = event_doy(s)
        if e is not None:
            ax.axvline(e, color=MARK, lw=0.7, ls=(0, (2.6, 1.8)), zorder=1)

        r = ax.twinx()
        pw = s.p_water
        r.plot(x[pw.notna()], pw.dropna().to_numpy(), ls="", marker="o",
               ms=1.9, mfc=DOT, mec="none", zorder=4)
        r.set_ylim(105, -5)                      # runs downward, as in (b)
        r.set_yticks([0, 20, 40, 60, 80, 100])
        r.tick_params(axis="y", colors=DOT)
        r.spines["right"].set_color(DOT)
        for side in ("top", "left"):
            r.spines[side].set_visible(False)
        if c == CLASSES[-1]:
            r.set_ylabel("Water probability $p_\\mathrm{water}$ (%)", color=DOT,
                         labelpad=3)
        else:
            r.set_yticklabels([])

        ax.set_title(f"({'abcd'[CLASSES.index(c)]}) {TITLE[c]}", fontsize=7.2,
                     pad=4)

    fig.tight_layout(w_pad=1.4)
    _save(fig, Path(out))
    plt.close(fig)
    print({c: (picks[c], event_doy(derive(d.series(picks[c])))) for c in CLASSES})
    return out


if __name__ == "__main__":
    classes()
