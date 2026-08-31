"""The figures built from the published result tables.

File names carry the paper's figure numbers:

    fig2_incumbent.pdf      Figure 2   input availability; head to head
    fig4_leadtime.pdf       Figure 4   per-class F1 vs cutoff; lead vs target
    fig5_generalisation.pdf Figure 5   per basin; cross-year
    fig6_errormodes.pdf     Figure 6   which pair of outcomes is confused, when

The other two are built elsewhere: fig_classes.py draws Figure 1 from the raw
record, and fig_pipeline.py draws Figure 3, the method schematic.

Class names come from CLASS_LABEL throughout, "rapid drainage" and "slow
drainage", never the `drain` / `drain_slow` data keys.
"""
from __future__ import annotations

import datetime as dt
from pathlib import Path

# Output goes to figures/, one level up from this package.
HERE = Path(__file__).resolve().parents[1]   # figures/

import numpy as np
import pandas as pd

from .plotting import (CLASS_COLOR, CLASS_LABEL, CLASS_STYLE, COL_WIDTH,
                       FULL_WIDTH, _save, despine, horizon_axis, use_paper_style)

CLASSES = ["drain", "drain_slow", "buried", "refreeze"]   # in T* order

# Dash patterns rather than markers: fourteen cutoffs would put fourteen
# markers on every curve, and where the classes converge above 92% they pile
# into an unreadable clump. The pattern is ordered rather than arbitrary, dot
# length growing with T*, so the four read as one graduated family and the
# figure stays legible in greyscale.
CLASS_DASH = {
    "drain": (0, (0.6, 1.3)),
    "drain_slow": (0, (1.4, 1.4)),
    "buried": (0, (2.6, 1.5)),
    "refreeze": (0, (4.4, 1.6)),
}


def ml_style():
    """Figure conventions: sans-serif text, a box frame, no grid, no markers.

    Identity rests on the colourblind-validated palette and on the graduated
    dash patterns above; the shaded bands carry uncertainty.
    """
    import matplotlib as mpl
    mpl.rcParams.update({
        # Sans-serif figure text beside serif body text, the usual convention
        # in ML venues. DejaVu Sans ships inside matplotlib and is a genuine
        # .ttf, so it embeds as clean TrueType under pdf.fonttype 42.
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
        "mathtext.fontset": "dejavusans",
        # Full box frame, ticks outward on the left and bottom only, no minor
        # ticks.
        "axes.linewidth": 0.8,
        "axes.grid": False,
        "axes.edgecolor": "#000000",
        "xtick.direction": "out", "ytick.direction": "out",
        "xtick.top": False, "ytick.right": False,
        "xtick.major.size": 3.5, "ytick.major.size": 3.5,
        "xtick.major.width": 0.8, "ytick.major.width": 0.8,
        "xtick.minor.visible": False, "ytick.minor.visible": False,
        "xtick.color": "#000000", "ytick.color": "#000000",
        # Set once here so every figure in the paper carries the same axis
        # text size; individual figures must not override these.
        "xtick.labelsize": 6.0, "ytick.labelsize": 6.0,
        "axes.labelsize": 6.4,
        "lines.linewidth": 1.3,
        # A thin boxed legend, which sits legibly over data.
        "legend.frameon": True,
        "legend.framealpha": 1.0,
        "legend.edgecolor": "#000000",
        "legend.fancybox": False,
        "legend.borderpad": 0.4,
    })

# The published stacked model's first runnable day: its last input is the
# 15 Oct - 1 Nov radar difference. Lead time is measured against this.
WALL = 288
THRESHOLD = 0.80
INK, MUTED, FAINT = "#0F161D", "#5A6672", "#AAB6C2"
GUIDE = "#8A94A0"


def _cal(doy: int) -> str:
    return (dt.date(2019, 1, 1) + dt.timedelta(days=int(doy) - 1)).strftime("%b %-d")


def _f1(df, horizons, classes=CLASSES):
    """Per-class F1 at each horizon, from the pooled confusion matrices."""
    order = sorted(classes)
    out = {c: [] for c in classes}
    for h in horizons:
        s = df[df.horizon == h]
        cm = np.array([[s[f"cm_{a}__{b}"].sum() for b in order] for a in order], float)
        for c in classes:
            i = order.index(c)
            r = cm[i, i] / cm[i].sum() if cm[i].sum() else 0.0
            p = cm[i, i] / cm[:, i].sum() if cm[:, i].sum() else 0.0
            out[c].append(2 * r * p / (r + p) if r + p else 0.0)
    return {c: np.array(v) for c, v in out.items()}


def _seed_sd(df, horizons):
    out = {c: [] for c in CLASSES}
    for h in horizons:
        per = {c: [] for c in CLASSES}
        for sd in sorted(df.seed.unique()):
            f = _f1(df[df.seed == sd], [h])
            for c in CLASSES:
                per[c].append(f[c][0])
        for c in CLASSES:
            out[c].append(np.std(per[c], ddof=1))
    return {c: np.array(v) for c, v in out.items()}


def _tstar(vals, horizons, th=THRESHOLD):
    return next((h for h, v in zip(horizons, vals) if v >= th), None)


def panel(ax, tag):
    """Panel letter drawn in axes coordinates.

    Never a title: a left-aligned title wider than its axes expands the tight
    bounding box at save time and silently squeezes every panel in the figure.
    The caption carries the wording; the axes carry only the letter.
    """
    ax.text(0.0, 1.045, tag, transform=ax.transAxes, fontsize=8,
            weight="bold", va="bottom", ha="left")


def curve(ax, x, y, color, ls="-", marker="o", lw=1.6, ms=3.0, sd=None,
          label=None, zorder=3, halo=True):
    """One measured series: a smooth guide through the points, and the points.

    Two presentation problems this solves. Straight segments between fourteen
    cutoffs make every crossing a hard corner, and where four classes converge
    above 92% the segments sit on top of each other and the figure reads as a
    tangle. A monotone cubic (PCHIP) guide is smooth and, unlike a spline,
    cannot overshoot -- it will not invent a peak between two measurements.

    The markers stay on the measured cutoffs, so nothing about where the data
    actually is gets hidden. The white halo behind each line keeps crossings
    legible without needing transparency, which would mud the colours.
    """
    from matplotlib import patheffects as pe
    from scipy.interpolate import PchipInterpolator

    x = np.asarray(x, float)
    xs = np.linspace(x[0], x[-1], 400)
    ys = PchipInterpolator(x, np.asarray(y, float))(xs)

    effects = [pe.Stroke(linewidth=lw + 1.6, foreground="white"), pe.Normal()]
    if sd is not None:
        lo = PchipInterpolator(x, np.asarray(y) - np.asarray(sd))(xs)
        hi = PchipInterpolator(x, np.asarray(y) + np.asarray(sd))(xs)
        ax.fill_between(xs, lo, hi, color=color, alpha=.15, lw=0, zorder=zorder - 1)
    # A series drawn without markers must carry its own legend label: attaching
    # it to an invisible marker artist yields a legend entry with no swatch.
    bare = (not marker) or ms == 0
    ax.plot(xs, ys, color=color, ls=ls, lw=lw, zorder=zorder,
            path_effects=effects if halo else None, solid_capstyle="round",
            dash_capstyle="round", label=label if bare else None)
    if not bare:
        ax.plot(x, y, ls="none", marker=marker, ms=ms, color=color, mfc="white",
                mew=.85, zorder=zorder + 1, label=label)
    return ax


def _headline(path="results/sweep_c1.parquet"):
    s = pd.read_parquet(path)
    return s[(s.varset == "sat") & (s.pipeline == "causal")]


# ===================================================================== fig 2


def leadtime(out: Path = HERE / "fig4_leadtime.pdf"):
    use_paper_style()
    ml_style()
    import matplotlib.pyplot as plt

    m = _headline()
    m = m[m.split == "kfold"]
    H = sorted(m.horizon.unique())
    f1, sd = _f1(m, H), _seed_sd(m, H)

    # Column width, not text width: these are single-column figures, and a
    # text-width canvas would be scaled down on the page along with its text.
    fig, (axa, axb) = plt.subplots(2, 1, figsize=(COL_WIDTH, 4.05))

    # ---- (a) the lead-time curve -------------------------------------
    X = horizon_axis(axa, H)
    axa.axhline(100 * THRESHOLD, color=GUIDE, lw=.6, ls=(0, (4, 3)), zorder=1)
    axa.text(X[0] + .25, 100 * THRESHOLD + 1.6, "80% F1", fontsize=6.2,
             color=GUIDE, va="bottom", ha="left")
    for c in CLASSES:
        curve(axa, X, 100 * f1[c], CLASS_COLOR[c], ls=CLASS_DASH[c], marker="",
              ms=0, lw=1.3, sd=100 * sd[c], zorder=3, halo=False)
        t = _tstar(f1[c], H)
        if t is not None:
            axa.plot([X[H.index(t)]], [100 * THRESHOLD], marker="v", ms=4.6,
                     color=CLASS_COLOR[c], clip_on=False, zorder=6)
    # A legend, not direct labels: the four curves converge above 92% by the
    # end of the season, so end-of-line labels would overlap.
    handles = [plt.Line2D([], [], color=CLASS_COLOR[c], lw=1.3,
                          ls=CLASS_DASH[c], dash_capstyle="round",
                          label=CLASS_LABEL[c]) for c in CLASSES]
    # Two columns rather than four rows, so the legend box does not land on
    # the curves it names.
    axa.legend(handles=handles, loc="lower right", fontsize=6.3, ncol=2,
               handlelength=1.9, columnspacing=.9, labelspacing=.28,
               borderaxespad=.4)
    axa.set_ylabel("Per-class F1 (%)")
    axa.set_xlabel("Prediction cutoff (2019)")
    axa.set_ylim(32, 102)
    axa.set_xlim(X[0] - .6, X[-1] + .5)
    panel(axa, "(a)")

    # ---- (b) what a stricter accuracy requirement costs ---------------
    # Lead is measured against 15 October, the earliest date an end-of-season
    # label can be produced at all. That is a property of the input calendar
    # rather than of any model, so the axis names the reference date rather
    # than framing the result as a margin over someone else's method.
    THS = (.75, .80, .85, .90)
    xs = np.array([100 * th for th in THS])
    for c in CLASSES:
        leads = [WALL - _tstar(f1[c], H, th) for th in THS]
        axb.plot(xs, leads, color=CLASS_COLOR[c], ls=CLASS_DASH[c], lw=1.3,
                 dash_capstyle="round", marker="o", ms=3.4, mfc="white",
                 mew=1.0, zorder=3)
        # Labelled inside the panel and set back from the last point, so each
        # label sits where its own curve is still higher. Hanging them off the
        # right-hand end would need a margin that squeezes the four thresholds
        # into the left third of the axes.
        dy = {"drain": 6, "drain_slow": -7, "buried": 11, "refreeze": -8}[c]
        axb.annotate(CLASS_LABEL[c], xy=(xs[-1], leads[-1]), xytext=(-3, dy),
                     textcoords="offset points", fontsize=6.2,
                     color=CLASS_COLOR[c], ha="right",
                     va="top" if dy < 0 else "bottom")

    axb.axhline(0, color="0.55", lw=.8, ls=(0, (4, 3)), zorder=1)
    # Two lines, in the empty lower-left wedge: refreeze does not drop below
    # 30 days until 80% F1, so nothing crosses this block.
    axb.annotate("no lead over\nend-of-season labeling", xy=(74.6, 2.0),
                 fontsize=6.2, color="0.35", va="bottom", ha="left",
                 linespacing=1.35)
    axb.set_xticks(xs)
    axb.set_xticklabels([f"{x:.0f}" for x in xs])
    axb.set_xlim(73.9, 91.1)
    axb.set_ylim(-20, 100)
    axb.set_xlabel("Per-class F1 required (%)")
    axb.set_ylabel("Lead time before 15 Oct (days)")
    panel(axb, "(b)")

    fig.tight_layout(h_pad=1.1)
    _save(fig, Path(out))
    plt.close(fig)
    print("T*:", {CLASS_LABEL[c]: _cal(_tstar(f1[c], H)) for c in CLASSES})
    return out


# ===================================================================== fig 3


# The three confusion pairs that matter, and how each is coloured. A pair has
# two classes, so no single class colour is the "right" one; the rule used here
# is the family the pair sits in -- both drainage outcomes take the drainage
# hue, both storage outcomes take the storage hue, and the cross-family pair
# takes the remaining one. Dash lengths graduate as in fig 2.
PAIRS = (("drain", "drain_slow"), ("buried", "refreeze"), ("drain_slow", "refreeze"))
PAIR_LABEL = {
    ("drain", "drain_slow"): "rapid vs slow drainage",
    ("buried", "refreeze"): "buried vs refreeze",
    ("drain_slow", "refreeze"): "slow drainage vs refreeze",
}
PAIR_STYLE = {
    ("drain", "drain_slow"): (CLASS_COLOR["drain"], (0, (1.0, 1.5))),
    ("buried", "refreeze"): (CLASS_COLOR["buried"], (0, (2.6, 1.5))),
    ("drain_slow", "refreeze"): (CLASS_COLOR["refreeze"], (0, (4.4, 1.6))),
}


def errormodes(out: Path = HERE / "fig6_errormodes.pdf"):
    """Which pair of outcomes the classifier confuses, and when it switches.

    The mechanism behind fig 2: the errors do not merely shrink, they change
    kind. Early the model cannot separate rapid from slow drainage, because
    only the rate distinguishes them and not enough has happened. From July it
    cannot separate buried from refrozen, because what distinguishes those is
    whether winter snow covers the lake, which has not happened yet.
    """
    use_paper_style()
    ml_style()
    import matplotlib.pyplot as plt

    m = _headline()
    m = m[m.split == "kfold"]
    H = sorted(m.horizon.unique())
    order = sorted(CLASSES)

    share, total = {p: [] for p in PAIRS}, []
    for h in H:
        s = m[m.horizon == h]
        cm = np.array([[s[f"cm_{a}__{b}"].sum() for b in order] for a in order], float)
        off = cm.sum() - np.trace(cm)
        total.append(off)
        for a, b in PAIRS:
            i, j = order.index(a), order.index(b)
            share[(a, b)].append(100 * (cm[i, j] + cm[j, i]) / off)

    # Taller than a default single-column panel, so the legend clears the 72%
    # peak without compressing the curves beneath it.
    fig, ax = plt.subplots(figsize=(COL_WIDTH, 3.0))
    X = horizon_axis(ax, H)

    named = sum(np.array(share[p]) for p in PAIRS)
    # Labelled on its own curve rather than in the legend: it runs alone along
    # the bottom from August, and removing it drops the legend to three rows.
    other = 100 - named
    curve(ax, X, other, "0.55", ls=(0, (0.6, 1.4)), marker="", ms=0, lw=1.3,
          zorder=2, halo=False)
    ax.annotate("all other pairs", xy=(X[12], other[12]), xytext=(0, 5),
                textcoords="offset points", fontsize=6.2, color="0.45",
                ha="center", va="bottom")
    for pair in PAIRS:
        col, dash = PAIR_STYLE[pair]
        curve(ax, X, share[pair], col, ls=dash, marker="", ms=0, lw=1.3,
              label=PAIR_LABEL[pair], zorder=3, halo=False)

    # The switch: first cutoff at which buried/refreeze overtakes the
    # drainage-rate confusion, drawn between the two cutoffs it happens across.
    d, b = np.array(share[("drain", "drain_slow")]), np.array(share[("buried", "refreeze")])
    k = int(np.argmax(b > d))
    xb = (X[k - 1] + X[k]) / 2
    ax.axvline(xb, color="0.25", lw=.8, ls=(0, (3, 2.4)), zorder=1)
    # Left of the break, where only the two flat early curves sit. To its right
    # the buried/refreeze curve is climbing steeply through the same band.
    ax.annotate("the errors\nchange kind", xy=(xb, 47), xytext=(-4, 0),
                textcoords="offset points", fontsize=6.4, color="0.25",
                va="center", ha="right", linespacing=1.35)

    ax.set_ylabel("Share of all errors (%)")
    ax.set_xlabel("Prediction cutoff (2019)")
    # Legend inside, upper left, over empty space: within the width it
    # occupies, the buried/refreeze curve stays below 65%.
    ax.set_ylim(0, 98)
    ax.set_yticks(range(0, 81, 20))
    ax.legend(loc="upper left", fontsize=6.2, handlelength=2.6,
              handletextpad=.7, labelspacing=.30, borderpad=.5,
              borderaxespad=.4)
    _save(fig, Path(out))
    plt.close(fig)
    print(f"switch between {_cal(H[k-1])} and {_cal(H[k])}; "
          f"errors {int(total[0])} -> {int(total[-1])}")
    return out


# ===================================================================== fig 4

# Ours against the published component. These are semantic roles rather than
# outcome classes; the hues are shared with the class palette but no figure
# shows both meanings at once, so nothing is ambiguous in place.
OURS, THEIRS = "#0072B2", "#AE017E"

# The three input groups on an ordered light-to-dark grey ramp. They are
# ordered by when they open, so a ramp is right; greys rather than a hue keeps
# them from competing with the outcome palette, and every step holds contrast
# against white at 1.6 pt line width.
#
# `needs_event` is the one genuinely continuous series -- it is the fraction of
# lakes that have drained by each date -- so it is smoothed. The other two are
# true step functions and are drawn as steps.
TIERS = [("always", "#C2C8CE", (0, (4.4, 1.6)), "available from 1 May", "step"),
         ("needs_event", "#79828B", (0, (2.6, 1.5)), "needs lake's drainage event", "smooth"),
         ("needs_october", "#2B3138", (0, (1.0, 1.5)), "needs the 15 Oct - 1 Nov window", "step")]

# The wall carries the panel's message, so it takes the one accent in the
# figure. Vermillion reads as a limit; no outcome classes appear here, so it
# cannot be confused with rapid drainage.
WALLC = "#D55E00"


def incumbent(out: Path = HERE / "fig2_incumbent.pdf"):
    """Why the question cannot currently be asked, and what happens where it can.

    (a) is structural rather than statistical: the stacked model needs all four
    of its input groups, and two of them cannot be computed until 15 October,
    so before that date it returns nothing at all. (b) is the comparison on the
    only lakes it can be scored on fairly -- fold 0, which the published models
    were not trained on.
    """
    use_paper_style()
    ml_style()
    import matplotlib.pyplot as plt

    av = pd.read_parquet("results/c5_availability.parquet").sort_values("horizon")
    c5 = pd.read_parquet("results/c5_truncation.parquet")
    sweep = _headline()
    H = av.horizon.tolist()

    # Column width, not text width: these are single-column figures, and a
    # text-width canvas would be scaled down on the page along with its text.
    fig, (axa, axb) = plt.subplots(2, 1, figsize=(COL_WIDTH, 4.05))

    # ---- (a) the availability wall -----------------------------------
    X = horizon_axis(axa, H)
    first = int(np.argmax(av.stacked_model_runnable.to_numpy().astype(bool)))
    # The line sits on 15 October, the day both October-dependent groups open.
    xw = X[first]
    axa.axvspan(X[0] - .6, xw, color=WALLC, alpha=.07, lw=0, zorder=0)
    axa.axvline(xw, color=WALLC, lw=1.1, zorder=4)

    # Steps, not smoothed curves: the October group is 0% at 1 Oct and 100% at
    # 15 Oct, and an interpolant would draw a climb that never happens. Panel
    # (b) is smoothed because it is a continuous quantity; this is not.
    # Anchor index, y, vertical offset, alignment, rotation. All three labels
    # are horizontal; at column width a rotated label would span most of the
    # panel height and collide with the others.
    LAB = {"always": (1, 100, -7, "left", "top", 0),
           "needs_event": (3, 30, -6, "left", "center", 30),
           "needs_october": (6, 4, 0, "center", "bottom", 0)}
    for key, col, dash, lab, mode in TIERS:
        y = 100 * av[key].to_numpy()
        if mode == "smooth":
            curve(axa, X, y, col, ls=dash, marker="", ms=0, lw=1.6, zorder=3,
                  halo=False)
        else:
            axa.plot(X, y, color=col, ls=dash, lw=1.6, drawstyle="steps-post",
                     dash_capstyle="round", zorder=3)
        i, yy, dy, ha, va, rot = LAB[key]
        axa.annotate(lab, xy=(X[i], yy), xytext=(7 if rot else 4, dy),
                     textcoords="offset points", fontsize=6.2, rotation=rot,
                     rotation_mode="anchor",
                     color="#8C939A" if key == "always" else col, ha=ha, va=va)

    axa.annotate("returns nothing\nbefore 15 Oct", xy=(xw, 120), xytext=(-5, 0),
                 textcoords="offset points", fontsize=6.2, color=WALLC,
                 ha="right", va="center", linespacing=1.35)
    axa.annotate("runnable", xy=(xw, 100), xytext=(6, -5), textcoords="offset points",
                 fontsize=6.4, color=WALLC, ha="left", va="top", weight="bold")
    axa.set_ylabel("Inputs available (%)", labelpad=2)
    axa.set_xlabel("Prediction cutoff (2019)")
    axa.set_ylim(-6, 132)
    axa.set_yticks(range(0, 101, 25))
    panel(axa, "(a)")

    # ---- (b) head to head on the same 200 lakes ----------------------
    f0 = sweep[(sweep.split == "kfold") & (sweep.fold == "0")]
    ours = np.array([100 * f0[f0.horizon == h].accuracy.mean() for h in H])
    osd = np.array([100 * f0[f0.horizon == h].accuracy.std(ddof=1) for h in H])
    s1 = c5[c5.model == "S1_rocket"].set_index("horizon")
    theirs = np.array([100 * s1.loc[h, "accuracy"] for h in H])
    tlo = np.array([100 * s1.loc[h, "accuracy_lo"] for h in H])
    thi = np.array([100 * s1.loc[h, "accuracy_hi"] for h in H])

    XB = horizon_axis(axb, H)
    axb.axhline(25, color="0.6", lw=.8, ls=(0, (1.0, 1.6)), zorder=1)
    # On the right, where both curves have long since left the chance level.
    # At the left edge the incumbent's bootstrap band still straddles 25%.
    axb.annotate("chance", xy=(XB[-1] - .2, 26), fontsize=6.2, color="0.45",
                 va="bottom", ha="right")
    # Tinted by whichever side is ahead, rather than one flat grey: the point
    # of the panel is that the sign changes partway through the season.
    axb.fill_between(XB, theirs, ours, where=ours >= theirs, color=OURS,
                     alpha=.13, lw=0, zorder=1, interpolate=True)
    axb.fill_between(XB, theirs, ours, where=ours < theirs, color=THEIRS,
                     alpha=.13, lw=0, zorder=1, interpolate=True)
    axb.fill_between(XB, tlo, thi, color=THEIRS, alpha=.28, lw=0, zorder=2)
    curve(axb, XB, theirs, THEIRS, ls=(0, (2.6, 1.5)), marker="", ms=0, lw=1.5,
          label="S1 radar classifier", zorder=3, halo=False)
    curve(axb, XB, ours, OURS, ls="-", marker="", ms=0, lw=1.5, sd=osd,
          label="This work", zorder=4, halo=False)

    cross = next(i for i, (o, t_) in enumerate(zip(ours, theirs)) if o < t_)
    axb.axvline(XB[cross], color="0.25", lw=.8, ls=(0, (3, 2.4)), zorder=1)
    axb.annotate("crossover", xy=(XB[cross], 76), xytext=(5, 0),
                 textcoords="offset points", fontsize=6.4, color="0.25",
                 ha="left", va="center")
    gap = ours[1] - theirs[1]
    axb.annotate(f"+{gap:.0f}%", xy=(XB[1], (ours[1] + theirs[1]) / 2),
                 xytext=(6, 0), textcoords="offset points", fontsize=6.6,
                 color="0.30", va="center", weight="bold")
    axb.set_ylabel("Accuracy (%)")
    axb.set_xlabel("Prediction cutoff (2019)")
    axb.set_ylim(12, 126)
    axb.set_yticks(range(20, 101, 20))
    axb.legend(loc="upper left", fontsize=6.2, handlelength=2.6,
               handletextpad=.7, labelspacing=.30, borderpad=.5, borderaxespad=.4)
    panel(axb, "(b)")

    fig.tight_layout(h_pad=1.1)
    _save(fig, Path(out))
    plt.close(fig)
    print(f"wall at {_cal(H[first])}; crossover {_cal(H[cross])}; "
          f"max lead +{(ours - theirs).max():.1f}")
    return out


# ===================================================================== fig 5

# Cross-year runs. Three distinct hues from the validated palette rather than
# two greys and an accent: each run changes one thing from the one above it, so
# they are peers, not a ramp.
XY_STYLE = {
    "A_2019_expert":  ("#0072B2", (0, ()),                   "2019, expert labels"),
    "B_2019_machine": ("#AE017E", (0, (2.6, 1.5)),           "2019, machine labels"),
    "C_2018_machine": ("#D55E00", (0, (5.5, 1.7, 1.1, 1.7)), "2018, machine labels"),
}

# Best and worst basin. Blue reads as the ceiling, vermillion as the floor.
BEST, WORST, RESTC = "#0072B2", "#D55E00", "#C3CBD3"


def generalisation(out: Path = HERE / "fig5_generalisation.pdf"):
    """Does the measurement survive a different place, and a different year?

    (a) holds out whole drainage basins. The finding is the spread and how it
    narrows, not six individual curves -- they converge above 94% by year end,
    so only the two extremes are named. (b) separates two costs a naive
    2019-versus-2018 comparison conflates: scoring against published rather
    than expert labels, and the melt season itself.
    """
    use_paper_style()
    ml_style()
    import matplotlib.pyplot as plt

    sweep = _headline()
    b = sweep[sweep.split == "basin"]
    H = sorted(b.horizon.unique())
    xy = pd.read_parquet("results/crossyear.parquet")

    # Column width, not text width: these are single-column figures, and a
    # text-width canvas would be scaled down on the page along with its text.
    fig, (axa, axb) = plt.subplots(2, 1, figsize=(COL_WIDTH, 4.05))

    # ---- (a) per basin -----------------------------------------------
    X = horizon_axis(axa, H)
    curves = {r: np.array([100 * b[(b.fold == r) & (b.horizon == h)].accuracy.mean()
                           for h in H]) for r in sorted(b.fold.unique())}
    stack = np.vstack([curves[r] for r in curves])
    lo, hi = stack.min(0), stack.max(0)
    k = H.index(196)
    best = max(curves, key=lambda r: curves[r][k])
    worst = min(curves, key=lambda r: curves[r][k])

    # Envelope and thin lines share the smoothing of the two highlighted
    # curves; mixing raw and smoothed in one panel reads as two kinds of data.
    from scipy.interpolate import PchipInterpolator
    xs = np.linspace(X[0], X[-1], 400)
    axa.fill_between(xs, PchipInterpolator(X, lo)(xs), PchipInterpolator(X, hi)(xs),
                     color=BEST, alpha=.09, lw=0, zorder=1)
    for r, y in curves.items():
        if r not in (best, worst):
            axa.plot(xs, PchipInterpolator(X, y)(xs), color=RESTC, lw=1.0, zorder=2)
    # Monochrome by design: the panel is about the width of the envelope, and
    # six hues would imply six identities the reader is not asked to track.
    curve(axa, X, curves[best], BEST, ls=(0, ()), marker="", ms=0, lw=1.7,
          zorder=4, halo=False)
    curve(axa, X, curves[worst], WORST, ls=(0, (4.4, 1.6)), marker="", ms=0,
          lw=1.7, zorder=4, halo=False)

    j = H.index(196)
    axa.annotate(f"{best}, best", xy=(X[j], curves[best][j]), xytext=(-3, 8),
                 textcoords="offset points", fontsize=6.4, weight="bold",
                 color=BEST, ha="right", va="bottom")
    axa.annotate(f"{worst}, worst", xy=(X[j], curves[worst][j]), xytext=(11, 5),
                 textcoords="offset points", fontsize=6.4, weight="bold",
                 color=WORST, ha="left", va="bottom")
    axa.text(.97, .04,
              f"spread across basins\n{hi[k] - lo[k]:.0f}% on 15 Jul"
              f"  $\\rightarrow$  {hi[-1] - lo[-1]:.0f}% by 31 Dec",
              transform=axa.transAxes, fontsize=6.2, color="#4A525A",
              ha="right", va="bottom", linespacing=1.35)
    axa.set_ylabel("Accuracy (%)", labelpad=2)
    axa.set_xlabel("Prediction cutoff (2019)")
    axa.set_ylim(30, 102)
    panel(axa, "(a)")

    # ---- (b) cross-year decomposition --------------------------------
    Hx = sorted(xy.horizon.unique())
    XB = horizon_axis(axb, Hx)
    def run(name):
        return np.array([100 * xy[(xy.horizon == h) & (xy.run == name)].macro_recall.mean()
                         for h in Hx])
    A, B, C = run("A_2019_expert"), run("B_2019_machine"), run("C_2018_machine")
    axb.fill_between(XB, B, A, color="#AE017E", alpha=.15, lw=0, zorder=1)
    axb.fill_between(XB, C, B, color="#D55E00", alpha=.15, lw=0, zorder=1)
    for y, key in ((A, "A_2019_expert"), (B, "B_2019_machine"), (C, "C_2018_machine")):
        col, dash, lab = XY_STYLE[key]
        curve(axb, XB, y, col, ls=dash, marker="", ms=0, lw=1.5, label=lab,
              zorder=3, halo=False)
    mid = len(Hx) // 2 + 1
    axb.annotate(f"label cost {np.mean(B - A):+.1f}%",
                 xy=(XB[-1], (A[-1] + B[-1]) / 2), xytext=(-2, 0),
                 textcoords="offset points", fontsize=6.2, color="#8A0064",
                 ha="right", va="center")
    # Inside its own band, at the left where the gap is widest: mid-season the
    # two costs sit close together and a label there reads as ambiguous.
    axb.annotate(f"season cost {np.mean(C - B):+.1f}%",
                 xy=(XB[0], (B[0] + C[0]) / 2), xytext=(2, 3),
                 textcoords="offset points", fontsize=6.2, color="#A04600",
                 ha="left", va="center")
    axb.set_ylabel("Macro-recall (%)", labelpad=2)
    axb.set_xlabel("Prediction cutoff")
    axb.set_ylim(30, 108)
    axb.set_yticks(range(40, 101, 20))
    axb.legend(loc="lower right", fontsize=6.2, handlelength=2.6,
               handletextpad=.7, labelspacing=.30, borderpad=.5, borderaxespad=.4)
    panel(axb, "(b)")

    fig.tight_layout(h_pad=1.1)
    _save(fig, Path(out))
    plt.close(fig)
    print(f"basin spread 15 Jul {hi[k] - lo[k]:.1f}%, 31 Dec {hi[-1] - lo[-1]:.1f}%; "
          f"best {best}, worst {worst}")
    return out
