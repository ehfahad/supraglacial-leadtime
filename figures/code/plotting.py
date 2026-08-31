"""Figures, built to conference standard.

Conventions applied throughout, because reviewers notice them:

* **Vector PDF with TrueType fonts** (``pdf.fonttype = 42``). Type-3 fonts are
  rejected by several venues and are the single most common figure desk-reject.
* **Drawn at final print size.** Each figure is created at the column width it
  will occupy, so nothing is rescaled and every label lands at its stated point
  size. Rescaling a figure in LaTeX is what produces the mismatched-font look.
* **No titles.** The caption carries them; a title inside the axes duplicates it
  and steals vertical space.
* **Serif text** matching a Times-set paper, via Nimbus Roman.
* **Uncertainty is always drawn.** Every mean is a mean over folds and seeds, so
  it ships with a band; a bare line would imply a precision we do not have.

The palette is not hand-picked. These four hues were validated computationally
under Machado-Oliveira-Fernandes 2009 colour-vision-deficiency simulation at
severity 1.0, all-pairs: normal-vision worst pair dE 18.7 (floor 15), protanopia
13.0 and deuteranopia 9.2 (floor 8), every hue inside the OKLCH lightness band
with chroma >= 0.10 and contrast >= 3:1 on white. Okabe-Ito's pink only reached
the 6-8 deuteranopia warning band, so it was replaced.

Colour is never the only channel: each class also carries a distinct line style
and marker, so the figures survive greyscale printing.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import numpy as np
import pandas as pd

# Validated 4-class categorical palette; fixed order, never cycled.
# Hue assignment is semantic, not arbitrary: rapid drainage is the outcome with
# consequences -- it delivers meltwater to the bed and can accelerate ice flow --
# so it takes the warm vermillion that reads as the alert colour. Slow drainage,
# the same process at a different rate, takes the adjacent magenta. Buried and
# refreeze, both storage outcomes, take the cool blue and green.
#
# The four hues are unchanged from the validated set (see the module docstring);
# only which class wears which has been reassigned, so the colour-vision-
# deficiency margins measured for the palette still hold.
CLASS_COLOR = {
    "drain": "#D55E00",
    "drain_slow": "#AE017E",
    "refreeze": "#009E73",
    "buried": "#0072B2",
}
CLASS_STYLE = {          # secondary encoding, so identity is never colour-alone
    "drain": ("-", "o"),
    "drain_slow": ("--", "s"),
    "refreeze": ("-.", "^"),
    "buried": (":", "D"),
}
CLASS_LABEL = {
    "drain": "Rapid drainage",
    "drain_slow": "Slow drainage",
    "refreeze": "Refreeze",
    "buried": "Buried",
}

INK = "#1a1a1a"
MUTED = "#6b6b6b"
GRID = "#dcdcdc"

# Single- and double-column widths for a two-column ACM/NeurIPS-style layout.
COL_WIDTH = 3.33
FULL_WIDTH = 6.9


def use_paper_style():
    mpl.rcParams.update({
        # ACM prohibits Type 3 fonts and requires everything embedded, so
        # fonttype 42 plus a genuine .ttf source. STIXGeneral is Times-metric
        # and ships inside matplotlib, so the figures rebuild anywhere.
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "font.family": "serif",
        "font.serif": ["STIXGeneral", "DejaVu Serif"],
        "mathtext.fontset": "stix",   # math in the same face as the text
        "font.size": 8,
        "axes.labelsize": 8,
        "axes.titlesize": 8,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 7,
        "axes.linewidth": 0.6,
        "axes.edgecolor": INK,
        "axes.labelcolor": INK,
        "text.color": INK,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "xtick.major.size": 2.5,
        "ytick.major.size": 2.5,
        "lines.linewidth": 1.3,
        "lines.markersize": 3.2,
        "legend.frameon": False,
        "legend.handlelength": 2.2,
        "figure.dpi": 200,
        "savefig.dpi": 400,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.01,
        "grid.color": GRID,
        "grid.linewidth": 0.5,
    })


def _save(fig, out: Path):
    """Write in whatever format the suffix asks for, defaulting to vector PDF."""
    out = Path(out)
    fig.savefig(out, format=out.suffix.lstrip(".") or "pdf")


def horizon_axis(ax, horizons, rotation=45, break_after=304, gap=2.5,
                 thin=True):
    """Index-based x axis with calendar labels and a real gap before the reference.

    The horizon grid is ~15 days apart through the melt season and then jumps 61
    days to 31 Dec, the full-year retrospective reference. A linear date axis
    strands the interesting region in the left third; dropping 31 Dec instead
    would hide a real measurement, and one that carries weight -- "waiting past
    mid-October buys nothing" rests on the two non-significant steps at the end.

    So: equal spacing inside the melt season, then a gap of ``gap`` index units
    before the reference point, with break marks drawn in it. The default 2.5 is
    a compromise -- 31 Oct to 31 Dec is 61 days against a typical 15-day step,
    so true proportional spacing would be ~4 units and would dominate the panel,
    while anything near 1.0 reads as equidistant and misleads.

    Returns the x position of each horizon, which callers must use for plotting.
    """
    import datetime as _dt
    import numpy as _np

    hs = list(horizons)
    xs, x = [], 0.0
    for i, h in enumerate(hs):
        xs.append(x)
        x += gap if (break_after is not None and h == break_after
                     and i < len(hs) - 1) else 1.0
    xs = _np.asarray(xs)

    labels = [(_dt.date(2019, 1, 1) + _dt.timedelta(days=int(h) - 1)).strftime("%b %-d")
              for h in hs]
    # Fourteen rotated labels do not fit across a single column, so the
    # mid-month cutoffs keep their ticks and lose their text. The ticks are what
    # locate a point on the curve; the text is only needed often enough to keep
    # the reader oriented in the calendar.
    if thin:
        labels = [l if not l.endswith(" 15") else "" for l in labels]
    ax.set_xticks(xs)
    ax.set_xticklabels(labels, rotation=rotation, ha="right" if rotation else "center")
    ax.set_xlim(xs[0] - .6, xs[-1] + .5)

    if break_after in hs and hs.index(break_after) < len(hs) - 1:
        k = hs.index(break_after)
        xb = (xs[k] + xs[k + 1]) / 2
        for dx in (-.075, .075):
            ax.plot([xb + dx, xb + dx], [-.030, .030],
                    transform=ax.get_xaxis_transform(), color=MUTED, lw=.8,
                    clip_on=False, zorder=6)
        ax.text(xb, -.055, "//", transform=ax.get_xaxis_transform(),
                ha="center", va="top", fontsize=6, color=MUTED, clip_on=False)
    return xs


def despine(ax, keep=("left", "bottom")):
    for side in ("top", "right", "left", "bottom"):
        ax.spines[side].set_visible(side in keep)
