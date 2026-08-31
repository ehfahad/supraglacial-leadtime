"""Figure 1: the end-to-end architecture, as a wired card diagram.

Modules are bordered cards carrying a drawn glyph, a title and a one-line
detail, wired left to right by explicit arrows: three input sources collect onto
a rail, run through the processing chain, fork at the evaluation protocol and
converge on the quantity the pipeline emits.

Three colour families, no more. Slate marks the inputs, vermillion the two
stages this paper contributes, and neutral grey the machinery that is inherited
rather than proposed. Glyphs are built from primitives so the figure stays a
self-contained vector PDF with no raster assets.

The leave-one-basin-out glyph is not a schematic. It is the real position of
every lake in the 2019 record, projected to NSIDC polar stereographic and
coloured by drainage basin, with one basin held out.

The figure shows architecture only: no date, accuracy or per-class result
appears in it. CARRA is absent because the headline model is satellite-only.
"""
from __future__ import annotations

from pathlib import Path

HERE = Path(__file__).resolve().parents[1]   # figures/
CACHE = Path(__file__).resolve().parent / "_greenland.npz"

import numpy as np

from .paperfigs import ml_style
from .plotting import FULL_WIDTH, _save, use_paper_style

# One hue per module type, and modules that do the same job share it: the three
# inputs are one family, the two split schemes are another, and each processing
# stage gets its own. Every entry is (fill, stroke, glyph), built to a single
# recipe so the five hues sit at the same lightness and saturation: pale fill,
# mid stroke, working-saturation glyph. Type stays near-black throughout, which
# is what keeps a multi-hue figure looking like a paper rather than a slide.
DATA  = ("#D3E4F9", "#7FB0DE", "#2A6FA8")    # the three input sources
PREP  = ("#CDEBE4", "#74C6B4", "#17897A")    # leakage-free preprocessing
TRUNC = ("#FBE6C4", "#EFBA6A", "#B87A12")    # truncation at the cutoff
MODEL = ("#E1DAF7", "#A99AE0", "#6B4FC0")    # representation and classifier
EVAL  = ("#FADAE3", "#EE9AB2", "#C04068")    # the two split schemes
OUT   = ("#DCEFC9", "#9CCB70", "#4F8A22")    # what the pipeline emits
# Green sits in the one gap the other five hues leave on the wheel, and is
# pushed yellow-green so it separates cleanly from the blue-green teal.

TITLE, BODY = "#1F2933", "#4E5862"
INK, MID, FAINT = "#1F262E", "#7C868E", "#CDD3D9"


# ---------------------------------------------------------------- primitives

def card(fig, x, y, w, h, fill, edge, lw=.7, r=.015, z=2):
    import matplotlib.patches as mp
    fig.patches.append(mp.FancyBboxPatch(
        (x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}",
        transform=fig.transFigure, facecolor=fill, edgecolor=edge,
        linewidth=lw, zorder=z))


def arrow(fig, p0, p1, color=MID, lw=1.0, rad=0.0, z=6):
    import matplotlib.patches as mp
    fig.patches.append(mp.FancyArrowPatch(
        p0, p1, transform=fig.transFigure, color=color, lw=lw,
        arrowstyle="-|>", mutation_scale=8,
        connectionstyle=f"arc3,rad={rad}", shrinkA=0, shrinkB=0, zorder=z))


def glyph(fig, cx, cy, w, h):
    """A bare unit canvas for one icon, aspect-corrected by the caller."""
    ax = fig.add_axes([cx - w / 2, cy - h / 2, w, h])
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_xticks([]); ax.set_yticks([])
    ax.patch.set_alpha(0)
    ax.set_zorder(5)
    return ax


def _rr(ax, x, y, w, h, ink, fill="none", lw=.9, r=.04, z=3, alpha=1.0):
    import matplotlib.patches as mp
    ax.add_patch(mp.FancyBboxPatch(
        (x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}",
        facecolor=fill, edgecolor=ink, lw=lw, zorder=z, alpha=alpha))


# -------------------------------------------------------------------- glyphs

def ic_radar(ax, ink):
    """Satellite with a dish and downward pulses."""
    import matplotlib.patches as mp
    _rr(ax, .41, .50, .18, .21, ink, fill=ink, lw=0, r=.03)
    for x in (.17, .61):
        _rr(ax, x, .54, .22, .13, ink, fill="white", lw=.7, r=.02)
        for k in (1, 2):
            ax.plot([x + .22 * k / 3] * 2, [.55, .66], color=ink, lw=.45)
    ax.add_patch(mp.Wedge((.50, .71), .12, 25, 155, width=.033,
                          facecolor=ink, edgecolor="none", zorder=4))
    for i, r in enumerate((.13, .21, .29)):
        ax.add_patch(mp.Arc((.50, .48), 2 * r, 2 * r, theta1=210, theta2=330,
                            color=ink, lw=.8, alpha=.85 - .22 * i))


def ic_optical(ax, ink):
    """Satellite with a lens and a light cone."""
    import matplotlib.patches as mp
    _rr(ax, .41, .56, .18, .20, ink, fill=ink, lw=0, r=.03)
    for x in (.17, .61):
        _rr(ax, x, .59, .22, .13, ink, fill="white", lw=.7, r=.02)
        for k in (1, 2):
            ax.plot([x + .22 * k / 3] * 2, [.60, .71], color=ink, lw=.45)
    ax.add_patch(mp.Polygon([[.43, .54], [.57, .54], [.73, .12], [.27, .12]],
                            closed=True, facecolor=ink, alpha=.15,
                            edgecolor="none", zorder=2))
    ax.add_patch(mp.Circle((.50, .535), .052, facecolor="white", edgecolor=ink,
                           lw=.8, zorder=4))


def ic_outline(ax, ink, poly):
    """A real lake outline inside a tile, with a label tag."""
    import matplotlib.patches as mp
    _rr(ax, .10, .16, .80, .66, ink, fill="white", lw=.8, r=.05)
    ax.add_patch(mp.Polygon(poly, closed=True, facecolor=ink, alpha=.28,
                            edgecolor=ink, lw=.8, zorder=3))
    _rr(ax, .58, .58, .27, .17, ink, fill="white", lw=.7, r=.03, z=4)
    for k in (0, 1):
        ax.plot([.62, .81], [.685 - .055 * k] * 2, color=ink, lw=.6, zorder=5)


def ic_causal(ax, ink):
    """A day reading a trailing window, with nothing after it to read.

    The forward half of the axis is drawn faint and left empty rather than
    crossed out: at this size a prohibition symbol reads as clutter, whereas an
    empty forward half says the same thing by construction.
    """
    import matplotlib.patches as mp
    ax.plot([.08, .52], [.32, .32], color=ink, lw=1.0, solid_capstyle="butt")
    ax.plot([.52, .92], [.32, .32], color=FAINT, lw=1.0, ls=(0, (1.6, 1.6)))
    ax.add_patch(mp.Rectangle((.18, .34), .34, .32, facecolor=ink, alpha=.20,
                              edgecolor="none", zorder=2))
    ax.plot([.52, .52], [.18, .80], color=ink, lw=1.1, zorder=4)
    ax.text(.52, .86, "$t$", ha="center", va="bottom", fontsize=6.6, color=ink)
    ax.annotate("", xy=(.19, .50), xytext=(.51, .50),
                arrowprops=dict(arrowstyle="-|>", color=ink, lw=1.0,
                                shrinkA=0, shrinkB=0, mutation_scale=8))
    for xx in (.62, .72, .82):
        ax.plot([xx], [.50], marker="o", ms=1.5, color=FAINT, zorder=3)


def ic_truncate(ax, ink):
    """A season kept to a cutoff and discarded after it."""
    x = np.linspace(0, 1, 200)
    y = .46 + .22 * np.sin(5.0 * x) * np.exp(-1.0 * x) + .11 * x
    k = x <= .55
    ax.fill_between(x[k], .10, y[k], color=ink, alpha=.13, lw=0, zorder=2)
    ax.plot(x[k], y[k], color=ink, lw=1.2, zorder=4, solid_capstyle="round")
    ax.plot(x[~k], y[~k], color=FAINT, lw=1.0, ls=(0, (1.3, 1.3)), zorder=3)
    ax.plot([.55, .55], [.10, .84], color=ink, lw=.9, ls=(0, (2.2, 1.5)),
            zorder=5)
    ax.text(.60, .88, "$T$", ha="left", va="top", fontsize=6.4, color=ink)
    ax.set_ylim(.04, 1.0)


def ic_model(ax, ink):
    """A bank of kernels feeding a linear decision boundary."""
    import matplotlib.patches as mp
    for i, (w, yy) in enumerate([(.30, .72), (.20, .54), (.34, .36), (.24, .18)]):
        _rr(ax, .04, yy, w, .11, ink, fill=ink, lw=0, r=.02,
            alpha=.45 + .18 * i)
    ax.annotate("", xy=(.55, .47), xytext=(.43, .47),
                arrowprops=dict(arrowstyle="-|>", color=ink, lw=.9,
                                shrinkA=0, shrinkB=0, mutation_scale=7))
    ax.add_patch(mp.Circle((.79, .47), .20, facecolor="white", edgecolor=ink,
                           lw=.8, zorder=3))
    ax.plot([.635, .945], [.33, .61], color=ink, lw=.9, zorder=4)
    for px, py in ((.71, .60), (.79, .66)):
        ax.plot([px], [py], marker="o", ms=2.1, color=ink, zorder=5)
    for px, py in ((.79, .29), (.88, .37)):
        ax.plot([px], [py], marker="s", ms=2.1, color=ink, zorder=5)


def ic_kfold(ax, ink):
    """Five equal blocks, one held out."""
    for i in range(5):
        held = i == 3
        _rr(ax, .05 + i * .19, .30, .15, .40, ink,
            fill=ink if held else "white", lw=.8, r=.02)


def ic_basins(ax, pts, regions, ink, held="NW"):
    """Every lake in the 2019 record, by basin, with one basin held out."""
    keep = regions == held
    ax.plot(pts[~keep, 0], pts[~keep, 1], ".", ms=.9, color=FAINT, zorder=3)
    ax.plot(pts[keep, 0], pts[keep, 1], ".", ms=1.1, color=ink, zorder=4)
    ax.set_xlim(pts[:, 0].min() - 6e4, pts[:, 0].max() + 6e4)
    ax.set_ylim(pts[:, 1].min() - 6e4, pts[:, 1].max() + 6e4)
    ax.set_aspect(1)


def ic_output(ax, ink):
    """A calendar with one day marked."""
    import matplotlib.patches as mp
    _rr(ax, .12, .12, .76, .70, ink, fill="white", lw=.9, r=.05)
    ax.add_patch(mp.Rectangle((.12, .64), .76, .18, facecolor=ink,
                              edgecolor="none", zorder=4))
    for x in (.30, .70):
        ax.plot([x, x], [.80, .92], color=ink, lw=1.2, zorder=5,
                solid_capstyle="round")
    for r in range(3):
        for c in range(4):
            cx, cy = .225 + c * .165, .50 - r * .155
            if (r, c) == (1, 2):
                ax.add_patch(mp.Circle((cx, cy), .062, facecolor=ink,
                                       edgecolor="none", zorder=5))
            else:
                ax.plot([cx], [cy], marker="o", ms=1.4, color=FAINT, zorder=5)


# ---------------------------------------------------------------- real data

def greenland():
    """Lake centroids in EPSG:3413 with their basin, cached after first build."""
    if CACHE.exists():
        z = np.load(CACHE, allow_pickle=True)
        return z["pts"], z["reg"]
    import geopandas as gpd
    g = gpd.read_file("data/raw/labels_2019_volumes.geojson")[["region", "geometry"]]
    c = g.to_crs(3413).geometry.centroid
    pts = np.c_[c.x.values, c.y.values]
    reg = g.region.to_numpy().astype(str)
    np.savez(CACHE, pts=pts, reg=reg)
    return pts, reg


def one_lake():
    """A single real lake outline, normalised into the unit square.

    Cached alongside the centroids so the schematic builds without data/raw.
    """
    if CACHE.exists() and "outline" in np.load(CACHE, allow_pickle=True):
        xy = np.load(CACHE, allow_pickle=True)["outline"]
    else:
        import geopandas as gpd
        g = gpd.read_file("data/raw/labels_2019_volumes.geojson")
        g = g[g.new_id == "CW2019_1541"].to_crs(3413)
        xy = np.asarray(g.geometry.iloc[0].exterior.coords)
    xy = xy - xy.min(0)
    xy = xy / xy.max()
    return .18 + .58 * xy


# --------------------------------------------------------------------- build

def build(out: Path = HERE / "fig3_pipeline.pdf"):
    use_paper_style()
    ml_style()
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    pts, reg = greenland()
    poly = one_lake()
    fig = plt.figure(figsize=(FULL_WIDTH, 2.46))
    ar = FULL_WIDTH / 2.46                       # to keep glyph canvases square

    def gl(cx, cy, h):
        return glyph(fig, cx, cy, h / ar, h)

    # ---- three input sources ----------------------------------------
    SRC = [(ic_radar, "Sentinel-1\nSAR backscatter", .790),
           (ic_optical, "Sentinel-2 and\nLandsat-8 optical", .500),
           (None, "Lake outlines\nand expert labels", .210)]
    sx, sw, sh = .004, .186, .238
    for fn, txt, cy in SRC:
        card(fig, sx, cy - sh / 2, sw, sh, DATA[0], DATA[1])
        a = gl(sx + .040, cy, .150)
        (fn or (lambda ax, ink: ic_outline(ax, ink, poly)))(a, DATA[2])
        fig.text(sx + .072, cy, txt, ha="left", va="center", fontsize=5.1,
                 color=TITLE, linespacing=1.45, zorder=7)

    bus = sx + sw + .016
    for _, _, cy in SRC:
        fig.add_artist(Line2D([sx + sw, bus], [cy, cy], color=MID, lw=1.0,
                              transform=fig.transFigure, zorder=6))
    fig.add_artist(Line2D([bus, bus], [SRC[0][2], SRC[2][2]], color=MID, lw=1.0,
                          transform=fig.transFigure, zorder=6))

    # ---- the processing chain ---------------------------------------
    STAGE = [(.214, ic_causal, PREP, "Leakage-free\npreprocessing",
              "LOCF gap fill\ntrailing windows\none-sided Hampel\nexpanding scaling"),
             (.376, ic_truncate, TRUNC, "Truncation\nat cutoff $T$",
              "14 cutoffs\n1 May to 31 Dec\nrefit at each"),
             (.538, ic_model, MODEL, "Representation\nand classifier",
              "MiniROCKET\n10,000 kernels, PPV\nridge classifier")]
    cw, cy0, ch = .146, .182, .636
    for cx, fn, pal, head, detail in STAGE:
        card(fig, cx, cy0, cw, ch, pal[0], pal[1])
        fn(gl(cx + cw / 2, cy0 + ch - .175, .200), pal[2])
        fig.text(cx + cw / 2, cy0 + ch - .315, head, ha="center", va="top",
                 fontsize=6.5, weight="bold", color=TITLE, linespacing=1.35,
                 zorder=7)
        fig.text(cx + cw / 2, cy0 + ch - .455, detail, ha="center", va="top",
                 fontsize=5.3, color=BODY, linespacing=1.55, zorder=7)

    arrow(fig, (bus, .500), (STAGE[0][0], .500))
    for a, b in zip(STAGE, STAGE[1:]):
        arrow(fig, (a[0] + cw, .500), (b[0], .500))

    # ---- the protocol fork ------------------------------------------
    fx, fw, fh = .704, .148, .330
    FORK = [(None, "Random 5-fold", .700), (None, "Leave-one-basin-out", .300)]
    card(fig, fx, FORK[0][2] - fh / 2, fw, fh, EVAL[0], EVAL[1])
    ic_kfold(gl(fx + fw / 2, FORK[0][2] + .046, .132), EVAL[2])
    card(fig, fx, FORK[1][2] - fh / 2, fw, fh, EVAL[0], EVAL[1])
    ic_basins(glyph(fig, fx + fw / 2, FORK[1][2] + .044, .064, .206), pts, reg,
                  EVAL[2])
    for _, txt, cy in FORK:
        fig.text(fx + fw / 2, cy - .076, txt, ha="center", va="top",
                 fontsize=5.9, weight="bold", color=TITLE, zorder=7)
    fig.text(fx + fw / 2, .500, "$\\times$ 5 seeds", ha="center", va="center",
             fontsize=5.4, color=MID, zorder=7)

    m_right = STAGE[-1][0] + cw
    arrow(fig, (m_right, .500), (fx, FORK[0][2]), rad=-.20)
    arrow(fig, (m_right, .500), (fx, FORK[1][2]), rad=.20)

    # ---- what the pipeline emits ------------------------------------
    ox, ow, oh = .872, .124, .470
    card(fig, ox, .500 - oh / 2, ow, oh, OUT[0], OUT[1], lw=.9)
    ic_output(gl(ox + ow / 2, .615, .135), OUT[2])
    fig.text(ox + ow / 2, .470, "Per-class\nknowability date\n$T^{\\ast}_c$",
             ha="center", va="top", fontsize=5.8, weight="bold", color=TITLE,
             linespacing=1.55, zorder=7)
    for _, _, cy in FORK:
        arrow(fig, (fx + fw, cy), (ox, .500), rad=(-.18 if cy > .5 else .18))

    _save(fig, Path(out))
    plt.close(fig)
    return out


if __name__ == "__main__":
    build()
