"""
_plot_style.py
--------------
Single source of truth for matplotlib styling across all experiment
figures. Imported at the top of every experiment script before any
pyplot call.
"""
import matplotlib as _mpl
import matplotlib.pyplot as _plt


# Colour palette (colour-blind safe, distinguishable in greyscale)
COLOR_IRLS = "#1f4e79"   # dark blue
COLOR_DSM  = "#c0392b"   # warm red
COLOR_FBAR = "#1c1c1c"   # near-black for record value overlay
COLOR_FCUR = "#e67e22"   # orange for current iterate value
COLOR_REF  = "#2c7a30"   # green for f* reference line
COLOR_AUX  = "#7f7f7f"   # mid-grey for asymptote / power-law guides

# Sequential colour ramps for parameter sweeps (light -> dark)
RAMP_BLUES   = "Blues"
RAMP_REDS    = "Reds"
RAMP_ORANGES = "Oranges"
RAMP_PURPLES = "Purples"


def apply_style() -> None:
    """Set rcParams for publication-quality figures.

    Sizes are tuned for one-figure-per-page rendering at \\textwidth in the
    LaTeX report: the report's default \\textwidth is ~430pt, so a 13in
    matplotlib figure at 220 dpi gives a printed width of ~6in (~430pt) and
    leaves typography legible without forced zoom.
    """
    _mpl.rcParams.update({
        # Fonts — larger across the board so the rendered figure is legible
        # at \\textwidth without zooming.
        "font.family":        "serif",
        "font.size":          13,
        "axes.titlesize":     15,
        "axes.labelsize":     13,
        "legend.fontsize":    11,
        "xtick.labelsize":    12,
        "ytick.labelsize":    12,
        "figure.titlesize":   16,
        # Math text rendered with the Computer Modern lookalike that
        # ships with matplotlib (no external LaTeX needed).
        "mathtext.fontset":   "cm",
        # Lines & markers
        "lines.linewidth":    1.8,
        "lines.markersize":   4.0,
        "axes.linewidth":     1.0,
        # Grid
        "axes.grid":          True,
        "grid.alpha":         0.25,
        "grid.linestyle":     "-",
        "grid.linewidth":     0.6,
        # Spines
        "axes.spines.top":    False,
        "axes.spines.right":  False,
        # Legend
        "legend.frameon":     True,
        "legend.framealpha":  0.92,
        "legend.edgecolor":   "#cccccc",
        "legend.handlelength": 2.0,
        # Figure
        "figure.dpi":         110,
        "savefig.dpi":        220,
        "savefig.bbox":       "tight",
        "savefig.pad_inches": 0.08,
    })


# Standard figure sizes (inches) — use these so all figures share scale.
SIZE_SINGLE   = (10.0, 5.5)   # one panel, full report width
SIZE_DOUBLE   = (14.0, 5.5)   # two panels side-by-side, full width
SIZE_TALL     = (10.0, 7.5)   # one panel, taller (annotated)
SIZE_QUAD     = (14.0, 9.0)   # 2x2 panel layout


def style_axes(ax) -> None:
    """Light per-axis polishing (consistent margins, tick orientation)."""
    ax.tick_params(direction="out", length=3.0, width=0.8)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color("#333333")
