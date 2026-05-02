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
    """Set rcParams for publication-quality figures."""
    _mpl.rcParams.update({
        # Fonts
        "font.family":        "serif",
        "font.size":          11,
        "axes.titlesize":     12,
        "axes.labelsize":     11,
        "legend.fontsize":    9,
        "xtick.labelsize":    10,
        "ytick.labelsize":    10,
        # Math text rendered with the Computer Modern lookalike that
        # ships with matplotlib (no external LaTeX needed).
        "mathtext.fontset":   "cm",
        # Lines & markers
        "lines.linewidth":    1.6,
        "lines.markersize":   3.5,
        "axes.linewidth":     0.8,
        # Grid
        "axes.grid":          True,
        "grid.alpha":         0.25,
        "grid.linestyle":     "-",
        "grid.linewidth":     0.5,
        # Spines
        "axes.spines.top":    False,
        "axes.spines.right":  False,
        # Legend
        "legend.frameon":     False,
        "legend.handlelength": 2.0,
        # Figure
        "figure.dpi":         110,
        "savefig.dpi":        220,
        "savefig.bbox":       "tight",
        "savefig.pad_inches": 0.05,
    })


def style_axes(ax) -> None:
    """Light per-axis polishing (consistent margins, tick orientation)."""
    ax.tick_params(direction="out", length=3.0, width=0.8)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color("#333333")
