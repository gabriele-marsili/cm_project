"""Shared matplotlib style and palette for all experiment figures."""
import matplotlib as _mpl
import matplotlib.pyplot as _plt


COLOR_IRLS = "#1f4e79"
COLOR_DSM  = "#c0392b"
COLOR_FBAR = "#1c1c1c"
COLOR_FCUR = "#e67e22"
COLOR_REF  = "#2c7a30"
COLOR_AUX  = "#7f7f7f"

RAMP_BLUES   = "Blues"
RAMP_REDS    = "Reds"
RAMP_ORANGES = "Oranges"
RAMP_PURPLES = "Purples"


def apply_style() -> None:
    _mpl.rcParams.update({
        "font.family":        "serif",
        "font.size":          13,
        "axes.titlesize":     15,
        "axes.labelsize":     13,
        "legend.fontsize":    11,
        "xtick.labelsize":    12,
        "ytick.labelsize":    12,
        "figure.titlesize":   16,
        "mathtext.fontset":   "cm",
        "lines.linewidth":    1.8,
        "lines.markersize":   4.0,
        "axes.linewidth":     1.0,
        "axes.grid":          True,
        "grid.alpha":         0.25,
        "grid.linestyle":     "-",
        "grid.linewidth":     0.6,
        "axes.spines.top":    False,
        "axes.spines.right":  False,
        "legend.frameon":     True,
        "legend.framealpha":  0.92,
        "legend.edgecolor":   "#cccccc",
        "legend.handlelength": 2.0,
        "figure.dpi":         110,
        "savefig.dpi":        220,
        "savefig.bbox":       "tight",
        "savefig.pad_inches": 0.08,
    })


SIZE_SINGLE = (10.0, 5.5)
SIZE_DOUBLE = (14.0, 5.5)
SIZE_TALL   = (10.0, 7.5)
SIZE_QUAD   = (14.0, 9.0)


def style_axes(ax) -> None:
    ax.tick_params(direction="out", length=3.0, width=0.8)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color("#333333")
