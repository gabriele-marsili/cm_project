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


# ---------------------------------------------------------------------------
# Shared plot panel for the SGPTL long-run figure.
# Used by experiment_sgptl_long_run.py (full run) and
# _replot_sgptl_long_run.py (CSV-only re-plot).
# ---------------------------------------------------------------------------
def plot_long_run_panel(
    ax,
    ks,
    obs_gaps,
    gap0: float,
    title: str,
    floor: float = None,
) -> None:
    """Draw one panel of the SGPTL long-run figure on `ax`.

    Args:
        ks: iteration indices (1-D array).
        obs_gaps: observed record gap at each k.
        gap0: initial gap, used to draw the g_0 / sqrt(i) envelope.
        title: panel title.
        floor: if given, clip obs_gaps from below and add an annotation at
            the first floored point (used by the CSV replot to handle the
            1e-300 underflow on diabetes).
    """
    import numpy as np

    ks = np.asarray(ks, dtype=float)
    obs = np.asarray(obs_gaps, dtype=float)
    if floor is not None:
        obs = np.maximum(obs, floor)

    ks_dense = np.geomspace(ks.min(), ks.max(), 200)
    env_dense = gap0 / np.sqrt(ks_dense)

    ax.loglog(
        ks_dense, env_dense,
        color="grey", linestyle="--", linewidth=1.4,
        label=r"$g_{0}^{\mathrm{rel}} / \sqrt{i}$  (theoretical envelope)",
    )
    ax.loglog(
        ks, obs,
        color=COLOR_DSM, marker="o", markersize=7, linewidth=1.8,
        label=r"SGPTL record  $(\bar{f}^{\,i} - f^{*})/|f^{*}|$",
    )
    if floor is not None:
        floored = np.where(obs <= floor * 5)[0]
        if len(floored) > 0:
            j = int(floored[0])
            ax.annotate(
                rf"$\leq 10^{{{int(np.log10(floor))}}}$",
                xy=(ks[j], obs[j]),
                xytext=(10, 10),
                textcoords="offset points",
                ha="left", fontsize=9, color=COLOR_DSM,
            )
        ax.set_ylim(floor / 5, max(env_dense.max(), obs.max()) * 5)

    ax.set_xlabel(r"Iteration $i$  (log scale)")
    ax.set_ylabel(r"relative gap  $(f - f^{*})/|f^{*}|$  (log scale)")
    ax.set_title(title)
    ax.legend(loc="lower left", fontsize=9)
    style_axes(ax)
