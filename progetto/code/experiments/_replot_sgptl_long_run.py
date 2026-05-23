"""Re-plot the SGPTL long-run result from the saved CSV (no re-compute).

The figure produced by experiment_sgptl_long_run.py shows the gap reach
floating-point underflow on diabetes, which stretches the y-axis to
~10^-279 and hides the relevant range. This script re-plots the same
points from results/tables/sgptl_long_run.csv with a sensible floor and
clean titles. It does NOT re-run the experiment.
"""
import csv
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir))
from _plot_style import apply_style, style_axes, COLOR_DSM, SIZE_DOUBLE  # noqa: E402

apply_style()

FIG_DIR = os.path.join(os.path.dirname(__file__), os.pardir, "results", "figures")
TAB_PATH = os.path.join(
    os.path.dirname(__file__), os.pardir, "results", "tables", "sgptl_long_run.csv"
)
FLOOR = 1e-12


def load_csv():
    rows = {}
    with open(TAB_PATH, newline="") as fh:
        rd = csv.DictReader(fh)
        for r in rd:
            d = r["dataset"]
            rows.setdefault(d, []).append(
                {
                    "k": int(r["k"]),
                    "observed": max(float(r["observed_gap"]), FLOOR),
                    "predicted": float(r["predicted_envelope"]),
                }
            )
    for d in rows:
        rows[d].sort(key=lambda x: x["k"])
    return rows


def main():
    rows = load_csv()
    order = ["diabetes", "california"]

    fig, axes = plt.subplots(1, 2, figsize=SIZE_DOUBLE)
    for ax, name in zip(axes, order):
        data = rows[name]
        ks = np.array([d["k"] for d in data], dtype=float)
        obs = np.array([d["observed"] for d in data], dtype=float)
        env = np.array([d["predicted"] for d in data], dtype=float)

        # Smooth envelope across the iteration range
        ks_dense = np.geomspace(ks.min(), ks.max(), 200)
        # g_0 implicit from CSV: predicted = g_0 / sqrt(k) => g_0 = predicted * sqrt(k)
        g0 = float(env[0]) * np.sqrt(ks[0])
        env_dense = g0 / np.sqrt(ks_dense)

        ax.loglog(
            ks_dense,
            env_dense,
            color="grey",
            linestyle="--",
            linewidth=1.4,
            label=r"$g_{0} / \sqrt{i}$ (theoretical envelope)",
        )
        ax.loglog(
            ks,
            obs,
            color=COLOR_DSM,
            marker="o",
            markersize=7,
            linewidth=1.8,
            label=r"SGPTL record  $\bar{f}^{\,i} - f^{*}$",
        )
        # annotate the first floored point only (avoids overlap)
        floored = np.where(obs <= FLOOR * 5)[0]
        if len(floored) > 0:
            j = int(floored[0])
            ax.annotate(
                r"$\leq 10^{-12}$",
                xy=(ks[j], obs[j]),
                xytext=(10, 10),
                textcoords="offset points",
                ha="left",
                fontsize=9,
                color=COLOR_DSM,
            )

        ax.set_xlabel(r"Iteration $i$  (log scale)")
        ax.set_ylabel(r"gap $\bar{f}^{\,i} - f^{*}$  (log scale)")
        ax.set_ylim(FLOOR / 5, max(env_dense.max(), obs.max()) * 5)
        i_max_str = f"{int(ks.max()):,}".replace(",", r"{,}")
        ax.set_title(
            rf"{name} cold start  ($H=200$, $\lambda=0.1$, $i_{{\max}}={i_max_str}$)"
        )
        ax.legend(loc="lower left", fontsize=9)
        style_axes(ax)

    fig.tight_layout()
    out = os.path.join(FIG_DIR, "sgptl_long_run.pdf")
    fig.savefig(out)
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
