"""Re-plot the SGPTL long-run figure from the saved CSV (no re-compute).

experiment_sgptl_long_run.py stores the full gap trajectory and produces the
figure in one shot. This script reproduces the same figure from
results/tables/sgptl_long_run.csv without re-running the experiment, which
takes ~30 minutes on california. It is useful when only the plot needs
restyling.

The CSV-only path needs a numerical floor because the diabetes trajectory
reaches floating-point underflow (~1e-300), which otherwise stretches the
y-axis and hides the relevant range.
"""

import csv
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir))
from _plot_style import (  # noqa: E402
    SIZE_DOUBLE,
    apply_style,
    plot_long_run_panel,
)

apply_style()

FIG_DIR = os.path.join(
    os.path.dirname(__file__), os.pardir, "results", "figures",
)
TAB_PATH = os.path.join(
    os.path.dirname(__file__), os.pardir, "results", "tables",
    "sgptl_long_run.csv",
)
FLOOR = 1e-12


def load_csv() -> dict:
    """Group the long-run CSV rows by dataset, sorted by iteration count.

    Returns dict: dataset -> [{k, observed, predicted}, ...] ordered by k.
    """
    rows: dict = {}
    with open(TAB_PATH, newline="") as fh:
        rd = csv.DictReader(fh)
        for r in rd:
            d = r["dataset"]
            rows.setdefault(d, []).append({
                "k": int(r["k"]),
                "observed": float(r["observed_gap"]),
                "predicted": float(r["predicted_envelope"]),
            })
    for d in rows:
        rows[d].sort(key=lambda x: x["k"])
    return rows


def main() -> None:
    # f* per dataset (relative gap = absolute gap / |f*|); matches the
    # long_run_cache npz f_star values.
    FSTAR = {"diabetes": 52.948763, "california": 2358.019449}
    rows = load_csv()
    fig, axes = plt.subplots(1, 2, figsize=SIZE_DOUBLE)
    for ax, name in zip(axes, ["diabetes", "california"]):
        fs = FSTAR[name]
        data = rows[name]
        ks = [d["k"] for d in data]
        obs = [d["observed"] / fs for d in data]
        # g_0 implicit from predicted envelope: predicted = g_0 / sqrt(k).
        g0 = data[0]["predicted"] * (ks[0] ** 0.5) / fs
        i_max_str = f"{int(max(ks)):,}".replace(",", r"{,}")
        title = (
            rf"{name} cold start  ($H=200$, $\lambda=0.1$, "
            rf"$i_{{\max}}={i_max_str}$)"
        )
        plot_long_run_panel(ax, ks, obs, g0, title, floor=FLOOR)

    fig.tight_layout()
    out = os.path.join(FIG_DIR, "sgptl_long_run.pdf")
    fig.savefig(out)
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
