"""Figure: where crossing the queue and ranking restarts by yield each add repairs.

Reads the budget curves written by queue_restart_decision.py (27B, calls) and
nebius_queue_restart.py (70B, trajectory messages) and draws the two paired
components with their 95% replicate bands. Writes paper/figures/queue_crossing.png.

Run: .ai/swebench/venv/bin/python analysis/queue_crossing_figure.py
"""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "paper/figures/queue_crossing.png"
SURFACE, INK, INK2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
SERIES = (("adaptive_minus_deferred_rate", "Crossing the queue", "#2a78d6"),
          ("deferred_rate_minus_ranked_breadth", "Ranking restarts by yield", "#eb6834"))
PANELS = (("queue_restart_decision.json", "Qwen3.8-27B, 2,464 held-out issues", "share of all-rollout calls"),
          ("nebius_queue_restart.json", "Llama 3 70B agent, 1,803 held-out issues", "share of all-rollout messages"))


def main():
    plt.rcParams.update({"font.family": "sans-serif", "font.size": 9, "axes.edgecolor": AXIS,
                         "axes.labelcolor": INK2, "xtick.color": MUTED, "ytick.color": MUTED})
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.5), dpi=200, facecolor=SURFACE)
    for ax, (name, title, unit) in zip(axes, PANELS):
        curve = json.loads((ROOT / "evidence/results" / name).read_text())["curve"]
        x = [100 * f for f in curve["fractions"]]
        ax.set_facecolor(SURFACE)
        ax.axhline(0, color=AXIS, lw=0.8, zorder=1)
        ax.axvline(100 / 3, color=GRID, lw=0.8, zorder=0)
        for key, label, color in SERIES:
            band = curve[key]
            ax.fill_between(x, band["lo"], band["hi"], color=color, alpha=0.10, lw=0, zorder=2)
            ax.plot(x, band["mean"], color=color, lw=2, solid_capstyle="round", zorder=3, label=label)
            peak = max(range(len(x)), key=lambda i: band["mean"][i])
            if band["lo"][peak] > 0:  # label a peak only when its band excludes zero
                ax.annotate(f"+{band['mean'][peak]:.0f} at {x[peak]:.0f}%", (x[peak], band["mean"][peak]),
                            xytext=(6, 4), textcoords="offset points", color=INK2, fontsize=8)
        ax.text(100 / 3 + 0.8, ax.get_ylim()[0], " one attempt per issue", color=MUTED, fontsize=7.5,
                va="bottom")
        ax.set_title(title, loc="left", color=INK, fontsize=9.5)
        ax.set_xlabel(f"Budget ({unit}, %)")
        ax.set_xlim(x[0], x[-1])
        ax.grid(axis="y", color=GRID, lw=0.6)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    axes[0].set_ylabel("Extra issues solved")
    axes[0].legend(loc="upper left", frameon=False, labelcolor=INK2, fontsize=8)
    fig.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, facecolor=SURFACE)
    print(OUT)


if __name__ == "__main__":
    main()
