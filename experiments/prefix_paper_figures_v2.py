"""Draw the compact recall comparison used by the paper."""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import StrMethodFormatter
import numpy as np

from experiments.prefix_figures_v2 import METHODS, best_at_recall


def draw(comparison, output):
    comparison, output = Path(comparison), Path(output)
    output.mkdir(parents=True, exist_ok=True)
    rows = json.loads((comparison / "main/settings.json").read_text())
    identity = json.loads((comparison / "inputs-manifest.json").read_text())["identity"]
    plt.rcParams.update({"font.size": 12, "axes.spines.top": False,
                         "axes.spines.right": False, "savefig.dpi": 180})
    figure, axes = plt.subplots(1, 2, figsize=(8.8, 3.8))
    for axis, k in zip(axes, [1, 100]):
        for method, (label, color) in METHODS.items():
            targets = sorted({.5, 1.0} | {row["recall"] for row in rows
                if row["qualified"] and row["method"] == method
                and row["top_k"] == k and row["recall"] >= .5})
            choices = [best_at_recall(rows, method, k, target) for target in targets]
            times = [row["p50_ms"] if row else np.nan for row in choices]
            axis.step(np.asarray(targets) * 100, times, where="pre", color=color,
                      label=label, linewidth=2)
        axis.set(title=f"Return {k} {'document' if k == 1 else 'documents'}",
                 xlabel="Required recall (%)", xlim=(50, 100), yscale="log")
        axis.yaxis.set_major_formatter(StrMethodFormatter("{x:g}"))
        axis.set_xticks([50, 70, 90, 100])
        axis.grid(alpha=.2)
    axes[0].set_ylabel("Median query time (ms)")
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="lower center", ncol=3, frameon=False, fontsize=10)
    figure.suptitle(f"{identity['documents']:,} documents · {identity['queries']:,} queries · "
                   f"{identity['dimensions']} bits", fontsize=13)
    figure.subplots_adjust(left=.085, right=.97, top=.79, bottom=.24, wspace=.29)
    for extension in ["png", "pdf"]:
        figure.savefig(output / f"qwen32-comparison.{extension}")
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("comparison", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    draw(args.comparison, args.output)


if __name__ == "__main__":
    main()
