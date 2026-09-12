"""Draw the tested recall/time choices and export the repeated comparisons."""

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

METHODS = {
    "scan": ("A: cluster then scan", "#286493"),
    "branch": ("B: Bitplanes", "#C36620"),
    "prefix": ("C: Backward Walk", "#7850A4"),
}


def best_at_recall(rows, method, top_k, target):
    """Choose a measured setting that actually meets this recall requirement."""
    eligible = [row for row in rows if row["method"] == method
                and row["top_k"] == top_k and row["qualified"]
                and row["recall"] >= target]
    return min(eligible, key=lambda row: row["p50_ms"]) if eligible else None


def repeated_comparisons(main, repeats, targets, top_ks):
    """Selection comes from the first sweep; repeat results cannot replace it."""
    by_id = {row["setting_id"]: row for row in repeats}
    output = []
    for k in top_ks:
        for target in targets:
            for method in METHODS:
                selected = best_at_recall(main, method, k, target)
                if selected is None:
                    continue
                repeated = by_id.get(selected["setting_id"])
                if repeated is None or not repeated["qualified"]:
                    continue
                output.append(dict(target=target, meets_target=repeated["recall"] >= target,
                                   **repeated))
    return output


def read_json(path):
    return json.loads(path.read_text())


def save_csv(path, rows):
    if not rows:
        return
    columns = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def draw_study(folder):
    config = read_json(folder / "configuration.json")
    identity = read_json(folder / "inputs-manifest.json")["identity"]
    rows = read_json(folder / "main/settings.json")
    repeat_path = folder / "repeat/settings.json"
    repeats = read_json(repeat_path) if repeat_path.exists() else []
    output = folder / "figures"
    output.mkdir(exist_ok=True)
    comparison = repeated_comparisons(rows, repeats, config["recall_targets"], config["top_ks"])
    save_csv(output / "repeated-comparisons.csv", comparison)
    (output / "repeated-comparisons.json").write_text(json.dumps(comparison, indent=2) + "\n")

    plt.rcParams.update({"font.size": 10, "axes.spines.top": False,
                         "axes.spines.right": False, "savefig.dpi": 180})
    counts = f"{identity['documents']:,} documents · {identity['queries']:,} queries · {identity['dimensions']} bits"
    for k in config["top_ks"]:
        fig, axis = plt.subplots(figsize=(8.2, 4.4), layout="constrained")
        for method, (label, color) in METHODS.items():
            targets = sorted({.5, 1.0} | {row["recall"] for row in rows
                             if row["qualified"] and row["method"] == method
                             and row["top_k"] == k and row["recall"] >= .5})
            chosen = [best_at_recall(rows, method, k, target) for target in targets]
            times = [row["p50_ms"] if row else np.nan for row in chosen]
            # These are measured qualifying choices, not an interpolated speed model.
            # A setting at90% remains eligible through90%, but not above it.
            axis.step(np.asarray(targets) * 100, times, where="pre", label=label, color=color, linewidth=2)
        axis.set(xlabel=f"Required recall@{k} (%)", ylabel="Median query time (ms)",
                 title=f"Fastest tested settings\n{counts}", xlim=(50, 100), yscale="log")
        axis.grid(alpha=.2)
        axis.legend(frameon=False)
        for extension in ("png", "pdf"):
            fig.savefig(output / f"recall-latency-k{k}.{extension}")
        plt.close(fig)

        fig, axes = plt.subplots(1, 3, figsize=(11, 3.7), layout="constrained", sharey=True)
        clusters = sorted({row["clusters"] for row in rows})
        for axis, target in zip(axes, [.8, .95, .99]):
            for method, (label, color) in METHODS.items():
                choices = [best_at_recall([row for row in rows if row["clusters"] == count],
                                          method, k, target) for count in clusters]
                values = [row["p50_ms"] if row else np.nan for row in choices]
                axis.plot(clusters, values, "o-", label=label, color=color, linewidth=1.5)
            axis.set(xlabel="Number of clusters", title=f"Recall@{k} ≥ {target:.0%}",
                     xscale="log", yscale="log")
            axis.grid(alpha=.2)
        axes[0].set_ylabel("Median query time (ms)")
        axes[0].legend(fontsize=8, frameon=False)
        fig.suptitle(counts)
        for extension in ("png", "pdf"):
            fig.savefig(output / f"clusters-k{k}.{extension}")
        plt.close(fig)

    lines = ["# Reading the figures", "", counts, "",
             "Each curve chooses the fastest tested configuration that reaches the required recall.",
             "A step means the best qualifying choice changed. It does not interpolate between experiments.",
             "Each method chooses its own clusters, probes and local settings, with the same inputs and RAM limit.",
             "A gap means no completed, qualifying setting was found there. A missing point is not a measured loss.",
             "Query time includes routing and native search. Query embeddings are cached.", "",
             "The comparison CSV uses configurations chosen by the first sweep, measured again in three process blocks.",
             "The minimum and maximum process medians show repeat variation; they are not confidence intervals.",
             "Logical index bytes count stored fields. Process peak includes runtime and index construction.", ""]
    (output / "README.md").write_text("\n".join(lines))
    return comparison


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("studies", type=Path, nargs="+")
    args = parser.parse_args()
    for folder in args.studies:
        comparisons = draw_study(folder)
        print(f"{folder.name}: {len(comparisons)} repeated target comparisons", flush=True)


if __name__ == "__main__":
    main()
