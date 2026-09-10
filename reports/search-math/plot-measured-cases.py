"""Plot the saved independent choices at one common size and recall requirement."""
import csv
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


REPORT = Path(__file__).resolve().parent
PROJECT = REPORT.parents[1]
SOURCES = [
    ("Real Nomic embeddings", "independent-evaluation-2026-09-10/report/targets.csv"),
    ("Fixed strong coordinates", "controlled-fixed-2026-09-10/evaluation/report/targets.csv"),
    ("Changing strong coordinates", "controlled-adaptive-2026-09-10/evaluation/report/targets.csv"),
]
METHODS = ["scan", "branch", "keys"]
COLORS = ["#3978bb", "#d77832", "#8860b6"]

fig, axes = plt.subplots(3, 1, figsize=(7, 7), sharey=True)
evidence = []
for ax, (title, relative) in zip(axes, SOURCES, strict=True):
    source = PROJECT / "results" / relative
    with source.open() as file:
        rows = [row for row in csv.DictReader(file)
                if int(row["documents"]) == 1000000
                and float(row["target"]) == .99 and row["selection"] == "conservative"]
    assert len(rows) == 3 and {row["method"] for row in rows} == set(METHODS)
    for position, method in enumerate(METHODS):
        row = next(row for row in rows if row["method"] == method)
        assert row["met_target"] == row["qualified_environment"] == "True"
        time = float(row["p50_ms"])
        low, high = float(row["p50_low"]), float(row["p50_high"])
        ax.errorbar(position, time, yerr=[[max(0, time-low)], [max(0, high-time)]],
                    fmt="o", color=COLORS[position], markersize=6, capsize=4)
        ax.annotate(f"{time:.4f} ms\n{100*float(row['recall']):.2f}% recall",
                    (position, time), xytext=(0, 8), textcoords="offset points",
                    ha="center", fontsize=10)
    ax.set_title(title, fontsize=11, pad=10)
    ax.set_xticks(range(3), ["A: scan", "B: bitplanes", "C: keys"])
    ax.set_xlim(-.55, 2.55)
    ax.set_yscale("log")
    ax.set_ylim(.006, 200)
    ax.tick_params(labelsize=9)
    ax.set_ylabel("Query time (ms)", fontsize=9)
    ax.grid(axis="y", alpha=.2)
    ax.spines[["top", "right"]].set_visible(False)
    evidence.append(dict(source=str(source.relative_to(PROJECT)),
                         sha256=hashlib.sha256(source.read_bytes()).hexdigest(), rows=rows))
fig.tight_layout()
fig.savefig(REPORT / "figures" / "measured-cases.pdf", bbox_inches="tight")
fig.savefig(REPORT / "figures" / "measured-cases.png", dpi=170, bbox_inches="tight")
(REPORT / "evidence" / "measured-cases.json").write_text(json.dumps(evidence, indent=2) + "\n")
