"""Render the fixed-grid scaling figures and Markdown report."""

import math
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter


DISPLAY_LABELS = {
    "scan": "Scan",
    "target95": "95% recall target",
    "target99": "99% recall target",
    "unlimited": "Unlimited branch",
}
LINE_STYLES = {
    "scan": {"linestyle": "-", "marker": "o"},
    "target95": {"linestyle": "--", "marker": "s", "markerfacecolor": "white"},
    "target99": {"linestyle": ":", "marker": "^", "markerfacecolor": "white"},
    "unlimited": {"linestyle": "-.", "marker": "D"},
}


def _format_count(value: int) -> str:
    if value >= 1_000_000:
        return f"{value / 1_000_000:g}M"
    if value >= 1_000:
        return f"{value / 1_000:g}k"
    return str(value)


def _set_pool_ticks(axis, pools: list[int]) -> None:
    axis.set_xscale("log")
    axis.set_xticks(pools, [_format_count(pool) for pool in pools])
    axis.xaxis.set_minor_formatter(NullFormatter())


def _series(
    pools: list[int],
    evaluation: list[dict],
    points: list[dict],
    label: str,
    field: str,
) -> tuple[list[float], list[bool]]:
    lookup = {
        (row["pool_size"], row["method"], row["node_budget"]): row
        for row in evaluation
    }
    values = []
    misses = []
    for pool in pools:
        if label == "scan":
            row = lookup.get((pool, "scan", 0))
            value = row.get(field) if row and row["status"] == "complete" else None
            miss = False
        elif label == "unlimited":
            row = lookup.get((pool, "branch", 0))
            value = row.get(field) if row and row["status"] == "complete" else None
            miss = False
        else:
            target = int(label.removeprefix("target")) / 100
            point = next(
                item
                for item in points
                if item["pool_size"] == pool and math.isclose(item["target"], target)
            )
            measured = lookup.get((pool, "branch", point["node_budget"]))
            value = point.get(field)
            if field not in point and measured and measured["status"] == "complete":
                value = measured.get(field)
            miss = value is not None and not point["achieved"]
        values.append(float(value) if value is not None else np.nan)
        misses.append(miss)
    return values, misses


def _plot_scaling(
    directory: Path,
    pools: list[int],
    evaluation: list[dict],
    points: list[dict],
) -> None:
    colors = {
        "scan": "#222222",
        "target95": "#2878b5",
        "target99": "#d65f2d",
        "unlimited": "#6b6b6b",
    }
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), constrained_layout=True)
    fields = ("api_p50_ms", "api_p95_ms")
    titles = ("Median search time", "95th-percentile search time")
    for axis, field, title in zip(axes, fields, titles):
        miss_label_added = False
        for label in ("scan", "target95", "target99", "unlimited"):
            values, misses = _series(pools, evaluation, points, label, field)
            axis.plot(
                pools,
                values,
                linewidth=1.8,
                label=DISPLAY_LABELS[label],
                color=colors[label],
                **LINE_STYLES[label],
            )
            for pool, value, miss in zip(pools, values, misses):
                if miss and math.isfinite(value):
                    axis.scatter(
                        pool,
                        value,
                        marker="x",
                        s=70,
                        linewidth=2.0,
                        color="#a40000",
                        zorder=5,
                        label="Below recall target" if not miss_label_added else None,
                    )
                    miss_label_added = True
        _set_pool_ticks(axis, pools)
        axis.set_yscale("log")
        axis.set_xlabel("Documents in pool")
        axis.set_ylabel("Search time (ms, log scale)")
        axis.set_title(title)
        axis.grid(alpha=0.2)
    axes[0].legend(frameon=False, ncol=2, fontsize=8)
    fig.suptitle("Cached test queries", fontsize=11)
    fig.savefig(directory / "scaling.png", dpi=180)
    fig.savefig(directory / "scaling.svg")
    plt.close(fig)


def _plot_speedup(directory: Path, pools: list[int], points: list[dict]) -> None:
    fig, axis = plt.subplots(figsize=(7.5, 4.5), constrained_layout=True)
    axis.axhline(1.0, color="#555555", linewidth=1, linestyle="--")
    miss_label_added = False
    for target, label, color in (
        (0.95, "target95", "#2878b5"),
        (0.99, "target99", "#d65f2d"),
    ):
        selected = [
            next(
                row
                for row in points
                if row["pool_size"] == pool and row["target"] == target
            )
            for pool in pools
        ]
        values = [
            row["speedup"] if row["speedup"] is not None else np.nan
            for row in selected
        ]
        low = [
            row["speedup_ci_low"] if row["speedup_ci_low"] is not None else np.nan
            for row in selected
        ]
        high = [
            row["speedup_ci_high"] if row["speedup_ci_high"] is not None else np.nan
            for row in selected
        ]
        axis.plot(
            pools,
            values,
            color=color,
            label=DISPLAY_LABELS[label],
            **LINE_STYLES[label],
        )
        axis.fill_between(pools, low, high, color=color, alpha=0.12)
        for pool, value, row in zip(pools, values, selected):
            if math.isfinite(value) and not row["achieved"]:
                axis.scatter(
                    pool,
                    value,
                    marker="x",
                    s=70,
                    linewidth=2,
                    color="#a40000",
                    zorder=5,
                    label="Below recall target" if not miss_label_added else None,
                )
                miss_label_added = True
    _set_pool_ticks(axis, pools)
    axis.set_xlabel("Pool size N")
    axis.set_ylabel("Scan median / branch median")
    axis.set_title("Median speedup on cached test queries")
    axis.grid(alpha=0.2)
    axis.legend(frameon=False)
    fig.savefig(directory / "speedup.png", dpi=180)
    plt.close(fig)


def _plot_work(
    directory: Path,
    pools: list[int],
    evaluation: list[dict],
    points: list[dict],
) -> None:
    fields = (
        ("mean_fraction_fully_scored", "Fraction fully scored"),
        ("mean_nodes", "Branch nodes"),
        ("mean_bitplane_words", "Split bitmap words"),
        ("native_peak_rss_bytes", "Worker peak memory (GiB)"),
    )
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), constrained_layout=True)
    for axis, (field, title) in zip(axes.flat, fields):
        for label in ("scan", "target95", "target99", "unlimited"):
            values, _ = _series(pools, evaluation, points, label, field)
            if field == "native_peak_rss_bytes":
                values = [value / 2**30 for value in values]
            axis.plot(
                pools,
                values,
                linewidth=1.4,
                label=DISPLAY_LABELS[label],
                **LINE_STYLES[label],
            )
        _set_pool_ticks(axis, pools)
        axis.set_title(title)
        axis.set_xlabel("Pool size N")
        axis.grid(alpha=0.2)
    axes.flat[0].legend(frameon=False, ncol=2, fontsize=8)
    fig.suptitle("Cached test queries", fontsize=11)
    fig.savefig(directory / "work.png", dpi=180)
    plt.close(fig)


def _plot_budgets(
    directory: Path,
    development: list[dict],
    pools: list[int],
    budgets: list[int],
) -> None:
    ordered = [budget for budget in budgets if budget != 0] + [0]
    positions = list(range(len(ordered)))
    lookup = {
        (row["pool_size"], row["method"], row["node_budget"]): row
        for row in development
    }
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), constrained_layout=True)
    for pool in pools:
        recall = []
        latency = []
        for budget in ordered:
            row = lookup[(pool, "branch", budget)]
            recall.append(
                row["mean_recall"] if row["status"] == "complete" else np.nan
            )
            latency.append(
                row["api_p50_ms"] if row["status"] == "complete" else np.nan
            )
        label = f"N={_format_count(pool)}"
        axes[0].plot(positions, recall, marker="o", linewidth=1.2, label=label)
        axes[1].plot(positions, latency, marker="o", linewidth=1.2, label=label)
    labels = ["unlimited" if budget == 0 else f"{budget:,}" for budget in ordered]
    titles = ("Development recall", "Development median search time")
    ylabels = ("Candidate recall", "Search call time (ms)")
    for axis, title, ylabel in zip(axes, titles, ylabels):
        axis.set_xticks(positions, labels, rotation=45, ha="right")
        axis.set_title(title)
        axis.set_xlabel("Tested node budget")
        axis.set_ylabel(ylabel)
        axis.grid(alpha=0.2)
    axes[0].legend(frameon=False, fontsize=8, ncol=2)
    fig.suptitle("Cached development queries", fontsize=11)
    fig.savefig(directory / "budgets.png", dpi=180)
    plt.close(fig)


def write_plots(
    directory: Path,
    pools: list[int],
    evaluation: list[dict],
    points: list[dict],
    development: list[dict],
    budgets: list[int],
) -> None:
    """Write all fixed report figures from prepared analysis values."""
    _plot_scaling(directory, pools, evaluation, points)
    _plot_speedup(directory, pools, points)
    _plot_work(directory, pools, evaluation, points)
    _plot_budgets(directory, development, pools, budgets)


def _crossovers(
    points: list[dict],
    evaluation: list[dict],
    percentile: int,
) -> dict[float, int | None]:
    scan = {row["pool_size"]: row for row in evaluation if row["method"] == "scan"}
    field = f"api_p{percentile}_ms"
    result = {}
    for target in sorted({row["target"] for row in points}):
        eligible = []
        for row in points:
            scan_row = scan.get(row["pool_size"])
            if (
                row["target"] == target
                and row["achieved"]
                and scan_row
                and scan_row["status"] == "complete"
                and row[field] < scan_row[field]
            ):
                eligible.append(row["pool_size"])
        result[target] = min(eligible) if eligible else None
    return result


def report_text(
    metadata: dict,
    points: list[dict],
    evaluation: list[dict],
    failures: list[dict],
) -> str:
    """Format the report from prepared measurements and saved failure records."""
    query_count = len(metadata["query_splits"]["evaluation"])
    p50_crossovers = _crossovers(points, evaluation, 50)
    p95_crossovers = _crossovers(points, evaluation, 95)
    lines = [
        "# Fixed-grid scan versus bitplane scaling",
        "",
        (
            f"This report uses {query_count} evaluation queries with three measured repeats per "
            "query. Development choices were fixed before the test run. Each choice is the fastest "
            "tested complete setting that reached its development target; it is not a global "
            "optimum."
        ),
        "",
        "## Results",
        "",
        (
            "| Pool | Target | Budget | Validation recall | Test recall | p50 ms | "
            "p95 ms | Achieved | Speedup |"
        ),
        "|---:|---:|---:|---:|---:|---:|---:|:---:|---:|",
    ]
    for row in points:
        budget = "unlimited" if row["node_budget"] == 0 else row["node_budget"]

        def shown(value, digits=3):
            return "—" if value is None else f"{value:.{digits}f}"

        budget_text = budget if budget is not None else "—"
        achieved = "yes" if row["achieved"] else "no"
        lines.append(
            f"| {row['pool_size']:,} | {row['target']:.0%} | {budget_text} | "
            f"{shown(row['development_recall'])} | {shown(row['evaluation_recall'])} | "
            f"{shown(row['api_p50_ms'])} | {shown(row['api_p95_ms'])} | {achieved} | "
            f"{shown(row['speedup'], 2)} |"
        )
    lines.extend(
        [
            "",
            (
                f"There are {len(failures)} incomplete or failed scheduled cases. They remain in "
                "[failures.csv](failures.csv) and [settings.csv](settings.csv), and they do not "
                "contribute full-case percentiles or choices."
            ),
            "",
        ]
    )
    if all(value is None for value in p50_crossovers.values()):
        lines.append(
            "No observed median crossover among target cases that achieved their evaluation target."
        )
    else:
        observed = ", ".join(
            f"{target:.0%}: {value:,}"
            for target, value in p50_crossovers.items()
            if value is not None
        )
        lines.append(f"Observed median crossovers: {observed}.")
    if all(value is None for value in p95_crossovers.values()):
        lines.append(
            "No observed p95 crossover among target cases that achieved their evaluation target."
        )
    else:
        observed = ", ".join(
            f"{target:.0%}: {value:,}"
            for target, value in p95_crossovers.items()
            if value is not None
        )
        lines.append(f"Observed p95 crossovers: {observed}.")
    lines.extend(
        [
            "",
            (
                "The speedup intervals use paired query bootstrap draws. Each draw samples "
                "original query IDs and keeps all three times for each sampled ID. The quality "
                "interval uses one recall observation per query. These intervals are exploratory "
                "and conditional on this run; they do not include hardware, population, or "
                "development-selection uncertainty."
            ),
            "",
            (
                "When two target labels choose the same budget, they point to the same physical "
                "measurement. [query-summary.csv.gz](query-summary.csv.gz) records that "
                "measurement once and lists both labels."
            ),
            "",
            "## Measurement limits",
            "",
            (
                "Worker peak memory includes Python, input arrays, the shared Index and temporary "
                "search buffers. The worker's own peak reading captures short peaks that periodic "
                "sampling can miss. Both methods build packed rows and bitplanes. The `bitplane_words` counter "
                "reports split bitmap words only; leaf bitmap reads are not counted. The plots do "
                "not claim to measure all memory traffic."
            ),
            "",
            (
                "Candidate recall compares each result with the exact scan from the same saved "
                "pool and score. A target miss on evaluation remains a miss; the report does not "
                "retune it."
            ),
            "",
            "## Saved evidence",
            "",
            (
                "- [settings.csv](settings.csv): every summarized development, pilot, and "
                "evaluation setting, including limited cases."
            ),
            (
                "- [target-points.csv](target-points.csv): frozen choices, achieved recall, "
                "latency, and paired intervals."
            ),
            "- [failures.csv](failures.csv): incomplete and failed scheduled cases.",
            (
                "- [query-summary.csv.gz](query-summary.csv.gz): one row per physical evaluation "
                "setting and original query."
            ),
            (
                "- [scaling.png](scaling.png), [speedup.png](speedup.png), "
                "[work.png](work.png), and [budgets.png](budgets.png): saved figures."
            ),
            "- [metadata.json](metadata.json): analysis source hash and input file hashes.",
            "",
        ]
    )
    return "\n".join(lines)
