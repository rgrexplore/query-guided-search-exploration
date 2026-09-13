"""Prepare a small probability comparison within Method B, without running it.

Usage:
  python -m experiments.prefix_probability_v2 prepare SOURCE_STUDY NEW_OUTPUT
  python -m experiments.prefix_study_v2 run NEW_OUTPUT
  python -m experiments.prefix_study_v2 summarize NEW_OUTPUT
  python -m experiments.prefix_study_v2 repeat NEW_OUTPUT
  python -m experiments.prefix_probability_v2 plot FINISHED_STUDY FIGURE_OUTPUT

The existing controller shares one 300-second allowance across main and repeat.
This compares B's probability settings; it is not a new A/B/C comparison.
"""

import argparse
from itertools import product
import json
from pathlib import Path
import random
import re
from statistics import mean, median

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter

from experiments import prefix_study_v2 as study
from experiments.prefix_exploration_v2 import split_schedule


PROBABILITIES = (0, .1, .2)
SEEDS = (7, 23, 42)
TOP_KS = (1, 100)
TARGETS = (.95, .8)


def build_jobs(rows, source_jobs, *, id_prefix):
    """Keep the best completed B setting that actually splits a bitmap."""
    templates = {variant["setting_id"]: (job, variant)
                 for job in source_jobs for variant in job["variants"]}
    usable = [row for row in rows if row["method"] == "branch" and row.get("complete")
              and row.get("qualified") and row.get("mean_bitplane_words", 0) > 0
              and row.get("recall") is not None and row.get("p50_ms") is not None
              and row["p50_ms"] > 0]
    jobs, selected, omitted = [], [], []
    number = 0
    rng = random.Random(42)
    for k in TOP_KS:
        candidates, chosen_target = [], None
        for target in TARGETS:
            candidates = [row for row in usable if row["top_k"] == k and row["recall"] >= target]
            if candidates:
                chosen_target = target
                break
        if not candidates:
            omitted.append(dict(top_k=k, reason="No completed, qualified B setting with split work meets 80% recall."))
            continue
        chosen = min(candidates, key=lambda row: (row["p50_ms"], row["setting_id"]))
        original_job, original_variant = templates[chosen["setting_id"]]
        # If the selected setting already uses 128 nodes, measure that budget once.
        budgets = sorted({128, original_variant["node_budget"]})
        variants = []
        for budget, probability, seed in product(budgets, PROBABILITIES, SEEDS):
            variants.append(dict(original_variant, setting_id=f"{id_prefix}-s{number:05d}",
                                 source_setting_id=chosen["setting_id"], source_target=chosen_target,
                                 node_budget=budget, exploration=probability, seed=seed, repetitions=1))
            number += 1
        rng.shuffle(variants)
        jobs.append(dict(original_job, phase="main", source_job_id=original_job["job_id"],
                         job_id=f"{id_prefix}-j{len(jobs):04d}", variants=variants))
        selected.append(dict(top_k=k, source_setting_id=chosen["setting_id"], target=chosen_target,
                             source_recall=chosen["recall"], source_p50_ms=chosen["p50_ms"],
                             source_mean_bitplane_words=chosen["mean_bitplane_words"],
                             router=original_job["router"], probes=original_variant["probes"],
                             leaf_size=original_variant["leaf_size"], node_budgets=budgets,
                             prefer_deeper_ties=original_variant.get("prefer_deeper_ties", False)))
    return jobs, dict(selected=selected, omitted=omitted, settings=number,
                      probabilities=list(PROBABILITIES), seeds=list(SEEDS))


def prepare(source, output):
    """Copy saved identities and layouts; no arrays, routers or models are built."""
    source, output = Path(source).resolve(), Path(output).resolve()
    if output.exists():
        raise FileExistsError("Use a new probability-study output directory.")
    names = ("configuration.json", "inputs-manifest.json", "layouts.json",
             "main/schedule.json", "main/settings.json")
    snapshots = {name: study.read_json(source / name) for name in names}
    source_hashes = {name: study.file_hash(source / name) for name in names}
    original = snapshots["configuration.json"]
    manifest = snapshots["inputs-manifest.json"]
    if manifest["identity"]["queries"] != 1000:
        raise ValueError("This probability experiment requires the complete 1,000-query study.")
    prefix = re.sub(r"[^A-Za-z0-9_-]", "-", output.name) or "probability"
    jobs, evidence = build_jobs(snapshots["main/settings.json"], snapshots["main/schedule.json"],
                                id_prefix=prefix)
    for job in jobs:
        if job["query_rows"] != list(range(1000)) or job["pool"] != manifest["pool"]:
            raise ValueError("Selected settings must keep the same pool and every saved query.")

    output.mkdir(parents=True)
    (output / "main").mkdir()
    (output / "source").mkdir()
    for name, value in snapshots.items():
        study.save_json(output / "source" / name.replace("/", "-"), value)
    # Only the saved schedule is executed. Do not carry an unrelated source grid.
    config = {name: original[name] for name in (
        "pool", "ram_budget_bytes", "worker_stop_bytes", "schedule_seed", "router_seed"
    ) if name in original}
    config.update(top_ks=list(TOP_KS), recall_targets=[.8, .95], repetitions=1,
                  global_seconds=300, job_seconds=min(original.get("job_seconds", 60), 60),
                  probability_source=str(source),
                  scope="Probability comparison within Method B; fixed documents, queries and selected router.")
    study.save_json(output / "configuration.json", config)
    study.save_json(output / "inputs-manifest.json", manifest)
    study.save_json(output / "layouts.json", snapshots["layouts.json"])
    study.save_json(output / "main/schedule-before-batching.json", jobs)
    batches = split_schedule(jobs)
    study.save_json(output / "main/schedule.json", batches)
    evidence.update(source=str(source), source_file_hashes=source_hashes, jobs=len(batches),
                    timing_budget_seconds=300,
                    policy="For each K, use the fastest qualified B setting with actual bitmap splits at 95% recall; fall back to 80%, otherwise omit K. Keep its router, probes, leaf size and tie order. Compare 128 nodes and its selected budget across every declared probability and seed.",
                    scope="This is an ablation within B, not an A/B/C comparison. Probability 0 is retained at all three seeds as the deterministic control. Compare all seeds when interpreting probability; the fastest seed alone is not evidence of an improvement.",
                    input_check="Preparation copies saved input and layout records. The study controller verifies their hashes before running.")
    study.save_json(output / "probability-selection.json", evidence)
    return dict(output=str(output), settings=evidence["settings"], jobs=len(batches),
                omitted=evidence["omitted"], timing_budget_seconds=300)


def group_probability_rows(rows, selection):
    """Summarize all declared groups, keeping missing or unqualified seeds visible."""
    expected = selection["seeds"]
    groups = []
    for chosen in selection["selected"]:
        fixed = [row for row in rows if row["method"] == "branch" and row["top_k"] == chosen["top_k"]
                 and all(row.get(name) == chosen[name]
                         for name in ("source_setting_id", "router", "probes", "leaf_size"))
                 and row.get("prefer_deeper_ties", False) == chosen.get("prefer_deeper_ties", False)]
        for budget, probability in product(chosen["node_budgets"], selection["probabilities"]):
            own = [row for row in fixed if row["node_budget"] == budget
                   and row["exploration"] == probability and row["seed"] in expected]
            found = sorted({row["seed"] for row in own})
            invalid = sorted({row["seed"] for row in own if not row.get("complete")
                              or not row.get("qualified") or row.get("recall") is None
                              or row.get("p50_ms") is None})
            ready = len(expected) == len(own) == 3 and set(found) == set(expected) and not invalid
            group = dict(top_k=chosen["top_k"], node_budget=budget, probability=probability,
                router=chosen["router"], probes=chosen["probes"], leaf_size=chosen["leaf_size"],
                seeds_found=found, missing_seeds=[seed for seed in expected if seed not in found],
                unqualified_seeds=invalid, plotted=bool(ready),
                recall_percent_mean=None, recall_percent_min=None, recall_percent_max=None,
                query_ms_median=None, query_ms_min=None, query_ms_max=None)
            if ready:
                recalls = [100 * row["recall"] for row in own]
                times = [row["p50_ms"] for row in own]
                group.update(recall_percent_mean=mean(recalls), recall_percent_min=min(recalls),
                    recall_percent_max=max(recalls), query_ms_median=median(times),
                    query_ms_min=min(times), query_ms_max=max(times))
            groups.append(group)
    return groups


def plot(source, output):
    """Draw finished studies from settings and selection metadata, without raw-query reads."""
    source, output = Path(source).resolve(), Path(output).resolve()
    selection = study.read_json(source / "probability-selection.json")
    groups = group_probability_rows(study.read_json(source / "main/settings.json"), selection)
    output.mkdir(parents=True, exist_ok=True)
    scope = ("Recall points are means across three seeds. Time points are medians of the three seed "
             "median query times. Bars show the minimum to maximum seed range, not a confidence interval. "
             "Only complete, qualified three-seed groups are plotted; gaps remain in the saved summary.")
    study.save_json(output / "probability-summary.json", dict(groups=groups,
        omitted=selection.get("omitted", []), source=str(source), scope=scope))
    if groups:
        study.write_csv(output / "probability-summary.csv", groups)
    else:
        (output / "probability-summary.csv").write_text("")
    plots = []
    for chosen in selection["selected"]:
        k = chosen["top_k"]
        fig, axes = plt.subplots(1, 2, figsize=(9.4, 4.7))
        fig.subplots_adjust(left=.08, right=.98, bottom=.22, top=.78, wspace=.28)
        x = [100 * value for value in selection["probabilities"]]
        for budget in chosen["node_budgets"]:
            own = [group for group in groups if group["top_k"] == k and group["node_budget"] == budget]
            if not any(group["plotted"] for group in own):
                continue
            label = "Unlimited nodes" if budget == 0 else f"{budget:,} nodes"
            for axis, middle, lower, upper in (
                (axes[0], "recall_percent_mean", "recall_percent_min", "recall_percent_max"),
                (axes[1], "query_ms_median", "query_ms_min", "query_ms_max")):
                values = [row[middle] if row["plotted"] else float("nan") for row in own]
                errors = [[row[middle] - row[lower] if row["plotted"] else float("nan") for row in own],
                          [row[upper] - row[middle] if row["plotted"] else float("nan") for row in own]]
                axis.errorbar(x, values, yerr=errors, fmt="o-", capsize=3, linewidth=1.5, label=label)
        for axis, title, ylabel in zip(axes, ("Recall", "Query time"),
                                       ("Mean recall across seeds (%)", "Median query time (ms)")):
            axis.set(xlabel="Exploration probability (%)", ylabel=ylabel, title=title)
            axis.set_xticks(x, [f"{value:g}" for value in x])
            formatter = ScalarFormatter(useOffset=False); formatter.set_scientific(False)
            axis.yaxis.set_major_formatter(formatter)
            axis.spines[["top", "right"]].set_visible(False)
            axis.grid(alpha=.2)
            if axis.get_legend_handles_labels()[0]:
                axis.legend(frameon=False, fontsize=9)
            else:
                axis.text(.5, .5, "No complete, qualified\nthree-seed group", transform=axis.transAxes,
                          ha="center", va="center")
        axes[0].set_ylim(0, 100)
        router = Path(chosen["router"]).name if chosen["router"] else "Global cluster"
        name = Path(selection.get("source", source.name)).name
        fig.suptitle(f"{name}: Method B, K={k}\n{router}; {chosen['probes']:,} clusters opened; leaf {chosen['leaf_size']:,}", fontsize=11)
        fig.text(.5, .045, "Points: mean recall and median of seed median times.\n"
                 "Bars: range across three seeds (minimum to maximum), not a confidence interval.",
                 ha="center", fontsize=9)
        for extension in ("png", "pdf"):
            path = output / f"probability-k{k}.{extension}"
            fig.savefig(path, dpi=180)
            plots.append(str(path))
        plt.close(fig)
    return dict(output=str(output), plots=plots, groups=len(groups),
                plotted_groups=sum(group["plotted"] for group in groups))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("prepare", "plot"))
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    action = prepare if args.stage == "prepare" else plot
    print(json.dumps(action(args.source, args.output)), flush=True)


if __name__ == "__main__":
    main()
