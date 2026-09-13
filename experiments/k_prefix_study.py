"""Run the K comparison using the existing cached data and study runner.

Example: python -m experiments.k_prefix_study prepare
Then run `pilot`, inspect its estimate, and run `sweep`.
"""

import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np

from experiments.prefix_exploration_v2 import prepare_exploration
from experiments.prefix_study_v2 import read_json, repeat, run, summarize
from experiments.fixed_data_comparison import save_json

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "results/k-prefix-study-2026-09-13"
POOL = ROOT / "data/prefix-study-2026-09-13/pools/qwen-quora-d32"
CONFIG = ROOT / "experiments/configs/k-prefix-study.json"
KS = [1, 2, 3, 5, 10, 20, 50, 100]
TARGETS = [.8, .9, .95, .99]


def prepare():
    branch = [dict(leaf_size=leaf, node_budget=budget, prefer_deeper_ties=True)
              for leaf in [32, 128, 2048] for budget in [1024, 8192]]
    # Include a direct scan through B's API, so extra machinery need not be forced.
    branch.append(dict(leaf_size=522931, node_budget=0, prefer_deeper_ties=False))
    config = dict(pool=str(POOL), top_ks=KS, recall_targets=TARGETS,
                  cluster_counts=[16, 64, 256, 1024, 4096],
                  probe_targets=[.8, .9, .95, .99, .999], include_all_clusters=True,
                  repetitions=1, ram_budget_bytes=64_000_000_000,
                  worker_stop_bytes=64_000_000_000, global_seconds=18000,
                  job_seconds=300, schedule_seed=47, router_seed=42,
                  routing_representation="binary",
                  branch=dict(settings=branch, exploration=0, seed=42),
                  prefix=dict(max_prefix_bits=32, settings=[
                      dict(start_depth=d, candidate_target=0, stop_when_exact=True)
                      for d in [0, 1, 4, 16, 32]]))
    save_json(CONFIG, config)
    jobs = prepare_exploration(CONFIG, OUTPUT / "sweep", batch_size=6)
    # The pilot keeps all 1,000 queries. It samples settings, not documents or queries.
    pilot = OUTPUT / "pilot"
    (pilot / "main").mkdir(parents=True)
    for name in ["inputs-manifest.json", "layouts.json"]:
        save_json(pilot / name, read_json(OUTPUT / "sweep" / name))
    save_json(pilot / "configuration.json", dict(config, top_ks=[1, 50], global_seconds=900))
    layouts = {row["clusters"]: row for row in read_json(pilot / "layouts.json")}
    pilot_jobs = []
    for job in jobs:
        if job["clusters"] not in [16, 4096]:
            continue
        choices = []
        for variant in job["variants"]:
            if variant["top_k"] not in [1, 50]:
                continue
            cutoffs = layouts[job["clusters"]]["cutoffs"][str(variant["top_k"])]
            probes = {row["probes"] for row in cutoffs if row["target"] in [.8, .99]}
            if variant["probes"] not in probes:
                continue
            if job["method"] == "branch" and (variant["leaf_size"], variant["node_budget"]) != (128, 8192):
                continue
            if job["method"] == "prefix" and variant["start_depth"] != 4:
                continue
            choices.append(variant)
        if choices:
            pilot_jobs.append(dict(job, variants=choices))
    save_json(pilot / "main/schedule.json", pilot_jobs)
    save_json(OUTPUT / "status.json", dict(stage="prepared", top_ks=KS,
        total_settings=sum(len(j["variants"]) for j in jobs),
        pilot_settings=sum(len(j["variants"]) for j in pilot_jobs)))
    print(json.dumps(read_json(OUTPUT / "status.json")), flush=True)


def pilot():
    run(OUTPUT / "pilot")
    rows = summarize(OUTPUT / "pilot")
    jobs = read_json(OUTPUT / "sweep/main/schedule.json")
    estimates = {}
    for method in ["scan", "branch", "prefix"]:
        own = [r for r in rows if r["method"] == method and r["qualified"]]
        # These are a range of observed costs, not a deadline guarantee.
        durations = [r["total_query_ms"] / 1000 for r in own]
        settings = sum(len(j["variants"]) for j in jobs if j["method"] == method)
        estimates[method] = dict(pilot_settings=len(own), scheduled_settings=settings,
            seconds_at_pilot_median=settings * float(np.median(durations)),
            seconds_at_pilot_max=settings * max(durations))
    estimate = dict(methods=estimates,
        note="Query-loop costs only. Fresh process startup, builds and JSON output add time.")
    save_json(OUTPUT / "pilot-estimate.json", estimate)
    print(json.dumps(estimate, indent=2), flush=True)


def sweep():
    started = perf_counter()
    save_json(OUTPUT / "status.json", dict(stage="sweep running", top_ks=KS))
    run(OUTPUT / "sweep")
    rows = summarize(OUTPUT / "sweep")
    save_json(OUTPUT / "status.json", dict(stage="sweep complete, repeating selections",
        complete=sum(r["qualified"] for r in rows), scheduled=len(rows)))
    repeated = repeat(OUTPUT / "sweep")
    save_json(OUTPUT / "status.json", dict(stage="sweep and selected repeats complete",
        complete=sum(r["qualified"] for r in rows), scheduled=len(rows),
        repeated=sum(r["qualified"] for r in repeated), seconds=perf_counter() - started))
    print(json.dumps(read_json(OUTPUT / "status.json")), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["prepare", "pilot", "sweep"])
    args = parser.parse_args()
    {"prepare": prepare, "pilot": pilot, "sweep": sweep}[args.stage]()


if __name__ == "__main__":
    main()
