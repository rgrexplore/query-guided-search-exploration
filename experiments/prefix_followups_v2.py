"""Prepare targeted baseline controls, then use the existing study controller.

Examples:
  python -m experiments.prefix_followups_v2 prepare SOURCE_STUDY NEW_OUTPUT
  python -m experiments.prefix_followups_v2 run NEW_OUTPUT
  python -m experiments.prefix_followups_v2 summarize NEW_OUTPUT
  python -m experiments.prefix_followups_v2 repeat NEW_OUTPUT

Preparation computes routing ranks only when explicitly invoked. It builds no new
layouts and runs no timing workers. The default follow-up timing budget is 15 minutes.
"""

import argparse
from copy import deepcopy
import json
from pathlib import Path
import random
import re

import numpy as np
from threadpoolctl import threadpool_limits

from experiments import prefix_study_v2 as study
from experiments.prefix_exploration_v2 import split_schedule
from experiments.prefix_study_v2 import choose_settings


def setting_key(row):
    return (row["router"], row["method"], row["top_k"], row["probes"],
            row.get("start_depth") if row["method"] == "prefix" else None,
            row.get("candidate_target") if row["method"] == "prefix" else None)


def apparent_advantages(rows, minimum=.03):
    """Keep positive-recall alternative frontier points with a sampled A comparator."""
    valid = [row for row in rows if row.get("complete") and row.get("qualified")
             and row.get("recall") is not None and 0 < row["recall"] <= 1
             and row.get("p50_ms") is not None and row["p50_ms"] > 0]
    alternatives = [row for row in valid if row["method"] in ("branch", "prefix")]
    selected = []
    for row in alternatives:
        if any(other["top_k"] == row["top_k"] and other["recall"] >= row["recall"]
               and other["p50_ms"] <= row["p50_ms"]
               and (other["recall"] > row["recall"] or other["p50_ms"] < row["p50_ms"])
               for other in alternatives):
            continue
        scans = [other for other in valid if other["method"] == "scan"
                 and other["top_k"] == row["top_k"] and other["recall"] >= row["recall"]]
        if not scans:
            continue
        best = min(scans, key=lambda other: (other["p50_ms"], other["setting_id"]))
        if row["p50_ms"] < (1 - minimum) * best["p50_ms"]:
            selected.append(dict(row, sampled_scan_setting_id=best["setting_id"],
                                 sampled_scan_p50_ms=best["p50_ms"],
                                 apparent_advantage=1 - row["p50_ms"] / best["p50_ms"]))
    return sorted(selected, key=lambda row: (row["top_k"], -row["recall"],
                                             row["p50_ms"], row["setting_id"]))


def build_followups(rows, source_jobs, layouts, config, resolve_probes, *,
                   minimum_advantage=.03, id_prefix="followup"):
    """Select and deduplicate controls; routing is supplied by the caller."""
    valid = [row for row in rows if row.get("complete") and row.get("qualified")]
    existing = {setting_key(row): row["setting_id"] for row in valid}
    proposed, skipped, routing_evidence = {}, [], []
    templates = {(job["router"], job["method"]): job for job in source_jobs}
    points = apparent_advantages(valid, minimum_advantage)

    def add(router, method, k, probes, reason):
        spec = dict(router=router, method=method, top_k=k, probes=probes,
                    repetitions=config.get("repetitions", 1))
        if method == "prefix":
            spec.update(start_depth=0, candidate_target=0)
        key = setting_key(spec)
        if key in existing:
            skipped.append(dict(existing_setting_id=existing[key], reason=reason))
            return
        if key not in proposed:
            proposed[key] = dict(spec, reasons=[], source_setting_ids=[])
        item = proposed[key]
        if reason not in item["reasons"]:
            item["reasons"].append(reason)
        identifier = reason["source_setting_id"]
        if identifier not in item["source_setting_ids"]:
            item["source_setting_ids"].append(identifier)

    cutoffs = {}
    for point in points:
        for layout in layouts:
            key = (layout["path"], point["top_k"], point["recall"])
            if key not in cutoffs:
                cutoffs[key] = resolve_probes(layout, point["top_k"], point["recall"])
                routing_evidence.append(dict(router=layout["path"], clusters=layout["clusters"],
                    top_k=point["top_k"], target=point["recall"], **cutoffs[key]))
            add(layout["path"], "scan", point["top_k"], cutoffs[key]["probes"],
                dict(kind="match_alternative_recall", source_setting_id=point["setting_id"],
                     target=point["recall"]))

    # These observed B cases spend all 512 visited nodes scoring whole cluster roots.
    # Their effective work needs a direct A-at-512 comparison, even if B is dominated.
    for row in valid:
        if (row["method"] == "branch" and row.get("node_budget") == 512
                and row.get("mean_bitplane_words") == 0 and row["probes"] >= 512
                and row["clusters"] >= 512):
            add(row["router"], "scan", row["top_k"], 512,
                dict(kind="zero_split_budget_512", source_setting_id=row["setting_id"]))

    best = choose_settings(valid, config["recall_targets"], config["top_ks"])
    for row in best:
        if row["method"] == "scan":
            add(row["router"], "prefix", row["top_k"], row["probes"],
                dict(kind="prefix_depth_zero_at_best_scan", source_setting_id=row["setting_id"],
                     target=row["target"]))

    jobs = {}
    for number, spec in enumerate(proposed.values()):
        group = (spec["router"], spec["method"])
        if group not in jobs:
            original = templates[group]
            jobs[group] = dict(original, phase="main", source_job_id=original["job_id"],
                               job_id=f"{id_prefix}-j{len(jobs):04d}", variants=[])
        variant = {key: value for key, value in spec.items() if key not in ("router", "method")}
        variant["setting_id"] = f"{id_prefix}-s{number:05d}"
        jobs[group]["variants"].append(variant)
    ordered = list(jobs.values())
    rng = random.Random(config.get("schedule_seed", 42) + 101)
    for job in ordered:
        rng.shuffle(job["variants"])
    rng.shuffle(ordered)
    evidence = dict(apparent_advantages=points, routing_cutoffs=routing_evidence,
                    skipped_completed_settings=skipped, settings=len(proposed), jobs=len(ordered),
                    minimum_advantage=minimum_advantage,
                    policy="Positive-recall nondominated B/C points, each compared with the fastest sampled A at at least that recall. New A cutoffs use every saved layout. Explicit zero-split budget512 controls and depth0/target0 controls at declared-target best A settings are added separately.")
    return ordered, evidence


def prepare(source, output, *, seconds=900, minimum_advantage=.03):
    source, output = Path(source).resolve(), Path(output).resolve()
    if output.exists():
        raise FileExistsError("Use a new follow-up output directory")
    study.verify_unchanged(source)
    names = ("configuration.json", "inputs-manifest.json", "layouts.json",
             "main/schedule.json", "main/settings.json")
    snapshots = {name: study.read_json(source / name) for name in names}
    hashes = {name: study.file_hash(source / name) for name in names}
    config = snapshots["configuration.json"]
    source_jobs = snapshots["main/schedule.json"]
    source_queries = source_jobs[0]["query_rows"]
    if any(job["query_rows"] != source_queries or job["pool"] != config["pool"] for job in source_jobs):
        raise ValueError("Source jobs do not share the same pool and query rows")
    ranks_by_router = {}

    def resolve(layout, k, target):
        router = layout["path"]
        if router not in ranks_by_router:
            ranks, scores, *_ = study.routing_ranks(dict(pool=config["pool"], router=router,
                                                         query_rows=source_queries))
            ranks_by_router[router] = (ranks, scores)
        ranks, scores = ranks_by_router[router]
        cutoff = study.probe_cutoff(ranks[:, :k], target)
        probes = study.cover_boundary_ties(scores, cutoff) if scores is not None else 1
        return dict(probes=probes, rank_cutoff=cutoff,
                    routing_recall=float(np.count_nonzero(ranks[:, :k] <= probes) / ranks[:, :k].size))

    prefix = re.sub(r"[^A-Za-z0-9_-]", "-", output.name) or "followup"
    with threadpool_limits(limits=1):
        jobs, evidence = build_followups(snapshots["main/settings.json"], source_jobs,
            snapshots["layouts.json"], config, resolve,
            minimum_advantage=minimum_advantage, id_prefix=prefix)
    output.mkdir(parents=True)
    (output / "main").mkdir()
    (output / "source").mkdir()
    for name, value in snapshots.items():
        study.save_json(output / "source" / name.replace("/", "-"), value)
    copied_config = dict(config, global_seconds=seconds,
                         job_seconds=min(config.get("job_seconds", seconds), seconds),
                         followup_source=str(source), followup_minimum_advantage=minimum_advantage)
    study.save_json(output / "configuration.json", copied_config)
    study.save_json(output / "inputs-manifest.json", snapshots["inputs-manifest.json"])
    layouts = deepcopy(snapshots["layouts.json"])
    for layout in layouts:
        for job in jobs:
            if job["router"] == layout["path"]:
                for variant in job["variants"]:
                    k = str(variant["top_k"])
                    layout["probes"][k] = sorted(set(layout["probes"].get(k, [])) | {variant["probes"]})
    study.save_json(output / "layouts.json", layouts)
    study.save_json(output / "main/schedule-before-batching.json", jobs)
    evidence["jobs_before_batching"] = len(jobs)
    jobs = split_schedule(jobs)
    evidence["jobs"] = len(jobs)
    study.save_json(output / "main/schedule.json", jobs)
    study.save_json(output / "followup-selection.json", dict(evidence, source=str(source),
                    source_file_hashes=hashes, timing_budget_seconds=seconds,
                    scope="Follow-up controls selected after inspecting the original results; input arrays, layouts and query rows remain fixed. Original setting IDs are retained as provenance, not reused for new measurements."))
    study.verify_unchanged(output)
    return dict(output=str(output), settings=evidence["settings"], jobs=evidence["jobs"],
                alternative_points=len(evidence["apparent_advantages"]), timing_budget_seconds=seconds)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("prepare", "run", "summarize", "repeat"))
    parser.add_argument("path", type=Path)
    parser.add_argument("output", type=Path, nargs="?")
    parser.add_argument("--seconds", type=float, default=900)
    parser.add_argument("--minimum-advantage", type=float, default=.03)
    args = parser.parse_args()
    if args.stage == "prepare":
        if args.output is None or args.seconds <= 0 or not 0 <= args.minimum_advantage < 1:
            parser.error("prepare needs SOURCE NEW_OUTPUT, positive seconds, and an advantage in [0,1)")
        print(json.dumps(prepare(args.path, args.output, seconds=args.seconds,
                                  minimum_advantage=args.minimum_advantage)), flush=True)
    else:
        {"run": study.run, "summarize": study.summarize, "repeat": study.repeat}[args.stage](args.path)


if __name__ == "__main__":
    main()
