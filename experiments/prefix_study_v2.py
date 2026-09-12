"""Prepare, run, summarize, and repeat a declared study on unchanged cached arrays."""

import argparse
from collections import defaultdict
from itertools import product
import json
import os
from pathlib import Path
import subprocess
import sys
from time import monotonic, sleep, time

import numpy as np
import psutil
from threadpoolctl import threadpool_limits

from experiments.analysis import write_csv
from experiments.fixed_data_comparison import file_hash, save_json, verify_arrays
from experiments.isolated import ROOT
from experiments.models import mean_recall
from experiments.probe_cutoffs import cover_boundary_ties, probe_cutoff, routing_ranks
from experiments.real_study import prepare_router

METHODS = ("scan", "branch", "prefix")
ROUTER_FILES = ("router.json", "assignments.npy", "centroids.npy", "labels.npy")
WORK_FIELDS = ("documents_scored", "nodes", "bitplane_words", "leaf_words",
               "prefix_levels", "prefix_lookups", "final_depth")


def read_json(path):
    return json.loads(Path(path).read_text())


def verify_pool(pool):
    pool = Path(pool).resolve()
    metadata = read_json(pool / "pool.json")
    hashes, shape = verify_arrays(pool), metadata["identity"]
    expected = {"codes": (shape["documents"], (shape["dimensions"] + 63) // 64),
                "documents": (shape["documents"], shape["dimensions"]),
                "queries": (shape["queries"], shape["dimensions"]),
                "reference": (shape["queries"], shape["top_k"])}
    for name, dimensions in expected.items():
        if np.load(pool / f"{name}.npy", mmap_mode="r").shape != dimensions:
            raise ValueError(f"prepared pool identity disagrees with {name}.npy")
    return dict(pool=str(pool), identity=shape, array_hashes=hashes,
                pool_metadata_sha256=file_hash(pool / "pool.json"))


def build_jobs(pool, layouts, top_ks, phase, *, shape=None, repetitions=1,
               ram_budget_bytes=32_000_000_000, branch=None, prefix=None, schedule_seed=42):
    shape = shape or dict(documents=1_000_000, dimensions=256, queries=200)
    branch, prefix = branch or {}, prefix or {}
    width = min(prefix.get("max_prefix_bits", 32), shape["dimensions"])
    depths = prefix.get("start_depths", [min(16, width)])
    if not all(0 <= depth <= width for depth in depths):
        raise ValueError("start_depths must fit the stored prefix width")
    policies = {"scan": [{}], "branch": [dict(leaf_size=leaf, node_budget=budget,
                 exploration=branch.get("exploration", 0), seed=branch.get("seed", 42))
                 for leaf, budget in product(branch.get("leaf_sizes", [32]),
                                              branch.get("node_budgets", [128, 0]))],
                "prefix": [dict(start_depth=depth, candidate_target=target)
                 for depth, target in product(depths, prefix.get("candidate_targets", [1000, 0]))]}
    jobs, number = [], 0
    rng = np.random.default_rng(schedule_seed)
    for layout in layouts:
        for method in METHODS:
            variants = []
            for k in top_ks:
                for probes, policy in product(sorted(set(layout["probes"][str(k)] +
                                                         [layout["clusters"]])), policies[method]):
                    variants.append(dict(setting_id=f"{phase}-s{number:05d}", top_k=k,
                                         probes=probes, repetitions=repetitions, **policy))
                    number += 1
            rng.shuffle(variants)
            jobs.append(dict(job_id=f"{phase}-j{len(jobs):04d}", phase=phase, pool=str(pool),
                router=layout["path"], clusters=layout["clusters"], method=method,
                dimensions=shape["dimensions"], documents=shape["documents"],
                query_rows=list(range(shape["queries"])), max_prefix_bits=width,
                ram_budget_bytes=ram_budget_bytes, variants=variants))
    rng.shuffle(jobs)
    return jobs


def prepare(config, output):
    output = Path(output); output.mkdir(parents=True, exist_ok=False)
    pool = Path(config["pool"]).resolve()
    manifest = verify_pool(pool)
    if not all(1 <= k <= manifest["identity"]["top_k"] for k in config["top_ks"]):
        raise ValueError("top_ks must fit the unchanged cached reference")
    config = dict(config, pool=str(pool))
    source = Path(config.get("router_source") or pool).resolve()
    for name in ("codes.npy", "documents.npy"):
        if file_hash(source / name) != manifest["array_hashes"][name]:
            raise ValueError("router source uses different document inputs")
    save_json(output / "configuration.json", config)
    save_json(output / "inputs-manifest.json", manifest)
    layouts = []
    with threadpool_limits(limits=1):
        for clusters in config["cluster_counts"]:
            if clusters == 1:
                layouts.append(dict(path=None, clusters=1, probes={str(k): [1] for k in config["top_ks"]}))
                continue
            seed = config.get("router_seed", 42)
            router = source / "routers" / f"ivf-{clusters}-seed{seed}"
            if not router.exists():
                router = prepare_router(pool, "ivf", clusters, seed)
            info = read_json(router / "router.json")
            if (info["kind"], info["clusters"], info["seed"]) != ("ivf", clusters, seed):
                raise ValueError("cached router does not match the declared layout")
            hashes = {name: file_hash(router / name) for name in ROUTER_FILES}
            for name in ("assignments", "centroids"):
                if hashes[f"{name}.npy"] != info[f"{name}_sha256"]:
                    raise ValueError(f"cached router {name} changed")
            ranks, scores, *_ = routing_ranks(dict(pool=str(pool), router=str(router),
                query_rows=list(range(manifest["identity"]["queries"]))))
            layout = dict(path=str(router), clusters=clusters, hashes=hashes, probes={}, cutoffs={})
            for k in config["top_ks"]:
                evidence, choices = [], {clusters}
                for target in config.get("probe_targets", config["recall_targets"]):
                    rank = probe_cutoff(ranks[:, :k], target)
                    probes = cover_boundary_ties(scores, rank)
                    choices.add(probes)
                    evidence.append(dict(target=target, rank_cutoff=rank, probes=probes,
                        routing_recall=int(np.count_nonzero(ranks[:, :k] <= probes)) / (len(ranks) * k)))
                layout["probes"][str(k)], layout["cutoffs"][str(k)] = sorted(choices), evidence
            layouts.append(layout)
    save_json(output / "layouts.json", layouts)
    jobs = build_jobs(pool, layouts, config["top_ks"], "main", shape=manifest["identity"],
        **{name: config[name] for name in ("repetitions", "ram_budget_bytes", "branch", "prefix",
                                          "schedule_seed") if name in config})
    (output / "main").mkdir()
    save_json(output / "main/schedule.json", jobs)
    verify_unchanged(output)
    return jobs


def verify_unchanged(output):
    manifest = read_json(output / "inputs-manifest.json")
    if verify_pool(manifest["pool"]) != manifest:
        raise ValueError("benchmark input identity changed")
    for layout in read_json(output / "layouts.json"):
        if layout["path"] and any(file_hash(Path(layout["path"]) / name) != digest
                                  for name, digest in layout["hashes"].items()):
            raise ValueError("derived routing inputs changed")


def run_one_job(job, folder, seconds, stop_bytes):
    folder.mkdir(parents=True, exist_ok=False)
    save_json(folder / "job.json", job)
    started, peak, failure = monotonic(), 0, None
    environment = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
    with (folder / "worker.log").open("w") as log:
        child = subprocess.Popen([sys.executable, "-m", "experiments.prefix_batch_worker_v2",
            str(folder / "job.json"), str(folder / "run")], cwd=ROOT, env=environment,
            stdout=log, stderr=subprocess.STDOUT)
        observation = dict(status="running", pid=child.pid, started_at=time(), seconds=0)
        save_json(folder / "process.json", observation)
        process = psutil.Process(child.pid)
        while child.poll() is None:
            try:
                peak = max(peak, process.memory_info().rss)
            except psutil.NoSuchProcess:
                break
            if peak > stop_bytes:
                failure = "memory_stop"
            elif monotonic() - started >= seconds:
                failure = "timeout"
            if failure:
                child.kill(); break
            sleep(.02)
        code = child.wait()
    status = failure or ("complete" if code == 0 else "failed")
    if status == "complete" and not (folder / "run/result.json").exists():
        status = "missing_result"
    observation.update(status=status, returncode=code, parent_sampled_peak=peak,
                       seconds=monotonic() - started)
    save_json(folder / "process.json", observation)
    return observation


def run(output, phase="main"):
    output = Path(output).resolve()
    verify_unchanged(output)
    config, folder = read_json(output / "configuration.json"), output / phase
    budget_path = output / "time-budget.json"
    previous = read_json(budget_path)["spent_seconds"] if budget_path.exists() else 0
    measured = sum(read_json(path).get("seconds", 0) for path in output.glob("*/cases/*/process.json"))
    spent, started, state = max(previous, measured), monotonic(), dict(status="complete")
    try:
        for job in read_json(folder / "schedule.json"):
            case = folder / "cases" / job["job_id"]
            if case.exists():
                path = case / "process.json"
                observed = read_json(path) if path.exists() else dict(status="interrupted", seconds=0)
                if observed["status"] == "running":
                    if psutil.pid_exists(observed["pid"]):
                        raise RuntimeError(f"job {job['job_id']} is still running; no parallel job was started")
                    observed.update(status="interrupted", seconds=time() - observed["started_at"])
                    save_json(path, observed)
                    spent = max(spent, sum(read_json(saved).get("seconds", 0)
                                for saved in output.glob("*/cases/*/process.json")))
                continue  # An attempted job is never silently retried.
            remaining = config["global_seconds"] - spent - (monotonic() - started)
            if remaining <= 0:
                state["status"] = "budget_exhausted"; break
            run_one_job(job, case, min(remaining, config.get("job_seconds", remaining)),
                        config.get("worker_stop_bytes", config.get("ram_budget_bytes", 32_000_000_000)))
    finally:
        save_json(budget_path, dict(spent_seconds=spent + monotonic() - started,
                                   global_seconds=config["global_seconds"]))
        save_json(folder / "run-state.json", state)
        verify_unchanged(output)
    return state


def choose_settings(rows, targets, top_ks):
    selected = []
    for k, target, method in product(top_ks, targets, METHODS):
        eligible = [row for row in rows if row["method"] == method and row["top_k"] == k
                    and row["qualified"] and row["recall"] is not None and row["recall"] >= target]
        if eligible:
            selected.append(dict(target=target, **min(eligible, key=lambda row: (row["p50_ms"], row["setting_id"]))))
    return selected


def load_observations(path):
    groups = defaultdict(lambda: dict(times=[], recalls=[], pairs=set(), work=defaultdict(float)))
    if path.exists():
        with path.open() as source:
            for line in source:
                if not line.endswith("\n"):
                    break  # Preserve earlier complete records after a killed worker's partial write.
                row = json.loads(line); group = groups[row["setting_id"]]
                group["times"].append(row["query_ms"]); group["recalls"].append(row["recall"])
                group["pairs"].add((row["query"], row["repetition"]))
                for name in WORK_FIELDS:
                    group["work"][name] += row.get(name, 0)
    return groups


def summarize(output, phase="main"):
    output = Path(output); folder = output / phase
    config = read_json(output / "configuration.json")
    by_setting, templates, failures = defaultdict(list), {}, []
    for job in read_json(folder / "schedule.json"):
        case = folder / "cases" / job["job_id"]
        process = read_json(case / "process.json") if (case / "process.json").exists() else {"status": "not_run"}
        try:
            result = read_json(case / "run/result.json") if (case / "run/result.json").exists() else {}
        except json.JSONDecodeError:
            result = {"status": "partial_result"}
        observations = load_observations(case / "run/queries.jsonl")
        if process["status"] != "complete" or result.get("status") != "complete":
            failures.append(dict(job_id=job["job_id"], result_status=result.get("status"), **process))
        for variant in job["variants"]:
            identifier = variant["setting_id"]
            templates[identifier] = dict(method=job["method"], clusters=job["clusters"],
                                          router=job["router"], **variant)
            rows = observations[identifier]
            expected = {(q, r) for q in job["query_rows"] for r in range(variant["repetitions"])}
            complete = (process["status"] == result.get("status") == "complete"
                        and identifier in result.get("completed_variants", [])
                        and rows["pairs"] == expected and len(rows["times"]) == len(expected))
            power = result.get("power_before", {})
            qualified = (complete and power.get("available", False) and power == result.get("power_after")
                         and result["memory"]["budget_status"] == "fits_by_lifetime_peak")
            by_setting[identifier].append(dict(rows=rows, result=result, complete=complete, qualified=qualified))
    settings = []
    for identifier, template in templates.items():
        items = by_setting[identifier]
        times = [value for item in items for value in item["rows"]["times"]]
        recalls = [value for item in items for value in item["rows"]["recalls"]]
        medians = [float(np.median(item["rows"]["times"])) for item in items if item["rows"]["times"]]
        complete = sum(item["complete"] for item in items)
        row = dict(template, measurements=len(times), complete_processes=complete,
            complete=complete == (3 if phase == "repeat" else 1),
            qualified=len(items) == (3 if phase == "repeat" else 1) and all(item["qualified"] for item in items),
            recall=mean_recall(recalls, template["top_k"]) if recalls else None,
            p50_ms=float(np.median(times)) if times else None,
            p95_ms=float(np.percentile(times, 95)) if times else None, total_query_ms=sum(times),
            process_p50_median_ms=float(np.median(medians)) if medians else None,
            process_p50_min_ms=min(medians) if medians else None,
            process_p50_max_ms=max(medians) if medians else None,
            logical_index_bytes=max((item["result"].get("storage", {}).get("logical_bytes", 0) for item in items)),
            lifetime_peak_bytes=max((item["result"].get("memory", {}).get("lifetime_peak_bytes", 0) for item in items)))
        row.update({f"mean_{name}": sum(item["rows"]["work"][name] for item in items) / len(times)
                    if times else None for name in WORK_FIELDS})
        settings.append(row)
    save_json(folder / "settings.json", settings)
    # Every method uses one column set, even though only some parameters apply to it.
    columns = list(dict.fromkeys(key for row in settings for key in row))
    if settings:
        write_csv(folder / "settings.csv", [{key: row.get(key) for key in columns} for row in settings])
    save_json(folder / "summary.json", dict(settings=settings, failures=failures,
        scope="All methods use the same cached queries. Repeat timings are separate from setting selection."))
    if phase == "main":
        save_json(output / "selected-settings.json", choose_settings(settings, config["recall_targets"], config["top_ks"]))
    return settings


def repeat(output):
    output = Path(output); folder = output / "repeat"
    if not folder.exists():
        folder.mkdir()
        selected = read_json(output / "selected-settings.json")
        save_json(folder / "selection.json", selected)
        chosen = {row["setting_id"] for row in selected}
        jobs, rng = [], np.random.default_rng(read_json(output / "configuration.json").get("schedule_seed", 42) + 1)
        for block in range(3):
            batch = [dict(job, phase="repeat", block=block, job_id=f"repeat-b{block}-{job['job_id']}",
                          variants=[variant for variant in job["variants"] if variant["setting_id"] in chosen])
                     for job in read_json(output / "main/schedule.json")]
            batch = [job for job in batch if job["variants"]]; rng.shuffle(batch); jobs.extend(batch)
        save_json(folder / "schedule.json", jobs)
    run(output, "repeat")
    return summarize(output, "repeat")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("prepare", "run", "summarize", "repeat"))
    parser.add_argument("path", type=Path)
    parser.add_argument("output", type=Path, nargs="?")
    args = parser.parse_args()
    if args.stage == "prepare":
        if args.output is None:
            parser.error("prepare requires CONFIG OUTPUT")
        prepare(read_json(args.path), args.output)
    else:
        {"run": run, "summarize": summarize, "repeat": repeat}[args.stage](args.path)


if __name__ == "__main__":
    main()
