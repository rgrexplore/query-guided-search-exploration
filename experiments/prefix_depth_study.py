"""Inspect every prefix depth, then time normal searches without recording traces.

The trace runs to the root. Its top-100 snapshots also contain the exact current
top 1, 2, 3, 5, 10, 20 and 50, without repeating the same document scoring eight times.
"""

import argparse
import csv
import gzip
import json
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from time import perf_counter

import bitplane_index
import faiss
import numpy as np
import psutil
from threadpoolctl import threadpool_limits

from experiments.fixed_data_comparison import save_json
from experiments.k_prefix_study import KS, OUTPUT, POOL
from experiments.prefix_batch_worker_v2 import file_hash, prepare_router, search_index
from experiments.worker import lifetime_peak_bytes, power_state

ROOT = Path(__file__).resolve().parents[1]
FOLDER = OUTPUT / "depth"
DATA = ROOT / "data/k-prefix-study-2026-09-13"
STARTS = [0, 1, 4, 8, 12, 16, 24, 32]
ROUTER = POOL / "routers/ivf-binary-1024-seed42"
PROBES = 78


def write_csv(path, rows):
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def arrays():
    return (np.load(POOL / "codes.npy", mmap_mode="r"),
            np.load(POOL / "queries.npy"),
            np.load(POOL / "reference.npy", mmap_mode="r"))


def prepare():
    FOLDER.mkdir(parents=True, exist_ok=True)
    DATA.mkdir(parents=True, exist_ok=True)
    codes, queries, reference = arrays()
    assignments, route, _, _ = prepare_router(ROUTER, len(codes))
    selected = np.concatenate([route(query[None, :], PROBES) for query in queries])
    scan = bitplane_index.Index(codes, assignments, queries.shape[1], build_bitplanes=False)
    local = scan.scan(queries, selected, candidate_limit=max(KS))["rows"]
    assert np.all(local >= 0)
    np.save(DATA / "selected.npy", selected)
    np.save(DATA / "local-reference.npy", local)
    opened = np.bincount(assignments)[selected].sum(axis=1)
    np.save(DATA / "opened-rows.npy", opened)
    inputs = [POOL / name for name in ["codes.npy", "queries.npy", "reference.npy"]]
    inputs += [ROUTER / name for name in ["assignments.npy", "centroids.npy", "labels.npy"]]
    inputs += [DATA / name for name in ["selected.npy", "local-reference.npy", "opened-rows.npy"]]
    save_json(FOLDER / "configuration.json", dict(documents=len(codes), queries=len(queries),
        dimensions=queries.shape[1], top_ks=KS, clusters=1024, probes=PROBES,
        mean_opened_rows=float(opened.mean()), starts=STARTS,
        input_hashes={str(p): file_hash(p) for p in inputs}))


def recall(rows, reference):
    return len(set(rows) & set(reference)) / len(reference)


def allows_stop(step, query_l1, dimensions, k):
    """Apply the existing native bound to the Kth score in a saved snapshot."""
    if step["depth"] == 0:
        return True  # The root has no unseen rows inside these clusters.
    if len(step["scores"]) < k:
        return False
    worst = step["scores"][k - 1]
    guard = 32 * np.finfo(np.float64).eps * (dimensions + 1) * (query_l1 + abs(worst) + 1)
    return step["upper_bound"] + guard < worst


def trace():
    codes, queries, reference = arrays()
    selected = np.load(DATA / "selected.npy")
    local = np.load(DATA / "local-reference.npy")
    assignments, _, _, _ = prepare_router(ROUTER, len(codes))
    index = bitplane_index.PrefixIndexV2(codes, assignments, queries.shape[1], max_prefix_bits=32)
    totals = defaultdict(lambda: defaultdict(float))
    gaps = []
    trace_path = FOLDER / "query-traces.jsonl.gz"
    with gzip.open(trace_path, "wt") as output:
        for qi, query in enumerate(queries):
            result = index.search(query[None, :], selected[qi:qi + 1], candidate_limit=max(KS),
                                  start_depth=32, candidate_target=0, stop_when_exact=False, trace=True)
            np.testing.assert_array_equal(result["rows"][0], local[qi])
            steps = result["prefix_trace"][0]
            assert [s["depth"] for s in steps] == list(range(32, -1, -1))
            l1 = float(np.abs(query.astype(np.float64)).sum())
            previous_scored = previous_lookups = 0
            for step in steps:
                step["new_documents_scored"] = step["documents_scored"] - previous_scored
                step["new_prefix_lookups"] = step["prefix_lookups"] - previous_lookups
                previous_scored, previous_lookups = step["documents_scored"], step["prefix_lookups"]
                if step["depth"]:
                    assert allows_stop(step, l1, len(query), max(KS)) == step["can_stop_exact"]
            output.write(json.dumps(dict(query=qi, query_values=query.tolist(),
                local_reference=local[qi].tolist(), global_reference=reference[qi].tolist(),
                steps=steps), allow_nan=False) + "\n")
            for k in KS:
                correct = next(s for s in steps if s["rows"][:k] == local[qi, :k].tolist())
                permitted = next(s for s in steps if allows_stop(s, l1, len(query), k))
                gaps.append(dict(query=qi, top_k=k, correct_depth=correct["depth"],
                    stop_depth=permitted["depth"], correct_documents_scored=correct["documents_scored"],
                    stop_documents_scored=permitted["documents_scored"],
                    extra_documents_scored=permitted["documents_scored"]-correct["documents_scored"],
                    extra_lookups=permitted["prefix_lookups"]-correct["prefix_lookups"]))
                for step in steps:
                    total = totals[(k, step["depth"])]
                    total["queries"] += 1
                    total["local_recall"] += recall(step["rows"][:k], local[qi, :k])
                    total["global_recall"] += recall(step["rows"][:k], reference[qi, :k])
                    for field in ["documents_scored", "new_documents_scored", "prefix_lookups", "new_prefix_lookups"]:
                        total[field] += step[field]
                    total["fraction_bound_allows_stop"] += allows_stop(step, l1, len(query), k)
            if qi % 100 == 0:
                print(f"Recorded prefix paths for {qi + 1}/{len(queries)} queries", flush=True)
    summary = [dict(top_k=k, depth=d, **{name: value/len(queries) if name != "queries" else int(value)
                for name, value in total.items()}) for (k, d), total in sorted(totals.items())]
    write_csv(FOLDER / "depth-summary.csv", summary)
    write_csv(FOLDER / "query-stopping-gaps.csv", gaps)
    save_json(FOLDER / "trace-checks.json", dict(queries=len(queries), depths_per_query=33,
        top_ks=KS, final_results_match_local_scan=True,
        note="Trace latency is excluded. All depths contain the same 1000 queries."))


def variants(method):
    settings = []
    for k in KS:
        if method == "scan":
            settings.append(dict(top_k=k, name=f"scan-k{k}"))
        elif method == "branch":
            for leaf, budget in [(32,8192), (128,8192), (2048,8192), (522931,0)]:
                settings.append(dict(top_k=k, name=f"branch-k{k}-leaf{leaf}",
                    leaf_size=leaf, node_budget=budget, prefer_deeper_ties=True))
        else:
            for depth in STARTS:
                settings.append(dict(top_k=k, name=f"prefix-k{k}-depth{depth}-exact",
                    start_depth=depth, candidate_target=0, stop_when_exact=True))
            for depth in [4,16,32]:
                for multiplier in [1,4,16,64]:
                    settings.append(dict(top_k=k, name=f"prefix-k{k}-depth{depth}-count{multiplier}k",
                        start_depth=depth, candidate_target=k*multiplier, stop_when_exact=True))
    return settings


def timing_worker(method, repetition):
    folder = FOLDER / f"timing-{method}-{repetition}"
    folder.mkdir(exist_ok=False)
    codes, queries, reference = arrays()
    selected = np.load(DATA / "selected.npy")
    local = np.load(DATA / "local-reference.npy")
    assignments, _, _, _ = prepare_router(ROUTER, len(codes))
    index = (bitplane_index.PrefixIndexV2(codes, assignments, queries.shape[1], max_prefix_bits=32)
             if method == "prefix" else bitplane_index.Index(
                 codes, assignments, queries.shape[1], build_bitplanes=method == "branch"))
    options = variants(method)
    np.random.default_rng(100 + repetition).shuffle(options)
    metadata = dict(method=method, repetition=repetition, power_before=power_state(),
        native_sha256=file_hash(bitplane_index.__file__), index=index.info(),
        memory_limit=64_000_000_000, timing="local search and returned arrays; routing precomputed")
    with (folder / "queries.jsonl").open("w") as output:
        for variant in options:
            k = variant["top_k"]
            for qi in range(10):
                search_index(index, method, queries[qi:qi+1], selected[qi:qi+1], variant)
            for qi in range(len(queries)):
                started = perf_counter()
                result = search_index(index, method, queries[qi:qi+1], selected[qi:qi+1], variant)
                milliseconds = (perf_counter() - started)*1000
                rows = result["rows"][0].tolist()
                exact = method == "scan" or (method == "prefix" and variant["candidate_target"] == 0)
                if exact:
                    np.testing.assert_array_equal(rows, local[qi,:k])
                record = dict(method=method, repetition=repetition, query=qi, **variant,
                    local_ms=milliseconds, local_recall=recall(rows,local[qi,:k]),
                    global_recall=recall(rows,reference[qi,:k]), rows=rows, **result["stats"][0])
                output.write(json.dumps(record, allow_nan=False) + "\n")
            output.flush()
    metadata.update(power_after=power_state(), peak_bytes=lifetime_peak_bytes())
    assert metadata["power_before"] == metadata["power_after"]
    assert metadata["peak_bytes"] <= metadata["memory_limit"]
    save_json(folder / "complete.json", metadata)


def run_all():
    if not (FOLDER / "configuration.json").exists():
        prepare()
    if not (FOLDER / "trace-checks.json").exists():
        trace()
    for repetition in range(3):
        methods = ["scan", "branch", "prefix"]
        methods = methods[repetition:] + methods[:repetition]
        for method in methods:
            if (FOLDER / f"timing-{method}-{repetition}/complete.json").exists():
                continue
            subprocess.run([sys.executable, "-m", "experiments.prefix_depth_study",
                            "worker", "--method", method, "--repetition", str(repetition)], check=True)
            print(f"Completed local timing: {method}, repeat {repetition + 1}", flush=True)
    config = json.loads((FOLDER / "configuration.json").read_text())
    assert all(file_hash(path)==digest for path,digest in config["input_hashes"].items())
    save_json(FOLDER / "complete.json", dict(status="complete", input_hashes_unchanged=True))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["run", "worker"])
    parser.add_argument("--method", choices=["scan", "branch", "prefix"])
    parser.add_argument("--repetition", type=int, default=0)
    args = parser.parse_args()
    faiss.omp_set_num_threads(1)
    with threadpool_limits(limits=1):
        if args.stage == "run":
            run_all()
        else:
            timing_worker(args.method, args.repetition)


if __name__ == "__main__":
    main()
