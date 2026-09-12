"""Build one index, then measure declared variants on every cached query.

The controller orders jobs and variants and enforces the process memory limit.
This worker writes raw whole-query timings and retains completed-variant progress.
"""

import argparse
import gc
import hashlib
import json
import os
import platform
import subprocess
import sys
from pathlib import Path
from time import perf_counter

import bitplane_index
import faiss
import numpy as np
import psutil
from threadpoolctl import threadpool_limits

from experiments.worker import lifetime_peak_bytes, power_state


def file_hash(path):
    with Path(path).open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def prepare_router(folder, documents):
    if folder is None:
        return (np.zeros(documents, dtype=np.int64),
                lambda query, probes: np.zeros((1, 1), dtype=np.int64), 0, 1)
    folder = Path(folder)
    kind = json.loads((folder / "router.json").read_text())["kind"]
    if kind not in ("ivf", "sign"):
        raise ValueError("batch worker requires centroid routing (ivf or sign)")
    assignments = np.load(folder / "assignments.npy", mmap_mode="r")
    labels = np.load(folder / "labels.npy")
    centroids = np.load(folder / "centroids.npy")
    quantizer = faiss.IndexFlatIP(centroids.shape[1])
    quantizer.add(centroids)
    payload_bytes = centroids.nbytes + labels.nbytes

    def route(query, probes):
        routing_query = query[:, :quantizer.d] if kind == "sign" else query
        _, positions = quantizer.search(np.ascontiguousarray(routing_query), min(probes, len(labels)))
        return np.ascontiguousarray(labels[positions], dtype=np.int64)

    return assignments, route, payload_bytes, len(labels)


def search_index(index, method, query, selected, variant):
    if method == "scan":
        return index.scan(query, selected, candidate_limit=variant["top_k"])
    if method == "branch":
        return index.search(query, selected, candidate_limit=variant["top_k"],
                            node_budget=variant["node_budget"], leaf_size=variant["leaf_size"],
                            explore_probability=variant.get("exploration", 0),
                            seed=variant.get("seed", 0),
                            prefer_deeper_ties=variant.get("prefer_deeper_ties", False))
    return index.search(query, selected, candidate_limit=variant["top_k"],
                        start_depth=variant["start_depth"],
                        candidate_target=variant["candidate_target"])


def measure_job(job, output):
    output.mkdir(parents=True, exist_ok=False)
    process = psutil.Process()
    project = Path(__file__).resolve().parents[1]
    sources = ["experiments/prefix_batch_worker_v2.py", "experiments/worker.py",
               "cpp/index.cpp", "cpp/index.hpp", "cpp/score.hpp", "cpp/prefix_index_v2.cpp",
               "cpp/prefix_index_v2.hpp", "cpp/bindings.cpp", "CMakeLists.txt"]
    memory = dict(rss_at_start=process.memory_info().rss, rss_before_queries=0,
                  sampled_query_peak=0, budget_bytes=job["ram_budget_bytes"])
    result = dict(status="running", job=job, pid=os.getpid(), memory=memory,
                  completed_variants=[], variant_summaries=[], measurements=0, exact_checks=0,
                  power_before=power_state(), python=sys.version, platform=platform.platform(),
                  source_commit=subprocess.run(["git", "rev-parse", "HEAD"], cwd=project,
                                               capture_output=True, text=True).stdout.strip(),
                  source_hashes={name: file_hash(project / name) for name in sources},
                  worker_sha256=file_hash(__file__), native_sha256=file_hash(bitplane_index.__file__),
                  native_module=str(bitplane_index.__file__))

    def save_progress():
        (output / "result.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")

    started = perf_counter()
    try:
        faiss.omp_set_num_threads(1)
        pool = Path(job["pool"])
        queries = np.load(pool / "queries.npy")
        if job["query_rows"] != list(range(len(queries))):
            raise ValueError("query_rows must include all cached queries in their original order")
        reference = np.load(pool / "reference.npy", mmap_mode="r")
        codes = np.load(pool / "codes.npy", mmap_mode="r")
        assignments, route, route_bytes, label_count = prepare_router(job["router"], len(codes))
        memory.update(routing_payload_bytes=route_bytes, routing_label_count=label_count)
        method = job["method"]
        if method not in ("scan", "branch", "prefix"):
            raise ValueError("method must be scan, branch, or prefix")
        build_start = perf_counter()
        if method == "prefix":
            index = bitplane_index.PrefixIndexV2(codes, assignments, job["dimensions"],
                                                max_prefix_bits=job.get("max_prefix_bits", 32))
        else:
            index = bitplane_index.Index(codes, assignments, job["dimensions"],
                                         build_bitplanes=method == "branch")
        result.update(build_ms=1000 * (perf_counter() - build_start), storage=index.info())
        # Native ownership lets us release preparation arrays before measuring RSS.
        del codes, assignments
        gc.collect()
        memory["build_lifetime_peak"] = lifetime_peak_bytes()
        with (output / "queries.jsonl").open("w") as records:
            for variant in job["variants"]:
                result["current_variant"] = variant["setting_id"]
                save_progress()
                top_k = variant["top_k"]
                if not 1 <= top_k <= reference.shape[1] or variant["repetitions"] < 1:
                    raise ValueError("top_k must fit the cached reference; repetitions must be positive")
                search_index(index, method, queries[:1], route(queries[:1], variant["probes"]), variant)
                rss_before = process.memory_info().rss
                memory["rss_before_queries"] = max(memory["rss_before_queries"], rss_before)
                memory["sampled_query_peak"] = max(memory["sampled_query_peak"], rss_before)
                variant_peak, count, exact_checks = rss_before, 0, 0
                variant_start = perf_counter()
                exact_local = (method == "scan" or (method == "branch" and variant["node_budget"] == 0)
                               or (method == "prefix" and variant["candidate_target"] == 0))
                all_clusters = job["router"] is None or variant["probes"] >= label_count
                for repetition in range(variant["repetitions"]):
                    for row, query in enumerate(queries):
                        query = query[None, :]
                        query_start = perf_counter()
                        routing_start = perf_counter()
                        selected = route(query, variant["probes"])
                        routing_ms = 1000 * (perf_counter() - routing_start)
                        search_start = perf_counter()
                        found = search_index(index, method, query, selected, variant)
                        search_ms = 1000 * (perf_counter() - search_start)
                        query_ms = 1000 * (perf_counter() - query_start)
                        variant_peak = max(variant_peak, process.memory_info().rss)
                        memory["sampled_query_peak"] = max(memory["sampled_query_peak"], variant_peak)
                        returned, truth = found["rows"][0], reference[row, :top_k]
                        if exact_local and all_clusters:
                            np.testing.assert_array_equal(returned, truth)
                            exact_checks += 1
                            result["exact_checks"] += 1
                        record = dict(setting_id=variant["setting_id"], query=int(row),
                                      repetition=repetition, recall=len(set(returned) & set(truth)) / top_k,
                                      query_ms=query_ms, routing_ms=routing_ms, search_ms=search_ms,
                                      returned=int(found["counts"][0]), **found["stats"][0])
                        records.write(json.dumps(record, allow_nan=False) + "\n")
                        count += 1
                        result["measurements"] += 1
                records.flush()
                result["completed_variants"].append(variant["setting_id"])
                result["variant_summaries"].append(dict(
                    setting_id=variant["setting_id"], measurements=count, exact_checks=exact_checks,
                    rss_before_queries=rss_before, sampled_query_peak=variant_peak,
                    query_loop_seconds=perf_counter() - variant_start))
                save_progress()
        result["status"] = "complete"
    except Exception as error:
        result.update(status="failed", error=dict(type=type(error).__name__, message=str(error)))
        raise
    finally:
        peak = lifetime_peak_bytes()
        budget = job["ram_budget_bytes"]
        budget_status = ("fits_by_lifetime_peak" if peak <= budget else
                         "resident_state_exceeds_budget" if memory["rss_before_queries"] > budget else
                         "query_peak_unresolved_build_peak_exceeds_budget")
        memory.update(lifetime_peak_bytes=peak, budget_status=budget_status,
                      scope="RSS includes runtime and evaluation arrays. Lifetime peak includes build. "
                            "Query peak is sampled; rss_before_queries is the largest variant start.")
        result.update(wall_seconds=perf_counter() - started, power_after=power_state())
        save_progress()
    return result


def run_job(job, output):
    with threadpool_limits(limits=1):
        return measure_job(job, Path(output))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("job", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    run_job(json.loads(args.job.read_text()), args.output)


if __name__ == "__main__":
    main()
