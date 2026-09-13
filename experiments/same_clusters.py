"""Compare local search methods after reusing the same cluster selections.

Run with: python -m experiments.same_clusters
The prepared local reference is a full scan of exactly those selected clusters.
"""

import argparse
import gc
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from time import perf_counter

import bitplane_index
import faiss
import numpy as np
import psutil
from threadpoolctl import threadpool_limits

from experiments.prefix_batch_worker_v2 import prepare_router, search_index
from experiments.worker import lifetime_peak_bytes, power_state

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "results/same-clusters-2026-09-13"
DATA = ROOT / "data/same-clusters-2026-09-13"
OLD_RESULTS = ROOT / "results/prefix-study-2026-09-13/comparison-final"
REPEATS = 3
RAM_LIMIT = 64_000_000_000


def save(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def digest(path):
    with path.open("rb") as file:
        return hashlib.file_digest(file, "sha256").hexdigest()


def prepare():
    """Reuse five existing layouts; save their selected IDs and local answers."""
    OUTPUT.mkdir(parents=True, exist_ok=True)
    DATA.mkdir(parents=True, exist_ok=True)
    requests = [("qwen-quora-d32", 1, method) for method in ["scan", "branch", "prefix"]]
    requests += [("qwen-quora-d256", 100, "scan"), ("nomic-msmarco-d256", 100, "scan")]
    setups = []
    for pool_name, top_k, routing_source in requests:
        table = json.loads((OLD_RESULTS / pool_name / "comparison-table.json").read_text())
        chosen = next(row for row in table if row["top_k"] == top_k
                      and row["target"] == .99 and row["method"] == routing_source)
        name = f"{pool_name}-{routing_source}-routing"
        folder = DATA / name
        folder.mkdir(exist_ok=True)
        pool = ROOT / "data/prefix-study-2026-09-13/pools" / pool_name
        dimensions = json.loads((pool / "pool.json").read_text())["identity"]["dimensions"]
        # Saved records may name their original worktree. The router directory
        # name is enough to find the same files in the shared main data cache.
        router = pool / "routers" / Path(chosen["router"]).name
        setup = dict(name=name, pool=str(pool), router=str(router), top_k=top_k,
                     dimensions=dimensions, clusters=chosen["clusters"], probes=chosen["probes"],
                     source_setting=chosen["setting_id"], folder=str(folder))
        record = OUTPUT / f"{name}.json"
        if record.exists():
            setups.append(json.loads(record.read_text()))
            continue
        queries = np.load(pool / "queries.npy")
        codes = np.load(pool / "codes.npy", mmap_mode="r")
        assignments, route, _, _ = prepare_router(router, len(codes))
        selected = np.concatenate([route(query[None, :], setup["probes"]) for query in queries])
        sizes = np.bincount(assignments)
        opened_rows = sizes[selected].sum(axis=1)
        assert np.all(opened_rows >= top_k)
        scan = bitplane_index.Index(codes, assignments, dimensions, build_bitplanes=False)
        local = np.empty((len(queries), top_k), dtype=np.int64)
        for start in range(0, len(queries), 32):
            stop = min(start + 32, len(queries))
            result = scan.scan(queries[start:stop], selected[start:stop], candidate_limit=top_k)
            local[start:stop] = result["rows"]
        assert np.all(local >= 0)
        np.save(folder / "selected.npy", selected)
        np.save(folder / "local-reference.npy", local)
        np.save(folder / "opened-rows.npy", opened_rows)
        paths = [pool / name for name in ["codes.npy", "queries.npy", "reference.npy"]]
        paths += [router / name for name in ["assignments.npy", "centroids.npy", "labels.npy"]]
        paths += [folder / name for name in ["selected.npy", "local-reference.npy", "opened-rows.npy"]]
        setup.update(queries=len(queries), documents=len(codes),
                     input_hashes={str(path.relative_to(ROOT)): digest(path) for path in paths})
        save(record, setup)
        setups.append(setup)
        print("Prepared", name, "opened rows:", round(float(opened_rows.mean()), 1), flush=True)
        del scan, codes, assignments
        gc.collect()
    return setups


def variants(method, setup):
    common = dict(top_k=setup["top_k"])
    if method == "scan":
        return [dict(common, name="scan")]
    if method == "branch":
        return [dict(common, name="branch-128", leaf_size=128, node_budget=8192,
                     prefer_deeper_ties=True, exploration=0, seed=42),
                dict(common, name="branch-scan", leaf_size=setup["documents"], node_budget=0,
                     prefer_deeper_ties=True, exploration=0, seed=42)]
    return [dict(common, name="prefix-4", start_depth=4, candidate_target=0, stop_when_exact=True),
            dict(common, name="prefix-0", start_depth=0, candidate_target=0, stop_when_exact=True)]


def measure(job, folder):
    """Build one method in a fresh process; time only local search and returned arrays."""
    setup = job["setup"]
    pool, prepared = Path(setup["pool"]), Path(setup["folder"])
    codes = np.load(pool / "codes.npy", mmap_mode="r")
    queries = np.load(pool / "queries.npy")
    assignments = np.load(Path(setup["router"]) / "assignments.npy", mmap_mode="r")
    selected = np.load(prepared / "selected.npy")
    local = np.load(prepared / "local-reference.npy")
    global_reference = np.load(pool / "reference.npy")[:, :setup["top_k"]]
    opened_rows = np.load(prepared / "opened-rows.npy")
    method = job["method"]
    power_before = power_state()
    if method == "prefix":
        index = bitplane_index.PrefixIndexV2(codes, assignments, setup["dimensions"], max_prefix_bits=32)
    else:
        index = bitplane_index.Index(codes, assignments, setup["dimensions"], build_bitplanes=method == "branch")
    storage = index.info()
    del codes, assignments
    gc.collect()
    process = psutil.Process()
    rows = []
    choices = variants(method, setup)
    if job["repeat"] % 2:
        choices.reverse()
    with (folder / "queries.jsonl").open("w") as file:
        for variant in choices:
            # Warm the call path before timing every query individually.
            for qi in range(5):
                search_index(index, method, queries[qi:qi+1], selected[qi:qi+1], variant)
            for qi, query in enumerate(queries):
                query = query[None, :]
                buckets = selected[qi:qi+1]
                started = perf_counter()
                result = search_index(index, method, query, buckets, variant)
                milliseconds = 1000 * (perf_counter() - started)
                count = int(result["counts"][0])
                ids = result["rows"][0, :count]
                exact = method in ("scan", "prefix") or variant["name"] == "branch-scan"
                if exact:
                    np.testing.assert_array_equal(ids, local[qi])
                row = dict(setup=setup["name"], method=method, variant=variant["name"],
                           repeat=job["repeat"], query=qi, opened_rows=int(opened_rows[qi]),
                           local_recall=len(set(ids) & set(local[qi])) / setup["top_k"],
                           global_recall=len(set(ids) & set(global_reference[qi])) / setup["top_k"],
                           local_ms=milliseconds, ids=ids.tolist(), **result["stats"][0])
                file.write(json.dumps(row, allow_nan=False) + "\n")
                rows.append(row)
            file.flush()
            print(variant["name"], "complete", flush=True)
    power_after = power_state()
    peak = lifetime_peak_bytes()
    assert peak < RAM_LIMIT
    save(folder / "result.json", dict(status="complete", measurements=len(rows), storage=storage,
         lifetime_peak_bytes=peak, rss_after=process.memory_info().rss,
         power_before=power_before, power_after=power_after, pid=os.getpid(),
         native_sha256=digest(Path(bitplane_index.__file__)), worker_sha256=digest(Path(__file__))))


def summarize():
    """Keep each setting's repeated query counts and timings together."""
    groups = {}
    for folder in sorted((OUTPUT / "cases").iterdir()):
        result_path = folder / "result.json"
        if not result_path.exists():
            continue
        result = json.loads(result_path.read_text())
        if result["status"] != "complete":
            continue
        job = json.loads((folder / "job.json").read_text())
        for line in (folder / "queries.jsonl").read_text().splitlines():
            row = json.loads(line)
            key = (row["setup"], row["variant"])
            group = groups.setdefault(key, dict(setup=job["setup"], method=row["method"], variant=row["variant"],
                                      rows=[], peaks=[], power=[]))
            group["rows"].append(row)
        for variant in variants(job["method"], job["setup"]):
            group = groups[(job["setup"]["name"], variant["name"])]
            group["peaks"].append(result["lifetime_peak_bytes"])
            group["power"].append(result["power_before"] == result["power_after"])
    summary = []
    for group in groups.values():
        rows = group.pop("rows")
        by_repeat = sorted({row["repeat"] for row in rows})
        fields = ["opened_rows", "documents_scored", "nodes", "bitplane_words", "leaf_words",
                  "prefix_levels", "prefix_lookups", "final_depth", "local_recall", "global_recall"]
        item = dict(setup=group["setup"]["name"], method=group["method"], variant=group["variant"],
                    measurements=len(rows), repetitions=len(by_repeat),
                    median_local_ms=float(np.median([row["local_ms"] for row in rows])),
                    p95_local_ms=float(np.percentile([row["local_ms"] for row in rows], 95)),
                    process_peak_bytes=max(group["peaks"]), power_stable=all(group["power"]))
        item.update({"mean_"+field:float(np.mean([row[field] for row in rows])) for field in fields})
        item["repeat_medians_ms"] = [float(np.median([row["local_ms"] for row in rows if row["repeat"] == repeat]))
                                      for repeat in by_repeat]
        item["complete"] = len(rows) == group["setup"]["queries"] * REPEATS and len(by_repeat) == REPEATS
        summary.append(item)
    save(OUTPUT / "summary.json", summary)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", type=Path)
    args = parser.parse_args()
    faiss.omp_set_num_threads(1)
    with threadpool_limits(limits=1):
        if args.case:
            measure(json.loads((args.case / "job.json").read_text()), args.case)
            return
        setups = prepare()
        cases = OUTPUT / "cases"
        cases.mkdir(exist_ok=True)
        methods = ["scan", "branch", "prefix"]
        save(OUTPUT / "configuration.json", dict(setups=setups, repetitions=REPEATS,
             ram_limit_bytes=RAM_LIMIT, method_order="rotated across repeat blocks",
             timing="local native call and result conversion; routing precomputed and excluded",
             reference="exhaustive scan of the saved selected clusters; global reference retained separately"))
        for repeat in range(REPEATS):
            for setup in setups:
                order = methods[repeat:] + methods[:repeat]
                for method in order:
                    folder = cases / f"{setup['name']}-{method}-r{repeat}"
                    if (folder / "result.json").exists():
                        continue
                    folder.mkdir(exist_ok=True)
                    save(folder / "job.json", dict(setup=setup, method=method, repeat=repeat))
                    print("Running", folder.name, flush=True)
                    with (folder / "worker.log").open("w") as log:
                        subprocess.run([sys.executable, "-m", "experiments.same_clusters", "--case", str(folder)],
                                       cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True, timeout=600)
                    summarize()
        # Recheck the actual inputs after all cases; the saved selection never changes.
        for setup in setups:
            for name, expected in setup["input_hashes"].items():
                assert digest(ROOT / name) == expected, name
        summarize()
        print("Complete:", OUTPUT / "summary.json", flush=True)


if __name__ == "__main__":
    main()
