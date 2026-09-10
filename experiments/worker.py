"""Measure one search configuration in its own process.

The worker releases build inputs before the query phase. Lifetime peak includes
construction; query RSS observations are sampled and may miss very brief peaks.
"""
import argparse
import gc
import hashlib
import json
import os
import platform
import re
import resource
import subprocess
import sys
from pathlib import Path
from time import perf_counter

import faiss
import numpy as np
import psutil
from threadpoolctl import threadpool_limits
import bitplane_index

from experiments.direct_routing import DirectPrefixRouter


def power_state():
    if sys.platform != 'darwin':
        return {'available': False}
    battery = subprocess.run(['pmset', '-g', 'batt'], capture_output=True, text=True)
    modes = subprocess.run(['pmset', '-g', 'custom'], capture_output=True, text=True)
    match = re.search(r"Now drawing from '([^']+)'", battery.stdout)
    return {'available': battery.returncode == modes.returncode == 0 and match is not None,
            'source': match.group(1) if match else None, 'settings': modes.stdout.strip()}


def lifetime_peak_bytes():
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(value if sys.platform == 'darwin' else value*1024)


def native_search(index, query, buckets, case):
    top_k = case['top_k']
    if case['method'] == 'scan':
        return index.scan(query, buckets, candidate_limit=top_k)
    if case['method'] == 'branch':
        return index.search(query, buckets, candidate_limit=top_k,
                            node_budget=case['node_budget'], leaf_size=case['leaf_size'],
                            explore_probability=case.get('exploration', 0.0), seed=case.get('seed', 0))
    return index.search(query, buckets, candidate_limit=top_k,
                        candidate_target=case['candidate_target'], key_limit=case['key_limit'])


def run_case(case, output):
    output.mkdir(parents=True, exist_ok=False)
    process = psutil.Process()
    power_before = power_state()
    rss_at_start = process.memory_info().rss
    pool = Path(case['pool'])
    codes = np.load(pool/'codes.npy', mmap_mode='r')
    query_rows = case['query_rows']
    queries = np.ascontiguousarray(np.load(pool/'queries.npy', mmap_mode='r')[query_rows], dtype=np.float32)
    reference = np.load(pool/'reference.npy', mmap_mode='r')[query_rows]
    quantizer = None
    direct_router = None
    labels = None
    routing_kind = 'none'
    route_bytes = 0
    if case['router'] is None:
        assignments = np.zeros(len(codes), dtype=np.int64)
    else:
        router_dir = Path(case['router'])
        route_config = json.loads((router_dir/'router.json').read_text())
        routing_kind = route_config['kind']
        assignments = np.load(router_dir/'assignments.npy', mmap_mode='r')
        labels = np.load(router_dir/'labels.npy')
        if routing_kind == 'direct':
            direct_router = DirectPrefixRouter(labels, route_config['routing_bits'])
            route_bytes = direct_router.payload_bytes
        else:
            centroids = np.load(router_dir/'centroids.npy')
            quantizer = faiss.IndexFlatIP(centroids.shape[1])
            quantizer.add(centroids)
            route_bytes = centroids.nbytes + labels.nbytes
            del centroids
    faiss.omp_set_num_threads(1)

    build_start = perf_counter()
    if case['method'] == 'keys':
        index = bitplane_index.KeyIndex(codes, assignments, case['dimensions'],
                                        case['key_bits'], case['key_offset'])
    else:
        index = bitplane_index.Index(codes, assignments, case['dimensions'],
                                     build_bitplanes=case['method']=='branch')
    build_ms = 1000*(perf_counter()-build_start)
    storage = index.info()
    # Native indexes own their arrays. Closing these mmap inputs removes their
    # resident file pages from the serving-state measurement.
    del codes, assignments
    gc.collect()
    build_peak = lifetime_peak_bytes()

    def route(query):
        if direct_router is not None:
            return direct_router.select(query, case['probes'])
        if quantizer is None:
            return np.zeros((1, 1), dtype=np.int64)
        routing_query = query[:, :quantizer.d] if routing_kind == 'sign' else query
        _, positions = quantizer.search(np.ascontiguousarray(routing_query), min(case['probes'], len(labels)))
        return np.ascontiguousarray(labels[positions], dtype=np.int64)

    # Lazy library initialization is kept out of measured query times.
    native_search(index, queries[:1], route(queries[:1]), case)
    rss_before_queries = process.memory_info().rss
    sampled_query_peak = rss_before_queries
    saved_rows = 0
    exact_checks = 0
    start = perf_counter()
    with (output/'queries.jsonl').open('w') as file:
        for repetition in range(case['repetitions']):
            for row, query in enumerate(queries):
                query = query[None, :]
                query_start = perf_counter()
                routing_start = perf_counter()
                selected = route(query)
                routing_ms = 1000*(perf_counter()-routing_start)
                search_start = perf_counter()
                result = native_search(index, query, selected, case)
                search_ms = 1000*(perf_counter()-search_start)
                query_ms = 1000*(perf_counter()-query_start)
                sampled_query_peak = max(sampled_query_peak, process.memory_info().rss)
                returned = result['rows'][0]
                truth = reference[row, :case['top_k']]
                recall = len(set(returned) & set(truth))/len(truth)
                exact_local = (case['method']=='scan'
                               or (case['method']=='branch' and case['node_budget']==0)
                               or (case['method']=='keys' and case['candidate_target']==case['key_limit']==0))
                all_clusters = case['router'] is None or case['probes'] >= len(labels)
                if exact_local and all_clusters:
                    np.testing.assert_array_equal(returned, truth)
                    exact_checks += 1
                record = dict(query=int(query_rows[row]), repetition=repetition, recall=recall,
                              routing_ms=routing_ms, search_ms=search_ms, query_ms=query_ms,
                              returned=int(result['counts'][0]), **result['stats'][0])
                file.write(json.dumps(record, allow_nan=False)+'\n')
                saved_rows += 1
    peak = lifetime_peak_bytes()
    budget = case['ram_budget_bytes']
    if peak <= budget:
        budget_status = 'fits_by_lifetime_peak'
    elif rss_before_queries > budget:
        budget_status = 'resident_state_exceeds_budget'
    else:
        budget_status = 'query_peak_unresolved_build_peak_exceeds_budget'
    memory = dict(rss_at_start=rss_at_start, build_lifetime_peak=build_peak,
                  rss_before_queries=rss_before_queries, sampled_query_peak=sampled_query_peak,
                  lifetime_peak_bytes=peak, routing_payload_bytes=route_bytes,
                  routing_label_count=1 if labels is None else len(labels),
                  budget_bytes=budget, budget_status=budget_status,
                  scope='RSS includes runtime and evaluation arrays. Lifetime peak includes build and is a conservative query bound. Sampled query peak is not a guaranteed maximum.')
    binary = Path(bitplane_index.__file__)
    source_commit = subprocess.run(['git', 'rev-parse', 'HEAD'], capture_output=True, text=True).stdout.strip()
    result = dict(status='complete', pid=os.getpid(), case=case, storage=storage, memory=memory,
                  build_ms=build_ms, query_loop_seconds=perf_counter()-start,
                  measurements=saved_rows, exact_checks=exact_checks,
                  source_commit=source_commit, worker_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  native_sha256=hashlib.sha256(binary.read_bytes()).hexdigest(), native_module=str(binary),
                  python=sys.version, platform=platform.platform(),
                  power_before=power_before, power_after=power_state())
    (output/'result.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('case', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    with threadpool_limits(limits=1):
        run_case(json.loads(args.case.read_text()), args.output)


if __name__ == '__main__':
    main()
