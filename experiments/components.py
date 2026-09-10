"""Check storage equations and native work counters before fitting timing models."""
import csv
import gc
import hashlib
import json
import platform
import sys
from pathlib import Path
from time import perf_counter

import numpy as np
from threadpoolctl import threadpool_limits
import bitplane_index

from experiments.models import bitplane_bytes, key_directory_bytes, packed_code_bytes


def pack_signs(signs):
    """Pack document rows exactly as the native API expects."""
    codes = np.zeros((len(signs), (signs.shape[1] + 63)//64), dtype=np.uint64)
    for dimension in range(signs.shape[1]):
        codes[:, dimension//64] |= signs[:, dimension].astype(np.uint64) << np.uint64(dimension % 64)
    return codes


def reference_rows(signs, queries, top_k):
    """Independent full scores, followed by row-ID tie ordering."""
    full = 2*signs.astype(np.float64)-1
    scores = queries.astype(np.float64) @ full.T
    return np.argsort(-scores, axis=1, kind='stable')[:, :top_k]


def search(index, method, query, buckets, settings, top_k):
    if method == 'scan':
        return index.scan(query, buckets, candidate_limit=top_k)
    if method == 'branch':
        return index.search(query, buckets, candidate_limit=top_k,
                            node_budget=settings['node_budget'], leaf_size=settings['leaf_size'])
    return index.search(query, buckets, candidate_limit=top_k,
                        candidate_target=settings['candidate_target'], key_limit=settings['key_limit'])


def run_components(config, output):
    output.mkdir(parents=True, exist_ok=False)
    data = config['data']
    settings = config['search']
    dimensions = data['dimensions']
    repetitions = config['measurement']['repetitions']
    rows, indexes = [], []
    rng = np.random.default_rng(data['seed'])
    started = perf_counter()
    for documents in data['sizes']:
        signs = rng.integers(0, 2, (documents, dimensions), dtype=np.uint8)
        queries = np.ascontiguousarray(rng.normal(size=(data['queries'], dimensions)), dtype=np.float32)
        codes = pack_signs(signs)
        assignments = np.zeros(documents, dtype=np.int64)
        selected = np.zeros((1, 1), dtype=np.int64)
        top_k = min(settings['top_k'], documents)
        with threadpool_limits(limits=1):
            truth = reference_rows(signs, queries, top_k)
        key_slice = signs[:, settings['key_offset']:settings['key_offset']+settings['key_bits']]
        occupied_keys = len(np.unique(key_slice, axis=0))

        for method in ('scan', 'branch', 'keys'):
            build_start = perf_counter()
            if method == 'keys':
                index = bitplane_index.KeyIndex(codes, assignments, dimensions,
                                                settings['key_bits'], settings['key_offset'])
            else:
                index = bitplane_index.Index(codes, assignments, dimensions, build_bitplanes=method=='branch')
            build_ms = 1000*(perf_counter()-build_start)
            storage = index.info()
            predicted_codes = packed_code_bytes(documents, dimensions)
            predicted_planes = bitplane_bytes([documents], dimensions) if method=='branch' else 0
            predicted_directory = key_directory_bytes(occupied_keys, offset_bytes=np.dtype(np.uintp).itemsize) if method=='keys' else 0
            assert storage['codes_bytes'] == predicted_codes
            assert storage['bitplanes_bytes'] == predicted_planes
            assert storage['key_directory_payload_bytes'] == predicted_directory
            assert storage['row_ids_bytes'] == documents*np.dtype(np.int64).itemsize
            indexes.append(dict(method=method, build_ms=build_ms,
                                predicted_codes_bytes=predicted_codes,
                                predicted_bitplanes_bytes=predicted_planes,
                                predicted_directory_bytes=predicted_directory, **storage))
            # Warmup is outside the saved query timings.
            search(index, method, queries[:1], selected, settings, top_k)
            for repetition in range(repetitions):
                for qi, query in enumerate(queries):
                    request_start = perf_counter()
                    result = search(index, method, query[None, :], selected, settings, top_k)
                    request_ms = 1000*(perf_counter()-request_start)
                    returned = result['rows'][0]
                    recall = len(set(returned) & set(truth[qi])) / top_k
                    exact_mode = (method=='scan' or (method=='branch' and settings['node_budget']==0)
                                  or (method=='keys' and settings['key_limit']==0 and settings['candidate_target']==0))
                    if exact_mode:
                        np.testing.assert_array_equal(returned, truth[qi])
                    stats = result['stats'][0]
                    rows.append(dict(method=method, documents=documents, dimensions=dimensions,
                                     query=qi, repetition=repetition, top_k=top_k, recall=recall,
                                     request_ms=request_ms, **stats))
            # This study records logical storage, not a process-memory comparison.
            # Keep only one native index live; input arrays still belong to the driver.
            del index
            gc.collect()

    write_csv(output/'measurements.csv', rows)
    write_csv(output/'indexes.csv', indexes)
    summaries = []
    for documents in data['sizes']:
        for method in ('scan', 'branch', 'keys'):
            selected_rows = [r for r in rows if r['documents']==documents and r['method']==method]
            # Repeats are timing observations, not additional independent quality queries.
            first_repeat = [r for r in selected_rows if r['repetition']==0]
            summaries.append(dict(method=method, documents=documents,
                                  recall=float(np.mean([r['recall'] for r in first_repeat])),
                                  p50_ms=float(np.median([r['request_ms'] for r in selected_rows])),
                                  p95_ms=float(np.percentile([r['request_ms'] for r in selected_rows],95)),
                                  mean_scored=float(np.mean([r['documents_scored'] for r in first_repeat])),
                                  mean_split_words=float(np.mean([r['bitplane_words'] for r in first_repeat])),
                                  mean_leaf_words=float(np.mean([r['leaf_words'] for r in first_repeat])),
                                  mean_key_attempts=float(np.mean([r['key_attempts'] for r in first_repeat]))))
    write_csv(output/'summary.csv', summaries)
    module_path=Path(bitplane_index.__file__)
    metadata=dict(config=config, platform=platform.platform(), machine=platform.machine(),
                  python=sys.version, native_module=str(module_path),
                  native_sha256=hashlib.sha256(module_path.read_bytes()).hexdigest(),
                  seconds=perf_counter()-started,
                  scope='Random-sign component study, one cluster, one CPU thread. No tuned winner, process-RAM claim, or real-data recall model.')
    (output/'environment.json').write_text(json.dumps(metadata, indent=2)+'\n')
    lines=['# Component checks', '', 'Storage equations matched native logical byte counts in every case.',
           'These timings are exploratory component observations on random signs, not a tuned comparison.',
           '', '| Documents | Method | Recall | p50 ms | Scored rows |', '|---:|---|---:|---:|---:|']
    for row in summaries:
        lines.append(f"| {row['documents']} | {row['method']} | {row['recall']:.3f} | {row['p50_ms']:.4f} | {row['mean_scored']:.1f} |")
    (output/'README.md').write_text('\n'.join(lines)+'\n')
    print('\n'.join(lines))


def write_csv(path, rows):
    with path.open('w', newline='') as file:
        writer=csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
