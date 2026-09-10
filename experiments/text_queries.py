"""Measure text -> embedding -> routing -> search with the frozen real-data settings."""
import argparse
import gc
import hashlib
import json
from pathlib import Path
from time import perf_counter

import faiss
import numpy as np
import psutil
import torch
from threadpoolctl import threadpool_limits
import bitplane_index

from embedding_model import NomicEncoder, matryoshka_vectors, normalized_prefix, package_versions
from experiments.components import reference_rows, pack_signs
from experiments.worker import native_search, power_state, lifetime_peak_bytes
from experiments.isolated import ROOT


def run(output, query_count=20, blocks=3, repeats=2):
    output.mkdir(parents=True, exist_ok=False)
    saved = ROOT / 'results/independent-evaluation-2026-09-10/frozen-shortlist.json'
    choices = json.loads(saved.read_text())
    cases = {choice['case']['method']: choice['case'] for choice in choices
             if choice['case']['documents']==1000000
             and dict(selection='conservative', target=.99) in choice['uses']}
    assert set(cases)=={'scan','branch','keys'}
    query_dir = ROOT / 'data/fresh-evaluation-2026-09-10'
    texts = [json.loads(line) for line in (query_dir/'queries.jsonl').read_text().splitlines()][:query_count]
    meta = json.loads((query_dir/'manifest.json').read_text())
    identity = json.loads(next((ROOT/'data/scaling').glob('*/embeddings/*/manifest.json')).read_text())['identity']
    process = psutil.Process()
    before = power_state()
    torch.set_num_threads(1)
    faiss.omp_set_num_threads(1)
    encoder = NomicEncoder(model_name=meta['model'], revision=meta['revision'], batch_size=1,
                           max_length=identity['max_length'], device='mps', cache_dir=ROOT/'data/model')

    def encode(text):
        raw = encoder.model.encode(['search_query: '+text], batch_size=1, show_progress_bar=False,
                                   convert_to_numpy=True, normalize_embeddings=False)
        _, full = matryoshka_vectors(raw, 768)
        return normalized_prefix(full, 256)

    # Warm each sequence shape before measuring steady-state requests.
    for text in texts:
        encode(text['text'])
    measurements, vectors, returned = [], [], []
    rng = np.random.default_rng(83)
    with threadpool_limits(limits=1):
        for block in range(blocks):
            for method in rng.permutation(['scan','branch','keys']):
                case = cases[method]
                pool = Path(case['pool'])
                router = Path(case['router'])
                codes = np.load(pool/'codes.npy', mmap_mode='r')
                assignments = np.load(router/'assignments.npy', mmap_mode='r')
                if method=='keys':
                    index = bitplane_index.KeyIndex(codes, assignments, 256, case['key_bits'], case['key_offset'])
                else:
                    index = bitplane_index.Index(codes, assignments, 256, build_bitplanes=method=='branch')
                centroids = np.load(router/'centroids.npy')
                labels = np.load(router/'labels.npy')
                quantizer = faiss.IndexFlatIP(centroids.shape[1]); quantizer.add(centroids)
                del codes, assignments, centroids
                gc.collect()

                def search(query):
                    routing_query = query[:,:quantizer.d] if case['router_kind']=='sign' else query
                    _, positions = quantizer.search(np.ascontiguousarray(routing_query), min(case['probes'],len(labels)))
                    buckets = np.ascontiguousarray(labels[positions], dtype=np.int64)
                    return native_search(index, query, buckets, case)

                search(encode(texts[0]['text']))
                for repetition in range(repeats):
                    for text in texts:
                        start = perf_counter()
                        query = encode(text['text'])
                        encoded_at = perf_counter()
                        result = search(query)
                        finished = perf_counter()
                        measurements.append(dict(method=str(method),block=block,repetition=repetition,
                            query_id=text['query_id'],characters=len(text['text']),
                            encoding_ms=1000*(encoded_at-start),retrieval_ms=1000*(finished-encoded_at),
                            total_ms=1000*(finished-start),rss_bytes=process.memory_info().rss))
                        vectors.append(query[0].copy())
                        returned.append(result['rows'][0].copy())
                del index, quantizer, labels
                gc.collect()
                print(f'block {block+1}/{blocks}: {method} complete', flush=True)

        # Compute references from the actual encoded vectors after timing. This
        # handles any batch/single-query numeric difference without changing truth.
        unique, inverse = np.unique(np.array(vectors), axis=0, return_inverse=True)
        pool = Path(cases['scan']['pool'])
        signs = np.load(pool/'documents.npy', mmap_mode='r') >= 0
        np.testing.assert_array_equal(pack_signs(signs), np.load(pool/'codes.npy', mmap_mode='r'))
        truth = reference_rows(signs, unique, 100)
        for i, row in enumerate(measurements):
            row['recall'] = len(set(returned[i]) & set(truth[inverse[i]]))/100
    np.save(output/'query-vectors.npy', np.array(vectors))
    np.save(output/'returned.npy', np.array(returned))
    np.save(output/'reference.npy', truth[inverse])
    (output/'measurements.json').write_text(json.dumps(measurements, indent=2)+'\n')
    (output/'configuration.json').write_text(json.dumps(dict(cases=cases,query_count=query_count,
        blocks=blocks,repeats=repeats,shortlist_sha256=hashlib.sha256(saved.read_bytes()).hexdigest()),indent=2)+'\n')
    summary = []
    for method in ('scan','branch','keys'):
        rows = [r for r in measurements if r['method']==method]
        summary.append(dict(method=method,observations=len(rows),recall=sum(round(r['recall']*100) for r in rows)/(len(rows)*100),
            encoding_p50_ms=float(np.median([r['encoding_ms'] for r in rows])),
            retrieval_p50_ms=float(np.median([r['retrieval_ms'] for r in rows])),
            total_p50_ms=float(np.median([r['total_ms'] for r in rows])),
            total_p95_ms=float(np.percentile([r['total_ms'] for r in rows],95))))
    (output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    (output/'environment.json').write_text(json.dumps(dict(model=meta['model'],revision=meta['revision'],
        recipe=meta['recipe'],device='mps',versions=package_versions(),power_before=before,power_after=power_state(),
        process_lifetime_peak_bytes=lifetime_peak_bytes(),mps_driver_allocated_bytes=torch.mps.driver_allocated_memory(),
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        scope='20 existing independent-query IDs; frozen conservative99% settings. Actual single-query encoding and search timed together; reference computed afterward from actual vectors. No retuning or speedup significance claim. RSS peak includes reference construction; MPS allocation may overlap RSS.'),indent=2)+'\n')
    print(json.dumps(summary,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',required=True,type=Path)
    args=parser.parse_args()
    run(args.output)
