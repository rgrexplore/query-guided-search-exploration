"""Reuse cached document embeddings and compare shared routing layouts."""
import hashlib
import itertools
import json
import math
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

from experiments.components import pack_signs, reference_rows
from experiments.isolated import ROOT, execute_cases
from routing import build_router


def file_hash(path):
    with path.open('rb') as file:
        return hashlib.file_digest(file, 'sha256').hexdigest()


def prepare_real_pool(config, documents):
    data = config['data']
    dimensions = data['dimensions']
    source = (ROOT/data['embedding_dir']).resolve()
    derived = source/'derived'/str(dimensions)
    query_source = (ROOT/data['query_dir']).resolve() if data.get('query_dir') else derived
    identity = dict(documents=documents, dimensions=dimensions, queries=data['queries'],
                    query_start=data['query_start'], source=str(source), query_source=str(query_source),
                    embedding_manifest=file_hash(source/'manifest.json'),
                    query_manifest=file_hash(query_source/'manifest.json'), top_k=config['search']['top_k'])
    pool = (ROOT/data['cache_dir']/f'n{documents}').resolve()
    if pool.exists():
        if json.loads((pool/'pool.json').read_text())['identity'] != identity:
            raise ValueError('prepared pool belongs to different inputs; choose a new cache_dir')
        return pool
    pool.mkdir(parents=True)
    full = np.load(source/'full-documents.npy', mmap_mode='r')
    vectors = np.ascontiguousarray(full[:documents, :dimensions])
    codes = np.array(np.load(derived/'codes.npy', mmap_mode='r')[:documents], copy=True)
    start = data['query_start']
    stop = start + data['queries']
    queries = np.array(np.load(query_source/'queries.npy', mmap_mode='r')[start:stop], copy=True)
    query_ids = json.loads((query_source/'manifest.json').read_text())['query_ids'][start:stop]
    assert len(vectors)==documents and len(queries)==data['queries']
    assert queries.shape[1]==dimensions and len(query_ids)==len(queries)
    signs = vectors >= 0
    # The independent reference starts from the original float vectors' signs,
    # not from the native index or its packed scoring helper.
    np.testing.assert_array_equal(codes, pack_signs(signs))
    with threadpool_limits(limits=1):
        reference = reference_rows(signs, queries, config['search']['top_k'])
    # Full cached vectors are already normalized. Normalize their shorter prefix
    # for spherical centroid training; this does not change any document signs.
    vectors = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
    for name, values in [('codes',codes), ('queries',queries), ('reference',reference), ('documents',vectors)]:
        np.save(pool/f'{name}.npy', values)
    hashes = {name: file_hash(pool/name) for name in ['codes.npy','queries.npy','reference.npy','documents.npy']}
    (pool/'pool.json').write_text(json.dumps(dict(identity=identity, hashes=hashes, query_ids=query_ids), indent=2)+'\n')
    print(f'Prepared {documents} documents and {len(queries)} query vectors', flush=True)
    return pool


def prepare_router(pool, kind, clusters, seed):
    folder = pool/'routers'/f'{kind}-{clusters}-seed{seed}'
    if folder.exists():
        return folder
    documents = np.load(pool/'documents.npy', mmap_mode='r')
    folder.mkdir(parents=True)
    if kind=='sign':
        bits = int(math.log2(clusters))
        if 2**bits != clusters:
            raise ValueError('sign-prefix cluster counts must be powers of two')
        router = build_router(documents, method='sign', routing_bits=bits)
        centroids = router.signed_codes
        labels = router.prefixes.astype(np.int64)
    else:
        bits = 0
        router = build_router(documents, method='ivf', clusters=clusters,
                              seed=seed, threads=1, retain_float_index=False)
        centroids = router.quantizer.reconstruct_n(0, clusters)
        labels = np.arange(clusters, dtype=np.int64)
    assignments = router.assignments
    np.save(folder/'assignments.npy', assignments)
    np.save(folder/'centroids.npy', centroids)
    np.save(folder/'labels.npy', labels)
    counts = np.bincount(assignments, minlength=clusters).tolist()
    metadata = dict(kind=kind, clusters=clusters, routing_bits=bits, seed=seed,
                    cluster_sizes=counts, build_ms=router.build_ms,
                    routing_payload_bytes=centroids.nbytes+labels.nbytes,
                    assignments_sha256=file_hash(folder/'assignments.npy'),
                    centroids_sha256=file_hash(folder/'centroids.npy'))
    (folder/'router.json').write_text(json.dumps(metadata, indent=2)+'\n')
    print(f'Prepared {kind} router with {clusters} clusters', flush=True)
    return folder


def study_cases(config, pool, documents, layouts):
    cases = []
    for layout in layouts:
        counts = sorted(set(min(p, layout['clusters']) for p in config['sweep']['probes']+[layout['clusters']]))
        for probes in counts:
            common = dict(pool=str(pool), documents=documents, dimensions=config['data']['dimensions'],
                          query_rows=list(range(config['data']['queries'])),
                          router=layout['path'], router_kind=layout['kind'], clusters=layout['clusters'],
                          probes=probes, top_k=config['search']['top_k'],
                          repetitions=config['measurement']['repetitions'],
                          ram_budget_bytes=config['limits']['ram_budget_bytes'])
            cases.append(dict(common, method='scan'))
            branch_settings = set(itertools.product(config['sweep']['node_budgets'], config['sweep']['leaf_sizes']))
            # Always include one complete branch setting, even if finite budgets
            # miss a high-recall target. Its full cost remains in the comparison.
            branch_settings.add((0, max(config['sweep']['leaf_sizes'])))
            for budget, leaf in sorted(branch_settings):
                cases.append(dict(common, method='branch', node_budget=budget, leaf_size=leaf))
            for bits, target in itertools.product(config['sweep']['key_bits'], config['sweep']['candidate_targets']):
                cases.append(dict(common, method='keys', key_bits=bits, key_offset=layout['routing_bits'],
                                  candidate_target=target, key_limit=config['sweep']['key_limit']))
    order = np.random.default_rng(config['measurement']['schedule_seed']).permutation(len(cases))
    return [cases[int(i)] for i in order]


def run_real_study(config, output):
    output.mkdir(parents=True, exist_ok=False)
    (output/'configuration.json').write_text(json.dumps(config, indent=2)+'\n')
    cases = []
    for documents in config['data']['sizes']:
        pool = prepare_real_pool(config, documents)
        layouts = [dict(path=None, kind='none', clusters=1, routing_bits=0)]
        for kind, clusters in itertools.product(config['sweep']['routers'], config['sweep']['clusters']):
            with threadpool_limits(limits=1):
                folder = prepare_router(pool, kind, clusters, config['sweep']['router_seed'])
            metadata = json.loads((folder/'router.json').read_text())
            layouts.append(dict(path=str(folder), kind=kind, clusters=clusters,
                                routing_bits=metadata['routing_bits']))
        cases.extend(study_cases(config, pool, documents, layouts))
    execute_cases(config, output, cases, 'Coarse tuning on cached MS MARCO embeddings; these queries are not a fresh final test.')


def run_refinement(config, output):
    """Check routing densely, then expand local search only above its recall ceiling.

    A local candidate search cannot recover a true neighbor excluded by routing.
    This removes impossible settings using the exact scan on the same tuning queries.
    It does not limit B/C to the fastest routing choice made by A.
    """
    output.mkdir(parents=True, exist_ok=False)
    (output/'configuration.json').write_text(json.dumps(config, indent=2)+'\n')
    all_cases = []
    for documents in config['data']['sizes']:
        pool = prepare_real_pool(config, documents)
        layouts = [dict(path=None, kind='none', clusters=1, routing_bits=0)]
        for kind, clusters in itertools.product(config['sweep']['routers'], config['sweep']['clusters']):
            with threadpool_limits(limits=1):
                folder = prepare_router(pool, kind, clusters, config['sweep']['router_seed'])
            meta = json.loads((folder/'router.json').read_text())
            layouts.append(dict(path=str(folder), kind=kind, clusters=clusters,
                                routing_bits=meta['routing_bits']))
        all_cases.extend(study_cases(config, pool, documents, layouts))
    routing_cases = [case for case in all_cases if case['method']=='scan']
    routing_output = output/'routing'
    routing_output.mkdir()
    (routing_output/'configuration.json').write_text(json.dumps(config, indent=2)+'\n')
    execute_cases(config, routing_output, routing_cases, 'Dense routing screen on tuning queries, using exact local scan.')
    qualifying = set()
    minimum_target = min(config['search']['recall_targets'])
    for number, case in enumerate(routing_cases):
        folder = routing_output/'cases'/f'{number:04d}'
        process = json.loads((folder/'process.json').read_text())
        if process['status'] != 'complete':
            continue
        rows = [json.loads(line) for line in (folder/'run/queries.jsonl').read_text().splitlines()]
        recall = np.mean([row['recall'] for row in rows if row['repetition']==0])
        if recall >= minimum_target:
            qualifying.add((case['pool'], case['router'], case['probes']))
    selected = [case for case in all_cases if (case['pool'], case['router'], case['probes']) in qualifying]
    search_output = output/'search'
    search_output.mkdir()
    (search_output/'configuration.json').write_text(json.dumps(config, indent=2)+'\n')
    print(f'{len(qualifying)} routing choices meet the lowest target; running {len(selected)} local configurations', flush=True)
    execute_cases(config, search_output, selected, 'Refined local search on tuning queries; fresh final evaluation is still required.')
