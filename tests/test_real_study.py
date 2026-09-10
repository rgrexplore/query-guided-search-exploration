"""Use a small fake vector cache to check the real-data preparation path."""
import json

import numpy as np

from experiments.components import pack_signs, reference_rows
from experiments.real_study import prepare_real_pool, prepare_router, study_cases


def test_cached_pool_preserves_codes_queries_and_independent_reference(tmp_path):
    source = tmp_path/'embeddings'
    derived = source/'derived'/'5'
    derived.mkdir(parents=True)
    rng = np.random.default_rng(17)
    documents = rng.normal(size=(32, 5)).astype(np.float32)
    queries = rng.normal(size=(4, 5)).astype(np.float32)
    np.save(source/'full-documents.npy', documents)
    np.save(derived/'codes.npy', pack_signs(documents >= 0))
    np.save(derived/'queries.npy', queries)
    (source/'manifest.json').write_text(json.dumps({'fixture': True}))
    (derived/'manifest.json').write_text(json.dumps({'query_ids': ['q0','q1','q2','q3']}))
    config = {'data': {'embedding_dir': str(source), 'cache_dir': str(tmp_path/'prepared'),
                       'dimensions': 5, 'queries': 3, 'query_start': 1},
              'search': {'top_k': 3}}
    pool = prepare_real_pool(config, 16)
    np.testing.assert_array_equal(np.load(pool/'codes.npy'), pack_signs(documents[:16] >= 0))
    np.testing.assert_array_equal(np.load(pool/'queries.npy'), queries[1:4])
    np.testing.assert_array_equal(np.load(pool/'reference.npy'),
                                  reference_rows(documents[:16]>=0, queries[1:4], 3))
    assert json.loads((pool/'pool.json').read_text())['query_ids'] == ['q1','q2','q3']
    for kind in ('sign', 'ivf'):
        router = prepare_router(pool, kind, 4, 7)
        assignments = np.load(router/'assignments.npy')
        centroids = np.load(router/'centroids.npy')
        labels = np.load(router/'labels.npy')
        vectors = np.load(pool/'documents.npy')
        if kind == 'sign':
            np.testing.assert_array_equal(assignments, (vectors[:,0]>=0) + 2*(vectors[:,1]>=0))
        else:
            np.testing.assert_array_equal(assignments, labels[(vectors @ centroids.T).argmax(axis=1)])


def test_every_router_choice_is_available_to_every_method():
    config = {'data': {'dimensions': 256, 'queries': 2}, 'search': {'top_k': 1},
              'measurement': {'repetitions': 1, 'schedule_seed': 9},
              'limits': {'ram_budget_bytes': 32_000_000_000},
              'sweep': {'probes': [1, 4], 'node_budgets': [128], 'leaf_sizes': [32],
                        'key_bits': [8], 'candidate_targets': [0], 'key_limit': 8192}}
    layouts = [{'path': None, 'kind': 'none', 'clusters': 1, 'routing_bits': 0},
               {'path': '/router', 'kind': 'sign', 'clusters': 4, 'routing_bits': 2}]
    cases = study_cases(config, '/pool', 100, layouts)
    for method in ('scan','branch','keys'):
        routes = {(c['router_kind'], c['clusters'], c['probes']) for c in cases if c['method']==method}
        assert routes == {('none',1,1), ('sign',4,1), ('sign',4,4)}
    assert all(c['key_offset']==2 for c in cases if c['method']=='keys' and c['router_kind']=='sign')
