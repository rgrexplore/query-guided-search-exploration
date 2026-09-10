import numpy as np
import bitplane_index

from experiments.components import pack_signs, reference_rows
from experiments.check_controlled import predicted_work, key_shell_work
from experiments.controlled import predict_one_path
from experiments.controlled_study import controlled_cases, evaluation_cases


def test_one_path_prediction_keeps_full_width_leaf_work():
    predicted=predict_one_path([1024,512,256,128],128,100,256)
    assert predicted==dict(nodes=4,bitplane_words=48,leaf_words=16,documents_scored=128,score_terms=8192)
    assert predict_one_path([1024,512,256,80],128,100,256) is None
    assert predict_one_path([1024,512,256,128],64,100,256) is None


def test_controlled_cases_share_layouts_and_leave_test_queries_unused():
    config = {
        'data': {'dimensions': 8, 'queries': 6, 'tuning_queries': 3},
        'search': {'top_k': 2},
        'sweep': {'leaf_sizes': [4, 8], 'key_bits': [2, 3], 'key_limit': 64},
        'measurement': {'repetitions': 1},
        'limits': {'ram_budget_bytes': 1000000},
    }
    layouts = [dict(path=None, kind='none', clusters=1, routing_bits=0, probes=[1]),
               dict(path='/direct', kind='direct', clusters=4, routing_bits=2, probes=[1, 4])]
    cases = controlled_cases(config, '/pool', 32, layouts)
    for layout in layouts:
        for probes in layout['probes']:
            chosen = [c for c in cases if c['router']==layout['path'] and c['probes']==probes]
            assert {c['method'] for c in chosen} == {'scan', 'branch', 'keys'}
    assert all(c['query_rows']==[0, 1, 2] for c in cases)
    assert all(c['candidate_target']==0 for c in cases if c['method']=='keys')
    assert all(c['key_offset']==2 for c in cases if c['method']=='keys' and c['router_kind']=='direct')


def test_controlled_evaluation_replays_frozen_settings_with_disjoint_queries():
    config = {'data': {'queries': 6, 'tuning_queries': 3},
              'measurement': {'blocks': 2, 'evaluation_repetitions': 3}}
    original = dict(method='keys', query_rows=[0, 1, 2], repetitions=1,
                    pool='/pool', key_bits=4, key_limit=128, candidate_target=0)
    choice = dict(setting_id='s0', case=original, evaluation_seeds=[0],
                  uses=[dict(selection='mean', target=.9)])
    cases = evaluation_cases(config, [choice])
    assert len(cases)==2
    assert {c['block'] for c in cases} == {0, 1}
    assert all(c['query_rows']==[3, 4, 5] and c['key_bits']==4 and c['key_limit']==128 for c in cases)
    assert original['query_rows']==[0, 1, 2] and original['repetitions']==1


def test_one_path_formula_matches_native_work_and_memory():
    # Every 8-bit document occurs once. Matching each chosen bit halves the set.
    rows = np.arange(256, dtype=np.uint64)
    signs = ((rows[:, None] >> np.arange(8, dtype=np.uint64)) & 1).astype(np.uint8)
    query = np.array([[1, -1, 1, .0001, -.0001, .0001, .0001, -.0001]], dtype=np.float32)
    index = bitplane_index.Index(pack_signs(signs), np.zeros(256, dtype=np.int64), 8)
    buckets = np.zeros((1, 1), dtype=np.int64)
    result = index.search(query, buckets, candidate_limit=4, node_budget=0, leaf_size=64)
    predicted = predict_one_path([256, 128, 64, 32], 64, 4, 8)
    stats = result['stats'][0]
    for field in ('nodes', 'bitplane_words', 'leaf_words', 'documents_scored'):
        assert stats[field] == predicted[field]
    # Two cuts: queued alternatives + parent + both new children = four masks.
    assert stats['peak_mask_bytes'] == 4 * 4 * 8
    np.testing.assert_array_equal(result['rows'], reference_rows(signs, query, 4))
    truncated = index.search(query, buckets, candidate_limit=4, node_budget=1, leaf_size=64)
    assert truncated['counts'][0] == 0


def test_work_prediction_does_not_apply_fixed_key_assumption_to_adaptive_weights():
    setup = dict(dimensions=8, strong_bits=3, weak_weight=.0001, support='fixed')
    metadata = {'identity': setup}
    case = dict(method='keys', router=None, key_bits=2, key_offset=0, top_k=4, dimensions=8,
                candidate_target=0, key_limit=0)
    prediction = predicted_work(case, metadata, [256,128,64,32])
    assert prediction == dict(keys_generated=1, key_attempts=1, documents_scored=64,
                              score_terms=128, recall=1.0)
    assert predicted_work(case, {'identity':dict(setup,support='adaptive')}, [256,128,64,32]) is None
    assert predicted_work(case, metadata, [256,128,64,3]) is None


def test_complete_strong_key_prediction_counts_empty_patterns_in_the_shell():
    # One exact match, one document in the first error layer, four in the next.
    # Both one-error keys must be tried even though only one contains a document.
    assert key_shell_work([1, 1, 4], 2, 2) == dict(
        keys_generated=3, key_attempts=3, documents_scored=2)
    assert key_shell_work([1, 1, 4], 3, 2) == dict(
        keys_generated=4, key_attempts=4, documents_scored=6)
