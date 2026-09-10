"""Independent examples for key lookup and method-specific storage."""
import itertools

import numpy as np
import pytest

import bitplane_index as native


def pack(signs):
    signs = np.asarray(signs, dtype=np.uint64)
    codes = np.zeros((len(signs), (signs.shape[1] + 63)//64), dtype=np.uint64)
    for bit in range(signs.shape[1]):
        codes[:, bit//64] |= signs[:, bit] << np.uint64(bit % 64)
    return codes


def expected(signs, assignments, query, selected, limit):
    rows = np.flatnonzero(np.isin(assignments, selected))
    scores = (2*signs.astype(np.float64)-1) @ query.astype(np.float64)
    order = rows[np.lexsort((rows, -scores[rows]))][:limit]
    return order, scores[order]


def test_scan_only_has_no_planes_and_preserves_scores():
    signs = np.array([[1, 0, 1], [0, 1, 0], [1, 1, 1]], dtype=np.uint8)
    labels = np.zeros(3, dtype=np.int64)
    query = np.array([[.7, -.4, .2]], dtype=np.float32)
    buckets = np.array([[0]], dtype=np.int64)
    old = native.Index(pack(signs), labels, 3)
    scan = native.Index(pack(signs), labels, 3, build_bitplanes=False)
    assert scan.info()['bitplanes_bytes'] == 0
    assert old.info()['bitplanes_bytes'] == 24
    np.testing.assert_array_equal(scan.scan(query, buckets)['rows'], old.scan(query, buckets)['rows'])
    with pytest.raises(RuntimeError, match='built without bitplanes'):
        scan.search(query, buckets)


@pytest.mark.parametrize('key_bits,offset', [(1, 0), (3, 0), (5, 0), (2, 2)])
def test_complete_key_search_matches_independent_scores(key_bits, offset):
    signs = np.array(list(itertools.product([0, 1], repeat=5)) + [[1]*5], dtype=np.uint8)
    labels = np.arange(len(signs), dtype=np.int64) % 3
    index = native.KeyIndex(pack(signs), labels, 5, key_bits, key_offset=offset)
    # Includes ties, zero weights, and a case where two cheap mistakes beat one costly mistake.
    queries = np.array([[.7, -.4, .2, .1, -.3], [4, 3, 2, 1, 1], [0]*5], dtype=np.float32)
    buckets = np.tile(np.array([2, 0, -1], dtype=np.int64), (3, 1))
    result = index.search(queries, buckets, candidate_limit=7, candidate_target=0, key_limit=0)
    for i, query in enumerate(queries):
        rows, scores = expected(signs, labels, query, [2, 0], 7)
        np.testing.assert_array_equal(result['rows'][i, :7], rows)
        np.testing.assert_allclose(result['scores'][i, :7], scores, atol=1e-12)
        assert result['stats'][i]['stop_reason'] in ('bound', 'exhausted')
    # With all-zero query weights, safe tie handling must not prune any keys.
    assert result['stats'][2]['key_attempts'] == 2 * 2**key_bits
    assert index.info()['bitplanes_bytes'] == 0


def test_quota_reads_whole_key_postings_across_selected_clusters():
    signs = np.array([[1, 1, 0], [1, 1, 1], [1, 1, 1], [0, 0, 0]], dtype=np.uint8)
    labels = np.array([0, 1, 0, 1], dtype=np.int64)
    index = native.KeyIndex(pack(signs), labels, 3, 2)
    result = index.search(np.array([[1, 1, 1]], dtype=np.float32),
                          np.array([[0, 1]], dtype=np.int64),
                          candidate_limit=2, candidate_target=1, key_limit=100)
    assert result['rows'][0].tolist() == [1, 2]
    stats = result['stats'][0]
    assert stats['documents_scored'] == 3  # A target is not a hard posting count cap.
    assert stats['key_attempts'] == 2
    assert stats['stop_reason'] == 'candidate_target'


def test_empty_lookup_attempts_consume_budget():
    signs = np.array([[0, 0, 0]], dtype=np.uint8)
    index = native.KeyIndex(pack(signs), np.array([0], dtype=np.int64), 3, 3)
    query = np.array([[3, 2, 1]], dtype=np.float32)
    buckets = np.array([[0]], dtype=np.int64)
    limited = index.search(query, buckets, candidate_limit=1, candidate_target=0, key_limit=2)
    assert limited['counts'][0] == 0
    assert limited['rows'][0, 0] == -1
    assert limited['stats'][0]['key_attempts'] == 2
    assert limited['stats'][0]['stop_reason'] == 'key_limit'
    complete = index.search(query, buckets, candidate_limit=1, candidate_target=0, key_limit=0)
    assert complete['rows'][0, 0] == 0
    assert complete['stats'][0]['key_attempts'] == 8


def test_leaf_word_count_is_separate_from_split_count():
    signs = np.ones((65, 5), dtype=np.uint8)
    index = native.Index(pack(signs), np.zeros(65, dtype=np.int64), 5)
    result = index.search(np.ones((1, 5), dtype=np.float32), np.array([[0]], dtype=np.int64),
                          candidate_limit=5, node_budget=0, leaf_size=100)
    stats = result['stats'][0]
    assert stats['bitplane_words'] == 0
    assert stats['leaf_words'] == 2
    assert stats['peak_mask_bytes'] == 16


def test_identical_codes_have_five_splits_and_one_full_width_leaf():
    signs = np.ones((65, 5), dtype=np.uint8)
    index = native.Index(pack(signs), np.zeros(65, dtype=np.int64), 5)
    result = index.search(np.ones((1, 5), dtype=np.float32), np.array([[0]], dtype=np.int64),
                          candidate_limit=5, node_budget=0, leaf_size=1)
    stats = result['stats'][0]
    assert stats['nodes'] == 6
    assert stats['bitplane_words'] == 10  # Five splits, two physical words each.
    assert stats['leaf_words'] == 2
    assert stats['peak_mask_bytes'] == 48  # Parent plus both allocated children.
