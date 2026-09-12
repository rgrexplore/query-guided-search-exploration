import itertools
import math

import bitplane_index
import numpy as np
import pytest

from experiments.components import pack_signs, reference_rows
from experiments.controlled import predict_one_path
from experiments.optima_model import (
    branch_depth_optimum,
    branch_depth_time,
    expected_occupied_keys,
    fixed_coordinate_stats,
    ideal_group_stats,
)


def test_ideal_group_recall_matches_every_tiny_corpus():
    # Two independent one-bit documents give four equally likely corpora.
    # The ideal key is 1. A missing ideal group contributes zero recall.
    corpora = list(itertools.product([0, 1], repeat=2))
    counts = [sum(corpus) for corpus in corpora]
    for top_k in [1, 2]:
        observed_recall = np.mean([min(count, top_k) / top_k for count in counts])
        stats = ideal_group_stats(2, 1, top_k)
        assert stats['mean_count'] == np.mean(counts)
        assert stats['std_count'] == pytest.approx(np.std(counts))
        assert stats['probability_fewer_than_k'] == np.mean([n < top_k for n in counts])
        assert stats['expected_ideal_only_recall'] == observed_recall
    # Replacing this calculation with min(E[count], K) would incorrectly give 1.
    assert ideal_group_stats(2, 1, 1)['expected_ideal_only_recall'] == 0.75


def test_occupied_key_expectation_matches_every_tiny_corpus():
    corpora = itertools.product(range(4), repeat=3)
    observed = np.mean([len(set(corpus)) for corpus in corpora])
    assert expected_occupied_keys(3, 2) == pytest.approx(observed)
    assert expected_occupied_keys(3, 0) == 1


def test_coordinate_overlap_matches_exhaustive_support_choices():
    # The first two positions form the fixed key; every three-position query
    # support among eight coordinates is equally likely.
    captured = [len(set(support) & {0, 1}) for support in itertools.combinations(range(8), 3)]
    stats = fixed_coordinate_stats(8, 3, 2)
    assert stats['probability_zero_overlap'] == pytest.approx(np.mean([s == 0 for s in captured]))
    assert stats['expected_retained_fraction'] == pytest.approx(np.mean([2.0**-s for s in captured]))
    assert stats['expected_compatible_keys'] == pytest.approx(np.mean([2.0**(2-s) for s in captured]))
    assert fixed_coordinate_stats(8, 3, 0)['expected_retained_fraction'] == 1


def test_leaf_suggestion_uses_occupancy_spread_without_claiming_a_guarantee():
    stats = ideal_group_stats(1_000_000, 12, 100)
    assert stats['mean_count'] == 244.140625
    assert stats['suggested_leaf_size'] == math.ceil(stats['mean_count'] + 3 * stats['std_count'])
    assert stats['probability_fewer_than_k'] < 1e-25
    # Adding two strong coordinates leaves too few ideal matches for top 100.
    assert ideal_group_stats(1_000_000, 14, 100)['expected_ideal_only_recall'] == pytest.approx(0.61035152)


@pytest.mark.parametrize('fixed_prefix_bits', [0, 2, 6])
@pytest.mark.parametrize('split_word_ms', [0.001, 0.1, 10.0])
def test_derivative_candidates_include_the_best_integer_depth(fixed_prefix_bits, split_word_ms):
    # Independent enumeration checks both the smooth optimum and the zero-cut
    # option, which matters when a fixed prefix must be traversed first.
    max_depth = 10
    prediction = branch_depth_optimum(6400, split_word_ms, 0.03, max_depth,
                                      fixed_prefix_bits=fixed_prefix_bits)
    costs = {
        depth: branch_depth_time(6400, depth, split_word_ms, 0.002, 0.03,
                                 fixed_prefix_bits=fixed_prefix_bits)
        for depth in range(max_depth + 1)
    }
    predicted_best = min(costs[depth] for depth in prediction['candidate_depths'])
    assert predicted_best == min(costs.values())


def test_prefix_fixed_coordinates_still_cost_native_splits():
    # Every 8-bit code occurs once. The selected cluster fixes the first two
    # signs, but today's traversal still visits them before the third sign.
    values = np.arange(256, dtype=np.uint64)
    signs = ((values[:, None] >> np.arange(8, dtype=np.uint64)) & 1).astype(np.uint8)
    assignments = np.ascontiguousarray(values.astype(np.int64) & 3)
    query = np.array([[1, -1, 1, .0001, -.0001, .0001, .0001, -.0001]], dtype=np.float32)
    index = bitplane_index.Index(pack_signs(signs), assignments, 8)
    selected = np.array([[1]], dtype=np.int64)
    result = index.search(query, selected, candidate_limit=4, leaf_size=32, node_budget=0)
    predicted = predict_one_path([64, 64, 64, 32], 32, 4, 8)
    for name in ['nodes', 'bitplane_words', 'leaf_words', 'documents_scored']:
        assert result['stats'][0][name] == predicted[name]
    assert predicted['bitplane_words'] == 3
    np.testing.assert_array_equal(result['rows'], reference_rows(signs, query, 4))

    # Unit prices turn the count formula into a directly checkable total:
    # 3 full-word splits, 1 leaf-word read, and 32 document scores.
    assert branch_depth_time(64, 3, 1, 1, 1, fixed_prefix_bits=2) == 36
    assert branch_depth_time(64, 3, 1, 1, 1) == 12
