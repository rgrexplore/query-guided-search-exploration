"""An optional equal-penalty order changes limited work, while preserving exact results."""

import itertools

import bitplane_index
import numpy as np
import pytest

from experiments.components import pack_signs
from experiments.prefix_batch_worker_v2 import search_index


def test_deeper_ties_reach_a_leaf_with_the_same_partial_node_budget():
    signs = np.ones((4, 3), dtype=np.uint8)
    index = bitplane_index.Index(pack_signs(signs), np.array([0, 0, 1, 1], dtype=np.int64), 3)
    query = np.ones((1, 3), dtype=np.float32)
    buckets = np.array([[0, 1]], dtype=np.int64)
    options = dict(candidate_limit=2, node_budget=4, leaf_size=1, trace=True)
    default = index.search(query, buckets, **options)
    fifo = index.search(query, buckets, prefer_deeper_ties=False, **options)
    deeper = index.search(query, buckets, prefer_deeper_ties=True, **options)
    np.testing.assert_array_equal(default["rows"], fifo["rows"])
    assert default["stats"][0]["trace"] == fifo["stats"][0]["trace"]
    assert fifo["counts"][0] == 0 and fifo["stats"][0]["documents_scored"] == 0
    assert [step["depth"] for step in fifo["stats"][0]["trace"]] == [0, 0, 1, 1]
    assert deeper["rows"][0].tolist() == [0, 1]
    assert [step["depth"] for step in deeper["stats"][0]["trace"]] == [0, 1, 2, 3]
    assert fifo["stats"][0]["nodes"] == deeper["stats"][0]["nodes"] == 4
    variant = dict(top_k=2, node_budget=4, leaf_size=1, prefer_deeper_ties=True)
    measured = search_index(index, "branch", query, buckets, variant)
    np.testing.assert_array_equal(measured["rows"], deeper["rows"])


@pytest.mark.parametrize("exploration", [0.0, 1.0])
def test_both_tie_orders_preserve_exact_scores_and_ids_with_random_removals(exploration):
    signs = np.array(list(itertools.product([0, 1], repeat=5)) + [[1] * 5, [0] * 5], dtype=np.uint8)
    labels = np.arange(len(signs), dtype=np.int64) % 4
    index = bitplane_index.Index(pack_signs(signs), labels, 5)
    queries = np.array([[0] * 5, [3, 2, 1, 0, 0], [.7, -.1, .6, -2, .3]], dtype=np.float32)
    buckets = np.tile(np.arange(4, dtype=np.int64), (len(queries), 1))
    scan = index.scan(queries, buckets, candidate_limit=8)
    for prefer_deeper in (False, True):
        result = index.search(queries, buckets, candidate_limit=8, node_budget=0, leaf_size=1,
                              explore_probability=exploration, seed=42,
                              prefer_deeper_ties=prefer_deeper)
        np.testing.assert_array_equal(result["rows"], scan["rows"])
        np.testing.assert_array_equal(result["scores"], scan["scores"])
        assert result["stats"][0]["documents_scored"] == len(signs)
        if exploration:
            assert result["stats"][0]["random_nodes"] > 0
