"""The paper's prefix policy, checked against direct bit membership and scores."""

import itertools

import numpy as np
import pytest

import bitplane_index as native
from experiments.prefix_reference_v2 import prefix_search


def pack(signs):
    codes = np.zeros((len(signs), (signs.shape[1] + 63) // 64), dtype=np.uint64)
    for bit in range(signs.shape[1]):
        codes[:, bit // 64] |= signs[:, bit].astype(np.uint64) << np.uint64(bit % 64)
    return codes


def binary_rows(*rows):
    return np.array([[int(bit) for bit in row] for row in rows], dtype=np.uint8)


def search(backend, signs, labels, queries, buckets, **options):
    if backend == "reference":
        return prefix_search(signs, labels, queries, buckets, **options)
    index = native.PrefixIndexV2(pack(signs), labels, signs.shape[1],
                                 max_prefix_bits=min(24, signs.shape[1]))
    return index.search(queries, buckets, **options)


@pytest.mark.parametrize("backend", ["reference", "native"])
def test_paper_example_scores_only_the_new_row_at_each_depth(backend):
    signs = binary_rows("10100", "11001", "11000")
    labels = np.zeros(3, dtype=np.int64)
    queries = np.array([[.7, .5, -.4, -.6, .3]], dtype=np.float32)
    buckets = np.array([[0]], dtype=np.int64)
    first = search(backend, signs, labels, queries, buckets,
                   candidate_limit=3, start_depth=5, candidate_target=1)
    second = search(backend, signs, labels, queries, buckets,
                    candidate_limit=3, start_depth=5, candidate_target=2)
    assert first["rows"][0].tolist() == [1, -1, -1]
    assert second["rows"][0].tolist() == [1, 2, -1]
    np.testing.assert_allclose(second["scores"][0, :2], [2.5, 1.9], atol=1e-7)
    assert first["stats"][0]["documents_scored"] == 1
    assert second["stats"][0]["documents_scored"] == 2
    assert second["stats"][0]["prefix_levels"] == 2
    assert second["stats"][0]["prefix_lookups"] == 4
    assert second["stats"][0]["final_depth"] == 4
    if backend == "reference":
        assert second["trace"][0] == [{"depth": 5, "new_rows": [1]},
                                      {"depth": 4, "new_rows": [2]}]


@pytest.mark.parametrize("backend", ["reference", "native"])
def test_empty_prefixes_show_approximate_miss_and_unlimited_control(backend):
    signs = binary_rows("1100", "1011", "0000")
    labels = np.zeros(3, dtype=np.int64)
    queries = np.array([[.6, .5, .4, .3]], dtype=np.float32)
    buckets = np.array([[0]], dtype=np.int64)
    limited = search(backend, signs, labels, queries, buckets,
                     candidate_limit=1, start_depth=4, candidate_target=1)
    complete = search(backend, signs, labels, queries, buckets,
                      candidate_limit=1, start_depth=4, candidate_target=0)
    assert limited["rows"][0, 0] == 0
    assert limited["stats"][0]["final_depth"] == 2
    assert limited["stats"][0]["prefix_levels"] == 3
    assert limited["stats"][0]["prefix_lookups"] == 6
    assert limited["stats"][0]["stop_reason"] == "candidate_target"
    assert complete["rows"][0, 0] == 1
    assert complete["scores"][0, 0] > limited["scores"][0, 0]
    assert complete["stats"][0]["documents_scored"] == 3
    assert complete["stats"][0]["final_depth"] == 0
    assert complete["stats"][0]["prefix_levels"] == 5
    assert complete["stats"][0]["prefix_lookups"] == 8
    assert complete["stats"][0]["stop_reason"] == "exhausted"


@pytest.mark.parametrize("backend", ["reference", "native"])
def test_candidate_target_finishes_the_depth_in_every_selected_cluster(backend):
    signs = binary_rows("110", "111", "000")
    labels = np.array([8, 2, 8], dtype=np.int64)
    queries = np.array([[.1, .2, 2]], dtype=np.float32)
    for selected in ([8, 2, -1, 99], [2, 8, -1, 99]):
        result = search(backend, signs, labels, queries,
                         np.array([selected], dtype=np.int64),
                         candidate_limit=2, start_depth=2, candidate_target=1)
        assert result["rows"][0].tolist() == [1, 0]
        assert result["stats"][0]["documents_scored"] == 2
        assert result["stats"][0]["prefix_levels"] == 1
        assert result["stats"][0]["prefix_lookups"] == 4
        assert result["stats"][0]["final_depth"] == 2


@pytest.mark.parametrize("backend", ["reference", "native"])
def test_first_embedding_bit_defines_prefix_even_when_its_weight_is_small(backend):
    signs = binary_rows("01", "10")
    result = search(backend, signs, np.zeros(2, dtype=np.int64),
                     np.array([[.1, 10]], dtype=np.float32),
                     np.array([[0]], dtype=np.int64),
                     candidate_limit=1, start_depth=1, candidate_target=1)
    assert result["rows"][0, 0] == 1
    assert result["stats"][0]["documents_scored"] == 1


@pytest.mark.parametrize("target", [1, 3, 9, 100, 0])
@pytest.mark.parametrize("start_depth", [0, 1, 5])
def test_native_membership_and_stopping_match_direct_reference(target, start_depth):
    signs = np.array(list(itertools.product([0, 1], repeat=5)) + [[1] * 5],
                     dtype=np.uint8)
    labels = np.arange(len(signs), dtype=np.int64) % 3
    queries = np.array([[.1, -.8, .3, -.2, 2], [0] * 5, [-1] * 5], dtype=np.float32)
    buckets = np.tile(np.array([2, 0, -1, 99], dtype=np.int64), (len(queries), 1))
    options = dict(candidate_limit=len(signs), start_depth=start_depth, candidate_target=target)
    expected = prefix_search(signs, labels, queries, buckets, **options)
    result = search("native", signs, labels, queries, buckets, **options)
    np.testing.assert_array_equal(result["rows"], expected["rows"])
    np.testing.assert_array_equal(result["counts"], expected["counts"])
    np.testing.assert_allclose(result["scores"], expected["scores"], atol=1e-12)
    for qi, stats in enumerate(result["stats"]):
        for key, value in expected["stats"][qi].items():
            assert stats[key] == value
        returned = result["rows"][qi, :result["counts"][qi]]
        assert len(set(returned)) == len(returned) == stats["documents_scored"]


@pytest.mark.parametrize("dimensions,max_prefix_bits", [(5, 5), (67, 24), (256, 32)])
@pytest.mark.parametrize("candidate_limit", [1, 10, 100])
def test_unlimited_matches_scan_including_ties_and_full_packed_codes(
        dimensions, max_prefix_bits, candidate_limit):
    rng = np.random.default_rng(421)
    signs = rng.integers(0, 2, size=(73, dimensions), dtype=np.uint8)
    signs[1] = signs[0]
    # Exercise both extreme stored keys, including the exclusive 2**32 high bound.
    signs[2, :max_prefix_bits] = 1
    signs[3, :max_prefix_bits] = 0
    codes = pack(signs)
    labels = np.arange(len(signs), dtype=np.int64) % 3
    queries = rng.normal(size=(4, dimensions)).astype(np.float32)
    queries[1] = 0
    queries[2] = 1
    queries[3] = -1
    buckets = np.tile(np.array([2, 0, 1], dtype=np.int64), (4, 1))
    index = native.PrefixIndexV2(codes, labels, dimensions,
                                 max_prefix_bits=max_prefix_bits)
    result = index.search(queries, buckets, candidate_limit=candidate_limit,
                           start_depth=max_prefix_bits, candidate_target=0)
    scan = native.Index(codes, labels, dimensions, build_bitplanes=False).scan(
        queries, buckets, candidate_limit=candidate_limit)
    np.testing.assert_array_equal(result["rows"], scan["rows"])
    np.testing.assert_array_equal(result["scores"], scan["scores"])
    direct = prefix_search(signs, labels, queries, buckets, candidate_limit=candidate_limit,
                            start_depth=max_prefix_bits, candidate_target=0)
    np.testing.assert_array_equal(result["rows"], direct["rows"])
    np.testing.assert_allclose(result["scores"], direct["scores"], atol=1e-12)
    assert all(stat["documents_scored"] == len(signs) for stat in result["stats"])
    assert all(stat["final_depth"] == 0 for stat in result["stats"])


def test_storage_counts_prefix_keys_full_codes_and_original_ids():
    signs = binary_rows("10100", "11001", "11000")
    index = native.PrefixIndexV2(pack(signs), np.zeros(3, dtype=np.int64), 5,
                                 max_prefix_bits=5)
    info = index.info()
    assert info["documents"] == 3
    assert info["dimensions"] == 5
    assert info["buckets"] == 1
    assert info["codes_bytes"] == 24
    assert info["row_ids_bytes"] == 24
    assert info["prefix_keys_bytes"] == 12
    assert info["logical_bytes"] == 60
    assert info["array_capacity_bytes"] >= info["logical_bytes"]
    assert info["bitplanes_bytes"] == info["key_directory_payload_bytes"] == 0


@pytest.mark.parametrize("max_prefix_bits", [0, 6, 25])
def test_rejects_unsupported_stored_prefix_width(max_prefix_bits):
    with pytest.raises(ValueError, match="max_prefix_bits"):
        native.PrefixIndexV2(pack(binary_rows("10100")), np.array([0], dtype=np.int64),
                             5, max_prefix_bits=max_prefix_bits)


@pytest.mark.parametrize("options,match", [
    ({"start_depth": -1}, "start_depth"),
    ({"start_depth": 6}, "start_depth"),
    ({"candidate_limit": 0}, "candidate_limit"),
    ({"candidate_target": -1}, "candidate_target"),
])
def test_rejects_invalid_search_options(options, match):
    index = native.PrefixIndexV2(pack(binary_rows("10100")),
                                 np.array([0], dtype=np.int64), 5, max_prefix_bits=5)
    arguments = dict(candidate_limit=1, start_depth=5, candidate_target=0)
    arguments.update(options)
    with pytest.raises(ValueError, match=match):
        index.search(np.ones((1, 5), dtype=np.float32),
                     np.array([[0]], dtype=np.int64), **arguments)


def test_reuses_numpy_validation_and_cluster_uniqueness_checks():
    codes = pack(binary_rows("10100"))
    labels = np.array([0], dtype=np.int64)
    with pytest.raises(TypeError, match="codes has the wrong dtype"):
        native.PrefixIndexV2(codes.astype(np.int64), labels, 5, max_prefix_bits=5)
    index = native.PrefixIndexV2(codes, labels, 5, max_prefix_bits=5)
    with pytest.raises(ValueError, match="same bucket twice"):
        index.search(np.ones((1, 5), dtype=np.float32),
                     np.array([[0, 0]], dtype=np.int64), start_depth=5)
    with pytest.raises(TypeError, match="queries has the wrong dtype"):
        index.search(np.ones((1, 5), dtype=np.float64),
                     np.array([[0]], dtype=np.int64), start_depth=5)
