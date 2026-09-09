import numpy as np
import pytest

from bitplane_index import Index


def pack(bits):
    bits = np.asarray(bits, dtype=np.uint64)
    codes = np.zeros((len(bits), (bits.shape[1] + 63) // 64), dtype=np.uint64)
    for dimension in range(bits.shape[1]):
        codes[:, dimension // 64] |= bits[:, dimension] << np.uint64(dimension % 64)
    return codes


def brute_force(bits, assignments, queries, buckets, limit):
    # Score the selected documents directly, without using native search helpers.
    rows = np.full((len(queries), limit), -1, dtype=np.int64)
    scores = np.full((len(queries), limit), -np.inf)
    signs = np.asarray(bits, dtype=np.float64) * 2 - 1
    for query_id, query in enumerate(queries):
        candidates = np.flatnonzero(np.isin(assignments, buckets[query_id][buckets[query_id] >= 0]))
        ranked = sorted(
            ((float(np.dot(query.astype(np.float64), signs[row])), int(row)) for row in candidates),
            key=lambda item: (-item[0], item[1]),
        )[:limit]
        for position, (score, row) in enumerate(ranked):
            rows[query_id, position] = row
            scores[query_id, position] = score
    return rows, scores


def assert_result(result, expected):
    rows, scores = expected
    np.testing.assert_array_equal(result["rows"], rows)
    np.testing.assert_allclose(result["scores"], scores, atol=1e-10, rtol=1e-12)
    np.testing.assert_array_equal(result["counts"], (rows >= 0).sum(axis=1))
    assert result["rows"].dtype == np.int64
    assert result["scores"].dtype == np.float64
    for stats in result["stats"]:
        assert stats["elapsed_ms"] >= 0
        assert stats["stop_reason"] in {"exhausted", "bound", "budget"}


def test_five_bit_counterexample():
    bits = np.array(
        [
            [0, 1, 0, 0, 1],
            [0, 0, 1, 0, 0],
            [1, 0, 1, 0, 0],
        ]
    )
    assignments = np.zeros(3, dtype=np.int64)
    queries = np.array([[0.7, 0.5, -0.4, -0.6, 0.3]], dtype=np.float32)
    buckets = np.array([[0]], dtype=np.int64)
    index = Index(pack(bits), assignments, 5)

    expected = brute_force(bits, assignments, queries, buckets, 3)
    np.testing.assert_allclose(expected[1][0], [1.1, 0.1, -1.3], atol=1e-7)
    assert_result(index.scan(queries, buckets, candidate_limit=3), expected)
    result = index.search(queries, buckets, candidate_limit=3, node_budget=0, leaf_size=1)
    assert_result(result, expected)


@pytest.mark.parametrize("dimensions", [1, 5, 63, 64, 65, 129])
@pytest.mark.parametrize("probability", [0.0, 0.5, 1.0])
def test_unlimited_search_matches_independent_scores(dimensions, probability):
    rng = np.random.default_rng(52)
    bits = rng.integers(0, 2, size=(37, dimensions))
    assignments = np.arange(37, dtype=np.int64) % 4
    queries = rng.normal(size=(5, dimensions)).astype(np.float32)
    queries[0] = 0
    buckets = np.tile(np.array([3, 1, 0, -1], dtype=np.int64), (5, 1))
    index = Index(pack(bits), assignments, dimensions)
    expected = brute_force(bits, assignments, queries, buckets, 9)
    assert_result(index.scan(queries, buckets, candidate_limit=9), expected)
    result = index.search(
        queries,
        buckets,
        candidate_limit=9,
        node_budget=0,
        leaf_size=2,
        explore_probability=probability,
        seed=19,
    )
    assert_result(result, expected)
    assert all(row["stop_reason"] != "budget" for row in result["stats"])


def test_duplicate_codes_at_terminal_depth_and_ties():
    bits = np.array([[1, 0, 1]] * 12)
    assignments = np.array([3, 8] * 6, dtype=np.int64)
    queries = np.array(
        [
            [0, 0, 0],
            [2, -1, 3],
        ],
        dtype=np.float32,
    )
    buckets = np.array(
        [
            [8, 3],
            [8, 3],
        ],
        dtype=np.int64,
    )
    index = Index(pack(bits), assignments, 3)
    result = index.search(queries, buckets, candidate_limit=5, node_budget=0, leaf_size=1)
    assert_result(result, brute_force(bits, assignments, queries, buckets, 5))
    np.testing.assert_array_equal(result["rows"], [[0, 1, 2, 3, 4]] * 2)


def test_missing_buckets_padding_and_empty_input():
    codes = pack([[1, 0], [0, 1]])
    index = Index(codes, np.array([7, 7], dtype=np.int64), 2)
    queries = np.array(
        [
            [1, 0],
            [0, 1],
        ],
        dtype=np.float32,
    )
    buckets = np.array(
        [
            [999, -1],
            [-1, -1],
        ],
        dtype=np.int64,
    )
    expected = (np.full((2, 4), -1, dtype=np.int64), np.full((2, 4), -np.inf))
    assert_result(index.scan(queries, buckets, candidate_limit=4), expected)
    assert_result(index.search(queries, buckets, candidate_limit=4), expected)
    empty = Index(np.empty((0, 1), dtype=np.uint64), np.empty(0, dtype=np.int64), 2)
    assert_result(empty.search(queries, buckets, candidate_limit=4), expected)
    result = index.search(np.empty((0, 2), dtype=np.float32), np.empty((0, 1), dtype=np.int64))
    assert result["rows"].shape == (0, 100)
    assert result["stats"] == []


def test_index_owns_its_arrays():
    codes = pack([[1, 0], [0, 1]])
    assignments = np.array([0, 1], dtype=np.int64)
    index = Index(codes, assignments, 2)
    codes[:] = 0
    assignments[:] = 99
    result = index.scan(np.array([[1, -1]], dtype=np.float32), np.array([[0]], dtype=np.int64))
    assert result["rows"][0, 0] == 0
    assert result["scores"][0, 0] == 2
    assert result["counts"].tolist() == [1]
    assert index.info()["documents"] == 2
    assert index.info()["dimensions"] == 2


def test_budget_is_global_and_reproducible():
    # Enumerate every eight-bit code, then spread them across three buckets.
    bits = np.array(
        [[int(value >> dimension & 1) for dimension in range(8)] for value in range(256)]
    )
    assignments = np.arange(256, dtype=np.int64) % 3
    index = Index(pack(bits), assignments, 8)
    queries = np.array([[0.7, 0.5, -0.4, -0.6, 0.3, -0.8, 0.2, 0.9]], dtype=np.float32)
    buckets = np.array([[0, 1, 2]], dtype=np.int64)
    settings = dict(
        candidate_limit=12,
        node_budget=19,
        leaf_size=1,
        explore_probability=0.6,
        seed=123,
    )
    first = index.search(queries, buckets, **settings)
    second = index.search(queries, buckets, **settings)
    np.testing.assert_array_equal(first["rows"], second["rows"])
    np.testing.assert_array_equal(first["scores"], second["scores"])
    assert first["stats"][0]["nodes"] == 19
    assert first["stats"][0]["random_nodes"] > 0
    assert first["stats"][0]["stop_reason"] == "budget"
    for field in (
        "nodes",
        "random_nodes",
        "bitplane_words",
        "documents_scored",
        "stop_reason",
    ):
        assert first["stats"][0][field] == second["stats"][0][field]
    one = index.search(queries, buckets, candidate_limit=12, node_budget=1, leaf_size=1)
    assert one["counts"][0] == 0
    assert one["stats"][0]["nodes"] == 1


def test_batch_seed_matches_separate_queries():
    rng = np.random.default_rng(21)
    bits = rng.integers(0, 2, size=(150, 12))
    index = Index(pack(bits), np.arange(150, dtype=np.int64) % 3, 12)
    queries = rng.normal(size=(4, 12)).astype(np.float32)
    buckets = np.tile(np.array([0, 1, 2], dtype=np.int64), (4, 1))
    settings = dict(candidate_limit=5, node_budget=17, leaf_size=4, explore_probability=0.6)
    batch = index.search(queries, buckets, seed=32, **settings)
    for row in range(len(queries)):
        single = index.search(
            queries[row : row + 1], buckets[row : row + 1], seed=32 + row, **settings
        )
        np.testing.assert_array_equal(batch["rows"][row], single["rows"][0])
        np.testing.assert_array_equal(batch["scores"][row], single["scores"][0])
        for field in (
            "nodes",
            "random_nodes",
            "bitplane_words",
            "documents_scored",
            "stop_reason",
        ):
            assert batch["stats"][row][field] == single["stats"][0][field]


@pytest.mark.parametrize("probability", [0.0, 0.5, 1.0])
@pytest.mark.parametrize("budget", [1, 17, 0])
def test_trace_records_work_without_changing_search(probability, budget):
    rng = np.random.default_rng(71)
    bits = rng.integers(0, 2, size=(75, 8))
    index = Index(pack(bits), np.arange(75, dtype=np.int64) % 2, 8)
    queries = rng.normal(size=(2, 8)).astype(np.float32)
    buckets = np.array(
        [
            [0, 1],
            [1, 0],
        ],
        dtype=np.int64,
    )
    settings = dict(
        candidate_limit=5,
        node_budget=budget,
        leaf_size=3,
        explore_probability=probability,
        seed=42,
    )
    plain = index.search(queries, buckets, **settings)
    traced = index.search(queries, buckets, trace=True, **settings)
    np.testing.assert_array_equal(plain["rows"], traced["rows"])
    np.testing.assert_array_equal(plain["scores"], traced["scores"])
    for query_row, (before, after) in enumerate(zip(plain["stats"], traced["stats"])):
        assert "trace" not in before
        for field in (
            "nodes",
            "random_nodes",
            "bitplane_words",
            "documents_scored",
            "stop_reason",
        ):
            assert before[field] == after[field]
        trace = after["trace"]
        assert len(trace) == after["nodes"]
        assert sum(step["random"] for step in trace) == after["random_nodes"]
        assert (
            sum(step["documents"] for step in trace if step["event"] == "score")
            == after["documents_scored"]
        )
        assert not budget or len(trace) <= budget
        query_l1 = np.abs(queries[query_row].astype(np.float64)).sum()
        for step in trace:
            assert step["documents"] > 0
            assert step["penalty"] >= 0
            assert step["upper_bound"] == pytest.approx(query_l1 - 2 * step["penalty"])
            if step["event"] == "score":
                assert step["dimension"] == -1
            else:
                assert step["event"] == "split"
                assert 0 <= step["dimension"] < 8
                assert step["documents"] > 3


def test_empty_search_has_empty_trace():
    index = Index(pack([[0, 1]]), np.array([0], dtype=np.int64), 2)
    result = index.search(
        np.array([[1, 1]], dtype=np.float32),
        np.array([[9]], dtype=np.int64),
        trace=True,
    )
    assert result["stats"][0]["trace"] == []


def test_bounds_survive_large_magnitudes_and_cancellation():
    rng = np.random.default_rng(97)
    bits = rng.integers(0, 2, size=(80, 9))
    assignments = np.arange(80, dtype=np.int64) % 2
    queries = np.array(
        [
            [1e30, -1e30, 0, 1, -1, 1e-20, -1e-20, 2, -2],
            [1e-30, 2e-30, -1e-30, 0, 0, 0, 0, 0, 0],
        ],
        dtype=np.float32,
    )
    buckets = np.array(
        [
            [0, 1],
            [1, 0],
        ],
        dtype=np.int64,
    )
    index = Index(pack(bits), assignments, 9)
    # The same binary scorer is the reference when cancellation makes sums order-sensitive.
    exact = index.scan(queries, buckets, candidate_limit=7)
    for probability in [0.0, 0.7, 1.0]:
        result = index.search(
            queries,
            buckets,
            candidate_limit=7,
            node_budget=0,
            leaf_size=1,
            explore_probability=probability,
        )
        assert_result(result, (exact["rows"], exact["scores"]))


def test_unused_packed_bits_do_not_affect_scores():
    bits = np.array(
        [
            [1, 0, 1],
            [0, 1, 0],
        ]
    )
    codes = pack(bits)
    codes |= np.uint64(0xFFFFFFFFFFFFFFF8)
    assignments = np.array([0, 0], dtype=np.int64)
    queries = np.array([[1, -2, 3]], dtype=np.float32)
    buckets = np.array([[0]], dtype=np.int64)
    index = Index(codes, assignments, 3)
    expected = brute_force(bits, assignments, queries, buckets, 4)
    assert_result(index.scan(queries, buckets, candidate_limit=4), expected)
    assert_result(
        index.search(queries, buckets, candidate_limit=4, node_budget=0, leaf_size=1),
        expected,
    )


@pytest.mark.parametrize(
    "bad_codes,bad_assignments,dimensions",
    [
        (np.zeros((2, 1), dtype=np.int64), np.zeros(2, dtype=np.int64), 3),
        (np.zeros((2, 1), dtype=np.uint64), np.zeros(2, dtype=np.int32), 3),
        (np.zeros(2, dtype=np.uint64), np.zeros(2, dtype=np.int64), 3),
        (np.zeros((2, 1), dtype=np.uint64), np.zeros(3, dtype=np.int64), 3),
        (np.zeros((2, 1), dtype=np.uint64), np.array([0, -1], dtype=np.int64), 3),
        (np.zeros((2, 1), dtype=np.uint64), np.zeros(2, dtype=np.int64), 0),
        (np.zeros((2, 1), dtype=np.uint64), np.zeros(2, dtype=np.int64), 65),
        (np.zeros((4, 2), dtype=np.uint64)[::2], np.zeros(2, dtype=np.int64), 65),
    ],
)
def test_rejects_invalid_construction(bad_codes, bad_assignments, dimensions):
    with pytest.raises((ValueError, TypeError)):
        Index(bad_codes, bad_assignments, dimensions)


@pytest.mark.parametrize("method", ["scan", "search"])
@pytest.mark.parametrize(
    "queries,buckets",
    [
        (np.zeros((1, 2), dtype=np.float64), np.array([[0]], dtype=np.int64)),
        (np.zeros((1, 2), dtype=np.float32), np.array([[0]], dtype=np.int32)),
        (np.array([[np.nan, 0]], dtype=np.float32), np.array([[0]], dtype=np.int64)),
        (np.array([[np.inf, 0]], dtype=np.float32), np.array([[0]], dtype=np.int64)),
        (np.zeros((1, 3), dtype=np.float32), np.array([[0]], dtype=np.int64)),
        (
            np.zeros((1, 2), dtype=np.float32),
            np.array(
                [
                    [0],
                    [1],
                ],
                dtype=np.int64,
            ),
        ),
        (np.zeros((1, 2), dtype=np.float32), np.array([[0, 0]], dtype=np.int64)),
        (np.zeros((1, 2), dtype=np.float32), np.array([[-2]], dtype=np.int64)),
        (np.zeros((1, 4), dtype=np.float32)[:, ::2], np.array([[0]], dtype=np.int64)),
    ],
)
def test_rejects_invalid_search_arrays(method, queries, buckets):
    index = Index(pack([[0, 1]]), np.array([0], dtype=np.int64), 2)
    with pytest.raises((ValueError, TypeError)):
        getattr(index, method)(queries, buckets)


@pytest.mark.parametrize(
    "settings",
    [
        {"candidate_limit": 0},
        {"node_budget": -1},
        {"leaf_size": 0},
        {"explore_probability": -0.1},
        {"explore_probability": 1.1},
        {"explore_probability": float("nan")},
    ],
)
def test_rejects_invalid_search_settings(settings):
    index = Index(pack([[0, 1]]), np.array([0], dtype=np.int64), 2)
    with pytest.raises((ValueError, TypeError)):
        index.search(
            np.array([[1, 1]], dtype=np.float32),
            np.array([[0]], dtype=np.int64),
            **settings,
        )
