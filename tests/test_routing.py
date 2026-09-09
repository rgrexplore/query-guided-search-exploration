import numpy as np
import pytest

from routing import build_router, exact_search


def test_sign_routes_follow_weighted_mismatches_and_skip_empty_prefixes():
    documents = np.array(
        [
            [1, -1, 1],
            [-1, 1, 1],
            [1, 1, 1],
        ],
        dtype=np.float32,
    )
    router = build_router(documents, method="sign", routing_bits=2)
    queries = np.array(
        [
            [0.2, -3, 1],
            [-3, 0.2, 1],
            [0, 0, 1],
        ],
        dtype=np.float32,
    )

    buckets, times = router.select(queries, probes=4)

    # Addresses 1, 2 and 3 exist. There is no document in address 0.
    np.testing.assert_array_equal(router.assignments, [1, 2, 3])
    np.testing.assert_array_equal(buckets, [[1, 3, 2, -1], [2, 3, 1, -1], [1, 2, 3, -1]])
    assert buckets.dtype == np.int64
    assert times.shape == (3,) and times.dtype == np.float64
    assert np.all(times >= 0)
    assert router.info["occupied_buckets"] == 3
    assert router.info["possible_buckets"] == 4


def test_ivf_routes_and_assignments_use_the_stored_centroids():
    documents = np.array(
        [
            [1, 0],
            [0.9, 0.1],
            [0, 1],
            [0.1, 0.9],
        ],
        dtype=np.float32,
    )
    router = build_router(documents, clusters=2, seed=7)
    centroids = router.quantizer.reconstruct_n(0, 2)
    np.testing.assert_array_equal(router.assignments, (documents @ centroids.T).argmax(axis=1))

    queries = np.array(
        [
            [1, 0.2],
            [0.1, 1],
        ],
        dtype=np.float32,
    )
    buckets, times = router.select(queries, probes=3)
    np.testing.assert_array_equal(buckets[:, :2], np.argsort(-(queries @ centroids.T), axis=1))
    np.testing.assert_array_equal(buckets[:, 2], -1)
    assert times.shape == (2,)
    assert router.build_ms >= 0
    assert router.assignments.flags.c_contiguous


def test_ivf_with_all_clusters_recovers_flat_inner_product_results():
    documents = np.array(
        [
            [1, 0],
            [0.8, 0.2],
            [0, 1],
            [-1, 0],
            [0, -1],
        ],
        dtype=np.float32,
    )
    queries = np.array(
        [
            [0.9, 0.1],
            [-0.2, 0.9],
        ],
        dtype=np.float32,
    )
    router = build_router(documents, clusters=2, seed=5)

    rows, scores, times = router.ivf_search(queries, candidate_limit=7, probes=2)
    exact_rows, exact_scores, _ = exact_search(documents, queries, top_k=7)

    np.testing.assert_array_equal(rows, exact_rows)
    np.testing.assert_allclose(scores, exact_scores)
    np.testing.assert_array_equal(rows[:, 5:], -1)
    assert times.shape == (2,) and np.all(times >= 0)


def test_exact_search_agrees_with_independent_matrix_scores():
    documents = np.array(
        [
            [1, 0],
            [0, 1],
            [-1, -1],
        ],
        dtype=np.float32,
    )
    queries = np.array(
        [
            [0.8, 0.2],
            [-0.4, 0.7],
        ],
        dtype=np.float32,
    )
    rows, scores, times = exact_search(documents, queries, top_k=2)
    expected_scores = queries @ documents.T
    expected_rows = np.argsort(-expected_scores, axis=1)[:, :2]
    np.testing.assert_array_equal(rows, expected_rows)
    np.testing.assert_allclose(scores, np.take_along_axis(expected_scores, expected_rows, axis=1))
    assert times.shape == (2,)


def test_ivf_seed_is_repeatable():
    documents = np.array(
        [
            [1, 0],
            [0.8, 0.2],
            [0, 1],
            [-1, 0],
            [0, -1],
        ],
        dtype=np.float32,
    )
    first = build_router(documents, clusters=2, seed=17)
    second = build_router(documents, clusters=2, seed=17)
    np.testing.assert_array_equal(first.assignments, second.assignments)


def test_ivf_keeps_empty_lists_in_the_probe_order():
    documents = np.array([[1, 0]] * 4, dtype=np.float32)
    router = build_router(documents, clusters=3)
    queries = np.array([[1, 0]], dtype=np.float32)
    buckets, _ = router.select(queries, probes=4)
    _, centroid_rows = router.quantizer.search(queries, 4)
    np.testing.assert_array_equal(buckets, centroid_rows)
    assert router.info["occupied_buckets"] < 3
    assert set(buckets[0, :3]) == {0, 1, 2}
    assert buckets[0, 3] == -1


@pytest.mark.parametrize(
    "documents",
    [
        np.ones((3, 2), dtype=np.float64),
        np.array([[np.nan, 1]], dtype=np.float32),
        np.empty((0, 2), dtype=np.float32),
        np.ones(3, dtype=np.float32),
    ],
)
def test_bad_document_arrays_are_rejected(documents):
    with pytest.raises((ValueError, TypeError)):
        build_router(documents, method="sign", routing_bits=1)


def test_bad_settings_and_queries_are_rejected():
    documents = np.eye(3, dtype=np.float32)
    for settings in (
        {"clusters": 4},
        {"clusters": 0},
        {"threads": 0},
        {"method": "missing"},
        {"method": "sign", "routing_bits": 4},
        {"method": "sign", "routing_bits": 0},
    ):
        with pytest.raises(ValueError):
            build_router(documents, **settings)
    router = build_router(documents, method="sign", routing_bits=2)
    for query in (
        np.ones((1, 2), dtype=np.float32),
        np.array([[np.inf, 0, 0]], dtype=np.float32),
    ):
        with pytest.raises(ValueError):
            router.select(query, probes=1)
    with pytest.raises(ValueError):
        router.select(documents, probes=0)
    with pytest.raises(ValueError):
        exact_search(documents, documents, top_k=0)


def test_empty_queries_keep_the_requested_output_shape():
    documents = np.eye(3, dtype=np.float32)
    queries = np.empty((0, 3), dtype=np.float32)
    router = build_router(documents, method="sign", routing_bits=2)
    buckets, times = router.select(queries, probes=5)
    assert buckets.shape == (0, 5) and times.shape == (0,)
    rows, scores, times = exact_search(documents, queries, top_k=5)
    assert rows.shape == scores.shape == (0, 5) and times.shape == (0,)
