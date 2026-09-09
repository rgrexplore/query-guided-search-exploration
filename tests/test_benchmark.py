import csv
import json
import math
from types import SimpleNamespace

import numpy as np
import pytest

import benchmark
from benchmark import (
    _build_cases,
    _build_schedule,
    _routing_diagnostics,
    neighbor_recall,
    plot_results,
    relevance_metrics,
    rerank,
    run_benchmark,
    summarize,
    write_trace,
)
from data import Dataset
from embeddings import EmbeddingArrays, encode_signs


def test_neighbor_recall_ignores_padding_and_duplicate_hits():
    assert neighbor_recall([2, 2, -1, 8], [2, 3, 4, -1]) == pytest.approx(1 / 3)
    assert neighbor_recall([], [2, 3]) == 0.0


def test_relevance_metrics_use_dataset_ids_and_all_known_positives():
    scores = relevance_metrics(["wrong", "paper-a"], {"paper-a": 1, "paper-b": 1}, 2)
    assert scores["precision"] == 0.5
    assert scores["recall"] == 0.5
    assert scores["ndcg"] == pytest.approx((1 / math.log2(3)) / (1 + 1 / math.log2(3)))


def test_rerank_only_scores_returned_candidates_and_breaks_ties_by_row():
    documents = np.array(
        [
            [1, 0],
            [0, 1],
            [1, 0],
            [-1, 0],
        ],
        dtype=np.float32,
    )
    query = np.array([1, 0], dtype=np.float32)
    rows = rerank([2, -1, 1, 0], documents, query, 2)
    assert rows.tolist() == [0, 2]
    assert rerank([-1], documents, query, 2).tolist() == []


def test_summary_keeps_seeds_separate_and_reports_query_time_percentiles():
    base = dict(
        method="ivf/branch",
        probes=2,
        node_budget=8,
        explore_probability=0.1,
        seed=7,
        float_recall=0.5,
        binary_recall=0.75,
        ndcg=0.6,
        precision=0.2,
        relevance_recall=0.4,
        candidates=8,
        nodes=8,
        random_nodes=1,
        documents_scored=10,
        core_ms=1.0,
        native_ms=0.8,
        routing_ms=0.1,
        rerank_ms=0.2,
        budget_exhausted=1,
    )
    # Repeated measurements share a summary only when their search seeds match.
    measurements = [
        dict(base, retrieval_ms=1.0),
        dict(base, retrieval_ms=3.0),
        dict(base, seed=8, retrieval_ms=9.0),
    ]
    summary = summarize(measurements)
    assert len(summary) == 2
    assert summary[0]["seed"] == 7
    assert summary[0]["retrieval_p50_ms"] == 2.0
    assert summary[0]["float_recall"] == 0.5
    assert summary[1]["retrieval_p50_ms"] == 9.0


@pytest.fixture
def hand_checked_inputs():
    documents = np.array(
        [
            [1, 0],
            [0, 1],
            [-1, 0],
            [0, -1],
            [0.8, 0.6],
            [0.6, 0.8],
        ],
        dtype=np.float32,
    )
    queries = np.array(
        [
            [1, 0],
            [0, 1],
        ],
        dtype=np.float32,
    )
    dataset = Dataset(
        corpus_ids=["a", "b", "c", "d", "e", "f"],
        titles=[""] * 6,
        texts=[""] * 6,
        query_ids=["qa", "qb"],
        queries=["a?", "b?"],
        qrels={
            "qa": {"a": 1},
            "qb": {"b": 1},
        },
        fingerprint="fixture",
        source="hand-checked vectors",
        full_document_count=6,
        full_query_count=2,
    )
    # These small vectors stand in for both embedding sizes in this pipeline check.
    arrays = EmbeddingArrays(
        documents=documents,
        queries=queries,
        full_documents=documents,
        full_queries=queries,
        codes=encode_signs(documents),
        metadata={"fixture": True},
    )
    config = {
        "routing": dict(
            methods=["ivf", "sign"],
            clusters=2,
            sign_bits=2,
            probes=[2],
            seed=7,
            threads=1,
        ),
        "search": dict(
            candidate_limit=4,
            top_k=2,
            node_budgets=[0, 3],
            leaf_size=1,
            explore_probabilities=[0.0, 0.5],
            seeds=[7],
        ),
        "benchmark": dict(warmup_queries=0, repetitions=1),
    }
    return dataset, arrays, config


def test_pipeline_writes_consistent_results_on_hand_checked_vectors(tmp_path, hand_checked_inputs):
    dataset, arrays, config = hand_checked_inputs
    output = tmp_path / "run"
    summary, metadata = run_benchmark(dataset, arrays, config, output)
    assert len(summary) == 11
    assert all(row["binary_recall"] == 1 for row in summary if row["method"].endswith("scan"))
    assert all(
        row["binary_recall"] == 1
        for row in summary
        if row["method"].endswith("branch") and row["node_budget"] == 0
    )
    assert metadata["packages"]["numpy"] == np.__version__
    assert metadata["status"] == "complete"
    assert metadata["expected_requests"] == 22
    assert metadata["measured_requests"] == 22
    assert metadata["unlimited_candidate_checks"] == 8

    with (output / "queries.csv").open(newline="") as stream:
        requests = list(csv.DictReader(stream))
    assert len(requests) == 22
    assert len({(row["case_id"], row["repeat"], row["query_id"]) for row in requests}) == 22
    for row in requests:
        query_row = int(row["query_row"])
        returned = int(row["candidates"])
        routed = int(row["routed_documents"])
        available = min(config["search"]["candidate_limit"], routed)
        assert row["query_id"] == dataset.query_ids[query_row]
        assert 0 <= int(row["selected_buckets"]) <= int(row["probes"])
        assert 0 <= float(row["routing_float_recall"]) <= 1
        assert int(row["empty_result"]) == int(returned == 0)
        assert int(row["fewer_than_k"]) == int(returned < config["search"]["top_k"])
        expected_fill = returned / available if available else 0.0
        assert float(row["candidate_fill_rate"]) == pytest.approx(expected_fill)

    plot_results(summary, metadata, output)
    assert (output / "tradeoffs.png").stat().st_size > 1000
    assert "Node budget" in (output / "report.md").read_text()
    write_trace(dataset, arrays, config, output, "qa")
    trace = json.loads((output / "trace.json").read_text())
    assert trace["query_id"] == "qa"
    assert len(trace["stats"]["trace"]) == trace["stats"]["nodes"]


@pytest.fixture
def dense_grid_config():
    return {
        "routing": {
            "methods": ["ivf", "sign"],
            "probes": [1, 2, 4, 8, 16, 32, 64],
            "seed": 42,
        },
        "search": {
            "node_budgets": [16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 0],
            "explore_probabilities": [0.0, 0.1],
            "seeds": [42, 43, 44],
        },
        "benchmark": {},
    }


@pytest.fixture
def router_counts():
    # IVF probes its possible lists; sign routing probes only occupied addresses.
    return {
        "ivf": SimpleNamespace(
            info={
                "possible_buckets": 64,
                "occupied_buckets": 64,
            }
        ),
        "sign": SimpleNamespace(
            info={
                "possible_buckets": 256,
                "occupied_buckets": 120,
            }
        ),
    }


def case_setting(case):
    return (
        case["routing_method"],
        case["probes"],
        case["kind"],
        case["node_budget"],
        case["explore_probability"],
        case["seed"],
    )


def test_dense_grid_contains_every_requested_setting_once(dense_grid_config, router_counts):
    cases = _build_cases(router_counts, dense_grid_config)
    expected = set()
    for routing_method in ["ivf", "sign"]:
        for probes in dense_grid_config["routing"]["probes"]:
            expected.add((routing_method, probes, "scan", 0, 0.0, 42))
            for budget in dense_grid_config["search"]["node_budgets"]:
                expected.add((routing_method, probes, "branch", budget, 0.0, 42))
                for seed in [42, 43, 44]:
                    expected.add((routing_method, probes, "branch", budget, 0.1, seed))
            if routing_method == "ivf":
                expected.add((routing_method, probes, "float", 0, 0.0, 42))

    assert len(cases) == 581
    assert len({case["case_id"] for case in cases}) == len(cases)
    assert len({case_setting(case) for case in cases}) == len(cases)
    assert {case_setting(case) for case in cases} == expected


def test_full_routing_adds_only_missing_sign_endpoints(dense_grid_config, router_counts):
    original = _build_cases(router_counts, dense_grid_config)
    dense_grid_config["benchmark"]["complete_routing_endpoint"] = True
    complete = _build_cases(router_counts, dense_grid_config)

    original_settings = {case_setting(case) for case in original}
    complete_settings = {case_setting(case) for case in complete}
    assert len(complete) == 583
    assert len({case["case_id"] for case in complete}) == 583
    assert len(complete_settings) == 583
    assert original_settings <= complete_settings
    assert complete_settings - original_settings == {
        ("sign", 120, "scan", 0, 0.0, 42),
        ("sign", 120, "branch", 0, 0.0, 42),
    }

    # The regular grid already searches all 64 IVF lists, so it needs no extra case.
    original_ivf = {case_setting(case) for case in original if case["routing_method"] == "ivf"}
    complete_ivf = {case_setting(case) for case in complete if case["routing_method"] == "ivf"}
    assert complete_ivf == original_ivf


def test_ivf_full_routing_counts_empty_lists(dense_grid_config, router_counts):
    dense_grid_config["routing"]["methods"] = ["ivf"]
    dense_grid_config["routing"]["probes"] = [1, 4, 16]
    router_counts["ivf"].info["occupied_buckets"] = 60
    original = _build_cases(router_counts, dense_grid_config)
    dense_grid_config["benchmark"]["complete_routing_endpoint"] = True
    complete = _build_cases(router_counts, dense_grid_config)

    assert {case_setting(case) for case in complete} - {
        case_setting(case) for case in original
    } == {
        ("ivf", 64, "scan", 0, 0.0, 42),
        ("ivf", 64, "branch", 0, 0.0, 42),
    }
    assert len(complete) == len(original) + 2


def test_schedule_is_repeatable_and_keeps_every_case_and_query_row_once():
    cases = [
        {"case_id": "ivf-scan"},
        {"case_id": "sign-scan"},
        {"case_id": "ivf-branch"},
        {"case_id": "sign-branch"},
        {"case_id": "ivf-float"},
    ]
    expected_ids = {case["case_id"] for case in cases}
    first = _build_schedule(cases, query_count=17, repetitions=3, seed=42)
    second = _build_schedule(cases, query_count=17, repetitions=3, seed=42)

    assert first == second
    assert len(first) == 3
    assert {repeat["repeat"] for repeat in first} == {0, 1, 2}
    for repeat in first:
        assert len(repeat["case_ids"]) == len(cases)
        assert set(repeat["case_ids"]) == expected_ids
        assert len(repeat["query_rows"]) == 17
        assert set(repeat["query_rows"]) == set(range(17))

    # Check variation without depending on one particular random sequence.
    assert len({tuple(repeat["case_ids"]) for repeat in first}) > 1
    assert len({tuple(repeat["query_rows"]) for repeat in first}) > 1


def test_routing_diagnostics_distinguish_population_from_reference_coverage():
    assignments = np.array([3, 3, 7, 9], dtype=np.int64)
    buckets = np.array(
        [
            [3, -1],
            [7, 9],
        ],
        dtype=np.int64,
    )
    full_reference_rows = np.array(
        [
            [0, 2],
            [2, 3],
        ],
        dtype=np.int64,
    )

    diagnostics = _routing_diagnostics(assignments, buckets, full_reference_rows)

    np.testing.assert_array_equal(diagnostics["routed_documents"], [2, 2])
    np.testing.assert_array_equal(diagnostics["selected_buckets"], [1, 2])
    np.testing.assert_allclose(diagnostics["routing_float_recall"], [0.5, 1.0])


def test_empty_budgeted_requests_remain_in_every_repeat_and_summary(
    tmp_path, hand_checked_inputs, monkeypatch
):
    dataset, arrays, config = hand_checked_inputs
    config["routing"].update(methods=["ivf"], clusters=1, probes=[1])
    config["search"].update(node_budgets=[1], seeds=[7, 8])
    config["benchmark"]["repetitions"] = 3

    # One node can split this bucket, but cannot reach a document-scoring leaf.
    observed_requests = []
    original_query_run = benchmark._query_run

    def record_query_run(
        index,
        router,
        query,
        full_query,
        full_documents,
        expected_buckets,
        probes,
        search,
        setting,
        query_number,
    ):
        observed_requests.append((setting["case_id"], query_number))
        np.testing.assert_array_equal(query, arrays.queries[query_number])
        np.testing.assert_array_equal(full_query, arrays.full_queries[query_number])
        return original_query_run(
            index,
            router,
            query,
            full_query,
            full_documents,
            expected_buckets,
            probes,
            search,
            setting,
            query_number,
        )

    monkeypatch.setattr(benchmark, "_query_run", record_query_run)
    output = tmp_path / "empty-results"
    summary, metadata = run_benchmark(dataset, arrays, config, output)

    with (output / "queries.csv").open(newline="") as stream:
        requests = list(csv.DictReader(stream))
    schedule = json.loads((output / "schedule.json").read_text())
    expected_order = [
        (repeat["repeat"], case_id, query_row)
        for repeat in schedule
        for case_id in repeat["case_ids"]
        for query_row in repeat["query_rows"]
    ]
    actual_order = [(int(row["repeat"]), row["case_id"], int(row["query_row"])) for row in requests]

    assert len(metadata["cases"]) == 5
    assert metadata["expected_requests"] == metadata["measured_requests"] == 30
    assert len(requests) == len(set(actual_order)) == 30
    assert actual_order == expected_order
    assert observed_requests == [(case_id, query_row) for _, case_id, query_row in expected_order]
    assert {int(row["repeat"]) for row in requests} == {0, 1, 2}
    assert all(row["measurements"] == 6 for row in summary)

    empty_requests = [row for row in requests if row["method"] == "ivf/branch"]
    assert len(empty_requests) == 18
    for row in empty_requests:
        assert row["query_id"] == dataset.query_ids[int(row["query_row"])]
        assert int(row["candidates"]) == 0
        assert int(row["routed_documents"]) == 6
        assert int(row["selected_buckets"]) == 1
        assert float(row["routing_float_recall"]) == 1.0
        assert int(row["empty_result"]) == 1
        assert int(row["fewer_than_k"]) == 1
        assert float(row["candidate_fill_rate"]) == 0.0
        assert float(row["float_recall"]) == 0.0
        assert float(row["binary_recall"]) == 0.0
        assert float(row["ndcg"]) == 0.0
        assert float(row["retrieval_ms"]) >= 0.0

    for row in summary:
        if row["method"] == "ivf/branch":
            assert row["empty_result"] == 1.0
            assert row["fewer_than_k"] == 1.0
            assert row["candidate_fill_rate"] == 0.0
            assert row["float_recall"] == 0.0
            assert row["ndcg"] == 0.0


def test_unlimited_search_must_match_the_reference_candidate_rows(
    tmp_path, hand_checked_inputs, monkeypatch
):
    dataset, arrays, config = hand_checked_inputs
    config["routing"].update(methods=["ivf"], clusters=1, probes=[1])
    config["search"].update(node_budgets=[0], explore_probabilities=[0.0])
    original_query_run = benchmark._query_run

    def return_reversed_candidates(*args, **kwargs):
        candidates, final_rows, stats, times = original_query_run(*args, **kwargs)
        setting = args[-2]
        if setting["kind"] == "branch":
            # The count and set still match; the ranked candidate list does not.
            candidates = candidates[::-1].copy()
        return candidates, final_rows, stats, times

    monkeypatch.setattr(benchmark, "_query_run", return_reversed_candidates)
    with pytest.raises(RuntimeError, match="Unlimited search disagrees with scan"):
        run_benchmark(dataset, arrays, config, tmp_path / "bad-unlimited-result")
