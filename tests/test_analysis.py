import csv
import hashlib
import json
from pathlib import Path

import pytest

import analyze


def write_csv(path, rows):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


@pytest.fixture
def recorded_run(tmp_path):
    cases = [
        dict(
            case_id="scan",
            routing_method="ivf",
            probes=1,
            kind="scan",
            node_budget=0,
            explore_probability=0.0,
            seed=42,
        ),
        dict(
            case_id="det",
            routing_method="ivf",
            probes=1,
            kind="branch",
            node_budget=16,
            explore_probability=0.0,
            seed=42,
        ),
    ]
    cases += [
        dict(
            case_id=f"random-{seed}",
            routing_method="ivf",
            probes=1,
            kind="branch",
            node_budget=16,
            explore_probability=0.1,
            seed=seed,
        )
        for seed in (42, 43, 44)
    ]
    metadata = dict(
        cases=cases,
        dataset=dict(documents=10, queries=2, is_subset=True),
        config=dict(benchmark=dict(repetitions=3), search=dict(top_k=2, candidate_limit=4)),
    )
    (tmp_path / "metadata.json").write_text(json.dumps(metadata))
    reference = [
        dict(
            query_id=f"q{q}",
            float_reference_ndcg=0.8,
            float_reference_precision=0.5,
            float_reference_recall=1.0,
        )
        for q in range(2)
    ]
    write_csv(tmp_path / "reference.csv", reference)
    rows = []
    for case in cases:
        values = [0.6, 0.8] if case["kind"] == "scan" else [0.4, 0.2]
        if case["explore_probability"]:
            values = [v + 0.1 * (case["seed"] - 42) for v in values]
        for q in range(2):
            for repeat in range(3):
                rows.append(
                    dict(
                        case_id=case["case_id"],
                        query_id=f"q{q}",
                        query_row=q,
                        repeat=repeat,
                        method="ivf/" + case["kind"],
                        probes=1,
                        node_budget=case["node_budget"],
                        explore_probability=case["explore_probability"],
                        seed=case["seed"],
                        float_recall=values[q],
                        binary_recall=1.0,
                        ndcg=values[q],
                        precision=0.5,
                        relevance_recall=0.5,
                        candidates=4,
                        nodes=16,
                        random_nodes=0,
                        documents_scored=8,
                        bitplane_words=12,
                        budget_exhausted=int(case["kind"] == "branch"),
                        routed_documents=8,
                        selected_buckets=1,
                        routing_float_recall=1.0,
                        empty_result=0,
                        fewer_than_k=0,
                        candidate_fill_rate=1.0,
                        stop_reason="budget" if case["kind"] == "branch" else "exhausted",
                        retrieval_ms=[1.0, 2.0, 9.0][repeat],
                        core_ms=0.5,
                        native_ms=0.4,
                        routing_ms=0.1,
                        rerank_ms=0.2,
                    )
                )
    write_csv(tmp_path / "queries.csv", rows)
    return tmp_path, rows


def test_timing_repeats_do_not_become_quality_samples(recorded_run):
    path, rows = recorded_run
    run = analyze.load_run(path)
    weights = analyze.bootstrap_weights(2, 100, 7)
    policies, summaries = analyze.summarize_settings(run, weights)
    random = next(row for row in summaries if row["explore_probability"] == 0.1)
    assert random["queries"] == 2
    assert random["search_seeds"] == 3
    assert random["requests"] == 18
    assert random["ndcg"] == pytest.approx(0.4)
    assert random["ndcg_seed_min"] == pytest.approx(0.3)
    assert random["ndcg_seed_max"] == pytest.approx(0.5)
    assert random["request_p95_ms"] == 9.0
    assert random["query_median_p95_ms"] == 2.0


def test_bootstrap_compares_the_same_queries_and_averages_seeds(recorded_run):
    path, _ = recorded_run
    run = analyze.load_run(path)
    weights = analyze.bootstrap_weights(2, 100, 7)
    policies, _ = analyze.summarize_settings(run, weights)
    differences = analyze.paired_differences(policies, weights)
    row = next(
        row
        for row in differences
        if row["comparison"] == "exploration-minus-deterministic" and row["metric"] == "ndcg"
    )
    assert row["difference"] == pytest.approx(0.1)
    assert row["ci_low"] == pytest.approx(0.1)
    assert row["ci_high"] == pytest.approx(0.1)


@pytest.mark.parametrize("fault", ["missing", "duplicate", "quality_changed", "nan", "wrong_query"])
def test_incomplete_or_inconsistent_measurements_are_rejected(recorded_run, fault):
    path, rows = recorded_run
    if fault == "missing":
        rows.pop()
    elif fault == "duplicate":
        rows.append(rows[0].copy())
    elif fault == "quality_changed":
        rows[1]["ndcg"] = 0.3
    elif fault == "nan":
        rows[0]["retrieval_ms"] = float("nan")
    else:
        rows[0]["query_id"] = "wrong"
    write_csv(path / "queries.csv", rows)
    with pytest.raises(ValueError):
        analyze.load_run(path)


def test_unlimited_budget_is_last_and_not_a_numeric_zero():
    budgets, labels = analyze.budget_axis([0, 64, 16, 32])
    assert budgets == [16, 32, 64, 0]
    assert labels == ["16", "32", "64", "Unlimited"]


def test_unlimited_budget_survives_a_generator():
    budgets, labels = analyze.budget_axis(value for value in [0, 64, 16, 32, 0])
    assert budgets == [16, 32, 64, 0]
    assert labels[-1] == "Unlimited"


@pytest.mark.parametrize("omitted", ["det", "random-44", "endpoint", "none"])
def test_declared_cases_cannot_hide_a_missing_planned_policy_or_seed(recorded_run, omitted):
    path, rows = recorded_run
    metadata = json.loads((path / "metadata.json").read_text())
    metadata["config"]["routing"] = dict(methods=["sign"], probes=[1], seed=42)
    metadata["config"]["search"].update(
        node_budgets=[16], explore_probabilities=[0.0, 0.1], seeds=[42, 43, 44]
    )
    metadata["config"]["benchmark"]["complete_routing_endpoint"] = omitted == "endpoint"
    metadata["builds"] = [
        dict(routing=dict(method="sign", possible_buckets=256, occupied_buckets=2))
    ]
    for case in metadata["cases"]:
        case["routing_method"] = "sign"
    for row in rows:
        row["method"] = row["method"].replace("ivf/", "sign/")
    metadata["cases"] = [case for case in metadata["cases"] if case["case_id"] != omitted]
    rows = [row for row in rows if row["case_id"] != omitted]
    (path / "metadata.json").write_text(json.dumps(metadata))
    write_csv(path / "queries.csv", rows)
    if omitted == "none":
        assert len(analyze.load_run(path).cases) == 5
    else:
        with pytest.raises(ValueError, match="planned"):
            analyze.load_run(path)


def test_empirical_envelope_keeps_only_measured_nondominated_points():
    points = [(1.0, 0.4), (2.0, 0.3), (2.0, 0.5), (2.0, 0.6), (3.0, 0.6), (4.0, 0.9)]
    assert analyze.nondominated_points(points) == [(1.0, 0.4), (2.0, 0.6), (4.0, 0.9)]


@pytest.mark.filterwarnings("error")
def test_analysis_writes_a_report_and_counts_failures(recorded_run):
    path, rows = recorded_run
    for row in rows:
        if row["case_id"] == "det" and row["query_id"] == "q0":
            row.update(
                candidates=0,
                empty_result=1,
                fewer_than_k=1,
                candidate_fill_rate=0,
                ndcg=0,
                precision=0,
                relevance_recall=0,
                float_recall=0,
                binary_recall=0,
            )
    write_csv(path / "queries.csv", rows)
    result = analyze.analyze_run(path, bootstrap_samples=30, seed=7)
    report = Path(result["report"])
    assert report.is_file()
    assert "648" not in report.read_text()
    assert "2 queries" in report.read_text()
    assert (report.parent / "completion.png").is_file()
    assert (report.parent / "overview.png").is_file()
    provenance = json.loads((report.parent / "metadata.json").read_text())
    assert provenance["bootstrap"] == {"samples": 30, "seed": 7}
    assert (
        provenance["analysis_sha256"]
        == hashlib.sha256(Path(analyze.__file__).read_bytes()).hexdigest()
    )
    for name in ("metadata.json", "queries.csv", "reference.csv"):
        assert (
            provenance["source_sha256"][name]
            == hashlib.sha256((path / name).read_bytes()).hexdigest()
        )
    with (report.parent / "settings.csv").open() as stream:
        settings = list(csv.DictReader(stream))
    deterministic = next(
        row for row in settings if row["kind"] == "branch" and row["explore_probability"] == "0.0"
    )
    assert float(deterministic["empty_result"]) == 0.5
    assert float(deterministic["fewer_than_k"]) == 0.5
