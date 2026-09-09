import csv
import gzip
import json
import shutil
from pathlib import Path

import numpy as np
import pytest

import scaling_analysis
import scaling_plots
from scaling_analysis import choose_case


def test_choose_case_uses_only_complete_qualifying_branch_measurements():
    rows = [
        {
            "method": "scan",
            "node_budget": 0,
            "status": "complete",
            "development_recall": 1.0,
            "api_p50_ms": 0.1,
        },
        {
            "method": "branch",
            "node_budget": 32,
            "status": "timeout",
            "development_recall": None,
            "api_p50_ms": None,
        },
        {
            "method": "branch",
            "node_budget": 128,
            "status": "complete",
            "development_recall": 0.94,
            "api_p50_ms": 0.5,
        },
        {
            "method": "branch",
            "node_budget": 512,
            "status": "complete",
            "development_recall": 0.96,
            "api_p50_ms": 2.0,
        },
    ]

    assert choose_case(rows, 0.95)["node_budget"] == 512
    assert choose_case(rows, 0.99) is None


def test_choose_case_ties_prefer_smaller_finite_budget_and_put_unlimited_last():
    rows = [
        {
            "method": "branch",
            "node_budget": budget,
            "status": "complete",
            "development_recall": 1.0,
            "api_p50_ms": 1.0,
        }
        for budget in (0, 512, 128)
    ]

    assert choose_case(rows, 0.99)["node_budget"] == 128


def test_choose_case_requires_a_finite_latency():
    rows = [
        {
            "method": "branch",
            "node_budget": 128,
            "status": "complete",
            "development_recall": 1.0,
            "api_p50_ms": float("nan"),
        }
    ]

    assert choose_case(rows, 0.95) is None


def _write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def _case(
    run_dir: Path,
    phase: str,
    pool_size: int,
    method: str,
    budget: int,
    repeat: int,
    query_rows: list[int],
    *,
    status: str = "complete",
    recalls: tuple[float, ...] | None = None,
    latency: float = 1.0,
) -> str:
    name = f"n{pool_size}-{method}-b{budget}-r{repeat}"
    folder = run_dir / "cases" / phase / name
    folder.mkdir(parents=True, exist_ok=True)
    case = {
        "phase": phase,
        "pool_size": pool_size,
        "method": method,
        "node_budget": budget,
        "repeat": repeat,
        "query_rows": query_rows,
        "run_path": str((run_dir / "metadata.json").resolve()),
        "reference_path": str((run_dir / "stale-reference.npz").resolve()),
        "output_dir": str(folder.resolve()),
    }
    _write_json(folder / "case.json", case)
    _write_json(
        folder / "summary.json",
        {
            "status": status,
            "completed_queries": len(query_rows) if status == "complete" else 0,
            "setup_ms": 2.0,
            "wall_ms": 5.0,
            "sampled_peak_rss_bytes": pool_size * 100,
            "native_peak_rss_bytes": pool_size * 80,
            **({} if status == "complete" else {"error": status}),
        },
    )
    _write_json(folder / "worker.json", {"setup_ms": 2.0, "wall_ms": 5.0})
    if status == "complete":
        assert recalls is not None
        reference_rows = {
            0: [0, 1],
            1: [1, 2],
            2: [2, 3],
            3: [3, 4],
        }
        lines = []
        for position, query_row in enumerate(query_rows):
            exact = reference_rows[query_row]
            rows = exact if recalls[position] == 1.0 else [exact[0], pool_size - 1]
            lines.append(
                json.dumps(
                    {
                        "phase": phase,
                        "pool_size": pool_size,
                        "method": method,
                        "node_budget": budget,
                        "repeat": repeat,
                        "query_row": query_row,
                        "query_id": f"q{query_row}",
                        "api_ms": latency + repeat * 0.1 + position * 0.01,
                        "native_ms": latency * 0.8 + repeat * 0.1 + position * 0.01,
                        "rows": rows,
                        "scores": [2.0, 1.0],
                        "count": 2,
                        "recall": recalls[position],
                        "nodes": budget or pool_size,
                        "random_nodes": 0,
                        "bitplane_words": 0 if method == "scan" else budget + 3,
                        "documents_scored": pool_size if method == "scan" else min(pool_size, 6),
                        "stop_reason": "exhausted" if budget == 0 else "budget",
                    },
                    separators=(",", ":"),
                )
            )
        (folder / "queries.jsonl").write_text("\n".join(lines) + "\n")
    return str((folder / "case.json").relative_to(run_dir))


def _reference_case(run_dir: Path, pool_size: int) -> str:
    folder = run_dir / "cases" / "reference" / f"n{pool_size}-scan-b0-r0"
    folder.mkdir(parents=True)
    case = {
        "phase": "reference",
        "pool_size": pool_size,
        "method": "scan",
        "node_budget": 0,
        "repeat": 0,
        "query_rows": [0, 1, 2, 3],
        "run_path": str((run_dir / "metadata.json").resolve()),
        "reference_path": None,
        "output_dir": str(folder.resolve()),
    }
    _write_json(folder / "case.json", case)
    _write_json(
        folder / "summary.json",
        {
            "status": "complete",
            "completed_queries": 4,
            "setup_ms": 2.0,
            "wall_ms": 5.0,
            "sampled_peak_rss_bytes": pool_size * 100,
            "native_peak_rss_bytes": pool_size * 80,
        },
    )
    rows = np.asarray([[0, 1], [1, 2], [2, 3], [3, 4]], dtype=np.int64)
    scores = np.asarray([[2.0, 1.0]] * 4, dtype=np.float64)
    np.savez(
        folder / "reference.npz",
        query_rows=np.asarray([0, 1, 2, 3], dtype=np.int64),
        query_ids=np.asarray(["q0", "q1", "q2", "q3"]),
        rows=rows,
        scores=scores,
        pool_size=np.asarray(pool_size),
        candidate_limit=np.asarray(2),
    )
    _write_json(folder / "verification.json", {"status": "pass"})
    return str((folder / "case.json").relative_to(run_dir))


@pytest.fixture
def saved_run(tmp_path):
    run_dir = tmp_path / "run"
    config = {
        "data": {
            "pool_sizes": [10, 20],
            "development_queries": 2,
            "evaluation_queries": 2,
        },
        "search": {"candidate_limit": 2},
        "tuning": {"targets": [0.95, 0.99], "node_budgets": [2, 4, 0]},
        "measurement": {
            "repetitions": 3,
            "bootstrap_samples": 40,
            "bootstrap_seed": 45,
        },
    }
    _write_json(
        run_dir / "metadata.json",
        {
            "config": config,
            "inputs": {
                "query_ids": ["q0", "q1", "q2", "q3"],
                "arrays": {
                    "codes": {"path": "/saved/codes.npy", "shape": [20, 1], "dtype": "uint64"},
                    "queries": {
                        "path": "/saved/queries.npy",
                        "shape": [4, 64],
                        "dtype": "float32",
                    },
                },
            },
            "query_splits": {"development": [0, 1], "evaluation": [2, 3]},
            "run_fingerprint": "fixture-run",
            "provenance": {"source": "fixture"},
        },
    )
    reference_schedule = [_reference_case(run_dir, pool) for pool in (10, 20)]
    development_schedule = []
    for pool in (10, 20):
        for repeat in range(3):
            development_schedule.append(
                _case(
                    run_dir,
                    "development",
                    pool,
                    "scan",
                    0,
                    repeat,
                    [0, 1],
                    recalls=(1.0, 1.0),
                    latency=1.0 if pool == 10 else 2.0,
                )
            )
            development_schedule.append(
                _case(
                    run_dir,
                    "development",
                    pool,
                    "branch",
                    2,
                    repeat,
                    [0, 1],
                    status="timeout" if pool == 20 and repeat == 2 else "complete",
                    recalls=(0.5, 0.5),
                    latency=0.5,
                )
            )
            development_schedule.append(
                _case(
                    run_dir,
                    "development",
                    pool,
                    "branch",
                    4,
                    repeat,
                    [0, 1],
                    recalls=(1.0, 1.0),
                    latency=2.0 if pool == 10 else 3.0,
                )
            )
            development_schedule.append(
                _case(
                    run_dir,
                    "development",
                    pool,
                    "branch",
                    0,
                    repeat,
                    [0, 1],
                    recalls=(1.0, 1.0),
                    latency=3.0 if pool == 10 else 4.0,
                )
            )
    _write_json(run_dir / "schedules" / "reference.json", reference_schedule)
    _write_json(run_dir / "schedules" / "development.json", development_schedule)

    pilot_schedule = []
    for pool in (10, 20):
        for method, budget in (("scan", 0), ("branch", 512), ("branch", 0)):
            pilot_schedule.append(
                _case(
                    run_dir,
                    "pilot",
                    pool,
                    method,
                    budget,
                    0,
                    [0, 1],
                    recalls=(1.0, 1.0),
                    latency=1.0,
                )
            )
    _write_json(run_dir / "schedules" / "pilot.json", pilot_schedule)

    evaluation_schedule = []
    for pool in (10, 20):
        for repeat in range(3):
            for method, budget, recalls, latency in (
                ("scan", 0, (1.0, 1.0), 1.0 if pool == 10 else 2.0),
                ("branch", 4, (0.5, 1.0) if pool == 10 else (1.0, 1.0), 2.0 if pool == 10 else 3.0),
                ("branch", 0, (1.0, 1.0), 3.0 if pool == 10 else 4.0),
            ):
                evaluation_schedule.append(
                    _case(
                        run_dir,
                        "evaluation",
                        pool,
                        method,
                        budget,
                        repeat,
                        [2, 3],
                        recalls=recalls,
                        latency=latency,
                    )
                )
    _write_json(run_dir / "schedules" / "evaluation.json", evaluation_schedule)
    return run_dir


def test_development_summary_requires_three_stable_repeats_and_keeps_limits(saved_run):
    rows = scaling_analysis.summarize_development(saved_run)

    chosen = choose_case([row for row in rows if row["pool_size"] == 10], 0.99)
    limited = next(
        row
        for row in rows
        if row["pool_size"] == 20 and row["method"] == "branch" and row["node_budget"] == 2
    )
    assert chosen["node_budget"] == 4
    assert chosen["queries"] == 2
    assert chosen["requests"] == 6
    assert limited["status"] == "timeout"
    assert limited["api_p50_ms"] is None


@pytest.mark.parametrize("fault", ["missing", "duplicate", "wrong_oracle"])
def test_complete_development_rows_reject_observed_query_or_oracle_errors(saved_run, fault):
    case_path = json.loads((saved_run / "schedules" / "development.json").read_text())[0]
    query_path = saved_run / Path(case_path).parent / "queries.jsonl"
    rows = [json.loads(line) for line in query_path.read_text().splitlines()]
    if fault == "missing":
        rows.pop()
    elif fault == "duplicate":
        rows.append(rows[0])
    else:
        rows[0]["recall"] = 0.5
    query_path.write_text("\n".join(json.dumps(row) for row in rows) + "\n")

    with pytest.raises(ValueError, match="query|recall"):
        scaling_analysis.summarize_development(saved_run)


def test_freeze_selection_ignores_evaluation_and_reuses_one_budget_for_two_targets(saved_run):
    frozen = scaling_analysis.freeze_selection(saved_run)
    selected = [row for row in frozen["targets"] if row["pool_size"] == 10]
    assert [row["node_budget"] for row in selected] == [4, 4]
    assert all(row["status"] == "selected" for row in selected)

    evaluation_case = json.loads((saved_run / "schedules" / "evaluation.json").read_text())[0]
    (saved_run / Path(evaluation_case).parent / "queries.jsonl").write_text("unfinished")
    assert scaling_analysis.freeze_selection(saved_run) == frozen

    development_case = json.loads((saved_run / "schedules" / "development.json").read_text())[0]
    summary_path = saved_run / Path(development_case).parent / "summary.json"
    summary = json.loads(summary_path.read_text())
    summary["wall_ms"] += 1
    _write_json(summary_path, summary)
    with pytest.raises(ValueError, match="development files"):
        scaling_analysis.freeze_selection(saved_run)


def test_paired_query_bootstrap_keeps_identical_ratios_exactly_one():
    result = scaling_analysis.paired_query_bootstrap(
        {2: [1.0, 2.0, 3.0], 3: [4.0, 5.0, 6.0]},
        {2: [1.0, 2.0, 3.0], 3: [4.0, 5.0, 6.0]},
        {2: 1.0, 3: 0.5},
        samples=1000,
        seed=45,
    )

    assert result["speedup"] == 1.0
    assert result["speedup_ci_low"] == 1.0
    assert result["speedup_ci_high"] == 1.0
    assert result["quality_mean"] == pytest.approx(0.75)


def test_target_work_curve_uses_the_selected_physical_measurement(saved_run):
    selection = scaling_analysis.freeze_selection(saved_run)
    metadata = scaling_analysis._metadata(saved_run)
    references = scaling_analysis._references(saved_run, metadata)
    evaluation, _, _ = scaling_analysis._summarize_phase(
        saved_run,
        metadata,
        references,
        "evaluation",
        scaling_analysis._expected_evaluation(metadata, selection),
    )
    points = scaling_analysis._target_points(selection, evaluation, metadata["config"])

    target_values, _ = scaling_plots._series(
        [10, 20], evaluation, points, "target95", "mean_nodes"
    )
    scan_values, _ = scaling_plots._series(
        [10, 20], evaluation, points, "scan", "mean_nodes"
    )
    assert target_values == [4.0, 4.0]
    assert scan_values == [10.0, 20.0]


def test_limited_evaluation_scan_stays_visible_without_a_crossover(saved_run):
    scaling_analysis.freeze_selection(saved_run)
    schedule = json.loads((saved_run / "schedules" / "evaluation.json").read_text())
    scan_case = next(
        relative
        for relative in schedule
        if "n10-scan-b0-r0/case.json" in relative
    )
    summary_path = saved_run / Path(scan_case).parent / "summary.json"
    summary = json.loads(summary_path.read_text())
    summary.update(status="timeout", completed_queries=0, error="timeout")
    _write_json(summary_path, summary)

    report = scaling_analysis.analyze_run(saved_run)

    assert report.is_file()
    assert "No observed median crossover" in report.read_text()
    with (report.parent / "failures.csv").open() as stream:
        failures = list(csv.DictReader(stream))
    assert any(
        row["phase"] == "evaluation"
        and row["method"] == "scan"
        and row["status"] == "timeout"
        for row in failures
    )


@pytest.mark.filterwarnings("error")
def test_analysis_writes_fixed_outputs_counts_once_and_survives_a_portable_copy(
    saved_run, tmp_path
):
    scaling_analysis.freeze_selection(saved_run)
    copied = tmp_path / "copied"
    shutil.copytree(saved_run, copied)
    for path in copied.glob("cases/*/*/queries.jsonl"):
        compressed = path.with_suffix(path.suffix + ".gz")
        with gzip.open(compressed, "wb") as stream:
            stream.write(path.read_bytes())
        path.unlink()

    report = scaling_analysis.analyze_run(copied)

    assert report == copied / "analysis" / "report.md"
    assert report.is_file()
    expected = {
        "settings.csv",
        "target-points.csv",
        "failures.csv",
        "query-summary.csv.gz",
        "scaling.png",
        "scaling.svg",
        "speedup.png",
        "work.png",
        "budgets.png",
        "report.md",
        "metadata.json",
    }
    assert expected <= {path.name for path in report.parent.iterdir()}
    provenance = json.loads((report.parent / "metadata.json").read_text())
    assert set(provenance["analysis_source_sha256"]) == {
        "scaling_analysis.py",
        "scaling_plots.py",
    }
    text = report.read_text()
    assert "2 evaluation queries" in text
    assert "No observed median crossover" in text
    assert "No observed p95 crossover" in text
    assert "shared Index" in text
    assert "split bitmap words" in text
    with (report.parent / "settings.csv").open() as stream:
        settings = list(csv.DictReader(stream))
    limited = next(
        row
        for row in settings
        if row["phase"] == "development"
        and row["pool_size"] == "20"
        and row["node_budget"] == "2"
    )
    assert limited["status"] == "timeout"
    with (report.parent / "target-points.csv").open() as stream:
        points = list(csv.DictReader(stream))
    shared = [row for row in points if row["pool_size"] == "10"]
    assert len({row["measurement_key"] for row in shared}) == 1
    assert all(row["achieved"] == "False" for row in shared)
    with gzip.open(report.parent / "query-summary.csv.gz", "rt", newline="") as stream:
        query_rows = list(csv.DictReader(stream))
    physical = [
        row
        for row in query_rows
        if row["pool_size"] == "10" and row["method"] == "branch" and row["node_budget"] == "4"
    ]
    assert len(physical) == 2
    assert all(row["labels"] == "target95|target99" for row in physical)
    provenance = json.loads((report.parent / "metadata.json").read_text())
    assert set(provenance["analysis_source_sha256"]) == {
        "scaling_analysis.py",
        "scaling_plots.py",
    }
