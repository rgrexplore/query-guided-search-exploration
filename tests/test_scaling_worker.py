import json
from pathlib import Path

import numpy as np
import pytest

import scaling_worker
from scaling_config import load_config
from scaling_worker import CorrectnessError, run_case, verify_native_sample


def pack(bit_rows):
    bits = np.asarray(bit_rows, dtype=np.uint64)
    codes = np.zeros((len(bits), (bits.shape[1] + 63) // 64), dtype=np.uint64)
    for dimension in range(bits.shape[1]):
        codes[:, dimension // 64] |= bits[:, dimension] << np.uint64(dimension % 64)
    return codes


def write_run(tmp_path, codes, queries, *, candidate_limit=5, leaf_size=1):
    codes_path = tmp_path / "codes.npy"
    queries_path = tmp_path / "queries.npy"
    np.save(codes_path, codes)
    np.save(queries_path, queries)
    config = load_config(Path("scaling.toml"))
    config["search"] = {
        **config["search"],
        "dimensions": queries.shape[1],
        "candidate_limit": candidate_limit,
        "leaf_size": leaf_size,
    }
    config["measurement"] = {**config["measurement"], "warmup_queries": 1}
    inputs = {
        "schema_version": 1,
        "input_hash": "a" * 64,
        "embedding_hash": "b" * 64,
        "dimensions": queries.shape[1],
        "query_ids": [f"q{row}" for row in range(len(queries))],
        "arrays": {
            "codes": {
                "path": str(codes_path.resolve()),
                "sha256": "c" * 64,
                "shape": list(codes.shape),
                "dtype": "uint64",
            },
            "queries": {
                "path": str(queries_path.resolve()),
                "sha256": "d" * 64,
                "shape": list(queries.shape),
                "dtype": "float32",
            },
        },
    }
    run_path = tmp_path / "metadata.json"
    run_path.write_text(json.dumps({"config": config, "inputs": inputs}))
    return run_path


def case_for(
    tmp_path,
    run_path,
    *,
    name,
    phase="pilot",
    method="scan",
    node_budget=0,
    query_rows=(1, 0),
    reference_path=None,
    pool_size=6,
):
    return {
        "phase": phase,
        "pool_size": pool_size,
        "method": method,
        "node_budget": node_budget,
        "repeat": 0,
        "query_rows": list(query_rows),
        "run_path": str(run_path.resolve()),
        "reference_path": None if reference_path is None else str(reference_path.resolve()),
        "output_dir": str((tmp_path / name).resolve()),
    }


@pytest.fixture
def native_inputs(tmp_path):
    bits = np.array(
        [
            [1] * 64,
            [1] * 64,
            [0] * 64,
            [1, 0] * 32,
            [0, 1] * 32,
            [0] * 63 + [1],
        ]
    )
    queries = np.array(
        [[0.0] * 64, [1.0] * 64],
        dtype=np.float32,
    )
    codes = pack(bits)
    return codes, queries, write_run(tmp_path, codes, queries)


def read_rows(output_dir):
    return [json.loads(line) for line in (output_dir / "queries.jsonl").read_text().splitlines()]


def test_real_scan_and_unlimited_branch_match_reference_with_stable_ties(
    tmp_path, native_inputs
):
    _, _, run_path = native_inputs
    reference_case = case_for(
        tmp_path, run_path, name="reference", phase="reference", query_rows=(0, 1)
    )
    reference_info = run_case(reference_case)
    reference_path = Path(reference_case["output_dir"]) / "reference.npz"

    scan_case = case_for(
        tmp_path, run_path, name="scan", reference_path=reference_path, query_rows=(1, 0)
    )
    branch_case = case_for(
        tmp_path,
        run_path,
        name="branch",
        phase="development",
        method="branch",
        node_budget=0,
        reference_path=reference_path,
        query_rows=(1, 0),
    )
    scan_info = run_case(scan_case)
    branch_info = run_case(branch_case)

    assert reference_info["query_count"] == 2
    assert scan_info["query_count"] == branch_info["query_count"] == 2
    scan_rows = read_rows(Path(scan_case["output_dir"]))
    branch_rows = read_rows(Path(branch_case["output_dir"]))
    assert set(scan_rows[0]) == {
        "phase",
        "pool_size",
        "method",
        "node_budget",
        "repeat",
        "query_row",
        "query_id",
        "api_ms",
        "native_ms",
        "rows",
        "scores",
        "count",
        "recall",
        "nodes",
        "random_nodes",
        "bitplane_words",
        "documents_scored",
        "stop_reason",
    }
    assert [row["query_id"] for row in scan_rows] == ["q1", "q0"]
    assert [row["rows"] for row in scan_rows] == [row["rows"] for row in branch_rows]
    assert scan_rows[1]["rows"] == [0, 1, 2, 3, 4]
    assert scan_rows[0]["scores"][0] == 64.0
    assert scan_rows[0]["scores"][-1] < 0
    assert all(row["recall"] == 1.0 for row in scan_rows + branch_rows)
    verification = json.loads(
        (Path(reference_case["output_dir"]) / "verification.json").read_text()
    )
    assert verification["status"] == "pass"
    assert verification["checked_documents"] == 6
    assert verification["checked_queries"] == 2
    with np.load(reference_path, allow_pickle=False) as reference:
        assert set(reference.files) == {
            "query_rows",
            "query_ids",
            "rows",
            "scores",
            "pool_size",
            "candidate_limit",
        }
        assert reference["query_rows"].shape == (2,)
        assert reference["query_ids"].shape == (2,)
        assert reference["rows"].shape == (2, 5)
        assert reference["scores"].shape == (2, 5)
    worker = json.loads((Path(scan_case["output_dir"]) / "worker.json").read_text())
    assert set(worker) == {"setup_ms", "wall_ms", "native_peak_rss_bytes"}


def test_reference_verification_uses_only_the_selected_pool(tmp_path, native_inputs):
    _, _, run_path = native_inputs
    reference_case = case_for(
        tmp_path,
        run_path,
        name="small-reference",
        phase="reference",
        query_rows=(0,),
        pool_size=5,
    )

    run_case(reference_case)

    verification = json.loads(
        (Path(reference_case["output_dir"]) / "verification.json").read_text()
    )
    assert verification["checked_documents"] == 5


def test_independent_score_check_detects_a_changed_native_score(monkeypatch, native_inputs):
    codes, queries, _ = native_inputs
    healthy = verify_native_sample(codes, queries, 64, candidate_limit=5)
    native_index = scaling_worker.bitplane_index.Index

    class ChangedScoreIndex:
        def __init__(self, *args):
            self.index = native_index(*args)

        def scan(self, *args, **kwargs):
            result = self.index.scan(*args, **kwargs)
            result["scores"][0, 0] += 1.0
            return result

    monkeypatch.setattr(scaling_worker.bitplane_index, "Index", ChangedScoreIndex)

    assert healthy["status"] == "pass"
    with pytest.raises(CorrectnessError, match="independent score"):
        verify_native_sample(codes, queries, 64, candidate_limit=5)


def test_timer_wraps_only_the_native_call():
    events = []
    times = iter([10.0, 10.007])

    class RecordingIndex:
        def scan(self, *args, **kwargs):
            events.append("native")
            return {"result": "returned"}

    def clock():
        events.append("clock")
        return next(times)

    result, api_ms = scaling_worker._timed_native_call(
        RecordingIndex(),
        "scan",
        np.zeros((1, 64), np.float32),
        np.array([[0]], np.int64),
        candidate_limit=5,
        node_budget=0,
        leaf_size=1,
        explore_probability=0.0,
        seed=42,
        clock=clock,
    )
    events.append("logging")

    assert result == {"result": "returned"}
    assert api_ms == pytest.approx(7.0)
    assert events == ["clock", "native", "clock", "logging"]


def test_finite_budget_empty_result_writes_strict_json(tmp_path, native_inputs):
    _, _, run_path = native_inputs
    reference_case = case_for(
        tmp_path, run_path, name="empty-reference", phase="reference", query_rows=(0,)
    )
    run_case(reference_case)
    result_case = case_for(
        tmp_path,
        run_path,
        name="empty-result",
        method="branch",
        node_budget=1,
        query_rows=(0,),
        reference_path=Path(reference_case["output_dir"]) / "reference.npz",
    )

    run_case(result_case)
    text = (Path(result_case["output_dir"]) / "queries.jsonl").read_text()
    row = json.loads(text, parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)))

    assert row["rows"] == []
    assert row["scores"] == []
    assert row["count"] == 0
    assert row["recall"] == 0.0


def test_branch_seed_uses_each_original_query_row(monkeypatch, tmp_path, native_inputs):
    _, _, run_path = native_inputs
    reference_case = case_for(
        tmp_path, run_path, name="seed-reference", phase="reference", query_rows=(0, 1)
    )
    run_case(reference_case)
    seeds = []

    class RecordingIndex:
        def __init__(self, codes, assignments, dimensions):
            self.count = min(5, len(codes))

        def search(self, queries, buckets, candidate_limit, **settings):
            seeds.append(settings["seed"])
            count = self.count
            return {
                "rows": np.array([list(range(count)) + [-1] * (candidate_limit - count)], np.int64),
                "scores": np.array([[0.0] * count + [-np.inf] * (candidate_limit - count)]),
                "counts": np.array([count], np.int64),
                "stats": [{
                    "elapsed_ms": 0.1,
                    "nodes": 1,
                    "random_nodes": 0,
                    "bitplane_words": 1,
                    "documents_scored": count,
                    "stop_reason": "exhausted",
                }],
            }

    monkeypatch.setattr(scaling_worker.bitplane_index, "Index", RecordingIndex)
    result_case = case_for(
        tmp_path,
        run_path,
        name="seed-result",
        method="branch",
        node_budget=512,
        query_rows=(1, 0),
        reference_path=Path(reference_case["output_dir"]) / "reference.npz",
    )

    run_case(result_case)

    assert seeds == [43, 43, 42]


def test_reference_requires_matching_pool_and_one_matching_query_id(
    tmp_path, native_inputs
):
    _, _, run_path = native_inputs
    reference_case = case_for(
        tmp_path, run_path, name="mapping-reference", phase="reference", query_rows=(0, 1)
    )
    run_case(reference_case)
    reference_path = Path(reference_case["output_dir"]) / "reference.npz"
    healthy_case = case_for(
        tmp_path, run_path, name="mapping-healthy", query_rows=(1,), reference_path=reference_path
    )

    run_case(healthy_case)

    with np.load(reference_path, allow_pickle=False) as reference:
        saved = {name: reference[name] for name in reference.files}
    wrong_pool = tmp_path / "wrong-pool.npz"
    np.savez(wrong_pool, **{**saved, "pool_size": np.array(5, np.int64)})
    wrong_id = tmp_path / "wrong-id.npz"
    ids = saved["query_ids"].copy()
    ids[1] = "not-q1"
    np.savez(wrong_id, **{**saved, "query_ids": ids})

    with pytest.raises(CorrectnessError, match="pool_size"):
        run_case(
            case_for(
                tmp_path,
                run_path,
                name="wrong-pool",
                query_rows=(1,),
                reference_path=wrong_pool,
            )
        )
    with pytest.raises(CorrectnessError, match="query ID"):
        run_case(
            case_for(
                tmp_path,
                run_path,
                name="wrong-id",
                query_rows=(1,),
                reference_path=wrong_id,
            )
        )
