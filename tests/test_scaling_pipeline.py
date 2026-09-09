import hashlib
import json
import shutil
import subprocess
import time
from pathlib import Path

import numpy as np
import psutil
import pytest

import scaling
import scaling_cache


def _write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def _config_text(cache_dir: Path) -> str:
    return f'''[data]
cache_dir = "{cache_dir}"
max_documents = 8
pool_sizes = [6, 8]
document_seed = 42
query_seed = 43
development_queries = 2
evaluation_queries = 2

[embedding]
model = "nomic-ai/nomic-embed-text-v1.5"
revision = "e9b6763023c676ca8431644204f50c2b100d9aab"
batch_size = 2
max_length = 64
device = "auto"
chunk_size = 3

[search]
dimensions = 64
candidate_limit = 2
leaf_size = 1
explore_probability = 0.0
seed = 42

[tuning]
targets = [0.95, 0.99]
node_budgets = [1, 4, 0]

[measurement]
repetitions = 3
warmup_queries = 0
schedule_seed = 44
bootstrap_samples = 20
bootstrap_seed = 45

[limits]
case_seconds = 10
rss_gib = 1
'''


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _selection(tmp_path: Path) -> dict:
    folder = tmp_path / "selection"
    folder.mkdir(parents=True)
    documents = folder / "documents.jsonl"
    queries = folder / "queries.jsonl"
    documents.write_text(
        "".join(
            json.dumps({"document_id": f"d{row}", "text": f"document {row}"}) + "\n"
            for row in range(8)
        )
    )
    queries.write_text(
        "".join(
            json.dumps({"query_id": f"q{row}", "text": f"query {row}"}) + "\n"
            for row in range(4)
        )
    )
    manifest_path = folder / "selection.json"
    return {
        "schema_version": 1,
        "selection_hash": "fixture-selection",
        "identity": {},
        "source": {},
        "parameters": {
            "max_documents": 8,
            "document_seed": 42,
            "query_seed": 43,
            "development_queries": 2,
            "evaluation_queries": 2,
        },
        "document_count": 8,
        "query_count": 4,
        "query_ids": ["q0", "q1", "q2", "q3"],
        "development_rows": [0, 1],
        "evaluation_rows": [2, 3],
        "files": {
            "documents": {"path": str(documents.resolve()), "sha256": _sha256(documents)},
            "queries": {"path": str(queries.resolve()), "sha256": _sha256(queries)},
            "manifest": {"path": str(manifest_path.resolve()), "sha256": "unused"},
        },
    }


class FakeEncoder:
    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def encode_prefixed(self, texts: list[str]) -> np.ndarray:
        base = np.arange(768, dtype=np.float32)
        return np.stack(
            [np.sin(base * ((sum(map(ord, text)) % 23 + 1) / 31.0)) for text in texts]
        ).astype(np.float32)


def test_phase_plan_has_global_case_order_and_one_query_order_per_repeat(tmp_path):
    config = {
        "data": {"pool_sizes": [10, 20, 30]},
        "tuning": {"node_budgets": [2, 4, 0]},
        "measurement": {"repetitions": 3, "schedule_seed": 44},
    }
    metadata = {
        "config": config,
        "query_splits": {"development": list(range(8)), "evaluation": list(range(8, 16))},
    }

    cases = scaling.plan_cases(tmp_path, metadata, "development")

    for repeat in range(3):
        repeated = [case for case in cases if case["repeat"] == repeat]
        assert len({tuple(case["query_rows"]) for case in repeated}) == 1
        assert {case["pool_size"] for case in repeated[:4]} != {10}
    assert scaling.plan_cases(tmp_path, metadata, "development") == cases


def test_macos_power_observation_distinguishes_active_low_power_mode(monkeypatch):
    monkeypatch.setattr(scaling.sys, "platform", "darwin")
    battery = "Now drawing from 'AC Power'\n"

    def observation(value: int):
        outputs = iter(
            [
                subprocess.CompletedProcess([], 0, battery, ""),
                subprocess.CompletedProcess(
                    [],
                    0,
                    f"Battery Power:\n lowpowermode 1\nAC Power:\n lowpowermode {value}\n",
                    "",
                ),
            ]
        )
        monkeypatch.setattr(scaling.subprocess, "run", lambda *args, **kwargs: next(outputs))
        return scaling.observe_power()

    normal = observation(0)
    low = observation(1)

    assert normal == {"source": "AC Power", "mode": "lowpowermode=0"}
    assert low == {"source": "AC Power", "mode": "lowpowermode=1"}
    assert not scaling._same_power(normal, low)


def _fake_worker(path: Path) -> None:
    path.write_text(
        '''import json, os, sys, time
from pathlib import Path
case = json.loads(Path(sys.argv[1]).read_text())
out = Path(case["output_dir"])
out.mkdir(parents=True, exist_ok=True)
(out / "pid.txt").write_text(str(os.getpid()))
if case["phase"] == "slow":
    row = {"query_row": case["query_rows"][0], "query_id": "q0"}
    (out / "queries.jsonl").write_text(json.dumps(row) + "\\n")
    time.sleep(30)
else:
    rows = [{"query_row": row, "query_id": f"q{row}"} for row in case["query_rows"]]
    (out / "queries.jsonl").write_text("".join(json.dumps(row) + "\\n" for row in rows))
    worker = {"setup_ms": 1.0, "wall_ms": 2.0, "native_peak_rss_bytes": 123}
    (out / "worker.json").write_text(json.dumps(worker))
'''
    )


def _process_case(tmp_path: Path, phase: str) -> Path:
    output = tmp_path / phase
    metadata = tmp_path / "metadata.json"
    _write_json(metadata, {"inputs": {"query_ids": ["q0", "q1"]}})
    case = {
        "phase": phase,
        "pool_size": 2,
        "method": "scan",
        "node_budget": 0,
        "repeat": 0,
        "query_rows": [0, 1],
        "run_path": str(metadata.resolve()),
        "reference_path": None,
        "output_dir": str(output.resolve()),
    }
    path = output / "case.json"
    _write_json(path, case)
    return path


def test_owned_child_timeout_retains_partial_rows_and_healthy_child_completes(
    tmp_path, monkeypatch
):
    worker = tmp_path / "fake_worker.py"
    _fake_worker(worker)
    monkeypatch.setattr(scaling, "WORKER_SCRIPT", worker)
    monkeypatch.setattr(scaling, "POLL_SECONDS", 0.01)
    monkeypatch.setattr(scaling, "observe_power", lambda: {"source": "fixed", "mode": "fixed"})

    slow_path = _process_case(tmp_path, "slow")
    slow = scaling.run_attempt(slow_path, {"case_seconds": 0.1, "rss_gib": 1})
    pid = int((slow_path.parent / "pid.txt").read_text())
    for _ in range(50):
        if not psutil.pid_exists(pid):
            break
        time.sleep(0.01)

    healthy_path = _process_case(tmp_path, "healthy")
    healthy = scaling.run_attempt(healthy_path, {"case_seconds": 2, "rss_gib": 1})

    assert slow["status"] == "timeout"
    assert slow["completed_queries"] == 1
    assert (slow_path.parent / "queries.jsonl").is_file()
    assert not psutil.pid_exists(pid)
    assert healthy["status"] == "complete"
    assert healthy["completed_queries"] == 2


def test_power_change_stops_the_owned_child_beside_a_constant_power_control(
    tmp_path, monkeypatch
):
    worker = tmp_path / "fake_worker.py"
    _fake_worker(worker)
    monkeypatch.setattr(scaling, "WORKER_SCRIPT", worker)
    monkeypatch.setattr(scaling, "POLL_SECONDS", 0.01)
    monkeypatch.setattr(scaling, "POWER_SAMPLE_SECONDS", 0.01)
    constant = {"source": "AC Power", "mode": "automatic"}
    monkeypatch.setattr(scaling, "observe_power", lambda: constant)
    healthy = scaling.run_attempt(
        _process_case(tmp_path, "healthy"), {"case_seconds": 2, "rss_gib": 1}, constant
    )

    observations = iter([constant, {"source": "Battery Power", "mode": "low"}])
    monkeypatch.setattr(scaling, "observe_power", lambda: next(observations))
    with pytest.raises(scaling.PowerChanged):
        scaling.run_attempt(
            _process_case(tmp_path, "slow"), {"case_seconds": 2, "rss_gib": 1}, constant
        )

    assert healthy["status"] == "complete"
    stopped = json.loads((tmp_path / "slow" / "summary.json").read_text())
    assert stopped["status"] == "interrupted"
    observations_path = tmp_path / "power-observations.json"
    assert any(row["point"] == "case" for row in json.loads(observations_path.read_text()))


def test_limited_reference_blocks_pilot_progress(tmp_path, monkeypatch):
    expected = {"source": "fixed", "mode": "fixed"}
    monkeypatch.setattr(scaling, "observe_power", lambda: expected)
    monkeypatch.setattr(
        scaling,
        "execute_case",
        lambda *args, **kwargs: {
            "status": "timeout",
            "completed_queries": 0,
            "setup_ms": None,
            "wall_ms": 1.0,
            "sampled_peak_rss_bytes": 1,
            "native_peak_rss_bytes": None,
        },
    )
    case = {
        "phase": "reference",
        "output_dir": str(tmp_path / "cases/reference/n10-scan-b0-r0"),
    }

    with pytest.raises(RuntimeError, match="reference.*timeout"):
        scaling._execute_phase(
            tmp_path,
            "reference",
            [case],
            {"case_seconds": 1, "rss_gib": 1},
            False,
            expected,
        )


def test_resume_keeps_finished_case_and_archives_interrupted_attempt(tmp_path, monkeypatch):
    run_dir = tmp_path / "run"
    case = {
        "phase": "pilot",
        "pool_size": 10,
        "method": "scan",
        "node_budget": 0,
        "repeat": 0,
        "query_rows": [0],
        "run_path": str((run_dir / "metadata.json").resolve()),
        "reference_path": str((run_dir / "reference.npz").resolve()),
        "output_dir": str((run_dir / "cases/pilot/n10-scan-b0-r0").resolve()),
    }
    case_path = Path(case["output_dir"]) / "case.json"
    _write_json(run_dir / "metadata.json", {"inputs": {"query_ids": ["q0"]}})
    _write_json(case_path, case)
    _write_json(
        case_path.parent / "summary.json",
        {"status": "complete", "completed_queries": 1},
    )
    (case_path.parent / "queries.jsonl").write_text('{"query_row":0,"query_id":"q0"}\n')
    calls = []
    monkeypatch.setattr(scaling, "run_attempt", lambda *args, **kwargs: calls.append(args) or {})

    scaling.execute_case(run_dir, case, {"case_seconds": 1, "rss_gib": 1}, resume=True)
    assert calls == []

    summary = json.loads((case_path.parent / "summary.json").read_text())
    summary["status"] = "interrupted"
    _write_json(case_path.parent / "summary.json", summary)
    scaling.execute_case(run_dir, case, {"case_seconds": 1, "rss_gib": 1}, resume=True)

    assert len(calls) == 1
    assert list((run_dir / "interrupted").glob("n10-scan-b0-r0-*"))
    assert case_path.is_file()


@pytest.mark.filterwarnings("error")
def test_fixture_prepare_pilot_tune_evaluate_report_and_resume_identity_checks(
    tmp_path, monkeypatch
):
    cache_dir = tmp_path / "cache"
    config_path = tmp_path / "scaling.toml"
    config_path.write_text(_config_text(cache_dir))
    selection = _selection(tmp_path)
    selection_path = Path(selection["files"]["manifest"]["path"])
    _write_json(selection_path, selection)
    monkeypatch.setattr(scaling.msmarco, "prepare_selection", lambda **kwargs: selection)
    monkeypatch.setattr(scaling_cache, "NomicEncoder", FakeEncoder)
    monkeypatch.setattr(scaling_cache, "choose_device", lambda requested: "cpu")
    monkeypatch.setattr(scaling_cache, "resolve_revision", lambda model, revision: revision)
    monkeypatch.setattr(
        scaling_cache,
        "package_versions",
        lambda: {"numpy": "test", "sentence-transformers": "test", "torch": "test"},
    )
    monkeypatch.setattr(scaling, "observe_power", lambda: {"source": "test", "mode": "test"})
    monkeypatch.setattr(scaling, "POLL_SECONDS", 0.01)
    monkeypatch.setattr(scaling, "POWER_SAMPLE_SECONDS", 0.05)

    scaling.run_stage(config_path, "prepare", prepare_check=True)
    assert not (cache_dir / "prepared.json").exists()
    pointer_path = scaling.run_stage(config_path, "prepare")
    pointer = json.loads(pointer_path.read_text())
    assert set(pointer) == {
        "selection_manifest_path",
        "full_manifest_path",
        "derived_manifest_path",
    }

    run_dir = tmp_path / "run"
    scaling.run_stage(config_path, "pilot", run_dir)
    scaling.run_stage(config_path, "tune", run_dir)
    scaling.run_stage(config_path, "evaluate", run_dir)
    report = scaling.run_stage(config_path, "report", run_dir)

    metadata = json.loads((run_dir / "metadata.json").read_text())
    assert set(metadata) >= {
        "config",
        "inputs",
        "query_splits",
        "run_fingerprint",
        "provenance",
        "manifest_paths",
    }
    assert metadata["inputs"]["query_ids"] == ["q0", "q1", "q2", "q3"]
    assert report == run_dir / "analysis" / "report.md"
    assert report.is_file()
    frozen = json.loads((run_dir / "selections.json").read_text())
    assert len(frozen["targets"]) == 4
    assert {row["node_budget"] for row in frozen["targets"]} == {0}
    evaluation = json.loads((run_dir / "schedules/evaluation.json").read_text())
    reference = json.loads((run_dir / "schedules/reference.json").read_text())
    pilot = json.loads((run_dir / "schedules/pilot.json").read_text())
    development = json.loads((run_dir / "schedules/development.json").read_text())
    assert len(reference) == 2
    assert len(pilot) == 2 * 3
    assert len(development) == 2 * 4 * 3
    assert len(evaluation) == 2 * 2 * 3
    assert len(evaluation) == len(set(evaluation))
    unlimited_case = next(relative for relative in evaluation if "branch-b0-r0" in relative)
    unlimited_rows = [
        json.loads(line)
        for line in (run_dir / Path(unlimited_case).parent / "queries.jsonl")
        .read_text()
        .splitlines()
    ]
    assert {row["query_id"] for row in unlimited_rows} == {"q2", "q3"}
    assert all(row["recall"] == 1.0 and row["count"] == 2 for row in unlimited_rows)

    scaling.run_stage(config_path, "tune", run_dir, resume=True)

    original_metadata = json.loads((run_dir / "metadata.json").read_text())
    changed = json.loads((run_dir / "metadata.json").read_text())
    changed["provenance"]["source_sha256"]["scaling.py"] = "changed"
    _write_json(run_dir / "metadata.json", changed)
    with pytest.raises(ValueError, match="source"):
        scaling.run_stage(config_path, "tune", run_dir, resume=True)
    _write_json(run_dir / "metadata.json", original_metadata)

    changed_config = tmp_path / "changed.toml"
    changed_config.write_text(
        _config_text(cache_dir).replace(
            "targets = [0.95, 0.99]", "targets = [0.94, 0.99]"
        )
    )
    with pytest.raises(ValueError, match="config"):
        scaling.run_stage(changed_config, "tune", run_dir, resume=True)

    codes_path = Path(metadata["inputs"]["arrays"]["codes"]["path"])
    saved_codes = codes_path.read_bytes()
    damaged = bytearray(saved_codes)
    damaged[-1] ^= 1
    codes_path.write_bytes(damaged)
    with pytest.raises(ValueError, match="input"):
        scaling.run_stage(config_path, "tune", run_dir, resume=True)
    codes_path.write_bytes(saved_codes)

    copied = tmp_path / "archived"
    shutil.copytree(run_dir, copied)
    shutil.rmtree(cache_dir)
    assert scaling.run_stage(config_path, "report", copied) == copied / "analysis/report.md"


def test_power_change_is_seen_between_short_cases(tmp_path, monkeypatch):
    normal = {"source": "AC Power", "mode": "lowpowermode=2"}
    low = {"source": "AC Power", "mode": "lowpowermode=1"}

    def run_phase(change):
        elapsed = 0

        def observe():
            return low if change and 10 <= elapsed < 40 else normal

        def execute(*args, **kwargs):
            nonlocal elapsed
            elapsed += 10  # Each case is shorter than the 15-second sampling interval.
            return {"status": "complete"}

        monkeypatch.setattr(scaling, "observe_power", observe)
        monkeypatch.setattr(scaling, "execute_case", execute)
        directory = tmp_path / ("changed" if change else "healthy")
        scaling._execute_phase(directory, "development", [{}] * 5, {}, False, normal)

    run_phase(False)
    with pytest.raises(scaling.PowerChanged):
        run_phase(True)
    assert (tmp_path / "changed/invalid-run.json").is_file()


def test_report_uses_saved_settings_with_a_copied_config(tmp_path, monkeypatch):
    original = tmp_path / "original.toml"
    original.write_text(_config_text(Path("cache")))
    run_dir = tmp_path / "run"
    _write_json(run_dir / "metadata.json", {"config": scaling._resolved_config(original)})
    copied_config = run_dir / "config.toml"
    shutil.copyfile(original, copied_config)
    report = run_dir / "analysis/report.md"
    monkeypatch.setattr(scaling.scaling_analysis, "analyze_run", lambda directory: report)

    assert scaling.run_stage(original, "report", run_dir) == report
    assert scaling.run_stage(copied_config, "report", run_dir) == report
