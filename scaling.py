"""Run the five stages of the fixed scan-versus-bitplane study."""

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import random
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import bitplane_index
import psutil

import msmarco
import scaling_analysis
import scaling_cache
from scaling_config import StudyConfig, identity_hash, load_config


WORKER_SCRIPT = Path(__file__).with_name("scaling_worker.py")
PROTOCOL_PATH = Path(__file__).with_name("docs") / "scaling-protocol-v2.md"
POLL_SECONDS = 0.25
POWER_SAMPLE_SECONDS = 15.0
TERMINATE_GRACE_SECONDS = 2.0
SOURCE_PATHS = (
    Path(__file__),
    Path(__file__).with_name("scaling_config.py"),
    Path(__file__).with_name("scaling_worker.py"),
    Path(__file__).with_name("scaling_analysis.py"),
    Path(__file__).with_name("cpp") / "index.cpp",
    Path(__file__).with_name("cpp") / "index.hpp",
    Path(__file__).with_name("cpp") / "bindings.cpp",
)


class PowerChanged(RuntimeError):
    """The observed power source or mode changed during a measured phase."""

    def __init__(self, expected: dict, observed: dict):
        self.expected = expected
        self.observed = observed
        super().__init__(f"power changed from {expected} to {observed}")


def _write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _resolved_config(config_path: Path) -> StudyConfig:
    config_path = Path(config_path).resolve()
    config = load_config(config_path)
    config["data"]["cache_dir"] = str(
        (config_path.parent / config["data"]["cache_dir"]).resolve()
    )
    return config


def _manifest_path_for_array(record: dict) -> Path:
    return Path(record["path"]).resolve().parent / "manifest.json"


def _prepare(config: StudyConfig, prepare_check: bool) -> Path:
    data = config["data"]
    cache_dir = Path(data["cache_dir"])
    selection = msmarco.prepare_selection(
        cache_dir=cache_dir,
        max_documents=data["max_documents"],
        document_seed=data["document_seed"],
        query_seed=data["query_seed"],
        development_queries=data["development_queries"],
        evaluation_queries=data["evaluation_queries"],
    )
    full = scaling_cache.prepare_cache(selection, config, check_only=prepare_check)
    if full["state"] != "complete":
        return Path(full["cache_dir"])
    derived = scaling_cache.derive_cache(full, config["search"]["dimensions"])
    pointer = {
        "selection_manifest_path": str(Path(full["selection_manifest_path"]).resolve()),
        "full_manifest_path": str(_manifest_path_for_array(full["arrays"]["documents"])),
        "derived_manifest_path": str(_manifest_path_for_array(derived["arrays"]["codes"])),
    }
    pointer_path = cache_dir / "prepared.json"
    _write_json(pointer_path, pointer)
    return pointer_path


def _check_array(record: dict, name: str) -> None:
    path = Path(record.get("path", ""))
    if not path.is_absolute() or not path.is_file() or _sha256(path) != record.get("sha256"):
        raise ValueError(f"input {name} does not match its prepared manifest")


def _load_prepared(config: StudyConfig) -> tuple[dict, dict, dict, dict]:
    pointer_path = Path(config["data"]["cache_dir"]) / "prepared.json"
    if not pointer_path.is_file():
        raise ValueError("prepared.json is missing; run the prepare stage first")
    pointer = _read_json(pointer_path)
    required = {
        "selection_manifest_path",
        "full_manifest_path",
        "derived_manifest_path",
    }
    if set(pointer) != required:
        raise ValueError("prepared.json has the wrong fields")
    selection = _read_json(Path(pointer["selection_manifest_path"]))
    full = _read_json(Path(pointer["full_manifest_path"]))
    derived = _read_json(Path(pointer["derived_manifest_path"]))
    data = config["data"]
    expected_parameters = {
        "max_documents": data["max_documents"],
        "document_seed": data["document_seed"],
        "query_seed": data["query_seed"],
        "development_queries": data["development_queries"],
        "evaluation_queries": data["evaluation_queries"],
    }
    if selection.get("parameters") != expected_parameters:
        raise ValueError("prepared selection does not match the config")
    embedding = config["embedding"]
    full_identity = full.get("identity", {})
    for field in ("model", "revision", "batch_size", "max_length", "chunk_size"):
        if full_identity.get(field) != embedding[field]:
            raise ValueError(f"prepared embedding {field} does not match the config")
    if embedding["device"] != "auto" and full_identity.get("device") != embedding["device"]:
        raise ValueError("prepared embedding device does not match the config")
    if (
        full.get("selection_hash") != selection.get("selection_hash")
        or derived.get("embedding_hash") != full.get("embedding_hash")
        or derived.get("dimensions") != config["search"]["dimensions"]
        or derived.get("query_ids") != selection.get("query_ids")
    ):
        raise ValueError("prepared model, selection, or search dimensions do not match")
    codes = derived.get("arrays", {}).get("codes", {})
    queries = derived.get("arrays", {}).get("queries", {})
    if codes.get("shape", [0])[0] != data["max_documents"]:
        raise ValueError("prepared document count does not match the config")
    if queries.get("shape", [0])[0] != data["development_queries"] + data["evaluation_queries"]:
        raise ValueError("prepared query count does not match the config")
    _check_array(codes, "codes")
    _check_array(queries, "queries")
    return pointer, selection, full, derived


def _source_hashes() -> dict[str, str]:
    root = Path(__file__).parent
    return {str(path.relative_to(root)): _sha256(path) for path in SOURCE_PATHS}


def _git_commit() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=Path(__file__).parent,
        capture_output=True,
        text=True,
        check=False,
    )
    return result.stdout.strip() if result.returncode == 0 else "unavailable"


def _package_versions() -> dict[str, str]:
    versions = {"python": platform.python_version()}
    for name in ("numpy", "psutil", "threadpoolctl"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = "unavailable"
    return versions


def observe_power() -> dict[str, str]:
    """Read the active macOS power source and its current power mode."""
    unavailable = {"source": "unavailable", "mode": "unavailable"}
    if sys.platform != "darwin":
        return unavailable
    battery = subprocess.run(
        ["pmset", "-g", "batt"], capture_output=True, text=True, check=False
    )
    settings = subprocess.run(
        ["pmset", "-g", "custom"], capture_output=True, text=True, check=False
    )
    if battery.returncode != 0 or settings.returncode != 0:
        return unavailable
    source = "unavailable"
    for line in battery.stdout.splitlines():
        if "Now drawing from" in line and "'" in line:
            source = line.split("'", 2)[1]
            break
    wanted_section = "AC Power" if source.startswith("AC") else "Battery Power"
    active = False
    mode = "unavailable"
    for line in settings.stdout.splitlines():
        stripped = line.strip()
        if stripped.endswith("Power:"):
            active = stripped[:-1] == wanted_section
        elif active and stripped.startswith(("lowpowermode ", "powermode ")):
            key, value = stripped.split(maxsplit=1)
            mode = f"{key}={value}"
            break
    return {"source": source, "mode": mode}


def _same_power(left: dict, right: dict) -> bool:
    return (left.get("source"), left.get("mode")) == (
        right.get("source"),
        right.get("mode"),
    )


def _provenance(initial_power: dict) -> dict:
    binary_path = Path(bitplane_index.__file__).resolve()
    return {
        "hardware": {
            "platform": platform.platform(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "cpu_count": os.cpu_count(),
        },
        "packages": _package_versions(),
        "git_commit": _git_commit(),
        "source_sha256": _source_hashes(),
        "native_binary": {"path": str(binary_path), "sha256": _sha256(binary_path)},
        "protocol_sha256": _sha256(PROTOCOL_PATH),
        "initial_power": initial_power,
    }


def _fingerprint(config: dict, inputs: dict, provenance: dict) -> str:
    return identity_hash(
        {
            "config": config,
            "input_hash": inputs["input_hash"],
            "input_array_sha256": {
                name: record["sha256"] for name, record in inputs["arrays"].items()
            },
            "source_sha256": provenance["source_sha256"],
            "native_binary_sha256": provenance["native_binary"]["sha256"],
            "protocol_sha256": provenance["protocol_sha256"],
        }
    )


def _create_run(
    config_path: Path,
    config: StudyConfig,
    run_dir: Path,
    pointer: dict,
    selection: dict,
    inputs: dict,
) -> dict:
    if run_dir.exists():
        raise ValueError("pilot requires a new run directory; use --resume for an existing run")
    run_dir.mkdir(parents=True)
    initial_power = observe_power()
    provenance = _provenance(initial_power)
    metadata = {
        "config": config,
        "inputs": inputs,
        "query_splits": {
            "development": selection["development_rows"],
            "evaluation": selection["evaluation_rows"],
        },
        "run_fingerprint": _fingerprint(config, inputs, provenance),
        "provenance": provenance,
        "manifest_paths": pointer,
    }
    _write_json(run_dir / "metadata.json", metadata)
    shutil.copyfile(config_path, run_dir / "config.toml")
    shutil.copyfile(PROTOCOL_PATH, run_dir / "protocol.md")
    return metadata


def _validate_run(config: StudyConfig, run_dir: Path, inputs: dict) -> dict:
    metadata = _read_json(run_dir / "metadata.json")
    if metadata.get("config") != config:
        raise ValueError("run config does not match the requested config")
    provenance = metadata.get("provenance", {})
    if provenance.get("source_sha256") != _source_hashes():
        raise ValueError("run source files do not match the measured source")
    binary_path = Path(bitplane_index.__file__).resolve()
    if provenance.get("native_binary", {}).get("sha256") != _sha256(binary_path):
        raise ValueError("run native binary does not match the measured input")
    if provenance.get("protocol_sha256") != _sha256(PROTOCOL_PATH):
        raise ValueError("run protocol does not match the measured source")
    for name, record in inputs["arrays"].items():
        _check_array(record, name)
    if metadata.get("inputs") != inputs:
        raise ValueError("run input manifest does not match prepared inputs")
    if metadata.get("run_fingerprint") != _fingerprint(config, inputs, provenance):
        raise ValueError("run fingerprint does not match config, input, or source")
    return metadata


def _reference_path(run_dir: Path, pool_size: int) -> Path:
    return run_dir / "cases" / "reference" / f"n{pool_size}-scan-b0-r0" / "reference.npz"


def _case(
    run_dir: Path,
    phase: str,
    pool: int,
    method: str,
    budget: int,
    repeat: int,
    query_rows: list[int],
) -> dict:
    name = f"n{pool}-{method}-b{budget}-r{repeat}"
    output_dir = run_dir / "cases" / phase / name
    return {
        "phase": phase,
        "pool_size": pool,
        "method": method,
        "node_budget": budget,
        "repeat": repeat,
        "query_rows": query_rows,
        "run_path": str((run_dir / "metadata.json").resolve()),
        "reference_path": (
            None if phase == "reference" else str(_reference_path(run_dir, pool).resolve())
        ),
        "output_dir": str(output_dir.resolve()),
    }


def plan_cases(
    run_dir: Path, metadata: dict, phase: str, selection: dict | None = None
) -> list[dict]:
    """Create the deterministic case order and shared query order for one phase."""
    config = metadata["config"]
    pools = config["data"]["pool_sizes"]
    if phase == "reference":
        rows = metadata["query_splits"]["development"] + metadata["query_splits"]["evaluation"]
        return [_case(run_dir, phase, pool, "scan", 0, 0, list(rows)) for pool in pools]
    if phase == "pilot":
        chosen_pools = {pools[0], pools[-1]}
        if 100_000 in pools:
            chosen_pools.add(100_000)
        pools = [pool for pool in pools if pool in chosen_pools]
        settings = [("scan", 0), ("branch", 512), ("branch", 0)]
        source_rows = metadata["query_splits"]["development"][:5]
        repetitions = 1
        offset = 0
    elif phase == "development":
        settings = [("scan", 0)] + [
            ("branch", budget) for budget in config["tuning"]["node_budgets"]
        ]
        source_rows = metadata["query_splits"]["development"]
        repetitions = config["measurement"]["repetitions"]
        offset = 1
    elif phase == "evaluation":
        if selection is None:
            raise ValueError("evaluation requires frozen development choices")
        choices: dict[int, set[int]] = {pool: set() for pool in pools}
        for row in selection["targets"]:
            if row["status"] == "selected":
                choices[row["pool_size"]].add(row["node_budget"])
        source_rows = metadata["query_splits"]["evaluation"]
        repetitions = config["measurement"]["repetitions"]
        offset = 2
    else:
        raise ValueError(f"unsupported phase {phase}")

    planned = []
    seed = config["measurement"]["schedule_seed"]
    for repeat in range(repetitions):
        rng = random.Random(seed + offset + repeat)
        query_rows = list(source_rows)
        rng.shuffle(query_rows)
        if phase == "evaluation":
            phase_settings = {
                pool: sorted(
                    {("scan", 0), ("branch", 0)}
                    | {("branch", budget) for budget in choices[pool]},
                    key=lambda value: (value[0], value[1]),
                )
                for pool in pools
            }
            repeated = [
                _case(run_dir, phase, pool, method, budget, repeat, list(query_rows))
                for pool in pools
                for method, budget in phase_settings[pool]
            ]
        else:
            repeated = [
                _case(run_dir, phase, pool, method, budget, repeat, list(query_rows))
                for pool in pools
                for method, budget in settings
            ]
        rng.shuffle(repeated)
        planned.extend(repeated)
    return planned


def _save_plan(run_dir: Path, phase: str, cases: list[dict], resume: bool) -> list[str]:
    relatives = [
        str((Path(case["output_dir"]) / "case.json").relative_to(run_dir)) for case in cases
    ]
    schedule_path = run_dir / "schedules" / f"{phase}.json"
    if schedule_path.exists():
        if not resume:
            raise ValueError(f"{phase} already has a saved schedule; use --resume")
        if _read_json(schedule_path) != relatives:
            raise ValueError(f"saved {phase} schedule does not match the requested run")
    else:
        if resume:
            raise ValueError(f"cannot resume {phase} without its saved schedule")
        _write_json(schedule_path, relatives)
    for case, relative in zip(cases, relatives, strict=True):
        case_path = run_dir / relative
        if case_path.exists() and _read_json(case_path) != case:
            raise ValueError(f"saved case does not match its schedule: {relative}")
        if not case_path.exists():
            _write_json(case_path, case)
    return relatives


def _read_completed_rows(case: dict) -> tuple[int, bool]:
    path = Path(case["output_dir"]) / "queries.jsonl"
    if not path.is_file():
        return 0, False
    query_ids = _read_json(Path(case["run_path"]))["inputs"]["query_ids"]
    found = []
    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            query_row = row.get("query_row")
            if (
                isinstance(query_row, int)
                and 0 <= query_row < len(query_ids)
                and row.get("query_id") == query_ids[query_row]
            ):
                found.append(query_row)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return len(found), False
    expected = case["query_rows"]
    return len(found), len(found) == len(expected) and set(found) == set(expected)


def _stop_child(process: subprocess.Popen) -> None:
    if process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=TERMINATE_GRACE_SECONDS)
    except subprocess.TimeoutExpired:
        if process.poll() is None:
            process.kill()
        process.wait()


def run_attempt(
    case_path: Path, limits: dict, expected_power: dict | None = None
) -> dict:
    """Run and supervise exactly one fixed worker process."""
    case_path = Path(case_path)
    case = _read_json(case_path)
    output_dir = Path(case["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    stdout_path = output_dir / "stdout.txt"
    stderr_path = output_dir / "stderr.txt"
    environment = os.environ.copy()
    thread_variables = (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    )
    for name in thread_variables:
        environment[name] = "1"
    started = time.monotonic()
    peak_rss = 0
    reason = None
    changed_power = None
    last_power_sample = started
    with stdout_path.open("w") as stdout, stderr_path.open("w") as stderr:
        process = subprocess.Popen(
            [sys.executable, str(WORKER_SCRIPT), str(case_path)],
            stdout=stdout,
            stderr=stderr,
            env=environment,
        )
        child = psutil.Process(process.pid)
        try:
            while process.poll() is None:
                now = time.monotonic()
                try:
                    peak_rss = max(peak_rss, child.memory_info().rss)
                except (psutil.NoSuchProcess, psutil.ZombieProcess):
                    pass
                if now - started >= float(limits["case_seconds"]):
                    reason = "timeout"
                    _stop_child(process)
                    break
                if peak_rss > float(limits["rss_gib"]) * 1024**3:
                    reason = "memory_limit"
                    _stop_child(process)
                    break
                if expected_power is not None and now - last_power_sample >= POWER_SAMPLE_SECONDS:
                    observed = observe_power()
                    last_power_sample = now
                    _record_power(
                        Path(case["run_path"]).parent,
                        case["phase"],
                        "case",
                        observed,
                    )
                    if not _same_power(expected_power, observed):
                        reason = "interrupted"
                        changed_power = observed
                        _stop_child(process)
                        break
                time.sleep(POLL_SECONDS)
        except BaseException:
            _stop_child(process)
            completed, _ = _read_completed_rows(case)
            summary = {
                "status": "interrupted",
                "completed_queries": completed,
                "setup_ms": None,
                "wall_ms": (time.monotonic() - started) * 1_000,
                "sampled_peak_rss_bytes": peak_rss,
                "native_peak_rss_bytes": None,
                "error": "parent interrupted",
            }
            _write_json(output_dir / "summary.json", summary)
            raise
        return_code = process.wait()

    completed, exact = _read_completed_rows(case)
    worker_path = output_dir / "worker.json"
    worker = _read_json(worker_path) if worker_path.is_file() else {}
    if reason is not None:
        status = reason
        error = "power changed" if changed_power is not None else reason
    elif return_code != 0:
        error_path = output_dir / "error.json"
        error_record = _read_json(error_path) if error_path.is_file() else {}
        status = error_record.get("status", "worker_error")
        if status not in {"worker_error", "correctness_failure"}:
            status = "worker_error"
        error = error_record.get("message", f"worker exited with status {return_code}")
    elif not exact:
        status = "correctness_failure"
        error = "worker did not write each expected query row and ID exactly once"
    else:
        status = "complete"
        error = None
    summary = {
        "status": status,
        "completed_queries": completed,
        "setup_ms": worker.get("setup_ms"),
        "wall_ms": worker.get("wall_ms", (time.monotonic() - started) * 1_000),
        "sampled_peak_rss_bytes": peak_rss,
        "native_peak_rss_bytes": worker.get("native_peak_rss_bytes"),
    }
    if error is not None:
        summary["error"] = error
    _write_json(output_dir / "summary.json", summary)
    if changed_power is not None:
        raise PowerChanged(expected_power, changed_power)
    return summary


def execute_case(
    run_dir: Path,
    case: dict,
    limits: dict,
    *,
    resume: bool,
    expected_power: dict | None = None,
) -> dict:
    """Keep a valid terminal case or archive one incomplete attempt and rerun it."""
    output_dir = Path(case["output_dir"])
    case_path = output_dir / "case.json"
    summary_path = output_dir / "summary.json"
    if summary_path.is_file():
        summary = _read_json(summary_path)
        status = summary.get("status")
        if not resume:
            raise ValueError(f"case already has a result: {case_path.relative_to(run_dir)}")
        if status in {"complete", "timeout", "memory_limit"}:
            if status == "complete":
                completed, exact = _read_completed_rows(case)
                if not exact or summary.get("completed_queries") != completed:
                    raise ValueError("saved complete case does not contain every expected query")
            return summary
        if status in {"worker_error", "correctness_failure"}:
            return summary
    existing = list(output_dir.iterdir()) if output_dir.exists() else []
    fresh_saved_case = (
        len(existing) == 1
        and existing[0] == case_path
        and _read_json(case_path) == case
    )
    if existing and not fresh_saved_case:
        if not resume:
            raise ValueError(f"case has an unfinished attempt: {case_path.relative_to(run_dir)}")
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        archive = run_dir / "interrupted" / f"{output_dir.name}-{timestamp}"
        archive.parent.mkdir(parents=True, exist_ok=True)
        output_dir.replace(archive)
    _write_json(case_path, case)
    return run_attempt(case_path, limits, expected_power)


def _record_power(run_dir: Path, phase: str, point: str, observation: dict) -> None:
    path = run_dir / "power-observations.json"
    rows = _read_json(path) if path.is_file() else []
    rows.append(
        {
            "phase": phase,
            "point": point,
            "observed_at": datetime.now(timezone.utc).isoformat(),
            **observation,
        }
    )
    _write_json(path, rows)


def _invalidate_power(run_dir: Path, error: PowerChanged) -> None:
    _write_json(
        run_dir / "invalid-run.json",
        {
            "reason": "power changed during measured work",
            "expected": error.expected,
            "observed": error.observed,
        },
    )


def _execute_phase(
    run_dir: Path, phase: str, cases: list[dict], limits: dict, resume: bool, expected: dict
) -> None:
    start_power = observe_power()
    _record_power(run_dir, phase, "start", start_power)
    if not _same_power(expected, start_power):
        error = PowerChanged(expected, start_power)
        _invalidate_power(run_dir, error)
        raise error
    try:
        for case in cases:
            summary = execute_case(
                run_dir, case, limits, resume=resume, expected_power=expected
            )
            if phase == "reference" and summary["status"] != "complete":
                raise RuntimeError(
                    f"reference case {Path(case['output_dir']).name} stopped: "
                    f"{summary['status']}"
                )
            if summary["status"] in {"worker_error", "correctness_failure", "interrupted"}:
                raise RuntimeError(
                    f"{case['phase']} case {Path(case['output_dir']).name} stopped: "
                    f"{summary['status']}"
                )
    except PowerChanged as error:
        _invalidate_power(run_dir, error)
        raise
    end_power = observe_power()
    _record_power(run_dir, phase, "end", end_power)
    if not _same_power(expected, end_power):
        error = PowerChanged(expected, end_power)
        _invalidate_power(run_dir, error)
        raise error


def _ensure_valid_run(run_dir: Path) -> None:
    if (run_dir / "invalid-run.json").is_file():
        raise ValueError("run was marked invalid after a power change; use a new run directory")


def run_stage(
    config_path: Path,
    stage: str,
    run_dir: Path | None = None,
    resume: bool = False,
    prepare_check: bool = False,
) -> Path:
    """Run one preparation, measurement, selection, evaluation, or report stage."""
    config_path = Path(config_path).resolve()
    config = _resolved_config(config_path)
    if stage == "prepare":
        if run_dir is not None:
            raise ValueError("preparation does not use a run directory")
        return _prepare(config, prepare_check)
    if prepare_check:
        raise ValueError("--prepare-check is only valid for preparation")
    if run_dir is None:
        raise ValueError(f"{stage} requires --run-dir")
    run_dir = Path(run_dir).resolve()
    if stage == "report":
        _ensure_valid_run(run_dir)
        metadata = _read_json(run_dir / "metadata.json")
        if metadata.get("config") != config:
            raise ValueError("run config does not match the requested config")
        return scaling_analysis.analyze_run(run_dir)

    pointer, selection, _, inputs = _load_prepared(config)
    if stage == "pilot" and not resume:
        metadata = _create_run(config_path, config, run_dir, pointer, selection, inputs)
    else:
        if not run_dir.is_dir():
            raise ValueError("the requested run directory does not exist")
        _ensure_valid_run(run_dir)
        metadata = _validate_run(config, run_dir, inputs)
    expected_power = metadata["provenance"]["initial_power"]

    if stage == "pilot":
        references = plan_cases(run_dir, metadata, "reference")
        pilot = plan_cases(run_dir, metadata, "pilot")
        _save_plan(run_dir, "reference", references, resume)
        _save_plan(run_dir, "pilot", pilot, resume)
        _execute_phase(run_dir, "reference", references, config["limits"], resume, expected_power)
        _execute_phase(run_dir, "pilot", pilot, config["limits"], resume, expected_power)
        return run_dir
    if stage == "tune":
        cases = plan_cases(run_dir, metadata, "development")
        _save_plan(run_dir, "development", cases, resume)
        _execute_phase(run_dir, "development", cases, config["limits"], resume, expected_power)
        scaling_analysis.freeze_selection(run_dir)
        return run_dir / "selections.json"
    if stage == "evaluate":
        selection_path = run_dir / "selections.json"
        if not selection_path.is_file():
            raise ValueError("evaluation requires frozen development choices")
        frozen = scaling_analysis.freeze_selection(run_dir)
        cases = plan_cases(run_dir, metadata, "evaluation", frozen)
        _save_plan(run_dir, "evaluation", cases, resume)
        _execute_phase(run_dir, "evaluation", cases, config["limits"], resume, expected_power)
        return run_dir
    raise ValueError(f"unsupported stage {stage}")


def _main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("scaling.toml"))
    parser.add_argument(
        "--stage", required=True, choices=("prepare", "pilot", "tune", "evaluate", "report")
    )
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--prepare-check", action="store_true")
    arguments = parser.parse_args(argv[1:])
    result = run_stage(
        arguments.config,
        arguments.stage,
        arguments.run_dir,
        arguments.resume,
        arguments.prepare_check,
    )
    print(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(_main(sys.argv))
