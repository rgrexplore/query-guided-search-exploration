"""Load and validate the shared scaling-study configuration and case files."""

import hashlib
import json
import math
import re
import tomllib
from pathlib import Path
from typing import Any


StudyConfig = dict[str, dict[str, Any]]
CaseSpec = dict[str, Any]

CONFIG_SECTIONS = {
    "data": {
        "cache_dir",
        "max_documents",
        "pool_sizes",
        "document_seed",
        "query_seed",
        "development_queries",
        "evaluation_queries",
    },
    "embedding": {
        "model",
        "revision",
        "batch_size",
        "max_length",
        "device",
        "chunk_size",
    },
    "search": {
        "dimensions",
        "candidate_limit",
        "leaf_size",
        "explore_probability",
        "seed",
    },
    "tuning": {"targets", "initial_budgets", "max_budget", "refinement_rounds"},
    "measurement": {
        "repetitions",
        "warmup_queries",
        "schedule_seed",
        "bootstrap_samples",
        "bootstrap_seed",
    },
    "limits": {
        "setup_seconds",
        "query_seconds",
        "attempt_seconds",
        "rss_gib",
        "poll_seconds",
    },
}

IDENTITY_FIELDS = {"run_hash", "input_hash", "protocol_hash", "code_hash"}
LIMIT_FIELDS = CONFIG_SECTIONS["limits"]
CASE_FIELDS = {
    "schema_version",
    *IDENTITY_FIELDS,
    "case_key",
    "attempt_id",
    "phase",
    "pool_size",
    "method",
    "node_budget",
    "repeat",
    "attempt_number",
    "query_rows",
    "query_ids",
    "candidate_limit",
    "dimensions",
    "leaf_size",
    "explore_probability",
    "seed",
    "run_manifest_path",
    "input_manifest_path",
    "code_path",
    "query_path",
    "reference_path",
    "warmup_queries",
    "limits",
    "output_path",
}


class ScalingConfigError(ValueError):
    """A named input error that callers can report without reading message text."""

    def __init__(self, code: str, field: str, message: str):
        self.code = code
        self.field = field
        super().__init__(f"{code} {field}: {message}")


def identity_hash(value: Any) -> str:
    """Hash one JSON value using the study's stable, strict representation."""
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def _invalid(code: str, field: str, message: str) -> None:
    raise ScalingConfigError(code, field, message)


def _check_fields(value: Any, expected: set[str], code: str, field: str) -> dict:
    if not isinstance(value, dict):
        _invalid(code, field, "must be a table")
    missing = expected - set(value)
    unknown = set(value) - expected
    if missing or unknown:
        _invalid(code, field, f"missing {sorted(missing)}, unknown {sorted(unknown)}")
    return value


def _integer(value: Any, code: str, field: str, minimum: int = 1) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        _invalid(code, field, f"must be an integer >= {minimum}")
    return value


def _number(value: Any, code: str, field: str, minimum: float | None = None) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        _invalid(code, field, "must be a finite number")
    if minimum is not None and value < minimum:
        _invalid(code, field, f"must be >= {minimum}")
    return float(value)


def _positive_number(value: Any, code: str, field: str) -> float:
    number = _number(value, code, field)
    if number <= 0:
        _invalid(code, field, "must be greater than zero")
    return number


def _nonempty_string(value: Any, code: str, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        _invalid(code, field, "must be a nonempty string")
    return value


def _hash(value: Any, code: str, field: str) -> str:
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        _invalid(code, field, "must be a lowercase SHA256 hash")
    return value


def _strictly_increasing_integers(
    values: Any,
    code: str,
    field: str,
    minimum: int = 1,
) -> list[int]:
    if not isinstance(values, list) or not values:
        _invalid(code, field, "must be a nonempty list")
    for value in values:
        _integer(value, code, field, minimum)
    if any(left >= right for left, right in zip(values, values[1:])):
        _invalid(code, field, "must contain unique values in increasing order")
    return values


def _validate_limits(value: Any, code: str = "CONFIG_INVALID") -> dict:
    limits = _check_fields(value, LIMIT_FIELDS, code, "limits")
    for name in LIMIT_FIELDS:
        _positive_number(limits[name], code, f"limits.{name}")
    return limits


def _validate_config(config: Any) -> StudyConfig:
    if not isinstance(config, dict) or set(config) != set(CONFIG_SECTIONS):
        found = set(config) if isinstance(config, dict) else set()
        _invalid(
            "CONFIG_INVALID",
            "sections",
            f"expected {sorted(CONFIG_SECTIONS)}, found {sorted(found)}",
        )
    for section, fields in CONFIG_SECTIONS.items():
        _check_fields(config[section], fields, "CONFIG_INVALID", section)

    data = config["data"]
    _nonempty_string(data["cache_dir"], "CONFIG_INVALID", "data.cache_dir")
    _integer(data["max_documents"], "CONFIG_INVALID", "data.max_documents")
    pools = _strictly_increasing_integers(
        data["pool_sizes"], "CONFIG_INVALID", "data.pool_sizes"
    )
    if pools[-1] > data["max_documents"]:
        _invalid("CONFIG_INVALID", "data.pool_sizes", "cannot exceed max_documents")
    for name in ("document_seed", "query_seed"):
        _integer(data[name], "CONFIG_INVALID", f"data.{name}", 0)
    for name in ("development_queries", "evaluation_queries"):
        _integer(data[name], "CONFIG_INVALID", f"data.{name}")

    embedding = config["embedding"]
    if embedding["model"] != "nomic-ai/nomic-embed-text-v1.5":
        _invalid("CONFIG_INVALID", "embedding.model", "is not supported")
    revision = embedding["revision"]
    if not isinstance(revision, str) or re.fullmatch(r"[0-9a-f]{40}", revision) is None:
        _invalid("CONFIG_INVALID", "embedding.revision", "must be a 40-character commit hash")
    _integer(embedding["batch_size"], "CONFIG_INVALID", "embedding.batch_size")
    max_length = _integer(
        embedding["max_length"], "CONFIG_INVALID", "embedding.max_length"
    )
    if max_length > 2_048:
        _invalid("CONFIG_INVALID", "embedding.max_length", "must be <= 2048")
    if embedding["device"] not in {"auto", "cpu", "mps", "cuda"}:
        _invalid("CONFIG_INVALID", "embedding.device", "must be auto, cpu, mps, or cuda")
    _integer(embedding["chunk_size"], "CONFIG_INVALID", "embedding.chunk_size")

    search = config["search"]
    if search["dimensions"] not in {64, 128, 256, 512, 768} or isinstance(
        search["dimensions"], bool
    ):
        _invalid("CONFIG_INVALID", "search.dimensions", "has an unsupported value")
    candidate_limit = _integer(
        search["candidate_limit"], "CONFIG_INVALID", "search.candidate_limit"
    )
    if candidate_limit > pools[0]:
        _invalid("CONFIG_INVALID", "search.candidate_limit", "cannot exceed the smallest pool")
    _integer(search["leaf_size"], "CONFIG_INVALID", "search.leaf_size")
    probability = _number(
        search["explore_probability"], "CONFIG_INVALID", "search.explore_probability"
    )
    if not 0 <= probability <= 1:
        _invalid("CONFIG_INVALID", "search.explore_probability", "must be between 0 and 1")
    _integer(search["seed"], "CONFIG_INVALID", "search.seed", 0)

    tuning = config["tuning"]
    targets = tuning["targets"]
    if not isinstance(targets, list) or not targets:
        _invalid("CONFIG_INVALID", "tuning.targets", "must be a nonempty list")
    checked_targets = [
        _number(target, "CONFIG_INVALID", "tuning.targets") for target in targets
    ]
    if any(target <= 0 or target > 1 for target in checked_targets):
        _invalid("CONFIG_INVALID", "tuning.targets", "must be between zero and one")
    if any(left >= right for left, right in zip(checked_targets, checked_targets[1:])):
        _invalid("CONFIG_INVALID", "tuning.targets", "must be unique and increasing")
    budgets = _strictly_increasing_integers(
        tuning["initial_budgets"], "CONFIG_INVALID", "tuning.initial_budgets"
    )
    max_budget = _integer(tuning["max_budget"], "CONFIG_INVALID", "tuning.max_budget")
    if max_budget < budgets[-1]:
        _invalid("CONFIG_INVALID", "tuning.max_budget", "cannot be below initial_budgets")
    _integer(
        tuning["refinement_rounds"], "CONFIG_INVALID", "tuning.refinement_rounds", 0
    )

    measurement = config["measurement"]
    _integer(measurement["repetitions"], "CONFIG_INVALID", "measurement.repetitions")
    _integer(
        measurement["warmup_queries"],
        "CONFIG_INVALID",
        "measurement.warmup_queries",
        0,
    )
    _integer(measurement["schedule_seed"], "CONFIG_INVALID", "measurement.schedule_seed", 0)
    _integer(
        measurement["bootstrap_samples"], "CONFIG_INVALID", "measurement.bootstrap_samples"
    )
    _integer(
        measurement["bootstrap_seed"], "CONFIG_INVALID", "measurement.bootstrap_seed", 0
    )
    _validate_limits(config["limits"])
    return config


def load_config(path: Path) -> StudyConfig:
    """Load one exact TOML schema and reject mistakes before expensive work begins."""
    try:
        with Path(path).open("rb") as stream:
            config = tomllib.load(stream)
    except (OSError, tomllib.TOMLDecodeError) as error:
        raise ScalingConfigError("CONFIG_INVALID", str(path), str(error)) from error
    return _validate_config(config)


def initial_node_budgets(config: StudyConfig) -> list[int]:
    """Return finite initial budgets followed by the native unlimited value."""
    _validate_config(config)
    return [*config["tuning"]["initial_budgets"], 0]


def _validate_case_identity(case: CaseSpec) -> None:
    case_identity = {
        "run_hash": case["run_hash"],
        "pool_size": case["pool_size"],
        "method": case["method"],
        "node_budget": case["node_budget"],
        "dimensions": case["dimensions"],
        "candidate_limit": case["candidate_limit"],
        "leaf_size": case["leaf_size"],
        "explore_probability": case["explore_probability"],
        "seed": case["seed"],
    }
    expected_case_key = identity_hash(case_identity)
    if case["case_key"] != expected_case_key:
        _invalid("CASE_INVALID", "case_key", "does not match the case identity")
    attempt_identity = {
        "case_key": case["case_key"],
        "phase": case["phase"],
        "repeat": case["repeat"],
        "attempt_number": case["attempt_number"],
    }
    if case["attempt_id"] != identity_hash(attempt_identity):
        _invalid("CASE_INVALID", "attempt_id", "does not match the attempt identity")


def validate_case(
    value: dict,
    *,
    expected_identity: dict[str, str] | None = None,
) -> CaseSpec:
    """Validate a worker case without opening its manifests or array files."""
    case = _check_fields(value, CASE_FIELDS, "CASE_INVALID", "fields")
    if case["schema_version"] != 1 or isinstance(case["schema_version"], bool):
        _invalid("CASE_INVALID", "schema_version", "must be 1")
    for name in IDENTITY_FIELDS | {"case_key", "attempt_id"}:
        _hash(case[name], "CASE_INVALID", name)

    if expected_identity is not None:
        expected = _check_fields(
            expected_identity,
            IDENTITY_FIELDS,
            "CASE_INVALID",
            "expected_identity",
        )
        for name in IDENTITY_FIELDS:
            _hash(expected[name], "CASE_INVALID", f"expected_identity.{name}")
            if case[name] != expected[name]:
                _invalid("CASE_INVALID", name, "does not match the expected identity")

    if case["phase"] not in {"reference", "pilot", "development", "evaluation"}:
        _invalid("CASE_INVALID", "phase", "has an unsupported value")
    if case["method"] not in {"scan", "branch"}:
        _invalid("CASE_INVALID", "method", "must be scan or branch")
    _integer(case["pool_size"], "CASE_INVALID", "pool_size")
    _integer(case["node_budget"], "CASE_INVALID", "node_budget", 0)
    if case["method"] == "scan" and case["node_budget"] != 0:
        _invalid("CASE_INVALID", "node_budget", "must be zero for scan")
    _integer(case["repeat"], "CASE_INVALID", "repeat", 0)
    _integer(case["attempt_number"], "CASE_INVALID", "attempt_number")
    candidate_limit = _integer(
        case["candidate_limit"], "CASE_INVALID", "candidate_limit"
    )
    if candidate_limit > case["pool_size"]:
        _invalid("CASE_INVALID", "candidate_limit", "cannot exceed pool_size")
    if case["dimensions"] not in {64, 128, 256, 512, 768} or isinstance(
        case["dimensions"], bool
    ):
        _invalid("CASE_INVALID", "dimensions", "has an unsupported value")
    _integer(case["leaf_size"], "CASE_INVALID", "leaf_size")
    probability = _number(
        case["explore_probability"], "CASE_INVALID", "explore_probability"
    )
    if not 0 <= probability <= 1:
        _invalid("CASE_INVALID", "explore_probability", "must be between 0 and 1")
    _integer(case["seed"], "CASE_INVALID", "seed", 0)
    _integer(case["warmup_queries"], "CASE_INVALID", "warmup_queries", 0)

    query_rows = case["query_rows"]
    if not isinstance(query_rows, list) or not query_rows:
        _invalid("CASE_INVALID", "query_rows", "must be a nonempty list")
    for row in query_rows:
        _integer(row, "CASE_INVALID", "query_rows", 0)
    if len(set(query_rows)) != len(query_rows):
        _invalid("CASE_INVALID", "query_rows", "must not overlap or repeat")

    query_ids = case["query_ids"]
    if not isinstance(query_ids, list) or len(query_ids) != len(query_rows):
        _invalid("CASE_INVALID", "query_ids", "must align with query_rows")
    for query_id in query_ids:
        _nonempty_string(query_id, "CASE_INVALID", "query_ids")
    if len(set(query_ids)) != len(query_ids):
        _invalid("CASE_INVALID", "query_ids", "must be unique")

    path_fields = {
        "run_manifest_path",
        "input_manifest_path",
        "code_path",
        "query_path",
        "output_path",
    }
    for name in path_fields:
        path = _nonempty_string(case[name], "CASE_INVALID", name)
        if not Path(path).is_absolute():
            _invalid("CASE_INVALID", name, "must be an absolute path")
    reference_path = case["reference_path"]
    if case["phase"] == "reference":
        if reference_path is not None:
            _invalid("CASE_INVALID", "reference_path", "must be null for reference creation")
    else:
        path = _nonempty_string(reference_path, "CASE_INVALID", "reference_path")
        if not Path(path).is_absolute():
            _invalid("CASE_INVALID", "reference_path", "must be an absolute path")

    _validate_limits(case["limits"], "CASE_INVALID")
    _validate_case_identity(case)
    return case
