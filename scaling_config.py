"""Load and validate the settings used by the scaling study."""

import hashlib
import json
import math
import re
import tomllib
from pathlib import Path
from typing import Any


StudyConfig = dict[str, dict[str, Any]]

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
    "tuning": {"targets", "node_budgets"},
    "measurement": {
        "repetitions",
        "warmup_queries",
        "schedule_seed",
        "bootstrap_samples",
        "bootstrap_seed",
    },
    "limits": {"case_seconds", "rss_gib"},
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


def _invalid(field: str, message: str) -> None:
    raise ScalingConfigError("CONFIG_INVALID", field, message)


def _check_fields(value: Any, expected: set[str], field: str) -> dict:
    if not isinstance(value, dict):
        _invalid(field, "must be a table")
    missing = expected - set(value)
    unknown = set(value) - expected
    if missing or unknown:
        _invalid(field, f"missing {sorted(missing)}, unknown {sorted(unknown)}")
    return value


def _integer(value: Any, field: str, minimum: int = 1) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        _invalid(field, f"must be an integer >= {minimum}")
    return value


def _number(value: Any, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        _invalid(field, "must be a finite number")
    return float(value)


def _positive_number(value: Any, field: str) -> float:
    number = _number(value, field)
    if number <= 0:
        _invalid(field, "must be greater than zero")
    return number


def _nonempty_string(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        _invalid(field, "must be a nonempty string")
    return value


def _increasing_integers(values: Any, field: str) -> list[int]:
    if not isinstance(values, list) or not values:
        _invalid(field, "must be a nonempty list")
    for value in values:
        _integer(value, field)
    if any(left >= right for left, right in zip(values, values[1:])):
        _invalid(field, "must contain unique values in increasing order")
    return values


def _validate_config(config: Any) -> StudyConfig:
    found = set(config) if isinstance(config, dict) else set()
    if found != set(CONFIG_SECTIONS):
        _invalid("sections", f"expected {sorted(CONFIG_SECTIONS)}, found {sorted(found)}")
    for section, fields in CONFIG_SECTIONS.items():
        _check_fields(config[section], fields, section)

    data = config["data"]
    _nonempty_string(data["cache_dir"], "data.cache_dir")
    max_documents = _integer(data["max_documents"], "data.max_documents")
    pools = _increasing_integers(data["pool_sizes"], "data.pool_sizes")
    if pools[-1] > max_documents:
        _invalid("data.pool_sizes", "cannot exceed max_documents")
    for name in ("document_seed", "query_seed"):
        _integer(data[name], f"data.{name}", 0)
    for name in ("development_queries", "evaluation_queries"):
        _integer(data[name], f"data.{name}")

    embedding = config["embedding"]
    if embedding["model"] != "nomic-ai/nomic-embed-text-v1.5":
        _invalid("embedding.model", "is not supported")
    if not isinstance(embedding["revision"], str) or re.fullmatch(
        r"[0-9a-f]{40}", embedding["revision"]
    ) is None:
        _invalid("embedding.revision", "must be a 40-character commit hash")
    _integer(embedding["batch_size"], "embedding.batch_size")
    max_length = _integer(embedding["max_length"], "embedding.max_length")
    if max_length > 2_048:
        _invalid("embedding.max_length", "must be <= 2048")
    device = _nonempty_string(embedding["device"], "embedding.device")
    if device not in {"auto", "cpu", "mps", "cuda"}:
        _invalid("embedding.device", "must be auto, cpu, mps, or cuda")
    _integer(embedding["chunk_size"], "embedding.chunk_size")

    search = config["search"]
    dimensions = _integer(search["dimensions"], "search.dimensions")
    if dimensions not in {64, 128, 256, 512, 768}:
        _invalid("search.dimensions", "has an unsupported value")
    candidate_limit = _integer(search["candidate_limit"], "search.candidate_limit")
    if candidate_limit > pools[0]:
        _invalid("search.candidate_limit", "cannot exceed the smallest pool")
    _integer(search["leaf_size"], "search.leaf_size")
    probability = _number(search["explore_probability"], "search.explore_probability")
    if not 0 <= probability <= 1:
        _invalid("search.explore_probability", "must be between 0 and 1")
    _integer(search["seed"], "search.seed", 0)

    targets = config["tuning"]["targets"]
    if not isinstance(targets, list) or not targets:
        _invalid("tuning.targets", "must be a nonempty list")
    checked_targets = [_number(target, "tuning.targets") for target in targets]
    if any(target <= 0 or target > 1 for target in checked_targets):
        _invalid("tuning.targets", "must be between zero and one")
    if any(left >= right for left, right in zip(checked_targets, checked_targets[1:])):
        _invalid("tuning.targets", "must be unique and increasing")

    budgets = config["tuning"]["node_budgets"]
    if not isinstance(budgets, list) or len(budgets) < 2:
        _invalid("tuning.node_budgets", "must contain finite budgets and unlimited")
    _increasing_integers(budgets[:-1], "tuning.node_budgets")
    _integer(budgets[-1], "tuning.node_budgets", 0)
    if budgets[-1] != 0 or budgets.count(0) != 1:
        _invalid("tuning.node_budgets", "must contain unlimited zero exactly once at the end")

    measurement = config["measurement"]
    _integer(measurement["repetitions"], "measurement.repetitions")
    _integer(measurement["warmup_queries"], "measurement.warmup_queries", 0)
    _integer(measurement["schedule_seed"], "measurement.schedule_seed", 0)
    _integer(measurement["bootstrap_samples"], "measurement.bootstrap_samples")
    _integer(measurement["bootstrap_seed"], "measurement.bootstrap_seed", 0)

    limits = config["limits"]
    _positive_number(limits["case_seconds"], "limits.case_seconds")
    _positive_number(limits["rss_gib"], "limits.rss_gib")
    return config


def load_config(path: Path) -> StudyConfig:
    """Load one exact TOML layout and reject mistakes before expensive work begins."""
    try:
        with Path(path).open("rb") as stream:
            config = tomllib.load(stream)
    except (OSError, tomllib.TOMLDecodeError) as error:
        raise ScalingConfigError("CONFIG_INVALID", str(path), str(error)) from error
    return _validate_config(config)
