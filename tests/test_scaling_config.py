import hashlib
import json
from pathlib import Path

import pytest

from scaling_config import identity_hash, initial_node_budgets, load_config, validate_case


EXPECTED_CONFIG = {
    "data": {
        "cache_dir": "data/scaling",
        "max_documents": 1_000_000,
        "pool_sizes": [1_000, 3_000, 10_000, 30_000, 100_000, 300_000, 1_000_000],
        "document_seed": 42,
        "query_seed": 43,
        "development_queries": 100,
        "evaluation_queries": 200,
    },
    "embedding": {
        "model": "nomic-ai/nomic-embed-text-v1.5",
        "revision": "e9b6763023c676ca8431644204f50c2b100d9aab",
        "batch_size": 32,
        "max_length": 512,
        "device": "auto",
        "chunk_size": 4_096,
    },
    "search": {
        "dimensions": 256,
        "candidate_limit": 100,
        "leaf_size": 32,
        "explore_probability": 0.0,
        "seed": 42,
    },
    "tuning": {
        "targets": [0.95, 0.99],
        "initial_budgets": [32, 128, 512, 2_048, 8_192, 32_768],
        "max_budget": 262_144,
        "refinement_rounds": 3,
    },
    "measurement": {
        "repetitions": 3,
        "warmup_queries": 3,
        "schedule_seed": 44,
        "bootstrap_samples": 1_000,
        "bootstrap_seed": 45,
    },
    "limits": {
        "setup_seconds": 120,
        "query_seconds": 10,
        "attempt_seconds": 1_800,
        "rss_gib": 32,
        "poll_seconds": 0.1,
    },
}


def changed_config(tmp_path, old, new):
    path = tmp_path / "changed.toml"
    original = Path("scaling.toml").read_text()
    assert old in original, f"Test setup could not find {old!r} in scaling.toml"
    path.write_text(original.replace(old, new))
    return path


def valid_case(tmp_path):
    values = {
        "schema_version": 1,
        "run_hash": "a" * 64,
        "input_hash": "b" * 64,
        "protocol_hash": "c" * 64,
        "code_hash": "d" * 64,
        "phase": "development",
        "pool_size": 1_000,
        "method": "branch",
        "node_budget": 512,
        "repeat": 0,
        "attempt_number": 1,
        "query_rows": [7, 2],
        "query_ids": ["query-7", "query-2"],
        "candidate_limit": 100,
        "dimensions": 256,
        "leaf_size": 32,
        "explore_probability": 0.0,
        "seed": 42,
        "run_manifest_path": str(tmp_path / "run.json"),
        "input_manifest_path": str(tmp_path / "input.json"),
        "code_path": str(tmp_path / "codes.npy"),
        "query_path": str(tmp_path / "queries.npy"),
        "reference_path": str(tmp_path / "reference.npz"),
        "warmup_queries": 3,
        "limits": EXPECTED_CONFIG["limits"].copy(),
        "output_path": str(tmp_path / "attempt"),
    }
    case_identity = {
        "run_hash": values["run_hash"],
        "pool_size": values["pool_size"],
        "method": values["method"],
        "node_budget": values["node_budget"],
        "dimensions": values["dimensions"],
        "candidate_limit": values["candidate_limit"],
        "leaf_size": values["leaf_size"],
        "explore_probability": values["explore_probability"],
        "seed": values["seed"],
    }
    values["case_key"] = identity_hash(case_identity)
    values["attempt_id"] = identity_hash(
        {
            "case_key": values["case_key"],
            "phase": values["phase"],
            "repeat": values["repeat"],
            "attempt_number": values["attempt_number"],
        }
    )
    return values


def test_committed_config_matches_the_initial_protocol_exactly():
    assert load_config(Path("scaling.toml")) == EXPECTED_CONFIG


def test_unlimited_budget_is_added_after_the_sorted_finite_budgets():
    config = load_config(Path("scaling.toml"))

    assert initial_node_budgets(config) == [32, 128, 512, 2_048, 8_192, 32_768, 0]
    assert config["tuning"]["initial_budgets"] == [32, 128, 512, 2_048, 8_192, 32_768]


@pytest.mark.parametrize(
    "old,new,setting",
    [
        ("pool_sizes = [1000, 3000", "pool_sizes = [3000, 1000", "pool_sizes"),
        ("1000000]", "300000]", "pool_sizes"),
        ("initial_budgets = [32, 128", "initial_budgets = [128, 32", "initial_budgets"),
        ("128, 512", "128, 128", "initial_budgets"),
        ("[32, 128, 512, 2048, 8192, 32768]", "[0, 32]", "initial_budgets"),
    ],
)
def test_config_rejects_unsorted_duplicate_or_unlimited_finite_lists(
    tmp_path, old, new, setting
):
    with pytest.raises(ValueError, match=f"CONFIG_INVALID.*{setting}"):
        load_config(changed_config(tmp_path, old, new))


@pytest.mark.parametrize(
    "old,new,setting",
    [
        ("document_seed = 42", "document_seed = true", "document_seed"),
        ("explore_probability = 0.0", "explore_probability = nan", "explore_probability"),
        ("poll_seconds = 0.1", "poll_seconds = inf", "poll_seconds"),
        ("candidate_limit = 100", "candidate_limit = 1001", "candidate_limit"),
        (
            'revision = "e9b6763023c676ca8431644204f50c2b100d9aab"',
            'revision = "main"',
            "revision",
        ),
    ],
)
def test_config_rejects_wrong_scalar_types_ranges_and_hashes(tmp_path, old, new, setting):
    with pytest.raises(ValueError, match=f"CONFIG_INVALID.*{setting}"):
        load_config(changed_config(tmp_path, old, new))


@pytest.mark.parametrize(
    "old,new,setting",
    [
        ("dimensions = 256", "dimensions = 256.0", "dimensions"),
        ('device = "auto"', 'device = ["auto"]', "device"),
        ("dimensions = 256", "dimensions = [256]", "dimensions"),
    ],
)
def test_config_rejects_float_integers_and_container_enum_values(
    tmp_path, old, new, setting
):
    with pytest.raises(ValueError, match=f"CONFIG_INVALID.*{setting}"):
        load_config(changed_config(tmp_path, old, new))


@pytest.mark.parametrize(
    "extra,setting",
    [
        ("\n[extra]\nvalue = 1\n", "sections"),
        ("\nunknown = 1\n", "limits"),
    ],
)
def test_config_rejects_unknown_sections_and_fields(tmp_path, extra, setting):
    path = tmp_path / "unknown.toml"
    path.write_text(Path("scaling.toml").read_text() + extra)

    with pytest.raises(ValueError, match=f"CONFIG_INVALID.*{setting}"):
        load_config(path)


def test_identity_hash_uses_strict_canonical_json():
    value = {"z": [3, 2, 1], "a": {"enabled": True}}
    canonical = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)

    assert identity_hash(value) == hashlib.sha256(canonical.encode()).hexdigest()
    assert identity_hash({"a": {"enabled": True}, "z": [3, 2, 1]}) == identity_hash(value)
    with pytest.raises(ValueError):
        identity_hash({"invalid": float("nan")})


def test_case_validator_accepts_the_settled_identity_and_paths(tmp_path):
    case = valid_case(tmp_path)
    expected = {
        name: case[name]
        for name in ("run_hash", "input_hash", "protocol_hash", "code_hash")
    }

    assert validate_case(case, expected_identity=expected) == case


def test_reference_case_requires_a_null_reference_path(tmp_path):
    case = valid_case(tmp_path)
    case.update(phase="reference", method="scan", node_budget=0, reference_path=None)
    case["case_key"] = identity_hash(
        {
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
    )
    case["attempt_id"] = identity_hash(
        {
            "case_key": case["case_key"],
            "phase": case["phase"],
            "repeat": case["repeat"],
            "attempt_number": case["attempt_number"],
        }
    )

    assert validate_case(case)["reference_path"] is None


@pytest.mark.parametrize(
    "change,setting",
    [
        (lambda case: case.update(input_hash="f" * 64), "input_hash"),
        (lambda case: case.update(protocol_hash="not-a-hash"), "protocol_hash"),
        (lambda case: case.update(query_rows=[7, 7]), "query_rows"),
        (lambda case: case.update(query_ids=["same", "same"]), "query_ids"),
        (lambda case: case.update(query_ids=["query-7"]), "query_ids"),
        (lambda case: case.update(pool_size=True), "pool_size"),
        (lambda case: case.update(explore_probability=float("nan")), "explore_probability"),
        (lambda case: case.update(code_path="relative/codes.npy"), "code_path"),
        (lambda case: case.update(reference_path=None), "reference_path"),
        (lambda case: case.update(case_key="e" * 64), "case_key"),
        (lambda case: case.update(attempt_id="e" * 64), "attempt_id"),
    ],
)
def test_case_validator_rejects_identity_shape_path_and_derived_id_errors(
    tmp_path, change, setting
):
    case = valid_case(tmp_path)
    expected = {
        name: case[name]
        for name in ("run_hash", "input_hash", "protocol_hash", "code_hash")
    }
    change(case)

    with pytest.raises(ValueError, match=f"CASE_INVALID.*{setting}"):
        validate_case(case, expected_identity=expected)


@pytest.mark.parametrize(
    "change,setting",
    [
        (lambda case: case.update(schema_version=1.0), "schema_version"),
        (lambda case: case.update(dimensions=256.0), "dimensions"),
        (lambda case: case.update(phase=[]), "phase"),
        (lambda case: case.update(method={}), "method"),
        (lambda case: case.update(dimensions=[256]), "dimensions"),
    ],
)
def test_case_validator_rejects_float_integers_and_container_enum_values(
    tmp_path, change, setting
):
    case = valid_case(tmp_path)
    change(case)

    with pytest.raises(ValueError, match=f"CASE_INVALID.*{setting}"):
        validate_case(case)


def test_case_validator_rejects_unknown_fields_in_the_case_limits_and_expected_identity(tmp_path):
    case = valid_case(tmp_path)
    case["surprise"] = True
    with pytest.raises(ValueError, match="CASE_INVALID.*fields"):
        validate_case(case)

    case = valid_case(tmp_path)
    case["limits"]["surprise"] = 1
    with pytest.raises(ValueError, match="CASE_INVALID.*limits"):
        validate_case(case)

    case = valid_case(tmp_path)
    expected = {
        name: case[name]
        for name in ("run_hash", "input_hash", "protocol_hash", "code_hash")
    }
    expected["surprise"] = "e" * 64
    with pytest.raises(ValueError, match="CASE_INVALID.*expected_identity"):
        validate_case(case, expected_identity=expected)
