import hashlib
import json
from pathlib import Path

import pytest

from scaling_config import ScalingConfigError, identity_hash, load_config


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
        "node_budgets": [
            32,
            128,
            512,
            2_048,
            8_192,
            32_768,
            65_536,
            131_072,
            262_144,
            0,
        ],
    },
    "measurement": {
        "repetitions": 3,
        "warmup_queries": 3,
        "schedule_seed": 44,
        "bootstrap_samples": 1_000,
        "bootstrap_seed": 45,
    },
    "limits": {"case_seconds": 1_800, "rss_gib": 32},
}


def changed_config(tmp_path, old, new):
    path = tmp_path / "changed.toml"
    original = Path("scaling.toml").read_text()
    assert old in original, f"Test setup could not find {old!r} in scaling.toml"
    path.write_text(original.replace(old, new))
    return path


def test_committed_config_matches_the_v2_protocol_exactly():
    assert load_config(Path("scaling.toml")) == EXPECTED_CONFIG


def test_node_budgets_are_sorted_finite_values_followed_by_one_unlimited_value():
    budgets = load_config(Path("scaling.toml"))["tuning"]["node_budgets"]

    assert budgets[:-1] == sorted(budgets[:-1])
    assert len(budgets[:-1]) == len(set(budgets[:-1]))
    assert budgets.count(0) == 1
    assert budgets[-1] == 0


@pytest.mark.parametrize(
    "old,new,setting",
    [
        ("pool_sizes = [1000, 3000", "pool_sizes = [3000, 1000", "pool_sizes"),
        ("1000000]", "300000]", "pool_sizes"),
        ("node_budgets = [32, 128", "node_budgets = [128, 32", "node_budgets"),
        ("128, 512", "128, 128", "node_budgets"),
        ("262144, 0]", "0, 262144]", "node_budgets"),
        ("262144, 0]", "262144, false]", "node_budgets"),
    ],
)
def test_config_rejects_invalid_order_counts_and_budget_layout(
    tmp_path, old, new, setting
):
    with pytest.raises(ScalingConfigError, match=f"CONFIG_INVALID.*{setting}"):
        load_config(changed_config(tmp_path, old, new))


@pytest.mark.parametrize(
    "old,new,setting",
    [
        ("dimensions = 256", "dimensions = 7", "dimensions"),
        ("candidate_limit = 100", "candidate_limit = 1001", "candidate_limit"),
        ("targets = [0.95, 0.99]", "targets = [0.99, 0.95]", "targets"),
        ("case_seconds = 1800", "case_seconds = 0", "case_seconds"),
        ("rss_gib = 32", "rss_gib = true", "rss_gib"),
    ],
)
def test_config_rejects_values_outside_the_consumed_ranges(tmp_path, old, new, setting):
    with pytest.raises(ScalingConfigError, match=f"CONFIG_INVALID.*{setting}"):
        load_config(changed_config(tmp_path, old, new))


@pytest.mark.parametrize(
    "extra,setting",
    [("\n[extra]\nvalue = 1\n", "sections"), ("\nunknown = 1\n", "limits")],
)
def test_config_rejects_wrong_section_or_setting_names(tmp_path, extra, setting):
    path = tmp_path / "unknown.toml"
    path.write_text(Path("scaling.toml").read_text() + extra)

    with pytest.raises(ScalingConfigError, match=f"CONFIG_INVALID.*{setting}"):
        load_config(path)


def test_identity_hash_remains_strict_and_order_independent():
    value = {"z": [3, 2, 1], "a": {"enabled": True}}
    canonical = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)

    assert identity_hash(value) == hashlib.sha256(canonical.encode()).hexdigest()
    assert identity_hash({"a": {"enabled": True}, "z": [3, 2, 1]}) == identity_hash(value)
    with pytest.raises(ValueError):
        identity_hash({"invalid": float("nan")})
