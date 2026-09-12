"""Check the study's data split and comparison scope before expensive runs."""
import copy
import hashlib
import json

import numpy as np
import pytest

from experiments.components import pack_signs, reference_rows
from experiments.optima_study import (
    default_config,
    final_cases,
    prepare_fresh_pool,
    study_cases,
    tuning_probes,
    verify_source_pool,
)


def test_every_method_receives_every_layout_and_probe_count():
    config = default_config()
    layouts = [
        dict(path=None, kind="none", clusters=1, routing_bits=0, probes=[1]),
        dict(path="/direct", kind="direct", clusters=1024, routing_bits=10, probes=[1, 8]),
        dict(path="/ivf", kind="ivf", clusters=256, routing_bits=0, probes=[240, 256]),
    ]
    cases = study_cases(config, "/pool", "adaptive", layouts)
    for layout in layouts:
        for probes in layout["probes"]:
            group = [case for case in cases
                     if case["router"] == layout["path"] and case["probes"] == probes]
            assert {case["method"] for case in group} == {"scan", "branch", "keys"}
            assert any(case["method"] == "branch" and case["leaf_size"] == 1_000_000
                       for case in group)
    assert all(case["query_rows"] == list(range(32)) for case in cases)
    assert all(case["node_budget"] == 0 for case in cases if case["method"] == "branch")
    assert all(case["candidate_target"] == case["key_limit"] == 0
               for case in cases if case["method"] == "keys")
    assert {case["key_bits"] for case in cases
            if case["method"] == "keys" and case["router"] is None} == {1, 8, 11, 12, 13, 14}
    assert {case["key_bits"] for case in cases
            if case["method"] == "keys" and case["router"] == "/direct"} == {1, 2, 4}
    assert all(case["key_offset"] == 10 for case in cases
               if case["method"] == "keys" and case["router"] == "/direct")


def test_probe_selection_reads_only_tuning_rows_and_keeps_analytic_control(monkeypatch):
    calls = []

    def fake_probes(base, targets):
        calls.append((base, targets))
        return [3, 5]

    monkeypatch.setattr("experiments.optima_study.layout_probes", fake_probes)
    config = default_config()
    layout = dict(path="/direct", kind="direct", clusters=16384, routing_bits=14)
    assert tuning_probes(config, "/pool", "fixed", layout) == [3, 4, 5]
    assert calls[0][0]["query_rows"] == list(range(32))
    assert 1.0 in calls[0][1]
    assert tuning_probes(config, "/pool", "adaptive", layout) == [3, 5]


def test_final_cases_preserve_settings_and_include_formula_choices():
    config = default_config()
    cases = study_cases(config, "/pool", "fixed", [
        dict(path=None, kind="none", clusters=1, routing_bits=0, probes=[1])])
    measured = dict(setting_id="s1", case=cases[0], evaluation_seeds=[0],
                    uses=[dict(selection="conservative", target=.99)])
    formula = dict(setting_id="formula-b", case=cases[1], evaluation_seeds=[0],
                   uses=[dict(selection="formula", target=.99)])
    before = copy.deepcopy([measured, formula])
    result = final_cases(config, [measured], [formula])
    assert [measured, formula] == before
    assert len(result) == 6
    assert {case["setting_id"] for case in result} == {"s1", "formula-b"}
    assert {case["block"] for case in result} == {0, 1, 2}
    for case in result:
        assert case["query_rows"] == list(range(32, 96))
        assert case["repetitions"] == 2
        original = measured["case"] if case["setting_id"] == "s1" else formula["case"]
        for field, value in original.items():
            if field not in {"query_rows", "repetitions"}:
                assert case[field] == value
    with pytest.raises(ValueError, match="unique"):
        final_cases(config, [measured], [dict(formula, setting_id="s1")])


def make_source(tmp_path):
    source = tmp_path / "original"
    source.mkdir()
    signs = np.random.default_rng(73).integers(0, 2, (128, 8), dtype=np.uint8)
    np.save(source / "codes.npy", pack_signs(signs))
    np.save(source / "documents.npy", (2 * signs.astype(np.float32) - 1) / np.sqrt(8))
    np.save(source / "queries.npy", np.zeros((2, 8), dtype=np.float32))
    hashes = {name: hashlib.sha256((source / name).read_bytes()).hexdigest()
              for name in ("codes.npy", "documents.npy")}
    (source / "pool.json").write_text(json.dumps(dict(
        identity=dict(documents=128, dimensions=8, seed=73), hashes=hashes)))
    return source, signs


def test_fresh_pool_uses_independent_reference_and_retains_every_query(tmp_path):
    source, signs = make_source(tmp_path)
    config = default_config()
    config["data"].update(documents=128, dimensions=8, strong_bits=3,
                           queries=8, tuning_queries=3)
    config["search"]["top_k"] = 2
    verified = verify_source_pool(source, config)
    pool = prepare_fresh_pool(config, tmp_path / "fresh", "adaptive", source, verified)
    queries = np.load(pool / "queries.npy")
    truth = np.load(pool / "reference.npy")
    metadata = json.loads((pool / "pool.json").read_text())
    np.testing.assert_array_equal(truth, reference_rows(signs, queries, 2))
    assert len(queries) == len(metadata["query_ids"]) == 8
    assert not set(metadata["tuning_query_ids"]) & set(metadata["evaluation_query_ids"])
    assert len(metadata["evaluation_query_ids"]) == 5
    assert metadata["identity"]["query_seed"] == 20260912
    assert (pool / "codes.npy").is_symlink()
    counts = np.load(pool / "preferred_counts.npy")
    for row, positions in enumerate(metadata["supports"]):
        active = np.ones(len(signs), dtype=bool)
        expected = [len(signs)]
        for position in positions:
            active &= signs[:, position] == (queries[row, position] >= 0)
            expected.append(int(active.sum()))
        assert counts[row].tolist() == expected


def test_reusing_documents_requires_matching_actual_file_hashes(tmp_path):
    source, _ = make_source(tmp_path)
    config = default_config()
    config["data"].update(documents=128, dimensions=8)
    verify_source_pool(source, config)
    codes = np.load(source / "codes.npy")
    codes[0, 0] ^= np.uint64(1)
    np.save(source / "codes.npy", codes)
    with pytest.raises(ValueError, match="codes.npy"):
        verify_source_pool(source, config)
