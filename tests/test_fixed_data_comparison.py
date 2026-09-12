"""Protect the unchanged benchmark inputs and equal access to settings."""
import copy
import hashlib
import json

import numpy as np
import pytest

from experiments.fixed_data_comparison import build_cases, choose_settings, repeat_cases, verify_arrays, derive_probes


def test_grid_uses_one_pool_and_all_query_rows_for_every_method():
    layouts = [dict(path=None, kind="none", clusters=1, probes=[1])]
    for clusters, probes in ((256, [16, 64, 128, 256]),
                             (4096, [128, 512, 1536, 4096]),
                             (8192, [160, 448, 1024, 8192])):
        layouts.append(dict(path=f"/ivf-{clusters}", kind="ivf", clusters=clusters, probes=probes))
    cases = build_cases("/unchanged-pool", layouts)
    assert len(cases) == 104
    assert {case["pool"] for case in cases} == {"/unchanged-pool"}
    assert all(case["query_rows"] == list(range(200)) for case in cases)
    assert all(case["dimensions"] == 256 and case["documents"] == 1_000_000 for case in cases)
    for layout in layouts:
        for probes in layout["probes"]:
            selected = [case for case in cases if case["router"] == layout["path"] and case["probes"] == probes]
            assert {case["method"] for case in selected} == {"scan", "branch", "keys"}
            assert {(case["leaf_size"], case["node_budget"]) for case in selected if case["method"] == "branch"} == {
                (128, 256), (128, 1024), (512, 0), (1_000_000, 0)}
            assert {case["key_bits"] for case in selected if case["method"] == "keys"} == {1, 4, 8}


def test_fast_low_recall_or_failed_setting_cannot_be_selected():
    rows = [dict(setting_id="low", method="scan", recall=.985, p50_ms=.01, qualified=True),
            dict(setting_id="fits", method="scan", recall=.99, p50_ms=.1, qualified=True),
            dict(setting_id="bad-memory", method="scan", recall=1, p50_ms=.001, qualified=False)]
    selected = choose_settings(rows, [.99])
    assert len(selected) == 1 and selected[0]["setting_id"] == "fits"
    assert len(rows) == 3


def test_repeat_preserves_inputs_and_search_settings():
    cases = build_cases("/unchanged-pool", [dict(path=None, kind="none", clusters=1, probes=[1])])
    choices = [dict(setting_id=cases[0]["setting_id"], target=.8),
               dict(setting_id=cases[0]["setting_id"], target=.99),
               dict(setting_id=cases[1]["setting_id"], target=.99)]
    before = copy.deepcopy(cases)
    repeated = repeat_cases(cases, choices, blocks=3)
    assert len(repeated) == 6
    assert cases == before
    for case in repeated:
        original = next(row for row in cases if row["setting_id"] == case["setting_id"])
        assert {key: value for key, value in case.items() if key != "block"} == original
    assert {case["block"] for case in repeated} == {0, 1, 2}


def test_hash_check_detects_modified_queries_and_preserves_healthy_arrays(tmp_path):
    arrays = {"codes.npy": np.array([[0], [1]], dtype=np.uint64),
              "queries.npy": np.array([[1], [-1]], dtype=np.float32),
              "reference.npy": np.array([[1], [0]], dtype=np.int64),
              "documents.npy": np.array([[-1], [1]], dtype=np.float32)}
    for name, array in arrays.items():
        np.save(tmp_path / name, array)
    metadata = dict(hashes={name: hashlib.sha256((tmp_path / name).read_bytes()).hexdigest() for name in arrays})
    (tmp_path / "pool.json").write_text(json.dumps(metadata))
    before = verify_arrays(tmp_path)
    assert set(before) == set(arrays)
    np.save(tmp_path / "queries.npy", np.zeros((2, 1), dtype=np.float32))
    with pytest.raises(ValueError, match="queries.npy"):
        verify_arrays(tmp_path)
    assert hashlib.sha256((tmp_path / "codes.npy").read_bytes()).hexdigest() == before["codes.npy"]


def test_probe_cutoffs_use_all_unchanged_queries(monkeypatch):
    seen = []
    ranks = np.tile(np.array([[1, 2, 3, 4]]), (200, 1))
    scores = np.tile(np.array([[4, 3, 2, 1]], dtype=np.float32), (200, 1))

    def fake_ranks(case):
        seen.append(case)
        return ranks, scores, None, None, None

    monkeypatch.setattr("experiments.fixed_data_comparison.routing_ranks", fake_ranks)
    probes, evidence, saved_ranks = derive_probes("/pool", "/ivf", 4, [.5, .8, .99])
    assert probes == [2, 4]
    assert seen[0]["query_rows"] == list(range(200))
    assert seen[0]["pool"] == "/pool"
    np.testing.assert_array_equal(saved_ranks, ranks)
    assert [row["probes"] for row in evidence["cutoffs"]] == [2, 4, 4]
    assert evidence["rank_cdf"] == [.25, .5, .75, 1.0]
