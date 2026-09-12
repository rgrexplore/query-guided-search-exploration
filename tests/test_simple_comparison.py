"""The comparison changes query weights, not documents or per-method inputs."""
import json

import numpy as np

from experiments.components import pack_signs, reference_rows
from experiments.simple_comparison import equal_weight_reference, make_query_examples, comparison_cases, summarize


def test_query_examples_share_signs_and_keep_every_query():
    examples = make_query_examples(16, 32, 4, 20260913)
    signs = None
    for queries, supports in examples.values():
        assert queries.shape == (32, 16)
        np.testing.assert_allclose(np.linalg.norm(queries, axis=1), 1, atol=1e-6)
        if signs is None:
            signs = queries >= 0
        else:
            np.testing.assert_array_equal(signs, queries >= 0)
    first, fixed_supports = examples["first12"]
    changing, changed_supports = examples["spread12"]
    equal, equal_supports = examples["equal"]
    assert all(list(row) == [0, 1, 2, 3] for row in fixed_supports)
    assert any(list(row) != list(fixed_supports[0]) for row in changed_supports)
    assert np.all((np.abs(first) > .1).sum(axis=1) == 4)
    assert np.all((np.abs(changing) > .1).sum(axis=1) == 4)
    assert len(np.unique(np.abs(equal))) == 1
    assert equal_supports is None


def test_equal_weight_reference_matches_full_score_and_id_ties():
    signs = np.array([[a, b, c, d] for a in (0, 1) for b in (0, 1)
                      for c in (0, 1) for d in (0, 1)], dtype=np.uint8)
    queries = np.array([[1, -1, 1, -1], [-1, -1, 1, 1]], dtype=np.float32) / 2
    expected = reference_rows(signs, queries, 7)
    actual = np.array([equal_weight_reference(pack_signs(signs), query, 7) for query in queries])
    np.testing.assert_array_equal(actual, expected)


def test_all_methods_get_the_same_inputs_and_layouts_with_no_query_split():
    layouts = [dict(path=None, kind="none", clusters=1, routing_bits=0, probes=1),
               dict(path="/direct", kind="direct", clusters=16, routing_bits=4, probes=16)]
    cases = comparison_cases("equal", "/same-pool", layouts)
    assert len(cases) == 14
    for layout in layouts:
        group = [case for case in cases if case["router"] == layout["path"]]
        assert {case["method"] for case in group} == {"scan", "branch", "keys"}
        assert all(case["query_rows"] == list(range(32)) for case in group)
        assert all(case["pool"] == "/same-pool" and case["documents"] == 1_000_000
                   for case in group)
    branches = [case for case in cases if case["method"] == "branch"]
    assert all(case["node_budget"] == (0 if case["leaf_size"] == 1_000_000 else 4096)
               for case in branches)
    keys = [case for case in cases if case["method"] == "keys"]
    assert all(case["candidate_target"] == case["key_limit"] == 0 for case in keys)


def test_fast_setting_below_recall_target_is_retained_but_not_selected(tmp_path, monkeypatch):
    run_folder = tmp_path / "runs"
    run_folder.mkdir()
    schedule, completed = [], []
    power = {"available": True, "source": "AC Power"}
    for setting, recall, time in (("too-few", .80, .01), ("enough", .995, .1)):
        for block in range(3):
            case = dict(setting_id=setting, example="first12", method="scan", router_kind="direct",
                        clusters=4096, probes=1 if setting == "too-few" else 2, block=block)
            schedule.append(case)
            row = dict(query=0, repetition=0, recall=recall, query_ms=time,
                       documents_scored=250, bitplane_words=0, leaf_words=0, key_attempts=0)
            result = dict(storage=dict(logical_bytes=100, array_capacity_bytes=120),
                          memory=dict(budget_status="fits_by_lifetime_peak", lifetime_peak_bytes=200),
                          power_before=power, power_after=power)
            completed.append(dict(case=case, result=result, queries=[row]))
    (run_folder / "schedule.json").write_text(json.dumps(schedule))
    monkeypatch.setattr("experiments.simple_comparison.load_cases", lambda _: (completed, []))
    summarize(tmp_path, dict(process_repeats=3, recall_target=.99))
    summary = json.loads((tmp_path / "summary.json").read_text())
    assert len(summary["settings"]) == 2
    assert [row["setting_id"] for row in summary["selected"]] == ["enough"]
    assert next(row for row in summary["settings"] if row["setting_id"] == "too-few")["eligible"] is False
