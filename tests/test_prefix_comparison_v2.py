"""Merged tables preserve input identity and repeat the setting actually selected."""

import json

import pytest

from experiments import prefix_comparison_v2 as comparison


def test_identity_ids_selection_storage_and_missing_repeat(tmp_path):
    router = tmp_path / "router"
    router.mkdir()
    (router / "router.json").write_text(json.dumps({"routing_payload_bytes": 128}))
    manifest = dict(pool="/data/example-pool", identity={"documents": 10, "queries": 1000, "dimensions": 32},
                    array_hashes={"codes.npy": "same-codes", "queries.npy": "same-queries"})

    def source(name, latency, *, repeated=False, probability=False):
        folder = tmp_path / name
        (folder / "main").mkdir(parents=True)
        config = dict(pool=manifest["pool"], top_ks=[1], recall_targets=[.95], ram_budget_bytes=10000)
        if probability:
            config["probability_source"] = "other-study"
        row = dict(setting_id="s1", method="branch", top_k=1, recall=.96, p50_ms=latency,
                   complete=True, qualified=True, router=str(router), probes=2, clusters=8,
                   logical_index_bytes=1000, mean_documents_scored=4)
        scan = dict(row, setting_id="scan", method="scan", p50_ms=2, mean_documents_scored=9)
        for filename, data in [("configuration.json", config), ("inputs-manifest.json", manifest),
                               ("main/settings.json", [row, scan])]:
            (folder / filename).write_text(json.dumps(data))
        if repeated:
            (folder / "repeat").mkdir()
            (folder / "repeat/settings.json").write_text(json.dumps([dict(row, p50_ms=1.1)]))
        return folder

    first, second, probability = source("first", 1, repeated=True), source("second", .5), source("probability", .01, probability=True)
    result = comparison.prepare([first, second, probability], tmp_path / "combined")
    group = tmp_path / "combined/example-pool"
    rows = json.loads((group / "main/settings.json").read_text())
    assert len(rows) == 4 and len({row["setting_id"] for row in rows}) == 4
    assert {row["setting_id"] for row in rows if row["original_setting_id"] == "s1"} == {"first::s1", "second::s1"}
    chosen = next(row for row in json.loads((group / "comparison-table.json").read_text()) if row["method"] == "branch")
    assert chosen["setting_id"] == "second::s1" and chosen["repeat_status"] == "missing"
    assert chosen["repeat_p50_ms"] is None  # Do not substitute first::s1's repeat.
    assert chosen["stored_fields_with_router_bytes"] == 1128
    assert chosen["mean_opened_rows"] == 9 and chosen["mean_documents_scored"] == 4
    repeats = json.loads((group / "repeat/settings.json").read_text())
    assert repeats[0]["setting_id"] == "first::s1"
    assert result["excluded_sources"][0]["source"] == str(probability.resolve())
    changed = dict(manifest, array_hashes={"codes.npy": "different", "queries.npy": "same-queries"})
    (second / "inputs-manifest.json").write_text(json.dumps(changed))
    with pytest.raises(ValueError, match="identical"):
        comparison.prepare([first, second], tmp_path / "invalid")
    assert not (tmp_path / "invalid").exists()
