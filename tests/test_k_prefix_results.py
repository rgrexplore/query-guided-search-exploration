import json

import pytest

from experiments import k_prefix_results


@pytest.mark.parametrize("other_time", [.1, .2])
def test_comparison_selects_the_same_timed_tie_as_the_runner(tmp_path, monkeypatch, other_time):
    folder = tmp_path / "study"
    (folder / "main").mkdir(parents=True)
    (folder / "repeat").mkdir()
    common = dict(method="branch", top_k=2, qualified=True, recall=1.0)
    selected = dict(common, setting_id="a", p50_ms=.1)
    other = dict(common, setting_id="b", p50_ms=other_time)
    (folder / "main/settings.json").write_text(json.dumps([other, selected]))
    (folder / "repeat/settings.json").write_text(json.dumps([selected]))
    monkeypatch.setattr(k_prefix_results, "OUTPUT", tmp_path)
    rows = k_prefix_results.comparisons([folder])
    assert all(row["status"] == "complete" for row in rows)
    assert {row["setting_id"] for row in rows} == {"a"}
