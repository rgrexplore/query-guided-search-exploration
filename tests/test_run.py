from pathlib import Path

import pytest

from run import load_config


def test_config_rejects_a_candidate_limit_smaller_than_final_results(tmp_path):
    original = Path("experiment.toml").read_text()
    path = tmp_path / "bad.toml"
    path.write_text(original.replace("candidate_limit = 100", "candidate_limit = 3"))
    with pytest.raises(ValueError, match="candidate_limit"):
        load_config(path)


def test_config_rejects_probability_outside_unit_interval(tmp_path):
    path = tmp_path / "bad.toml"
    path.write_text(Path("experiment.toml").read_text().replace("[0.0, 0.1]", "[0.0, 1.1]"))
    with pytest.raises(ValueError, match="explore_probabilities"):
        load_config(path)


def test_config_keeps_unlimited_budget_zero(tmp_path):
    path = tmp_path / "exact.toml"
    path.write_text(Path("experiment.toml").read_text().replace("[32, 128, 512]", "[0]"))
    assert load_config(path)["search"]["node_budgets"] == [0]


def test_config_accepts_explicit_sweep_analysis_settings(tmp_path):
    path = tmp_path / "analysis.toml"
    path.write_text(
        Path("experiment.toml").read_text()
        + "\ncomplete_routing_endpoint = true\n"
        + "bootstrap_samples = 20\n"
        + "analysis_seed = 17\n"
    )

    benchmark = load_config(path)["benchmark"]

    assert benchmark["complete_routing_endpoint"] is True
    assert benchmark["bootstrap_samples"] == 20
    assert benchmark["analysis_seed"] == 17


@pytest.mark.parametrize(
    "setting,value",
    [
        ("complete_routing_endpoint", "1"),
        ("bootstrap_samples", '"many"'),
        ("analysis_seed", "1.5"),
        ("analysis_seed", "true"),
    ],
)
def test_config_rejects_wrong_types_for_sweep_analysis_settings(tmp_path, setting, value):
    path = tmp_path / "bad-analysis.toml"
    path.write_text(Path("experiment.toml").read_text() + f"\n{setting} = {value}\n")

    with pytest.raises(ValueError, match=setting):
        load_config(path)
