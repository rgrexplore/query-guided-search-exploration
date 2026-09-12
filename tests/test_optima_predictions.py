import math
import json

import pytest

import experiments.optima_predictions as prediction_module
from experiments.optima_predictions import (
    WORK_FIELDS,
    analytic_work,
    features,
    fit_model,
    global_branch_derivative,
    predict_search,
    summarize_configuration,
)


def work(scored, **changes):
    return dict(dict.fromkeys(WORK_FIELDS, 0), documents_scored=scored, **changes)


def case(method='branch', **changes):
    values = dict(method=method, support='fixed', documents=256, dimensions=8,
                  router_kind='none', clusters=1, probes=1, top_k=4,
                  leaf_size=32, node_budget=0, key_bits=3, key_offset=0,
                  candidate_target=0, key_limit=0)
    return dict(values, **changes)


DATA = dict(documents=256, dimensions=8, strong_bits=3, weak_weight=.0001)


def test_fit_uses_one_configuration_median_not_all_query_rows():
    queries = [dict(work(count), search_ms=time, routing_ms=.1,
                    query_ms=time + .12) for count, time in [(100, 1), (300, 3), (200, 2)]]
    first = dict(case=case('scan'), queries=queries)
    repeated = dict(case=case('scan'), queries=queries * 40)
    assert summarize_configuration([first]) == summarize_configuration([repeated])
    summary = summarize_configuration([first])
    assert summary['search_ms'] == 2
    assert summary['work']['documents_scored'] == 200
    assert summary['composition_gap_ms'] == pytest.approx(.02)


def test_fit_recovers_positive_costs_from_configuration_medians():
    observations = []
    for count in [4, 8, 16, 32, 64, 128]:
        values = features('scan', work(count), 4)
        milliseconds = .5 + .01 * count + .25 * math.log(count / 4)
        observations.append(dict(features=values, search_ms=milliseconds, composition_gap_ms=.02))
    model = fit_model(observations)
    assert model['training_configurations'] == 6
    assert model['matrix_rank'] == 3
    assert model['identifiable']
    assert model['coefficients'] == pytest.approx(dict(fixed=.5, scored_rows=.01, log_scored_ratio=.25))
    assert predict_search(model, 'scan', work(24), 4) == pytest.approx(.5 + .24 + .25 * math.log(6))
    assert model['max_absolute_fit_error_percent'] < 1e-6


def test_constant_lookup_terms_are_reported_as_unidentifiable():
    observations = []
    for count in [4, 8, 16, 32, 64]:
        values = features('keys', work(count, key_attempts=1, keys_generated=1), 4)
        observations.append(dict(features=values, search_ms=1 + .01 * count,
                                 composition_gap_ms=0))
    model = fit_model(observations)
    assert not model['identifiable']
    assert model['matrix_rank'] == 3
    assert model['omitted_columns'] == ['key_lookups', 'key_queue_work']
    assert model['fit_within_threshold']


def test_analytic_prefix_branch_charges_fixed_bits_and_global_control_halves():
    selected = case(router_kind='direct', clusters=4)
    predicted = analytic_work(selected, DATA)
    assert predicted == work(32, nodes=4, bitplane_words=3, leaf_words=1)
    global_prediction = analytic_work(case(support='adaptive'), DATA)
    assert global_prediction == work(32, nodes=4, bitplane_words=12, leaf_words=4)
    assert analytic_work(case(leaf_size=16), DATA) is None


def test_fixed_key_and_direct_scan_can_select_the_same_document_group():
    by_key = analytic_work(case('keys'), DATA)
    by_route = analytic_work(case('scan', router_kind='direct', clusters=8), DATA)
    assert by_key['documents_scored'] == by_route['documents_scored'] == 32
    assert by_key['key_attempts'] == 1
    # A changing-support short key has a mixture of row counts. Do not use
    # its expected row count as a prediction of the median query time.
    assert analytic_work(case('keys', support='adaptive'), DATA) is None
    assert analytic_work(case(router_kind='ivf', clusters=4), DATA) is None


def test_global_derivative_requires_identifiable_costs():
    prices = dict(fixed=.1, scored_rows=.01, log_scored_ratio=.002,
                  split_words=.001, leaf_words=.001, nodes=.002)
    model = dict(identifiable=False, coefficients=prices)
    assert not global_branch_derivative(model, DATA, 4)['available']
    model['identifiable'] = True
    derivative = global_branch_derivative(model, DATA, 4)
    assert derivative['available']
    full_costs = []
    for depth in range(DATA['strong_bits'] + 1):
        counts = work(256 / 2**depth, bitplane_words=depth * 4,
                      leaf_words=4, nodes=depth + 1)
        full_costs.append(predict_search(model, 'branch', counts, 4))
    assert min(full_costs[i] for i in derivative['candidate_depths']) == min(full_costs)


def test_frozen_check_keeps_time_and_recall_misses_without_refitting(tmp_path, monkeypatch):
    output = tmp_path / 'predictions'
    output.mkdir()
    report = tmp_path / 'fixed' / 'evaluation' / 'report'
    report.mkdir(parents=True)
    (output / 'provenance.json').write_text(json.dumps(dict(supports=['fixed'], p50_error_threshold_percent=20)))
    (output / 'models.json').write_text('{}')
    prediction = dict(predicted_p50_ms=1, work_source='empirical_tuning_medians',
                      **{'predicted_' + name: value for name, value in work(100).items()})
    choice = dict(support='fixed', setting_id='formula-A', case=case('scan'),
                  prediction=prediction, measured_best_setting='s0', same_as_measured_best=True,
                  model_status='within_tuning_tolerance')
    (output / 'shortlist.json').write_text(json.dumps([choice]))
    (report / 'settings.csv').write_text(
        'setting_id,p50_ms,recall,qualified_environment\n'
        'formula-A,2,0.98,True\ns0,2,0.98,True\n')
    query = dict(work(125), search_ms=1.8, routing_ms=.1, query_ms=2)
    observed = dict(case=dict(case('scan'), setting_id='formula-A'), queries=[query])
    monkeypatch.setattr(prediction_module, 'load_cases', lambda path: ([observed], []))

    def refuse_refit(*args, **kwargs):
        raise AssertionError('final evaluation must not fit new coefficients')

    monkeypatch.setattr(prediction_module, 'fit_model', refuse_refit)
    prediction_module.check(tmp_path)
    result = json.loads((output / 'check.json').read_text())['rows'][0]
    assert result['p50_error_percent'] == -50
    assert not result['within_frozen_tolerance']
    assert not result['met_target']
    assert result['same_as_measured_best']
    assert result['work']['documents_scored']['error_percent'] == pytest.approx(-20)
