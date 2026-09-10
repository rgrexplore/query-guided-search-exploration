"""Check the units and fit/check separation of the cost model."""
import pytest

from experiments.models import fit_cost_model, predict_search_ms


def test_scan_fit_recovers_known_cost_and_predicts_an_unused_size():
    rows = [dict(dimensions=256, documents_scored=n, search_ms=.01 + n*64*.0000004)
            for n in (1000, 4000, 16000)]
    model = fit_cost_model(rows, 'scan')
    held_out = dict(dimensions=256, documents_scored=8000)
    assert model['identifiable']
    assert predict_search_ms(model, held_out) == pytest.approx(.01 + 8000*64*.0000004)
    assert model['coefficients']['score_terms'] == pytest.approx(.0000004)
