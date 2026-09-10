"""Selection must preserve quality, RAM constraints and recorded case IDs."""
from experiments.compare import best_settings, selected_inputs


def test_selection_rejects_faster_settings_that_miss_the_constraints():
    base = dict(documents=100, method='scan', budget_status='fits_by_lifetime_peak', power_unchanged=True)
    rows = [dict(base, case_id=2, recall=.91, p50_ms=.3),
            dict(base, case_id=20, recall=.89, p50_ms=.1),
            dict(base, case_id=21, recall=.95, p50_ms=.1, budget_status='resident_state_exceeds_budget')]
    selected = best_settings(rows, [.9])
    assert len(selected)==1 and selected[0]['case_id']==2


def test_selected_case_ids_do_not_depend_on_filtered_list_positions():
    selected = [dict(target=.9, case_id=20)]
    cases = [dict(case_id=5, case={'name':'first'}), dict(case_id=20, case={'name':'chosen'})]
    assert selected_inputs(selected, cases)==[dict(target=.9, case_id=20, case={'name':'chosen'})]
