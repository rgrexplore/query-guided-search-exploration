"""The follow-up includes useful pruning cases, not only exact scan-like settings."""
from experiments.exploration import select_bases


def test_exploration_candidates_include_near_target_pruning_and_exclude_far_cases():
    rows = [dict(key='best', documents=1000, recall=.96, p50_ms=1, scored=1000, routed=1000, splits=0),
            dict(key='pruned', documents=1000, recall=.93, p50_ms=.8, scored=600, routed=1000, splits=10),
            dict(key='poor', documents=1000, recall=.5, p50_ms=.1, scored=20, routed=1000, splits=10)]
    assert set(select_bases(rows,[.95]))=={'best','pruned'}
