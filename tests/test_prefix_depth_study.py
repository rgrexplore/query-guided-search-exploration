import pytest

from experiments.prefix_depth_study import allows_stop, recall


def test_saved_scores_use_the_requested_k_for_the_stop_check():
    step = dict(depth=2, upper_bound=.8, scores=[1.0, .7])
    assert allows_stop(step, 1.8, 4, 1)
    assert not allows_stop(step, 1.8, 4, 2)
    assert not allows_stop(step, 1.8, 4, 3)
    # An equal score is not sufficient because an unseen ID can win the tie.
    assert not allows_stop(dict(step, scores=[.8]), 1.8, 4, 1)


def test_recall_counts_the_requested_reference_ids():
    assert recall([7], [7]) == 1
    assert recall([8], [7]) == 0
    assert recall([7, 8], [7, 9]) == pytest.approx(.5)
    assert recall([], [7]) == 0
