import numpy as np

from experiments.prefix_representation_v2 import full_float_best, candidate_agreement


def test_full_score_gap_candidate_recovery_and_tolerance_are_distinct():
    near = 1 - 5e-6
    documents = np.array([[1, 0], [1, 0], [0, 1], [.6, .8],
                          [near, np.sqrt(1 - near * near)]], dtype=np.float32)
    queries = np.array([[1, 0], [0, 1], [.6, .8], [1, 0]], dtype=np.float32)
    best_scores, best_rows = full_float_best(documents, queries, threads=1, batch_size=2)
    # Two candidates keep this numeric example tiny; the analysis command uses
    # exactly the first 100 saved binary-reference rows for every real query.
    references = np.array([[3, 1], [0, 1], [3, 0], [4, 1]], dtype=np.int64)
    rows = candidate_agreement(documents, queries, references, best_scores, best_rows, batch_size=2)
    np.testing.assert_allclose(best_scores, [1, 1, 1, 1], atol=1e-7)
    np.testing.assert_allclose([row["top1_full_score_gap"] for row in rows],
                              [.4, 1, 0, 5e-6], atol=1e-7)
    assert [row["top1_agrees"] for row in rows] == [False, False, True, True]
    assert [row["candidates_agree"] for row in rows] == [True, False, True, True]
    assert rows[0]["candidate_best_row"] == 1  # A tied full-score best is accepted.
    assert rows[1]["candidate_best_full_score"] == 0
    assert rows[3]["top1_full_score_gap"] > 0  # Keep the small raw gap; do not clamp it away.
