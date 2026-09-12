"""A fast low-recall result must not appear as a high-recall winner."""

from experiments.prefix_figures_v2 import best_at_recall, repeated_comparisons


def test_curve_uses_a_qualifying_measured_result():
    rows = [dict(setting_id="fast", method="prefix", top_k=100, qualified=True, recall=.5, p50_ms=.1),
            dict(setting_id="enough", method="prefix", top_k=100, qualified=True, recall=.95, p50_ms=1),
            dict(setting_id="over-ram", method="prefix", top_k=100, qualified=False, recall=1, p50_ms=.01)]
    assert best_at_recall(rows, "prefix", 100, .9)["setting_id"] == "enough"
    assert best_at_recall(rows, "prefix", 100, .5)["setting_id"] == "fast"
    assert best_at_recall(rows, "prefix", 100, .99) is None


def test_repeats_do_not_reselect_the_fastest_setting():
    main = [dict(setting_id="chosen", method="scan", top_k=10, qualified=True, recall=.9, p50_ms=1),
            dict(setting_id="other", method="scan", top_k=10, qualified=True, recall=.99, p50_ms=2)]
    repeats = [dict(main[0], p50_ms=3), dict(main[1], p50_ms=.1)]
    rows = repeated_comparisons(main, repeats, [.9], [10])
    assert len(rows) == 1 and rows[0]["setting_id"] == "chosen"
    assert rows[0]["p50_ms"] == 3
