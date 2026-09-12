import numpy as np

from experiments.prefix_exact_prediction_v2 import predict_rows


def test_paper_example_and_zero_ties_predict_actual_prefix_work():
    # Original-coordinate signs: 10100, 11001, 11000. These integers use bit0
    # for the first coordinate, independently of the predictor's sorted keys.
    codes = np.array([[5], [19], [3]], dtype=np.uint64)
    queries = np.array([[.7, .5, -.4, -.6, .3], [0, 0, 0, 0, 0]], dtype=np.float32)
    bounds = []
    for qi, query in enumerate(queries):
        scores = [sum(float(value) * (1 if (int(code) >> bit) & 1 else -1)
                      for bit, value in enumerate(query)) for code in codes[:, 0]]
        bounds.append(dict(query=qi, top_k=2, query_l1=sum(abs(float(x)) for x in query),
                           kth_score=sorted(scores, reverse=True)[1]))
    rows = predict_rows(codes, queries, bounds, starts=(1, 4, 5), top_ks=(2,))
    found = {(row["query"], row["start_depth"]): row for row in rows}
    expected = {1: (1, 3, 1, 2), 4: (4, 2, 1, 2), 5: (4, 2, 2, 4)}
    for start, counts in expected.items():
        row = found[(0, start)]
        assert row["largest_valid_depth"] == 4
        assert tuple(row[name] for name in ("final_depth", "documents_scored",
                                           "prefix_levels", "prefix_lookups")) == counts
        # Independent direct membership count: no sorting or key ranges.
        final = row["final_depth"]
        direct = sum(all(bool((int(code) >> bit) & 1) == bool(queries[0, bit] >= 0)
                         for bit in range(final)) for code in codes[:, 0])
        assert row["documents_scored"] == direct
    for start in (1, 4, 5):
        row = found[(1, start)]
        assert row["largest_valid_depth"] == row["final_depth"] == 0
        assert row["documents_scored"] == 3
        assert row["prefix_levels"] == start + 1
        assert row["prefix_lookups"] == 2 * start
        assert row["predicted_stop_reason"] == "exhausted"
