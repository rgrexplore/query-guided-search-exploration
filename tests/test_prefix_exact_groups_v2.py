"""Exact-code membership partitions queries without using measured performance."""

import pytest

from experiments import prefix_exact_groups_v2 as analysis


def test_partition_same_queries_and_weighted_means():
    ids = ["q0", "q1", "q2", "q3"]
    geometry = [dict(query=i, depth=32, top_k=1, matched_rows=count)
                for i, count in enumerate([2, 0, 1, 0])]
    groups = analysis.query_groups(geometry, ids)
    assert groups == {"all": [0, 1, 2, 3], "exact-code-present": [0, 2], "exact-code-absent": [1, 3]}
    records = [dict(method=method, block=block, query=q, repetition=0,
                    query_ms=(q + 1) * multiplier + block, recall=1,
                    documents_scored=10 + q, nodes=q)
               for method, multiplier in [("scan", 1), ("branch", 2), ("prefix", 3)]
               for block in range(3) for q in range(4)]
    rows, checks = analysis.summarize_groups(records, groups, len(ids))
    assert len(rows) == 9 and all(check["matches"] for check in checks)
    overall = next(row for row in rows if row["method"] == "scan" and row["group"] == "all")
    assert overall["mean_query_ms"] == 3.5 and overall["mean_documents_scored"] == 11.5
    assert overall["process_median_min_ms"] == 2.5 and overall["process_median_max_ms"] == 4.5
    assert all(row["queries"] == 2 and row["fraction"] == .5 for row in rows if row["group"] != "all")
    with pytest.raises(ValueError, match="partition"):
        analysis.query_groups(geometry[:-1], ids)
