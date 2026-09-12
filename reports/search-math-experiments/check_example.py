"""Check the report's three-row example against the existing C++ search code.

This checks scores, work counts and stored bytes. It is not a speed benchmark.
"""
import json
from pathlib import Path

import numpy as np
import bitplane_index as native


def main():
    # Leftmost displayed bit is position 1; native packing starts at bit zero.
    bits = np.array([[1, 0, 1, 0, 0],
                     [1, 1, 0, 0, 1],
                     [1, 1, 0, 0, 0]], dtype=np.uint64)
    query = np.array([[0.7, 0.5, -0.4, -0.6, 0.3]], dtype=np.float32)
    codes = np.zeros((3, 1), dtype=np.uint64)
    for position in range(5):
        codes[:, 0] |= bits[:, position] << np.uint64(position)
    assignments = np.zeros(3, dtype=np.int64)
    opened = np.array([[0]], dtype=np.int64)

    # Compute the expected answer directly, without using the table scorer.
    scores = (2 * bits.astype(np.float64) - 1) @ query[0].astype(np.float64)
    expected_rows = np.lexsort((np.arange(3), -scores))[:2]
    np.testing.assert_array_equal(expected_rows, [1, 2])
    np.testing.assert_allclose(scores, [0.1, 2.5, 1.9], atol=1e-7)

    scan = native.Index(codes, assignments, 5, build_bitplanes=False)
    branch = native.Index(codes, assignments, 5)
    keys = native.KeyIndex(codes, assignments, 5, key_bits=2, key_offset=1)
    results = {
        "A": scan.scan(query, opened, candidate_limit=2),
        "B": branch.search(query, opened, candidate_limit=2, leaf_size=2,
                           node_budget=0, explore_probability=0, trace=True),
        "C": keys.search(query, opened, candidate_limit=2,
                         candidate_target=0, key_limit=0),
    }
    expected = {
        "A": dict(documents_scored=3),
        "B": dict(documents_scored=2, bitplane_words=3, leaf_words=1,
                  nodes=4, peak_mask_bytes=24),
        "C": dict(documents_scored=2, key_attempts=1, keys_generated=1,
                  peak_key_queue_bytes=16),
    }
    packed_keys = (bits[:, 1] | (bits[:, 2] << np.uint64(1)))
    stored_order = np.lexsort((np.arange(3), packed_keys))
    np.testing.assert_array_equal(stored_order, [1, 2, 0])
    records = {}
    for name, index in [("A", scan), ("B", branch), ("C", keys)]:
        result = results[name]
        np.testing.assert_array_equal(result["rows"][0], expected_rows)
        np.testing.assert_allclose(result["scores"][0], scores[expected_rows], atol=1e-12)
        stats = result["stats"][0]
        for field, value in expected[name].items():
            assert stats[field] == value, (name, field, stats[field], value)
        info = index.info()
        assert info["logical_bytes"] == (48 if name == "A" else 88)
        # Do not publish single-query timings as performance evidence.
        records[name] = dict(rows=result["rows"][0].tolist(),
                             scores=result["scores"][0].tolist(),
                             counts={k: stats[k] for k in expected[name]},
                             storage=info)
    assert [step["dimension"] for step in results["B"]["stats"][0]["trace"]] == [0, 3, 1, -1]
    assert results["B"]["stats"][0]["stop_reason"] == "bound"
    assert results["C"]["stats"][0]["stop_reason"] == "bound"
    common_bytes = 5 * 4 + 2 * 16 * 8 + 2 * 16
    assert common_bytes == 308
    output = dict(query=query[0].tolist(), documents=bits.tolist(), methods=records,
                  table_entries=32, table_slot_visits=128,
                  score_table_terms=dict(A=6, B=4, C=4),
                  counted_query_fields_bytes=dict(A=308, B=372, C=340),
                  counted_index_plus_query_bytes=dict(A=356, B=460, C=428),
                  scope="Named fields only; not total allocated process RAM. Teaching example, not benchmark data.")
    target = Path(__file__).parent / "evidence" / "three-document-example.json"
    target.write_text(json.dumps(output, indent=2) + "\n")
    print("All three methods return IDs 1 and 2; work and storage counts match the report.")


if __name__ == "__main__":
    main()
