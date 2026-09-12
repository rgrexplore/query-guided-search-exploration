"""Direct prefix membership and scores for small correctness examples.

Inputs are unpacked 0/1 signs and the same valid query/cluster arrays used by
the native API. This reference deliberately scans the bits at every depth;
it does not use sorted keys, binary range searches, or a score lookup table.
"""

import numpy as np


def prefix_search(signs, assignments, queries, buckets, candidate_limit=100,
                  start_depth=16, candidate_target=1000):
    """Return ranked rows, work counts, and the newly scored rows at each depth."""
    rows = np.full((len(queries), candidate_limit), -1, dtype=np.int64)
    scores = np.full((len(queries), candidate_limit), -np.inf, dtype=np.float64)
    counts = np.zeros(len(queries), dtype=np.int64)
    stats = []
    traces = []

    for qi, query in enumerate(queries):
        selected = [int(label) for label in buckets[qi]
                    if label >= 0 and np.any(assignments == label)]
        seen = set()
        scored = []
        trace = []
        lookups = 0
        stop_reason = "exhausted"
        for depth in range(start_depth, -1, -1):
            new_rows = []
            for label in selected:
                if depth:
                    lookups += 2  # Native implementation uses two binary bounds.
                for row in np.flatnonzero(assignments == label):
                    row = int(row)
                    if row in seen:
                        continue
                    if not all(bool(signs[row, bit]) == bool(query[bit] >= 0)
                               for bit in range(depth)):
                        continue
                    score = sum(float(query[bit]) * (1 if signs[row, bit] else -1)
                                for bit in range(len(query)))
                    seen.add(row)
                    new_rows.append(row)
                    scored.append((row, score))
            trace.append({"depth": depth, "new_rows": new_rows})
            # A complete depth may expose more rows than the target requests.
            if candidate_target and len(seen) >= candidate_target:
                stop_reason = "candidate_target"
                break

        ranked = sorted(scored, key=lambda item: (-item[1], item[0]))[:candidate_limit]
        counts[qi] = len(ranked)
        for position, (row, score) in enumerate(ranked):
            rows[qi, position] = row
            scores[qi, position] = score
        stats.append({
            "prefix_levels": len(trace),
            "prefix_lookups": lookups,
            "final_depth": depth,
            "documents_scored": len(seen),
            "stop_reason": stop_reason,
        })
        traces.append(trace)

    return {"rows": rows, "scores": scores, "counts": counts,
            "stats": stats, "trace": traces}
