"""Follow one query through scan, bitplane branching and weighted key lookup.

After building/installing the native extension, run:
    python examples/small_search.py
"""
import numpy as np

from bitplane_index import Index, KeyIndex


# Each character is one coordinate, in the same order as the query below.
# A stored 1 scores as +1; a stored 0 scores as -1.
document_bits = ["01001", "00100", "10100", "11001", "11000", "00011"]
query = np.array([[0.7, 0.5, -0.4, -0.6, 0.3]], dtype=np.float32)
top_k = 2
dimensions = query.shape[1]

signs = np.array([[int(bit) for bit in text] for text in document_bits], dtype=np.uint8)
codes = np.zeros((len(signs), 1), dtype=np.uint64)
for dimension in range(dimensions):
    codes[:, 0] |= signs[:, dimension].astype(np.uint64) << np.uint64(dimension)

# The first bit assigns each document to cluster 0 or cluster 1. Open both
# clusters here, so routing cannot remove a correct result from the example.
assignments = np.ascontiguousarray(signs[:, 0], dtype=np.int64)
selected_clusters = np.array([[1, 0]], dtype=np.int64)

# Calculate the answer separately from the packed index. Sorting by score and
# then document ID gives every method the same rule when scores tie.
full_scores = (2 * signs.astype(np.float64) - 1) @ query[0].astype(np.float64)
correct_rows = np.lexsort((np.arange(len(signs)), -full_scores))[:top_k]

print("Query: [" + ", ".join(f"{value:.1f}" for value in query[0]) + "]")
print("Ideal signs:", "".join("1" if value >= 0 else "0" for value in query[0]))
print("Split coordinates, largest magnitude first:", (np.argsort(-np.abs(query[0])) + 1).tolist())
print("\nStored documents:")
for row, bits in enumerate(document_bits):
    print(f"  ID {row}: bits={bits}, cluster={assignments[row]}, full score={full_scores[row]:.2f}")
print("\nCorrect top documents:", correct_rows.tolist())

scan_index = Index(codes, assignments, dimensions, build_bitplanes=False)
branch_index = Index(codes, assignments, dimensions, build_bitplanes=True)

# Coordinate 1 is already fixed by the cluster. Use coordinates 2 and 3 for
# the local key. The full five-coordinate score still decides the final order.
key_index = KeyIndex(codes, assignments, dimensions, key_bits=2, key_offset=1)
results = [
    ("Full scan", scan_index,
     scan_index.scan(query, selected_clusters, candidate_limit=top_k)),
    ("Bitplane branching", branch_index,
     branch_index.search(query, selected_clusters, candidate_limit=top_k,
                         node_budget=0, leaf_size=2)),
    ("Weighted key lookup", key_index,
     key_index.search(query, selected_clusters, candidate_limit=top_k,
                      candidate_target=0, key_limit=0)),
]

for name, index, result in results:
    np.testing.assert_array_equal(result["rows"][0], correct_rows)
    stats = result["stats"][0]
    print(f"\n{name}")
    print("  Returned IDs:", result["rows"][0].tolist())
    print("  Documents fully scored:", stats["documents_scored"])
    print("  Split-word visits:", stats["bitplane_words"])
    print("  Leaf-word visits:", stats["leaf_words"])
    print("  Directory lookups, including misses:", stats["key_attempts"])
    print("  Logical index payload:", index.info()["logical_bytes"], "bytes")

print("\nAll three match the direct score calculation.")
print("Logical payload excludes allocator and runtime memory. This example does not measure latency.")
