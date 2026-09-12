"""Run the paper's three-document example: python examples/prefix_search.py."""

import numpy as np

from bitplane_index import Index, PrefixIndexV2


document_bits = ["10100", "11001", "11000"]
query = np.array([[0.7, 0.5, -0.4, -0.6, 0.3]], dtype=np.float32)
dimensions = query.shape[1]
top_k = 2

# The leftmost character is coordinate zero. Pack that sign into the low bit
# of the first word. Each document uses one word in this five-bit example.
signs = np.array([[int(bit) for bit in text] for text in document_bits])
codes = np.zeros((len(signs), 1), dtype=np.uint64)
for position in range(dimensions):
    codes[:, 0] |= signs[:, position].astype(np.uint64) << np.uint64(position)

# Open the only cluster, so all three methods start with the same documents.
assignments = np.zeros(len(signs), dtype=np.int64)
opened = np.array([[0]], dtype=np.int64)
scores = (2 * signs - 1) @ query[0].astype(np.float64)
reference = np.lexsort((np.arange(len(signs)), -scores))[:top_k]

print("Query: [" + ", ".join(f"{value:.1f}" for value in query[0]) + "]")
print("Ideal signs:", "".join("1" if value >= 0 else "0" for value in query[0]))
for row, bits in enumerate(document_bits):
    print(f"ID {row}: {bits}, full score {scores[row]:.1f}")

scan = Index(codes, assignments, dimensions, build_bitplanes=False)
branch = Index(codes, assignments, dimensions, build_bitplanes=True)
prefix = PrefixIndexV2(codes, assignments, dimensions, max_prefix_bits=5)

results = [
    ("A: full scan", scan, scan.scan(query, opened, candidate_limit=top_k)),
    ("B: Bitplanes", branch,
     branch.search(query, opened, candidate_limit=top_k, node_budget=0, leaf_size=2)),
    ("C: Backward Walk", prefix,
     prefix.search(query, opened, candidate_limit=top_k, start_depth=5, candidate_target=2)),
]

for name, index, found in results:
    np.testing.assert_array_equal(found["rows"][0], reference)
    stats = found["stats"][0]
    print(f"\n{name}")
    print("  Returned IDs:", found["rows"][0].tolist())
    print("  Documents scored:", stats["documents_scored"])
    print("  Split-word visits:", stats["bitplane_words"])
    print("  Boundary searches:", stats["prefix_lookups"])
    print("  Stored fields:", index.info()["logical_bytes"], "bytes")

print("\nAll three match the direct calculation: IDs 1 and 2.")
print("Stored fields exclude process overhead. This small example does not measure latency.")
