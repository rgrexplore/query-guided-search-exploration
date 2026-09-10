# Native search code

The paper is in `../reports/search-math/`. Start with its four-bit score example, then follow
these functions:

| Paper topic | Code |
|---|---|
| Binary score and top-K ordering | `score.hpp`: `ScoreTable`, `TopCandidates` |
| Clustered scan | `index.cpp`: `Index::scan` |
| Bitplane construction and branching | `index.cpp`: `Index::Index`, `Index::search` |
| Short-key layout and weighted lookup | `key_index.cpp`: `KeyIndex::KeyIndex`, `KeyIndex::search` |
| Python arguments and returned arrays | `bindings.cpp` |
| Independent small examples | `../tests/test_key_index.py`, `../tests/test_index.py` |

## Python API

All code arrays are contiguous `uint64` with shape `(documents, ceil(dimensions / 64))`.
Assignments are contiguous `int64`, one cluster ID per document. Query arrays are contiguous
`float32`; selected cluster IDs are `int64` with one row per query. Use -1 for padding.

```python
from bitplane_index import Index, KeyIndex

scan = Index(codes, assignments, dimensions, build_bitplanes=False)
branch = Index(codes, assignments, dimensions)
keys = KeyIndex(codes, assignments, dimensions, key_bits=12, key_offset=0)

scan_result = scan.scan(queries, clusters, candidate_limit=100)
branch_result = branch.search(
    queries, clusters, candidate_limit=100, node_budget=2048, leaf_size=32
)
key_result = keys.search(
    queries, clusters, candidate_limit=100, candidate_target=1000, key_limit=8192
)
```

`candidate_limit` is the number of output rows retained. For keys, `candidate_target` is the
number of full document scores to collect before stopping. A posting range is read completely,
so this target may be exceeded. `key_limit` counts actual directory lookups, including empty
ones. Zero disables either stopping limit. A valid full-score bound can still stop exact search.

Keys use a contiguous set of 1–24 coordinates starting at `key_offset`. With sign-prefix routing,
an offset after the routing bits avoids indexing the same fixed bits twice. For centroid routing,
o sign positions are fixed automatically. Lookup visits a key across all selected clusters
before applying the candidate target. A key-attempt limit may stop partway through those clusters.

The directory contains occupied keys only. Rows and full packed codes are reordered together;
the stored original IDs are also the postings. This layout does not need a second ID permutation.

## What the counters mean

- `bitplane_words`: physical bitmap words visited by split loops.
- `leaf_words`: physical bitmap words visited by leaf loops.
- `peak_mask_bytes`: simultaneously live bitmap vector capacities, including parent and both
  children. It excludes queue metadata and other query buffers.
- `key_attempts`: occupied or empty directory lookups across selected clusters.
- `keys_generated`: weighted key patterns taken from the enumeration queue.
- `peak_key_queue_bytes`: largest allocated enumeration-vector capacity in bytes.
- `documents_scored`: complete packed-code scores evaluated.

`info()` reports logical code/ID/plane/directory payload, array capacities and directory counts.
It does not report whole-process RAM. Hash-node padding, allocator overhead, routing arrays,
query scratch and the Python runtime need separate measurement. Scanning an ordinary `Index`
still retains its planes; use `build_bitplanes=False` for scan-only storage.

The search methods share the same scorer and row-ID tie order. Complete key lookup and unlimited
branching are checked against an independent full-score sort. A finite work budget can return
fewer or different rows; the stop reason and counters remain part of the result.
