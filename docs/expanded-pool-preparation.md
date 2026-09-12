# Expanded query sets

The Nomic study reuses the existing one million MS MARCO document embeddings.
Only the additional queries need encoding. A separate pool is written for each
vector length, so a 256-dimensional experiment never changes a 768-dimensional
experiment's inputs.

## Data flow

1. Read `queries.dev.small.tsv` from the cached official MS MARCO archive.
2. Check that member's SHA-256, sort query IDs, and select 1,000 IDs using seed 42.
3. Save the selected IDs and text before encoding or search.
4. Encode `search_query: <text>` with the pinned Nomic model and its original recipe.
5. Take the first 256 or 768 coordinates from the normalized full vectors, then
   normalize that prefix. The cached documents already have Nomic's layer
   normalization; we do not apply it again.
6. Pack document signs into binary codes. Scan every document for every query to
   obtain the exact binary-score top 100.
7. Save the final array hashes in `pool.json`. Every method reads those same files.

Selection does not use relevance labels, search results, or query difficulty.
This measures recovery of the nearest binary-score neighbors. It is separate
from whether MS MARCO considers an answer relevant; some relevance judgments
refer to passages outside the retained one million documents.

## Commands

Run these from the project directory with its Python environment. Paths below
are placeholders for the existing cache and the new output folder.

```sh
python -m experiments.prepare_prefix_pools_v2 select \
  data/scaling/collectionandqueries.tar.gz \
  data/prefix-study-2026-09-13/nomic-msmarco/queries \
  --queries 1000 --seed 42

python -m experiments.prepare_prefix_pools_v2 encode \
  data/prefix-study-2026-09-13/nomic-msmarco/queries \
  --cache-dir data/model

python -m experiments.prepare_prefix_pools_v2 pool \
  FULL_DOCUMENTS.npy FULL_QUERIES.npy QUERY_SELECTION.json OUTPUT_FOLDER \
  --dimensions 256 --top-k 100 --provenance EMBEDDING_MANIFEST.json \
  --reference-limit 10
```

The first two commands prepare Nomic queries. The last command accepts any pair
of normalized full embedding arrays. For example, Qwen document and query arrays
can use it too; Qwen's own model preparation happens before this command.
`QUERY_SELECTION.json` is a list of IDs or an object with a `query_ids` field.
The provenance file records the model, revision, and source selection.

`--reference-limit 10` provides a small timing pilot. Its
`reference-progress.json` lists the number of queries scanned and the time for
each chunk. Divide scan seconds by completed queries and multiply by the planned
query count for a first estimate. Index build time is recorded separately.
Run the same command without the limit to finish all queries. The final
`pool.json` appears only when every reference query is complete.

## Saved progress

- Query selection: `selection.json`, `queries.jsonl`.
- Query encoding: `full-queries.npy`, `encoding-progress.json`, then `embedding.json`.
- Pool preparation: `request.json` and hashes in `arrays.json`.
- Exact search: `reference.npy` and `reference-progress.json`.
- Completed inputs: `pool.json` with dimensions, row counts, IDs, provenance and
  hashes for `documents.npy`, `queries.npy`, `codes.npy`, and `reference.npy`.

Encoding and exact search save each completed chunk. A resumed run begins at
that saved row. Array preparation is simpler: if interrupted before all arrays
are saved, it repeats that preparation step. Completed pools are checked and
reused; a different query selection or embedding input requires a new folder.

## Checks

`tests/test_prepare_prefix_pools_v2.py` uses tiny examples to check deterministic
selection, source hashes, Nomic's query prefix and normalization, exact score
ties, and interrupted-reference resumption. The independent reference in the
test calculates all float64 scores and uses a stable sort by row order.
The real reference calculation uses the native exhaustive scan to avoid a
query-by-document score matrix and a full sort of a million scores per query.

The reference is exact for the stored binary document codes and float query
vectors. It is not a claim that the original float document vectors have the
same nearest neighbors.

## Prepared on 2026-09-13

All four pools are complete. Each has 1,000 queries and an exact binary-score
top-100 reference. The two dimensions for each model use the same document IDs
and query IDs. The Nomic and Qwen experiments use different datasets and are
reported as separate experiments.

| Pool | Documents | Dimensions | Pilot estimate for reference | Actual reference | Total preparation |
|---|---:|---:|---:|---:|---:|
| `nomic-msmarco-d256` | 1,000,000 | 256 | 38.65 s | 39.60 s | 46.11 s |
| `nomic-msmarco-d768` | 1,000,000 | 768 | 137.54 s | 137.63 s | 150.24 s |
| `qwen-quora-d256` | 522,931 | 256 | 20.70 s | 20.78 s | 24.61 s |
| `qwen-quora-d1024` | 522,931 | 1024 | 102.35 s | 102.61 s | 111.46 s |

The estimate used the first ten queries. Total preparation includes array writes,
hashes, index construction, the reference pilot and its continuation; it excludes
model encoding. The saved data records are:

- `data/prefix-study-2026-09-13/pool-preparation-summary.json` for timings and final manifest hashes.
- `data/prefix-study-2026-09-13/pools/<pool>/pool.json` for exact array hashes and provenance.
- `data/prefix-study-2026-09-13/nomic-msmarco/queries/embedding.json` for the Nomic query recipe.
- `data/prefix-study-2026-09-13/quora-qwen-full/manifest.json` for the full Qwen encoding recipe.

The Nomic query selection contains 1,000 of the official 6,980 dev-small queries.
The selected-text SHA-256 is
`29ef20e59d0316f29a1739d8f329d841abbbe3ec03ebec9464424c5e483e8c9d`.
Selection happened before encoding and before any new search results.
