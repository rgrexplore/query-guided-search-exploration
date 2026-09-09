# Bitplane search

This compares two ways to search binary document embeddings: scan the rows, or walk through
bitplanes and score smaller groups. Both use float queries and the same binary documents.
Faiss supplies the coarse buckets and a separate float-IVF baseline.

[Latest full FiQA run](results/fiqa-2026-09-09-dense/README.md): 583 settings, clearer curves,
paired uncertainty and every measured request. Limited branching improves some sign-routing
tradeoffs across probe counts; exact branching is slower than scanning the same buckets.
[Earlier runs and comparison](results/README.md) are preserved too.

The path through the code is:

**documents → embeddings → buckets → binary candidates → float reranking → measurements**

## Run it

Python 3.12 and a C++20 compiler are needed. On macOS, the compiler comes with the Xcode
command-line tools. The native search runs on CPU; embedding preparation can use MPS or CUDA.
`requirements.lock` records the tested Python environment. Paths in the TOML file are relative
to that file's folder.

```bash
uv venv --python 3.12
uv pip install --python .venv/bin/python -e '.[test]'
.venv/bin/python -m pytest -q
.venv/bin/python run.py --config experiment.toml
```

The first run downloads FiQA and Nomic's model weights, then encodes 57,638 documents and 648
queries. Later runs reuse matching arrays. Model revision, text IDs, preprocessing and dimensions
are recorded in the cache. Changing the prefix dimension reuses the full vectors.

To prepare the arrays without running the comparisons:

```bash
.venv/bin/python run.py --prepare-only
```

## Run the fixed scaling study

The scaling study uses a fixed sample of MS MARCO passages and separate development and
evaluation queries. Run its stages in order. Pick a new run folder for each timing run.

```bash
.venv/bin/python scaling.py --config scaling.toml --stage prepare
.venv/bin/python scaling.py --config scaling.toml --stage pilot --run-dir runs/scaling-1
.venv/bin/python scaling.py --config scaling.toml --stage tune --run-dir runs/scaling-1
.venv/bin/python scaling.py --config scaling.toml --stage evaluate --run-dir runs/scaling-1
.venv/bin/python scaling.py --config scaling.toml --stage report --run-dir runs/scaling-1
```

Use `--prepare-check` with the prepare stage to encode one document chunk and all queries before
committing to the full preparation. Run prepare again without that flag to continue from saved
chunks. Completed full vectors are reused when the search width changes; only the short query
vectors and packed document signs are created again. Preparation writes `prepared.json`, which
points to the selected text, full-vector and search-input manifests. Later stages read those
manifests and never start the encoder.

If a measured command is interrupted, repeat that command with `--resume`. Finished cases and
cases that reached a time or memory limit stay saved. An unfinished case is moved under the run's
`interrupted/` folder before it is run again. A pilot without `--resume` always requires a new run
folder, so an older result cannot be overwritten by accident.

The main settings in `scaling.toml` are:

- `pool_sizes`: nested document counts tested from the same saved sample.
- `dimensions`: the number of leading embedding values converted to document sign bits.
- `candidate_limit`: how many binary candidates each method returns.
- `leaf_size`: the group size at which branching scores documents directly.
- `node_budgets`: the fixed amounts of branch work tested during development; zero means no limit.
- `targets`: the mean development recall levels used to select a budget.
- `repetitions`: repeated timings of each saved setting.
- `case_seconds` and `rss_gib`: the whole-case time and sampled memory limits.

Read a completed run in this order: `metadata.json` for the inputs and machine record,
`selections.json` for the frozen development choices, `analysis/report.md` for the main findings,
then `analysis/target-points.csv` and `analysis/settings.csv` for the exact values. Case folders
under `cases/` retain the raw query rows and terminal status. Schedules contain relative case paths,
so a copied run can be reported again without the vector cache.

Recall here is overlap with the exact scan's best binary candidates from the same document pool.
For example, recall 0.95 with 100 candidates means 95 of the scan's candidates were returned. It
does not mean that 95% of all relevant passages were found. Query vectors are already cached, so
query encoding time is excluded from search timing. The selected budget is the fastest complete
setting in the saved fixed grid that reached the target on development queries. It is not a claim
that the budget is the best possible setting outside that tested grid.

To try fewer evaluation queries while keeping the full document collection and embedding cache:

```bash
.venv/bin/python run.py --eval-queries 16
```

The data limits in `experiment.toml` change the data that gets encoded. Keep those at zero for
the full collection. A subset run is marked as such; its score is not the full FiQA result.

## Exploring the next version

[Latency options](docs/latency-options.md) examines dimensions, precision, cluster-free search,
CPU bit-sliced scoring and Apple GPU matrix operations. It separates measured bottlenecks from
proposed experiments; those alternatives have not been implemented or benchmarked here.

## A denser comparison

`experiment-dense.toml` expands the probe and node-budget grids while keeping the index,
embeddings and candidate allowance fixed. The [protocol](docs/dense-sweep.md) defines the
comparisons and plotting choices before the run.

```bash
.venv/bin/python run.py --config experiment-dense.toml
```

The runner saves the shuffled schedule, every measured request, routing coverage, and empty or
short results. Unlimited branching is checked against the exact binary scan after its timer stops.
`analysis/report.md` separates routing, probe counts, branch budgets and exploration differences.
Quality intervals resample queries; repeating a query for timing does not count as more evidence.

To redraw an existing run without measuring search again:

```bash
.venv/bin/python analyze.py runs/RUN_FOLDER
```

The analysis records its own code and input hashes, so a plotting change can be traced separately
from the search measurements. The original scatter and per-seed table remain available too.

## Where things live

| File | What it does |
|---|---|
| `run.py` | Calls the stages in order. Start here. |
| `data.py` | Loads the corpus, test queries and relevance labels; keeps their IDs intact. |
| `embeddings.py` | Encodes text, derives the shorter vectors and packs their signs. |
| `routing.py` | Builds IVF or sign-prefix buckets and chooses which ones to search. |
| `benchmark.py` | Builds the schedule, times requests, shares the float reranker and writes measurements. |
| `analyze.py` | Checks completeness, compares paired queries and draws the detailed plots. |
| `cpp/index.hpp` | The native index interface and result types. |
| `cpp/index.cpp` | Bucket storage, lookup-table scan and branch search. |
| `cpp/bindings.cpp` | Checks NumPy inputs so Python can call the native code. |

`data/` holds downloads and prepared arrays. Each run writes `queries.csv`, `summary.csv`,
`metadata.json`, `schedule.json`, `reference.csv`, `report.md`, `trace.md`/`trace.json`, and PNG/SVG plots into a new folder under `runs/`.
The trace records one query's branch choices after timing is finished. Use `--trace-query ID`
to pick a different query from the selected evaluation set. It uses the first routing method
and probe count, and the last configured node budget and exploration probability.

## Calling the index

`codes` is a contiguous `uint64` matrix, with one document per row. `assignments` is an `int64`
bucket ID per document. Queries are `float32`; selected bucket IDs are an `int64` matrix with
one row per query. Use `-1` to pad missing bucket selections.

```python
from bitplane_index import Index

index = Index(codes, assignments, dimensions=256)
result = index.search(
    query_vectors,
    selected_buckets,
    candidate_limit=100,
    node_budget=128,
    leaf_size=32,
    explore_probability=0.1,
    seed=42,
)
```

`result["rows"]` contains document row numbers, `scores` contains binary scores, and `counts`
says how many entries are valid for each query. The remaining row slots are `-1`. `stats` has
work counts and the stopping reason. `scan` takes the same arrays and candidate limit, without
branch settings. `run.py` and `benchmark.py` show where these arrays come from.

In the C++ files, `std::vector` is a resizable owned array and `std::uint64_t` is a word holding
64 bits. A `const float*` points at an existing float buffer without copying it. The index owns
its document storage, so it doesn't depend on the original document arrays staying alive.
Query buffers are borrowed only for the duration of a call.

The binding file has three steps for each operation: check the arrays, call C++, and turn the
result into NumPy arrays. The registration at the bottom gives those functions their Python
names. `py::gil_scoped_release` lets other Python threads run during the C++ work; the surrounding
block reacquires the lock before creating Python result objects.

## The two comparisons

First hold the buckets fixed and compare **LUT scan vs bitplane search**. The scan computes
float-query/binary-document dot products using four-bit lookup tables. Bitplane search visits
groups in order of sign-mismatch cost, then uses that same scorer on small groups.

Then compare **IVF routing vs sign-prefix routing**. These are separate changes: a poor bucket
choice can lose a document before either fine-search method sees it.

The `ivf/float` result is Faiss IVF-Flat using float scores to choose candidates. It is useful as
an outside baseline, but differs from the binary scorer. Every method gets the same candidate
limit and the same 768-dimensional float reranker.

## Knobs worth changing

| Setting | Meaning |
|---|---|
| `routing.probes` | Number of coarse buckets to search. |
| `routing.sign_bits` | Prefix length for sign routing; these are not learned IVF centroids. |
| `search.node_budgets` | Maximum popped branch nodes per query, across all selected buckets. `0` means unlimited. |
| `search.leaf_size` | Score a node directly once it holds this many documents or fewer. |
| `search.explore_probabilities` | Probability of choosing a random alternative queued node instead of the cheapest one. |
| `search.seeds` | Seeds used for probabilistic exploration. Each query adds its row number. |
| `search.candidate_limit` | Number of candidates allowed into float reranking. |
| `search.top_k` | Number of final results. |

Start with one probe count and vary the node budget. Then vary probes with other settings fixed.
The default sweep includes three seeds for random exploration and just one for deterministic runs.

### What the probability changes

Even at probability `0`, search can revisit an opposite-sign branch. It simply picks the pending
branch with the lowest total mismatch penalty first.

At `0.1`, roughly one in ten choices instead picks a random non-best pending branch, when one
exists. Both children remain in the queue. The probability changes the order of work; it does
not delete branches or represent a confidence score.

With no node limit, all policies reach the same binary top candidates within the selected
buckets. With a limit, their results can differ. More random exploration is not necessarily better.
Keep the seed rows separate when comparing it.

## Read the numbers

- **Binary recall:** overlap with the exact binary scan's candidates in the same buckets.
- **Float recall:** final results recovered from an exhaustive full-vector search of the corpus.
- **nDCG / precision / relevance recall:** results compared with FiQA's relevance labels.
- **Retrieval time:** routing, the native call and shared float reranking, with query embeddings
  already cached. This is not text-to-answer latency.
- **Native time:** query table/order preparation, traversal/scoring and top-result selection.
  Python validation and result allocation are counted by the outer call timer instead.

Build and embedding-preparation costs are separate. Logical native storage counts codes, bitplanes
and row IDs; it is not process RSS or peak frontier memory. A short node budget can return fewer
candidates than requested. That is visible in the measurements, with no hidden fallback.

## A few details that matter

The default encoder is `nomic-ai/nomic-embed-text-v1.5`: full 768-D output, layer normalization,
then prefix slicing and L2 normalization. Queries use `search_query:` and documents use
`search_document:`. Search uses 256 signs; reranking uses the full normalized vectors. Shorter
Matryoshka vectors do not guarantee that their first signs make good routing buckets.

The index is static. Each bucket owns a compact local bitmap, so a bitplane operation doesn't
walk a bitmap covering the whole corpus. Small tests cover packing, score bounds, ties and the
case where a greedy sign choice misses the best document. See [the algorithm](docs/algorithm.md).

The earlier Exa description uses binary documents, float queries, clustering and lookup tables.
This repository reconstructs those public ingredients for comparison; it does not reproduce
Exa's private index or compare against its API timings.

Sources: [Exa's database post](https://exa.ai/blog/building-web-scale-vector-db),
[BEIR / FiQA data](https://github.com/beir-cellar/beir),
[Nomic model and preprocessing](https://huggingface.co/nomic-ai/nomic-embed-text-v1.5),
[Faiss index types](https://github.com/facebookresearch/faiss/wiki/Faiss-indexes).
