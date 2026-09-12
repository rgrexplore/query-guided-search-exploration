# Search binary embeddings

This project compares three ways to search the same binary document codes:

| Method | What happens after selecting clusters |
|---|---|
| A: full scan | Score every document in the opened clusters. |
| B: Bitplanes | Split document masks by query signs, keep alternative branches, and score the remaining rows. |
| C: Backward Walk | Start with the query's sign prefix, shorten it one bit at a time, and score only newly included rows. |

The starting point is [Exa's vector database description](https://exa.ai/blog/building-web-scale-vector-db):
binary document vectors, floating-point query vectors, and a lookup table for scoring.
The code here is a CPU experiment using one common scorer for all three methods.

Start with the [paper](output/pdf/search-methods-prefix-v2.pdf), then try the
[three-document example](examples/prefix_search.py). It prints the returned IDs,
document scores, mask work, prefix lookups and stored bytes. It needs no dataset download.

## Install and run the example

Use Python 3.12 and a C++20 compiler. On macOS, the Xcode command-line tools provide
the compiler. `requirements.lock` records the tested Python environment.

```sh
uv venv --python 3.12
uv pip install --python .venv/bin/python -e '.[test]'
.venv/bin/python examples/prefix_search.py
```

The example returns IDs 1 and 2 with all three methods. A scores three rows;
B and C score two. B also visits three split words; C makes four boundary searches.
Those are different kinds of work, so fewer document scores alone do not prove lower latency.

## From the paper to the code

| Paper topic | Code |
|---|---|
| Binary scoring and top-K results | [cpp/score.hpp](cpp/score.hpp) |
| A: scan | [Index::scan in cpp/index.cpp](cpp/index.cpp) |
| B: split masks and keep branches | [Index::search in cpp/index.cpp](cpp/index.cpp) |
| C: sorted prefixes and newly included ranges | [cpp/prefix_index_v2.cpp](cpp/prefix_index_v2.cpp) |
| Python interface | [cpp/bindings.cpp](cpp/bindings.cpp) |
| Small numerical example | [examples/prefix_search.py](examples/prefix_search.py) |
| Encode and prepare exact references | [experiments/prepare_prefix_pools_v2.py](experiments/prepare_prefix_pools_v2.py) |
| Time one index on the fixed query set | [experiments/prefix_batch_worker_v2.py](experiments/prefix_batch_worker_v2.py) |
| Prepare settings, run, summarize and repeat | [experiments/prefix_study_v2.py](experiments/prefix_study_v2.py) |
| Check storage and loop formulas | [experiments/prefix_work_checks_v2.py](experiments/prefix_work_checks_v2.py) |
| Measure prefix counts and sign agreement | [experiments/prefix_geometry_v2.py](experiments/prefix_geometry_v2.py) |
| Draw recall and latency graphs | [experiments/prefix_figures_v2.py](experiments/prefix_figures_v2.py) |
| LaTeX source | [reports/search-math-prefix-v2](reports/search-math-prefix-v2) |

## Run an experiment

The flow is:

**texts → cached embeddings → binary codes and exact references → cluster layouts → A/B/C searches → measurements**

Data and model preparation are described in [expanded-pool-preparation.md](docs/expanded-pool-preparation.md)
and [model-data-pilot.md](docs/model-data-pilot.md). Large arrays live under the ignored `data/` folder.
Each completed pool contains document codes, query vectors, exact reference IDs and a manifest.

Once the pool named in a configuration exists, run these commands from the project directory:

```sh
.venv/bin/python -m experiments.prefix_study_v2 prepare \
  configs/prefix-study-v2/qwen-quora-d256.json results/my-quora-run
.venv/bin/python -m experiments.prefix_study_v2 run results/my-quora-run
.venv/bin/python -m experiments.prefix_study_v2 summarize results/my-quora-run
.venv/bin/python -m experiments.prefix_study_v2 repeat results/my-quora-run
.venv/bin/python -m experiments.prefix_figures_v2 results/my-quora-run
```

Use a new output folder for a new experiment. Running the `run` command again skips
attempted batches; time-limited attempts stay recorded. The small-batch preparation
command in [prefix_exploration_v2.py](experiments/prefix_exploration_v2.py) keeps the
same settings but saves progress in smaller groups.

## Parameters

All methods can change the number of clusters and how many clusters a query opens
(the probe count). `candidate_limit` is how many final IDs to return.

B adds `leaf_size`, the row count at which a set is scored, and `node_budget`, how many
sets can be visited. Budget zero means no visit limit. `explore_probability` changes
which waiting branch is visited next; probability zero still keeps both branches.
`prefer_deeper_ties` optionally visits the deeper set first when two penalties are equal.

C adds `start_depth`, the first prefix length, and `candidate_target`, how many rows
we want to score before stopping. It completes each depth across all opened clusters,
so the count can exceed the target. Target zero continues to the whole opened set.
Starting at depth zero skips widening. `max_prefix_bits` controls the stored key width,
up to 32 bits; full scores still use every bit in the document code.

## Reading the results

The dated study is saved under [results/prefix-study-2026-09-13](results/prefix-study-2026-09-13).
Within each collection, model and code length, all methods receive the same documents,
queries and exact binary-score references. Each method can choose its fastest tested
settings that reach the required recall and fit the common memory limit.

Recall here means agreement with the exhaustive binary ranking. Recovering 95 of
100 reference IDs gives 95% recall. It does not measure whether those documents answer
the text query. Changing the code length changes that reference ranking and starts a
separate comparison.

Query latency includes selecting clusters and searching them, with the query embedding
already cached. Stored field sizes and peak process RAM are reported separately.
The saved configurations, input hashes, per-query records and repeats make the choices
traceable. Earlier studies remain in their own report and result folders.
