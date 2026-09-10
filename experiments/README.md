# Reproduce the component checks

Start with the paper in `../reports/search-math/`, then use this small run to check its storage
formulas and the native search results. It uses random signs, one cluster and cached queries.
It does not download data or measure semantic retrieval quality.

## Run

Build the native extension in the project environment, then run:

```bash
uv pip install --python .venv/bin/python --no-build-isolation --no-deps -e .
.venv/bin/python -m experiments.run \
  --config experiments/configs/small.toml \
  --output results/my-component-check
```

This assumes the project's build dependencies are already installed. For a fresh environment,
follow the root README's installation step first. Use a new output folder for each run; saved
measurements are not overwritten.

The configuration controls document counts, dimensions, queries, random seed, key width,
branch budget, leaf size, candidate target, key-attempt limit, and timing repeats. Zero work
limits let the searches finish. The small example uses exact search to check the implementations.

## Paper to code

| Question | Formula | Native operation | Evidence |
|---|---|---|---|
| How many bytes do packed codes need? | `models.packed_code_bytes` | `Index::info`, `KeyIndex::info` | `indexes.csv` |
| How much does padding each cluster's bitplanes cost? | `models.bitplane_bytes` | `Index` construction | `indexes.csv` |
| What does the occupied-key directory store? | `models.key_directory_bytes` | `KeyIndex` construction | `indexes.csv` |
| Do all methods return the correct top K? | Independent full scores in `reference_rows` | `scan`, `search` | `measurements.csv`, correctness tests |
| Which work does branching perform? | Split and leaf counts kept separate | `Index::search` | `bitplane_words`, `leaf_words` |
| How many unsuccessful keys are tried? | Enumerated keys versus occupied postings | `KeyIndex::search` | `key_attempts`, `documents_scored` |

`run.py` reads the configuration. `components.py` builds one method at a time, checks its logical
storage, runs the queries and writes the evidence. `models.py` contains the formulas independently
of the C++ implementation.

## Reading the result

- `config.toml`: the exact input settings.
- `environment.json`: Python, platform, native module location and binary hash.
- `indexes.csv`: predicted and native logical storage, allocated array capacity, build time.
- `measurements.csv`: one row per query and timing repeat, with recall and native work counts.
- `summary.csv` and `README.md`: compact comparison.

Logical byte counts omit allocator overhead, hash-node padding, routing state and runtime memory.
The driver still owns its preparation arrays. These values must not be presented as measured
whole-process RAM or proof of feasibility under a serving RAM limit.

Timings here are exploratory component observations. This run does not tune each method,
interleave independent workers, establish stable power conditions, or fit a latency predictor.
Those are the next study stages. A component test passing does not establish that its method
is fastest on real embeddings.
