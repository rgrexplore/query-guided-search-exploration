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

## Isolated memory and timing calibration

The next stage runs one configuration per process. It compares a cost model with sizes
excluded from fitting:

```bash
.venv/bin/python -m experiments.run \
  --config experiments/configs/calibration.toml \
  --output results/my-calibration
.venv/bin/python -m experiments.analysis results/my-calibration
```

The default fit minimizes squared percentage error. For comparison, the original absolute-error
fit can be reproduced without overwriting it:

```bash
.venv/bin/python -m experiments.analysis results/my-calibration \
  --loss absolute --output-name analysis-absolute
```

The first recorded run is `../results/isolated-calibration-2026-09-10/`. Its original
`analysis/` directory keeps the initial fit; `analysis-relative/` keeps the revised fit.
The relative fit improved the diagnostic check, but one branch setting still missed the 20%
criterion. These sizes have now been examined, so they cannot also count as a fresh final test.

Freeze the model, then run different sizes and a new random seed:

```bash
.venv/bin/python -m experiments.run \
  --config experiments/configs/confirmation.toml \
  --output results/my-confirmation
.venv/bin/python -m experiments.analysis results/my-confirmation \
  --models results/my-calibration/analysis-relative/models.json \
  --output-name analysis-frozen
```

`--models` loads the saved coefficients and does not refit them on the confirmation data.
The original data preparation arrays live under the ignored `data/` cache. Every case has
its input configuration, process status, raw per-query observations and completed worker result.
Timeouts and memory stops remain in the result; they are not silently retried.

### What the timing model calculates

`models.cost_features` translates the paper's work counts into these terms:

- Scan: fixed work and full score-table terms.
- Bitplanes: fixed work, full score-table terms, physical bitmap-word visits and visited nodes.
- Keys: fixed work, full score-table terms, directory lookups and an approximate queue-work term.

The coefficients are fitted combined costs in milliseconds. They include memory access and
result selection; they are not timings for individual machine instructions. Predictions use
measured work counts and measured routing time. Predicting how much work a new query will need
is a separate distribution-model problem.

`analysis.py` adds the predicted search time to each query's routing time and an estimated
small enclosing-call cost, then computes the latency percentiles. It does not add stage medians.
The CSV retains failed prediction settings. Parameter tuning must use actual measured recall
and latency wherever the model is not reliable enough to distinguish the choices.

### Reading memory measurements

- `logical_bytes`: code, ID, plane and directory payload defined by the layout.
- `array_capacity_bytes`: the native vectors' retained capacity; still not total allocation.
- `rss_before_queries`: resident process memory after construction, releasing input mappings,
  and warmup. Allocators may retain memory from earlier allocations.
- `sampled_query_peak`: RSS observed after queries. It can miss short-lived allocations.
- `lifetime_peak_bytes`: the OS's process peak, including construction and warmup.

A lifetime peak below the RAM budget establishes a conservative fit for that whole measured
run. If only the build peak exceeds the budget, the worker reports query feasibility as
unresolved rather than declaring that the serving index cannot fit. Evaluation arrays and
runtime memory are charged to the process. Build time, index payload and query observations
remain separately available.

The benchmark records power configuration before and after every case. Equal settings do not
prove identical CPU frequency or a completely idle machine. Randomized case order, repeated
queries and the later confirmation help expose variation; these measurements apply to the
recorded machine and execution setup.

Recorded confirmation: `../results/isolated-confirmation-2026-09-10/README.md`. It contains
52 completed settings on new sizes and a new seed. All scan and key settings met the 20% p50
criterion; 21 of 24 branch settings did. The remaining branch errors are retained. Both
calibration and confirmation use one cluster, so they do not calibrate a learned router.
