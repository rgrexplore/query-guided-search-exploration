# Costs at larger buffer sizes

The search studies count bitmap words and scored documents. This small profile checks what
those operations cost on the same M5 Max. It uses constructed buffers and the shared C++ scorer.
It does not measure recall, train embeddings or claim a billion-document ANN benchmark.

## Run

From the project environment on macOS:

```bash
python -m experiments.buffer_costs --output results/my-buffer-costs
python -m experiments.buffer_costs --output results/my-gather-costs \
  --words 15625 15625000 --depth 8 --rows 128 --dimensions 256 \
  --gather-pools 1000000 4000000 16000000
python -m experiments.buffer_costs --output results/my-fresh-costs \
  --words 16 --depth 3 --rows 128 --dimensions 256 \
  --gather-pools 1000000 4000000 16000000 --fresh-subsets
```

The driver builds `experiments/cost_kernels.cpp`, runs its independent checks, shuffles the
case order and records the raw CSV, compiler, source/binary hashes and OS peak RAM. Source
copies under each result folder preserve the measured code. The Python runner uses the full
checkout; the C++ snapshots can be built with their saved headers. `/usr/bin/time -l` reads
macOS memory statistics and may need to run outside a restricted sandbox.

The default profile tests bitmap widths from 512 bytes to 125 MB, 13 splits per path,
128–4M sequential rows, dimensions 128/256/512/768, top 100, and five measured trials after
warmup. These are command-line parameters, not fixed assumptions in the mathematical model.

## What is timed

- `root`: allocate and fill the initial active bitmap.
- `split`: allocate two children, calculate both masks and the matching count, retain the
  alternative, and release the parent. Every step keeps the original bitmap width.
- `leaf`: enumerate set bits and calculate a checksum; full document scoring is excluded.
- `release`: release the remaining masks after the path finishes.
- `score`: build the score table, score contiguous document/ID rows and return top K.
- `gather`: the actual leaf-scoring loop, including bitmap enumeration, scattered document
  and ID reads, the shared scorer and top-K selection. The same candidate bitmap is reused.
- `gather_fresh`: the same scoring loop, with a different candidate bitmap prepared before
  every trial. Bitmap generation is outside the timer; its effect on cache history remains.

The path profile follows the 1 side of each split. It does not implement priority-queue
traversal or a recall stopping rule. Its retained alternatives and final surviving bitmap
are checked to form a complete, non-overlapping partition. The scorer is checked against
an independent direct sum; gathered results are also checked against the selected rows.

This standalone build is a component measurement. Compiler context, cache history, query
preparation, routing and the Python interface can change the complete native/API cost.
Composition still has to be checked against actual search requests.

## Recorded observations

`results/buffer-costs-2026-09-11/` has 34 cases. The 125 MB bitmap corresponds to one bit over
1B document positions. Thirteen planes plus live path masks used about 3.50 GB of process
RAM at peak. This is only the measured buffer set, not the full 256-bit search index.

The median split cost was roughly 0.34–0.37 ns per physical word for the larger buffers.
At 256 dimensions, contiguous scoring cost about 27 ns per document at 1M/4M rows. Small
candidate sets cost more per row because preparation and top-K selection are less amortized.

`results/buffer-gather-2026-09-11/` has 15 cases. With a 16M-document pool, reusing a roughly
1,957-row candidate set took 0.159 ms. A roughly 62,692-row set took 4.400 ms.

`results/buffer-fresh-2026-09-11/` has 14 cases. With the same pool size and selection densities,
changing the subset each trial took 0.455 ms for a median 1,958 rows and 8.271 ms for 62,492.
The inputs have the same distribution, not identical selected IDs. This comparison makes
cache reuse visible; it does not directly count hardware cache misses or fix CPU frequency.

All recorded cases completed with unchanged observed power configuration. Peak RAM includes
construction and input generation. It is not just the sum of the logical payload fields.

## Keep the failed approximations

A simple leaf model is:

`time = fixed cost + bitmap words × word cost + scored rows × gathered-row cost`.

Fit its nonnegative coefficients using only the 1M/4M pools, then check the 16M pool:

```bash
python -m experiments.summarize_buffer_costs \
  results/buffer-costs-2026-09-11 results/buffer-gather-2026-09-11 \
  --output results/my-warm-analysis
python -m experiments.summarize_buffer_costs results/buffer-fresh-2026-09-11 \
  --operation gather_fresh --output results/my-fresh-analysis
```

The repeated-subset model underestimated one 16M case by 45.2%. The changing-subset model
had errors of -12.8%, -18.5%, -27.6% and -8.7% across the four checked densities. One case still
misses the earlier 20% criterion. Both fits and failures are preserved in the analysis folders.

Coefficients combine costs. For example, a fitted zero intercept does not mean that score-table
preparation is free. A constant cost per scored row is insufficient across all these memory
conditions. Do not use either fit as a certified billion-document latency prediction.

## What this establishes for the RAM example

At 1B documents, 256-bit codes require 32 GB and 64-bit document IDs require another 8 GB.
The 40 GB payload already exceeds a 32 GB budget for the current layouts. No cluster or probe
choice removes that storage requirement. B adds at least another 32 GB of bitplanes, before
padding, containers and query masks. Those are whole-index costs; the 3.50 GB path-profile
peak cannot be substituted for them.

Larger-RAM projections can use the measured bitmap width, but must still identify extrapolated
scattered-row costs and omitted query stages. The largest measured document-row pool here is
640 MB, not 40 GB. Available system memory was about 29 GB after the profiles, so a 40 GB raw-row
experiment was not started. The earlier real/controlled retrieval results remain unchanged.
