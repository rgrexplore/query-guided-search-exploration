# Recall, latency and RAM

The current page adapts the supplied `ann-cost.html` layout. Its unmodified snapshot is
`archive/claude-original.html`. The simpler four-control page remains in Git at `f3de964`;
its modules remain available for the earlier calculation checks.

## Run

From the project root:

```bash
python3 -m http.server 8767 --bind 127.0.0.1
```

Open **http://127.0.0.1:8767/calculator/**. No build step is needed. The new page uses
`optimizer-v2.mjs`, `optimizer-v2-ui.mjs` and `optimizer-v2.css`. It keeps the supplied
sidebar, three cards, diagrams, cost bars and expandable explanations. The default theme is
dark; the theme button also supports a light version.

## What the configuration search answers

For fixed N, D, RAM and concurrent query count, evaluate a finite grid and choose the lowest
estimated latency meeting each **modeled** recall target and the RAM check. It is not a
measured global optimum. The main graph's horizontal axis is modeled recall@100, and its
vertical axis is estimated query latency. Click **Inspect** to load a target-table point.

A missing target means no configuration in this grid passed both tests. It does not prove
that the method cannot reach the target. The CSV keeps all configurations, including memory
rejections. Graph binning includes the method in its key so one method cannot erase another.

Grid:

- A/B: both sign-prefix and k-means routing; cluster counts 1, 2^6, 2^8, …, 2^20; probes are
  powers of two through all clusters.
- B: leaf sizes 32, 128, 512; total node budgets 128, 2,048, 32,768, or unlimited.
- C: global hash table; 8, 12, 16, 24, 32, 48, 64-bit keys; candidate targets 100, 1K, 10K,
  100K, 1% of N, 10% of N, N, with duplicates removed and targets capped at N.

The default 1B / 256-bit / 1 TB / one-query grid has 2,974 configurations. At the current
assumptions 2,066 pass the RAM check and 908 are excluded. Counts can change with inputs.
The model can favor B at one target and A at another. That is a hypothesis to benchmark,
not a claim that B has demonstrated a speedup.

## Corrections to the supplied calculation

These failures were reproduced before replacing the calculation:

| Problem in the supplied page | Correction |
|---|---|
| Global C changed its target from 10,000 to 59 when A/B clusters changed from 4,096 to 2^24 | C has a global target capped only by N; it is swept once and has no cluster/probe settings |
| C's 1B-document memory total was 32.27 GB although 256-bit codes plus 64-bit IDs already need 40 GB | Count IDs/postings, hash slot capacity, query scratch and reserve |
| Card C used global memory while the memory panel added clustered memory | One `evaluate()` result feeds cards, memory, graph, target selection and CSV |
| At 1M / budget 32,768, B counted 256M words using half the nodes as splits | Use archived split counts and count leaf mask visits too; the held-out reference has 352,466,875 split and 159,533,125 leaf words |
| Float centroid dot products were priced as binary table lookups | Count float terms separately with an explicit assumed rate, plus cluster selection |
| Weighted key search was priced using an unweighted Hamming-ball count | Count combinations by a representative quantized weighted penalty |
| Radius to enough candidates was called the first-hit radius | Separate the expected first-hit penalty from the candidate-target penalty |
| Recall conditioned on just the rank-100 boundary score | Average over the entire top-100 score tail in the hypothetical normal model |
| A one-number RAM check ignored query-time allocation | Include frontier / enumeration queue bounds and concurrent queries |
| “Fastest” was shown across unequal recalls and RAM-infeasible settings | Compare only qualifying points at a specified modeled recall target |

## Time model

The existing `measurements.json` and its source CSV are unchanged. Rates are fitted on the
archived validation rows, not the held-out test rows:

- Fixed setup: about 0.0199 ms.
- Four-sign score term: about 0.3977 ns, including average result selection.
- Bitmap word visit: about 0.4963 ns, averaged over split and leaf work.
- Node handling: about 87.77 ns, allowing for frontier handling and allocations.

The largest absolute relative timing error over 23 held-out test summaries is about 27.8%,
**given their observed work counts**. This does not bound error for extrapolated work or new
structures. The 1M reference remains 25.14 ms scan versus 244.98 ms branching at 99.26%
weighted-binary recall. The source scanner is not Exa's production implementation.

Unmeasured allowances are shown on the page: float centroid term 0.5 ns, cluster-selection
entry 2 ns, configurable hash lookup cost, key word 2 ns, enumeration comparison 5 ns, posting
read 2 ns, and scattered scoring 1.2 times scan. All are estimates, not measured instruction
latencies. Latency is for an isolated single-thread CPU query; concurrency only scales memory.

Branch work uses the closest archived flat-pool size. Node budget per probe is translated
into that pool's scale, then document, split and recall counts are interpolated between saved
validation settings. Leaf sizes other than 32 rescale node work by 32/leaf size. Selected
cluster loads and this leaf-size adjustment are assumptions; no new clustering experiment
was run. Multiplying router recall by transferred within-cluster recall is also an assumption.

Both split and leaf visits multiply by the modeled selected-cluster bitmap width. A node visit
is not a single Boolean operation. Logical traffic includes two child initializations, active
and bitplane reads, child writes, leaf reads and candidate code reads. It is not DRAM traffic.

## Memory model

All displayed memory totals come from the same result used by the optimizer:

```text
RAM check = stored index + concurrent queries × query scratch bound + 10% RAM reserve
```

The reserve is capacity kept free, not additional index data. The page separately shows index,
scratch and reserve. GB/TB use decimal bytes. The check excludes building the index, stored
original float document vectors, embedding execution, and a separate reranker.

Stored data:

- Packed codes: N × 8 × ceil(D/64), including 64-bit padding.
- Document IDs/postings: 8N bytes for every method, including global C.
- Cluster offsets and float centroids where used.
- Bitplanes: D × 8 × floor((N + 63C)/64), a padding bound for C nonempty clusters.
- C: up to min(N, 2^h) occupied keys; 32-byte hash slots at 70% load, plus the ID postings.

Query memory includes input/table buffers and result heaps. A allows space for routing scores
and indices. B allows pending masks and node structures, vector capacity growth and child
buffers. Pending nonempty nodes are disjoint sets, so the count is bounded by the selected
pool and by P + visited nodes. To avoid treating the average cluster as the largest cluster,
the scratch calculation uses the whole modeled selected pool as its largest-cluster bound.
For unlimited visits, the bound can be very loose. C allows a 24-byte enumeration node per
key attempt with a capacity-growth allowance. Candidate IDs can be streamed from postings.

A configuration exceeding this bound is excluded from the conservative search, but is not
proven impossible in a tighter implementation. Conversely, “fits” is conditional on the modeled
selected pool and these layouts, not a measured process-RAM guarantee. Peak memory needs a
real run before choosing a service configuration.

## Recall and the evidence mismatch

The provided `router256.py` defines truth using **float top 100 at 256 dimensions**.
`shortcode.py` defines truth using **float cosine top 100 at 768 dimensions**. The scaling
CSV uses **weighted binary top 100**. Those observations must not be combined as a validated
binary-recall curve. The scripts do not include saved raw results and their full provenance
in this repository; reported figures remain source-reported.

The new curve is explicitly a **hypothetical score-distribution model**. It assumes a joint
normal full score S and routing or partial-key score R with correlation rho. Recall is:

```text
P(R exceeds its selected-fraction threshold | S is in its top-100 tail)
```

The model averages over 48 midpoint quantiles across that tail. This fixes conditioning on
only its boundary score, but it does not make the distribution assumption true. Routing
correlations (.348/.808 at 4,096 clusters) and load curves are inherited transfer assumptions,
not an independently validated fit. Their dependence on cluster count and N is uncertain.
The input D changes costs; it does not supply evidence about embedding quality.

C assumes an independent-bit mismatch distribution using reported marginal rates. A
representative half-normal magnitude profile is quantized to positive integer weights.
Dynamic counting computes both the number of flip combinations and their probability mass
at each weighted penalty. All combinations tied at the stopping penalty are included, so
candidate counts can exceed the requested target. This differs from an unweighted Hamming
shell and from flipping one new bit at a time. The representative weights, independence,
quantization and partial-score correlation sqrt(h/D) are all assumptions.

The target table therefore cannot certify 80%, 90% or 99% real recall. It proposes concrete
configurations for a benchmark. A result that reads all documents can be exact for the binary
score, but its latency and memory remain modeled.

## Checks

```bash
node --test calculator/model.test.mjs calculator/optimizer-v2.test.mjs
```

Seventeen checks cover the original source hash and baseline formulas, global C invariance,
IDs and scratch in RAM, concurrency, recorded split/leaf counts, probing all clusters,
weighted key counts against exhaustive enumeration, candidate/first-hit separation, normal
model limits, Pareto selection and rejection of every 1B / 32 GB configuration.

The page is also exercised in the browser: initial render, target inspection, low-RAM empty
state, numeric/slider changes, expandable steps, theme, mobile layout and CSV download.
The original snapshot is evidence of the starting point, not a second supported calculator.
