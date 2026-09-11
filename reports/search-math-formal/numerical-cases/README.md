> Historical assumed-cost examples. The current implementation uses integrated C postings
> and a different measured cost model. Use `experiments/project_costs.py` and
> `results/projected-cases-2026-09-11/` for the current conditional scenarios.

# Numerical cases under a 32 GB RAM budget

These are calculated what-if cases, not new benchmark results. They instantiate the report's
formulas and give concrete conditions to test later. N=1,000,000 matches the size of the cached
scaling dataset; the favorable score distributions below are not claimed to describe that dataset.

## Common assumptions

- RAM budget: 32 GB decimal, including a 3.2 GB runtime reserve. Index plus query scratch must
  therefore fit within 28.8 GB. One query at a time; query embedding is cached.
- Return the top 100 by the same full binary score within each case. Different code widths are
  separate scenarios, not automatically equally useful semantic representations.
- Method 1: flat centroid IVF routing and full scoring of all selected rows. Routing dimension
  is 256, with an assumed 130 ns per centroid including rough selection cost.
- Full scoring: 0.4 ns per four-bit lookup group. Common setup: 20 microseconds.
- Method 2: 0.5 ns per bitmap word visit, 100 ns per node, and 2 ns per estimated sorting comparison.
- Methods 2 and 3 both pay an extra assumed 100 ns per scattered document, in addition to the
  scoring kernel. This keeps candidate gathers from being treated as free.
- Method 3: 100 ns per hash lookup, 2 ns per posting ID, weighted-prefix ordering, then only
  the remaining coordinates need scoring. The favorable cases stop after one occupied key
  because a full-score bound certifies the answer.
- IDs and additional C postings use eight bytes each. Hash slots use 32 bytes at 70% load.
  The reference C layout keeps separate postings; an integrated row-order layout could save them.
- Query buffers: 64 KiB, plus routing arrays or frontier masks. B's frontier uses a conservative
  capacity allowance. No original float document vectors, embedding model weights, construction
  buffers, network work or final float reranker are included in these search estimates.

The rounded lookup, word and node costs are near the earlier cost model, but these exact
coefficients and the new workloads have not been measured. Routing and gather costs are
especially important assumptions.

## Results

All times below are milliseconds. Budgeted RAM includes the 3.2 GB reserve.

| Case | N | Bits | Method 1 | Method 2 | Method 3 | Preferred under this case |
|---|---:|---:|---:|---:|---:|---|
| Little useful pruning; assume 32 probes are needed for 95% recall | 1M | 256 | 0.673 | 0.680, effectively a scan | Best fallback is scan | 1: simplest, least extra storage |
| Thirteen strong query-dependent coordinates outside the fixed short prefix | 1M | 768 | **at least 0.220** | **0.167** | No useful fixed-prefix shortcut; scan fallback | 2, if its assumed costs hold |
| Strong fixed first 12 bits | 1M | 256 | **at least 0.135** | 0.158 | **0.051** | 3 |
| Strong fixed first 16 bits | 100M | 256 | **at least 1.174** | 13.499 | **0.213** | 3 |
| Strong fixed first 20 bits | 500M | 256 | **at least 2.600** | **Does not fit** | **0.080** | 3 |
| Codes alone fill the budget | 1B | 256 | Does not fit | Does not fit | Does not fit | Change storage/representation or RAM |

**Why Method 1 says “at least”:** in favorable cases we optimize its cluster count while
pretending one probed cluster is enough for the recall target. That is an optimistic lower
bound in this particular flat-centroid cost model. It does not assert that a one-probe IVF
index really attains the target. Requiring more probes only raises its optimized modeled time.
The first case instead explicitly assumes 32 probes are needed.

**What “preferred” does not mean:** this is not a global optimum over every possible ANN
implementation, routing dimension, kernel, or data distribution. A hierarchical router,
faster centroid kernel, a different representation or a more integrated layout can change it.

## 1. Derivative: choosing the scan's cluster count

Let `b = ceil(d / 4) × 0.4 ns/document` and `a = 130 ns/centroid`.
For fixed probes P and balanced clusters:

\[
T_A(C)=T_0+aC+bNP/C,\qquad
C^*=\sqrt{bNP/a}.
\]

The script checks nearby integers and the RAM/candidate-count bounds. For N=1M, d=256, P=32:

- C* is about 2,510 clusters.
- About 12,749 documents are scored across 32 probes.
- Routing costs about 0.326 ms and scoring about 0.326 ms.
- With setup, total is **0.673 ms**.
- Budgeted RAM is **3.243 GB**.

If splitting avoids little scoring, Method 2 should use a large leaf size or call the scanner.
With leaf size 512 here, it costs approximately **0.680 ms** and **3.279 GB** including the
reserve. The tiny timing difference is not a convincing performance claim; the practical
point is that there is no reason to build/traverse extra bitplanes for this workload.

A fixed short prefix that provides no useful discrimination also gains nothing from key
search; its sensible fallback is the same scan. Giving it an arbitrarily low candidate quota
would make it faster by lowering recall, which is not a valid win.

## 2. A sufficient condition for the favorable cases

We deliberately use a strong, transparent condition so that we do not invent a recall percentage.
Let the h important query coordinates have minimum magnitude `a_min`, and let

\[
\sum_{i\notin I_h}|q_i| < a_{\min}.
\]

Assume the matching h-bit pattern has at least 100 documents. Then every document matching all
h important signs outranks every document mismatching any of them:

\[
\min S_{\mathrm{match}}\ge A_{\mathrm{important}}-A_{\mathrm{tail}}
> A_{\mathrm{important}}-2a_{\min}+A_{\mathrm{tail}}
\ge\max S_{\mathrm{mismatch}}.
\]

Score all documents with the matching pattern and keep their top 100. This gives **100% recall
for the binary score**, so it qualifies for 95% or 99% targets. We assume balanced occupancy of
these h-bit patterns: each contains floor(N/2^h) or ceil(N/2^h) rows.

This is a deliberately favorable, highly concentrated score distribution. It is much stronger
than merely saying “Matryoshka” or “80% prefix variance.” A real approximate-recall win may
require less concentration, but that needs measurement. This condition proves these particular
math cases; it does not claim our cached queries satisfy it.

## 3. When bitplanes win

Set N=1M, d=768. Put the thirteen important coordinates outside the fixed short lookup prefix,
but within the 256 routing dimensions. Their positions may vary across queries, so a single
fixed short-key table cannot simply follow them. Bitplanes can use the query's magnitude order.

Choose one cluster and probe it, leaf size 128. The matching pattern has at most 123 documents:

- 13 splits + one leaf = **14 visited nodes**.
- Each mask has ceil(1M/64) = **15,625 words**.
- Total word visits = **218,750**.
- Score **123 full 768-bit documents**; include their scattered-read penalty.
- Include query-dimension sorting and node handling.
- Total: **0.167 ms**.
- Budgeted RAM: **3.404 GB**.

Method 1's optimistic one-probe optimum is 769 clusters and **0.220 ms**. It may require more
probes for real recall. Under the model, B is about **1.31× faster even than that optimistic
scan estimate**. A short fixed prefix with none of these important coordinates cannot obtain
that pruning benefit; it should fall back to scanning or use a different query-adaptive index.

**Sensitivity:** B's advantage disappears against the one-probe scan bound if centroid routing
cost falls below about **70.6 ns per centroid**, or if the bitmap-word cost rises above about
**0.740 ns**, holding other costs fixed. These are concrete values to measure next, not guarantees.

## 4. When walk backwards wins

Put the important coordinates in a fixed prefix that can be indexed in advance. Under the score
gap above, the ideal prefix key's posting list contains every possible top-100 result. One lookup
and full-score evaluation of that list are enough; a bound excludes all other prefix keys.

| N | Strong prefix | Rows from ideal key, upper count | Keys tried | C latency | C budgeted RAM |
|---:|---:|---:|---:|---:|---:|
| 1M | 12 bits | 245 | 1 | 0.051 ms | 3.248 GB |
| 100M | 16 bits | 1,526 | 1 | 0.213 ms | 8.003 GB |
| 500M | 20 bits | 477 | 1 | 0.080 ms | 27.248 GB |

The 500M case is faster than the 100M case because it assumes a **stronger 20-bit prefix instead
of 16 bits** and retrieves fewer candidates. It is not a same-parameter claim that adding data
makes the query faster.

The win comes from avoiding both a centroid sweep and repeated full-cluster bitmap reads.
It is not due to ignoring empty lookups: this particular score/density condition needs only
the first lookup. With a weak prefix or a sparse key space, that condition fails and the empty
lookup count can dominate instead.

**Architectural overlap:** these C wins are against the stated centroid-IVF full-scan baseline.
If Method 1 is allowed to use the same direct prefix-key routing and posting layout, it becomes
very similar to Method 3. The result then identifies a good architecture—direct key routing for
that workload—not a categorical victory of one name over every other implementation.

## 5. RAM becomes decisive

At 500M documents and 256 bits, B's codes, IDs and second bitplane copy already need about
**36 GB**, before query RAM or the reserve. Its modeled budgeted total is about 41.9 GB, so
it is excluded. The reference C layout, including its extra posting references and hash slots,
fits at about **27.25 GB** including the reserve.

At 1B documents, 256-bit codes alone take **32 GB**. Eight-byte IDs add another **8 GB**.
There is no room in the stated budget for the chosen representation, even before queries.

Ignoring directories and scratch gives these *upper ceilings*, not guaranteed capacities,
when 28.8 GB is available after reserve:

| Reference layout | Bytes per document | Approximate ceiling |
|---|---:|---:|
| A: code + ID | 40 | 720M documents |
| B: code + ID + bitplanes | 72 | 400M documents |
| C: code + ID + extra posting | 48 | 600M documents |

Padding, centroids, hash slots and query RAM reduce the ceilings. A different layout can change
them; in particular C may integrate postings with row order. This table is not a lower bound
for all conceivable compressed representations.

## Reproduce

```bash
python3 reports/search-math/numerical-cases/calculate.py
```

`results.json` records every chosen configuration, individual memory terms, assumptions and
calculated time. The script checks the scan derivative's nearby integer choices and the large-N
RAM rejection. No corpus is loaded and no new benchmark is run. The next step is to measure
whether the favorable work counts, score concentration, routing cost and key occupancy occur
in actual data; the existing PDF and benchmarks have not been changed.
