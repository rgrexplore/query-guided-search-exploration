# Search cost explorer

A local calculator for scan, branching, and key-probing ideas. Change the inputs, inspect the
work counts, and compare the time model with the saved MS MARCO measurements.

From the project root:

```bash
python3 -m http.server 8767 --bind 127.0.0.1
```

Open **http://127.0.0.1:8767/calculator/**. No packages or build step are needed to view it.
Serve the project root so the links to the original measurements work too.

## Try these first

1. Keep the default one-million-document case. Select **Dense bitplanes** and expand its
   calculation: 352 million split-word visits plus 160 million leaf-word visits.
2. Compare that with **Packed scan**: 64 million four-sign lookup terms. A word visit and a
   score term are different kinds of work, so their counts cannot be compared as equal-cost instructions.
3. Change the first stage to similarity clusters or sign-bit buckets. Adjust the number of
   visited clusters. Use the measured-work table to load a smaller-pool branch profile if needed.
4. Try sparse bitplanes. Change the assumed active documents per split and sparse-word cost.
   Any modeled advantage depends on those assumptions; this implementation has not been benchmarked.
5. Try short keys and substring tables. Watch both expected candidates and stored payload.
   A fast lookup that finds almost nothing does not solve the retrieval problem.
6. In the small query example, change random exploration and step through the visits.
   The node budget stays fixed. Even at zero probability, alternative branches can still be visited.

The graph varies N while keeping other controls fixed. It does **not** retune each method to
matched recall. Branch survivor counts are capped by leaf capacity. A workload with more
nonempty leaves than scored documents is inconsistent; its curve is omitted and its table
entry is marked. Feasible counts are a necessary check, not proof that a real search will
produce that workload.

## Files

| File                      | Job                                                                             |
| ------------------------- | ------------------------------------------------------------------------------- |
| `model.mjs`               | Work counts, probability calculations, timing model, and the small query search |
| `app.mjs`                 | Controls, SVG plot, tables, and query steps                                     |
| `style.css`, `index.html` | Layout and page text                                                            |
| `measurements.json`       | 100 complete validation/test setting summaries from the saved run               |
| `export_measurements.py`  | Regenerates the data and fits four time coefficients                            |
| `model.test.mjs`          | Checks formulas against enumeration, source bytes, and known search results     |

## What the time estimate means

All estimates are for **one query, one CPU thread, cached embeddings, top 100 binary candidates**.
They exclude embedding, network calls, building the index, and float reranking. The small
browser example uses top 10 instead.

```text
time = fixed setup + routing
     + score terms × score cost
     + bitmap work × bitmap cost
     + visited nodes × node cost
     + key lookup / enumeration / dedup work
```

Costs in ns become ms by dividing by 1,000,000. These are effective measured or assumed costs,
not CPU instruction latencies. Scoring includes average top-100 selection work. The dense node
term includes an average allowance for allocations and the pending-node queue. Input size,
cache behavior, query weights, heap replacements, and compiler optimizations can change them.
Moving to GPU or batching queries needs different measurements; there is no universal CPU-to-GPU multiplier.

`export_measurements.py` fits nonnegative coefficients on **validation rows only**:

- Scan: fixed ms and ns per four-sign lookup term.
- Branch: additional ns per bitmap word visit and ns per visited node, with scan costs held fixed.
- Each residual is divided by that row's measured time, so the largest cases do not drown out
  the small cases. The fitted model minimizes squared relative residuals.
- Counters are means, while the time target is a median; this is an aggregate approximation.

The browser checks the model against all 23 held-out test setting summaries. At the original
coefficients the largest absolute relative error is about **27.84%**. This is a check of time
**given observed work**, not a prediction of how many branches a new query needs. The bounds
of that observed error are not confidence intervals for new methods or larger datasets.

The actual one-million-document test measurements are 25.1365 ms for scan and 244.9848 ms for
branching at budget 32,768 and recall 0.9926. The current model predicts about 25.475 ms and
261.16 ms from those counts. Inspect the page for the exact current coefficients and errors.

Sparse word access, counter operations, hashing, key enumeration, and routing have editable
**unmeasured** costs. The calculator shows them so we can decide what to measure next. It
cannot establish a speedup from those assumptions. Statements copied into `some-requests.md`
about other prototypes are not treated as verified benchmark evidence here.

## Work formulas

Let N be total documents, D dimensions, C clusters, and P visited clusters. With balanced
clusters, n = N/C documents per cluster and M = nP selected documents. Sign routing with r bits
has C = 2^r possible buckets. Similarity clustering has its own C; it is not first-r-bit routing.
With no routing, C = P = 1. Routing time is supplied separately.

Packed row bytes are `8 × ceil(D/64)`. A dense bitplane occupies W = ceil(n/64) 64-bit words.
The calculator uses expected bucket sizes; real empty buckets, imbalance, and correlations
need observed counts. More clusters alone cannot predict routing recall.

### Packed scan

Score M documents, each with G = ceil(D/4) table lookups. There are 16 table entries for each
group of four query values. The source scanner builds those entries once per query. The fixed
cost is fitted at D=256, so its setup component becomes less reliable when changing D.

Logical code bytes read: `M × 8 × ceil(D/64)`. Stored payload: packed codes plus one 8-byte ID
per document. This ideal scan layout omits the bitplanes that the current shared native Index
also builds for scan; it is not the measured process memory.

### Dense bitplanes

B means visits per probed cluster. A supplied fraction splits; the rest are leaves. Both are
workload assumptions, or observed means loaded from a saved setting. They are not inferred
from a recall target. With S splits and L leaves over all visited clusters:

```text
split word visits = S × W
leaf word visits  = L × W
fully scored documents = min(M × scored fraction, L × leaf size)
```

Per split word, the source performs two ANDs, a population count, complements and an addition,
reads the active mask and plane, and writes both children. Creating the two vectors also
zero-initializes them. The logical byte estimate counts **48 bytes per split word** (16 bytes
read, 16 written, 16 initialized), **8 bytes per leaf word**, and packed code reads for scoring.
Compilers, cache reuse and allocation behavior can change actual traffic. It is not a measured
DRAM-byte counter. Queue traffic and score-table accesses are not included in that byte ledger;
their average time is covered by the effective coefficients.

Stored payload adds `C × D × W × 8` bitplane bytes. One active dense node needs `W × 8` mask
bytes. The pending queue can hold many such nodes, so this per-node figure is not peak RAM.
The saved runs record process peak RAM separately.

### Sparse bitplanes

Retain `(word ID, mask)` pairs only where a node still has active documents. The model assumes
16 bytes per pair. With a active documents uniformly scattered through a bucket, the expected
number of occupied full words is approximately:

```text
floor(n/64) × [1 − (1 − a/n)^64]
```

The remaining partial word is counted using its actual size. This uses independent occupancy
as an approximation. If survivors are clustered or correlated it can be wrong. The assumed
mean active documents per split controls split visits; leaf occupancy uses scored documents
per leaf. Scattering is not the same as compacting all survivors into adjacent words.

The logical traffic allowance is 64 bytes per visited split word and 16 per leaf word, plus
code reads. Sparse ns/word must include pair access, building children, and less sequential
access. Using the dense node coefficient is another explicit approximation, not calibration
of a sparse implementation. Packed codes and dense document bitplanes are still stored.

### Bitplane prefilter

Read the K largest-magnitude query dimensions. Quantize each absolute query value to b bits;
for example b=4 allows integer weights from 0 to 15 after a shared scale. Add mismatch weights
in bit-sliced counters, compare their sums with a threshold, then score survivors with the
original float query. The threshold's survivor fraction is an input here, not derived from K.

Maximum counter width: `c = ceil(log2(K × (2^b − 1) + 1))`.
A simple conservative carry schedule is counted as:

```text
plane words = K × W × P
logical operations = 5 × plane words × b × c + 3 × W × P × c
```

A full-adder has five logical operations; the estimate allows a full-width add for each
quantized weight bit. The last term allows a threshold comparison. An actual implementation
can skip zero weight bits, shorten carries, or choose another adder. This is a specified cost
schedule, **not** a claim that every optimized implementation needs this many operations.
The traffic allowance is 8 bytes per input plane word plus 24 per counter operation and
survivor code reads. Counter scratch per cluster is `c × W × 8` bytes.

### Full-key probing

The ideal key has the signs of the query. Flip sets are ordered by the sum of their |query|
values. If H distinct full keys are checked, expected documents found under independent,
uniform signs are `M × H / 2^D`. H includes empty lookups. This does not assume that a stored
nearest neighbor is actually uniformly random; it is an occupancy example.

Each key costs a hash lookup plus key-word access. The model adds
`H × log2(H+1) × queue-comparison cost` for enumerating flip combinations, per visited cluster.
It is an approximate priority-queue schedule. More sophisticated key enumeration may differ.
Hash-table overhead is assumed to be 32 bytes per stored document in addition to codes and IDs.
Candidate-ID scratch is shown, but the enumeration queue and hash allocator add more memory.

### Short-key probing

Keep h bits for an inner key. Probe all keys with at most t flipped bits:

```text
V(h,t) = sum of choose(h,j), for j = 0..min(t,h)
lookups = P × V(h,t)
expected candidate documents = M × V(h,t) / 2^h
```

The model uses a sorted posting array with a dense offset directory: `(2^h + 1) × 8` bytes per
outer cluster. The directory can become expensive even if queries visit only a few clusters.
Hamming shells count flips equally. Weighted best-first short-key probes would use a different
schedule; they cannot reuse this radius formula as a weighted recall guarantee.

### Multi-index hashing

Divide D across m disjoint substrings, distributing the remainder one bit at a time. Probe
radius t in each table, read posting IDs, remove duplicates, and score the union. With substring
hit probabilities pj = V(dj,t)/2^dj, expected unique candidates are:

```text
M × [1 − product(1 − pj)]
```

This assumes independent signs between disjoint substrings. Posting reads count the sum of
hits before deduplication, and dedup gets a lookup-cost allowance. Payload adds 8N bytes of
postings per table and 32 bytes per possible occupied key, capped at N per table. This is a
sparse hash-directory size allowance, not an exact allocator layout.

Short/full-key traffic allows 32 bytes of metadata plus the key words per lookup, posting IDs,
and candidate codes. Hash collisions, queue traffic, sorting and cache misses need measurements.
The current page estimates a fixed-radius substring probe, not the paper's full adaptive
exact-neighbor algorithm.

## Quality

Three different things appear separately:

- **Measured binary recall:** overlap with exact top 100 from the same dataset, pool and score.
- **Small-search recall:** exact top-10 comparison for the generated 128-document browser example.
- **Hypothetical inclusion probability:** assume a relevant document mismatches each sign
  independently with probability p, then compute binomial inclusion for key radii. This is
  conditional on the document already being in the selected outer clusters.

A normal query-value distribution does not determine document/query correlations, the true
nearest neighbors, or semantic relevance labels. There is no justified real recall curve for
new branching, prefilter or clustering parameters without running them on real data.

The toy uses fixed illustrative query values and seeded generated document signs. It implements
penalty ordering and random pending-node selection, not every native optimization or bound.
Its exact reference sorts every toy document with the same score and row-ID tie break.

## Checks and data regeneration

```bash
node --test calculator/model.test.mjs
.venv/bin/python calculator/export_measurements.py
```

The exported JSON records a SHA-256 of the original CSV. Tests verify it, compare Hamming
volumes with exhaustive enumeration, verify the one-million-document counters, check sparse
occupancy, and compare a complete small search with exact top-10 results. Validation data
fits the coefficients; held-out test rows are displayed without refitting.

The calculator uses plain static files. It does not change the native index or launch a new
benchmark. New search structures should get their own measured experiment before their
assumed time rates are promoted to evidence.

## Sources

- [Exa: how we built a web-scale vector database](https://exa.ai/blog/building-web-scale-vector-db)
  describes similarity clustering, binary documents, float queries and subvector lookup tables.
- [Norouzi, Punjani and Fleet: Fast Exact Search in Hamming Space with Multi-Index Hashing](https://arxiv.org/abs/1307.2982)
  is the substring-hashing reference. Its Hamming guarantees do not automatically transfer to
  query-weighted mismatch costs.
- [Saved scaling report](../results/scaling-msmarco-1m/README.md) and
  [original summary CSV](../results/scaling-msmarco-1m/analysis/settings.csv) own the measurements.
