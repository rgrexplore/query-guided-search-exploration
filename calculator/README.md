# Current calculator

The current page is the configurable recall–latency–RAM calculator. Read
[OPTIMIZER.md](OPTIMIZER.md) for its formulas, assumptions, configuration grid and checks.
The guide below describes the earlier four-control version preserved at `f3de964`.

---

# Three ways to search

Four inputs: total documents, number of groups, groups opened per search, and bits per document.
The page compares:

- Baseline: cluster, then full scan.
- Cluster, then bitplane branching.
- Cluster, then walk backwards from the query's sign key.

From the project root:

```bash
python3 -m http.server 8767 --bind 127.0.0.1
```

Open **http://127.0.0.1:8767/calculator/**. There is no build step or browser dependency.

## Follow the example

With one million documents in 100 equally sized groups, each group has 10,000 documents.
Opening one group makes all three methods start with those same 10,000 documents.

A scores all of them. B splits by bits, visits the more promising side first, and later scores
small remaining groups. Its masks still span the entire opened group. C tries the query's own
key, then different combinations of flipped bits. It can spend its budget on empty keys.

The small visual examples use 16 documents and an 8-bit query. “Next step” follows actual
branch visits. “Try next key” enumerates all 256 flip sets by total query-weight penalty.
The other side of a split stays queued; it is not automatically thrown away.

## What is estimated

All methods target 100 binary candidates, with the query embedding already available and one
CPU thread. Embedding, building the index, final float reranking and network time are excluded.
Grouping adds a fixed **assumed 0.05 ms**, unless there is only one group. The page does not
claim to reproduce Exa's production timing.

`measurements.json` contains 100 complete validation/test setting summaries from the original
MS MARCO run. `export_measurements.py` fits nonnegative coefficients on validation rows:

```text
scan time = fixed setup + score terms × score cost
branch time = scan cost for surviving documents
            + bitmap words visited × word cost
            + visited nodes × node cost
```

One score term handles four signs. The fitted coefficients include average memory and result
selection work; they are not individual CPU instruction latencies. The branch word rate is an
average over splits and leaf visits. The node rate allows for the queue and allocations.
The fit minimizes squared relative timing errors, so large cases do not dominate it.

To estimate branch work in a group, the calculator takes the closest measured group size and
uses its fastest validation setting that reached 95% binary recall. It scales that setting's
node and scored-document counts to the requested group size, then multiplies the bitmap work
by `ceil(documents per group / 64)`. Other bit lengths reuse these counts.

That is a **workload assumption**, not a prediction of recall in new clusters. The chart shows
recorded group sizes, plus the current size when it exceeds the measured range. It does not
retune the new scenario to matched recall. Each opened group contributes work; a shared setup
cost is added once. Cache effects and the merging of per-group results can differ in a real run.

The maximum absolute relative timing error on the 23 held-out test settings is about **27.8%**,
given their observed work counts. That range is not an uncertainty bound for unbuilt methods.
The actual one-million-document test timings were 25.1365 ms for scan and 244.9848 ms for
branching at 99.26% binary recall. These are shown separately from estimates.

## Bitmap work and memory

For S splits, L leaves and W words per group:

- Split-word visits: `S × W`. Each does two ANDs, a bit count, complements, an addition,
  active/plane reads and two child writes. Initializing the child arrays costs work too.
- Leaf-word visits: `L × W`, even when most words are zero.
- Logical traffic estimate: `48 × split words + 8 × leaf words + survivor code bytes`.
  The 48 bytes include active/plane reads, two child writes and their zero-initialization.
  This is not measured DRAM traffic; cache reuse matters.
- Extra bitplane payload: `groups × bits × W × 8` bytes. At ten billion 256-bit documents
  without padding overhead, that is **320 GB**. GB here means one billion bytes.
- One queued dense mask needs `W × 8` bytes. Many nodes can be queued, so the extra payload
  shown on the card is not process peak RAM.

The “one split costs about X document scores” number compares a fitted average word visit
plus node overhead with full scoring. It is a rough comparison, not proof that splitting will
save time: the other branch may still be visited, and leaf reads cost time too.

## Walk backwards

The query's signs give an ideal full key. Flipping bits costs the sum of their absolute query
values. Enumerating combinations in that order is different from just accumulating flips.

The page fixes the attempt budget at **65,536 different keys per opened group**. Under
independent, uniformly distributed document signs, expected matches are:

```text
selected documents × 65,536 / 2^bits
```

For long full keys, this is extremely small. The card shows the time spent trying, not a
successful search time. This is a distribution assumption, not a claim about real recall.
The model uses unmeasured allowances of 100 ns per lookup, 2 ns per key word and 5 ns per queue
comparison, with about `log2(65,537)` comparisons per attempt. It includes expected candidate
scoring. Hash metadata allows 32 extra bytes per document; enumeration queues are additional.

A shorter lookup key is a different design. It is not silently substituted for the full-key
method to make the numbers look better.

## Source and checks

`model.mjs` owns the calculations and the small query search. `app.mjs` connects those results
to the four controls, cards and SVG graph. `index.html` and `style.css` own the page.

```bash
node --test calculator/model.test.mjs
.venv/bin/python calculator/export_measurements.py
```

Tests check the source CSV hash, the grouping example, the one-million-document split/leaf
counts, bitmap storage, empty-key expectations, available input ranges, and the complete small
search against an exact score sort. The JSON can be regenerated from the source CSV; no
embeddings or new searches are needed.

- [Original scaling measurements](../results/scaling-msmarco-1m/analysis/settings.csv)
- [Scaling report](../results/scaling-msmarco-1m/README.md)
- [Exa's published approach](https://exa.ai/blog/building-web-scale-vector-db): similarity
  groups, binary document codes, float queries and lookup-based scoring. Its clusters are not
  defined by taking the first few sign bits.

The previous seven-method calculator is preserved in commit `97a16be`. This page keeps the
three requested methods and moves fixed modeling assumptions out of the controls.
