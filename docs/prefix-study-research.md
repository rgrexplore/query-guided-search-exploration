# Prefix and bitplane search: evidence for the next fixed-data experiment

Research date: 2026-09-13. Native source inspected at `7628258` on
`prefix-experiments-v2`. This is a research note, not a benchmark result.

**The literature makes these methods worth testing, but does not establish that B or C
beats the tuned clustered scan.** The most plausible opening for B is a smaller requested
result count and enough work to reach useful leaves. The actual data gives substantial
counterevidence for C at high recall: matching early sign bits loses many reference
neighbors, while the surviving prefixes can contain very many documents.

## Scope

1. Produce source-backed hypotheses and small read-only checks of the existing arrays.
2. Do not change documents, queries, reference answers, scoring, or original bit order in C.
3. New embedding training, favorable constructed queries, new hash forests, new bitmap
   layouts, and a learned per-query method selector are outside this experiment.
4. It is appropriate to vary existing index/search parameters, retain failed settings,
   compare equal recall requirements, and explain why a setting succeeds or fails.

A is a scan of selected clusters. B is weighted bitplane branch-and-bound with masks
whose width stays equal to the cluster's row count. C is original-order prefix relaxation:
`101101 → 10110* → 1011**`, scoring only newly exposed rows. Historical weighted-key
enumeration is `C_key`, a separate method.

## What the primary sources support

**Exa's public baseline.** Exa describes truncating trained embeddings to 256 coordinates,
binarizing document coordinates, retaining a floating-point query, and using small
dot-product lookup tables. Its public system searches selected similarity clusters and
then reranks additional results with uncompressed data. Thus A should receive its own
cluster/probe tuning; comparing B or C only with a global scan is too weak. The public
article describes a larger production system and register-level optimizations, not our
native implementation or its expected latency.
[Exa, Building a web-scale vector database](https://exa.ai/blog/building-web-scale-vector-db).

**Matryoshka supports useful float prefixes, not exact sign-prefix matches.** The paper
trains representations at multiple prefix lengths. Its adaptive retrieval first ranks
candidates using a smaller float representation, then reranks with larger representations.
It does not establish that nearest neighbors share every sign in a short prefix, or that
earlier coordinates always have larger query magnitudes.
[Kusupati et al., NeurIPS 2022, §4.3.1](https://papers.nips.cc/paper_files/paper/2022/file/c32319f4868da7613d78af9993100e42-Paper-Conference.pdf).
Nomic's own card reports float dimensions 64/128/256/512/768; it supplies no recall claim
for widening a 16-bit exact sign match to 15 bits.
[Nomic Embed Text v1.5 model card](https://huggingface.co/nomic-ai/nomic-embed-text-v1.5#adjusting-dimensionality).

**LSH Forest is the closest prefix precedent.** Locality-sensitive hashing, or LSH,
uses hash functions chosen so closer points collide more often. Bawa, Condie and Ganesan
build several independent prefix trees, descend toward the longest query match, and
then ascend synchronously across trees until enough candidates have been collected.
Actual distances rank those candidates. This directly supports studying prefix depth,
candidate count, and level-by-level expansion. Its probability guarantees depend on a
suitable hash family and the number of independent trees; the original paper explicitly
qualifies their strength. Our disjoint clusters sharing one original bit order are not
independent LSH trees. Their guarantee cannot be attached to C.
[LSH Forest, WWW 2005, §5.1 and Figures 2–3](https://www.cs.princeton.edu/courses/archive/spring06/cos592/bib/LSHForest-bawa05.pdf).

**PUFFINN adds a more directly relevant implementation.** Its implementation stores
indices in arrays sorted by hash code, retrieves newly matching ranges, and shortens the
prefix one bit per round. It also uses small random-hyperplane bit sketches to filter
candidates, with a threshold adjusted using current top-K results. Its theoretical
stopping condition uses collision probabilities; the paper distinguishes theory from
engineering approximations. Additional RAM can buy additional hash repetitions in that
design. C has no such independent repetitions, sketches, or probability-based stopping
rule: a candidate target is not a recall guarantee. These are useful precedents and
separate future designs, not reasons to add features to the current comparison.
[Aumüller et al., PUFFINN, §3–5](https://arxiv.org/pdf/1906.12211).

**Multi-probe LSH supports informed alternatives, not arbitrary random walks.** Lv et al.
probe additional buckets using a sequence derived from the query's position inside hash
cells. Their implemented hash family targets Euclidean distance. They reduce the number
of separate tables needed compared with other LSH methods; the paper explicitly does
not present a comparison with all other index families. Its comparison with randomly
perturbed query sampling favors carefully ordered, distinct bucket probes. It provides
no guarantee for original-order Nomic signs, B's random queue selection, or a speedup
over A. The transferable question is whether an alternative visit order finds good
candidates sooner for a fixed amount of work.
[Multi-Probe LSH, VLDB 2007, §1 and §4](https://vldb.org/conf/2007/papers/research/p950-lv.pdf).

**Weighted binary search is established, but its data layout matters.** Weng et al.
generate bucket keys in increasing weighted Hamming distance. For long codes they use
multiple substring indexes and merge candidates with a stopping rule. Their experiments
include K=1/10/100 on image descriptors, with codes up to 128 bits. This is directly
relevant to the score identity used here and to `C_key`; it is not prefix relaxation
or fixed-width bitplane traversal. The paper also explains why long exact keys can waste
work on empty buckets. Its linear-scan comparisons do not establish a win over our
tuned clustered scan on 256-bit Nomic codes.
[Weng, Zhu and Liu, Fast Search on Binary Codes by Weighted Hamming Distance, §III–V](https://arxiv.org/html/2009.08591v2).
The earlier peer-reviewed version uses weighted multi-index bucket finding and merging.
[Weng and Zhu, AAAI 2020](https://ojs.aaai.org/index.php/AAAI/article/view/6919).

**Ordinary Hamming search is a different objective.** Norouzi et al.'s multi-index hashing
retrieves exact neighbors by unweighted Hamming distance using substring tables. A small
number of mismatches does not imply a small sum of query-dependent mismatch weights.
Using ordinary Hamming results as candidates can lose a weighted-score neighbor before
reranking, so its exactness cannot transfer unchanged.
[Norouzi, Punjani and Fleet, Fast Exact Search in Hamming Space](https://arxiv.org/abs/1307.2982).

**Score bounds have useful precedents beyond hashing.** Block-Max WAND uses upper bounds
to skip portions of posting lists while retaining exact top results. It demonstrates
that avoiding full scores can pay when the bound and skip operation are cheap. The
paper also shows why a naive substitution of local bounds can be incorrect. Its sparse
term postings and block movement differ from B's dense, full-width bitmap operations.
It is support for measuring bound quality and skip cost, not a reason to copy its
speedups or add another index here.
[Ding and Suel, SIGIR 2011, §3 and §5](https://research.engineering.nyu.edu/~suel/papers/bmw.pdf).

## Direct checks on the unchanged data

The following checks read all 200 query vectors and their existing top-100 reference IDs.
They use packed document codes in place. No timing comparison, new data generation,
query selection, or benchmark run was performed. Reference overlap means agreement with
the exact floating-query/binary-document score, not human relevance.

Pool: `data/real-evaluation-2026-09-10/n1000000/`.

| File | Shape | SHA-256 |
|---|---|---|
| `codes.npy` | 1,000,000 × 4 uint64 | `14c1f435857b13ab177c22c0e74aecf1432630e1ba67289686bf8a1159273d0e` |
| `queries.npy` | 200 × 256 float32 | `d462a5c3eb9f1e082fb1a626b49f4a8050cb9f24b9d2751eaec7dd53589df93c` |
| `reference.npy` | 200 × 100 IDs | `dc2f32b0c78dadef21d6fba6e4013ae36b6113ffb3a31d988326b207f780c632` |

For depth `h`, count every document whose first `h` bits match the query signs. The recall
column is the fraction of existing exact top-K IDs satisfying the same condition. These
are global fixed-depth counts, not outcomes of adaptive C with cluster routing.

| Depth | Mean matching documents | Median matching documents | Top-1 survival | Top-10 survival | Top-100 survival |
|---:|---:|---:|---:|---:|---:|
| 1 | 632,833.00 | 765,666 | 80.50% | 80.75% | 77.545% |
| 2 | 503,192.05 | 707,016 | 72.00% | 71.85% | 67.400% |
| 4 | 330,116.19 | 539,101 | 59.00% | 58.85% | 52.675% |
| 8 | 29,643.84 | 15,809.5 | 25.50% | 21.65% | 15.390% |
| 12 | 3,217.13 | 1,183.5 | 11.00% | 7.55% | 4.790% |
| 16 | 493.67 | 117 | 5.00% | 3.05% | 1.370% |

This falsifies the balanced-independent-bit count model for this data. At depth 16,
`N / 2^16` predicts 15.26 documents, while the observed mean is 493.67. At depth 4,
52 of 200 queries retain at least 80% of their exact top-100; at depth 8, none do.
If every C query stops at depth 1 or deeper, its aggregate top-100 recall cannot exceed
77.545%, even when every cluster is open. Therefore reaching 80/90/95/99% requires some
queries to reach depth 0. Cluster selection can reduce candidates but cannot restore
neighbors excluded by a surviving prefix constraint. Different stop depths across
queries can change the aggregate tradeoff; the table is not a prediction of that curve.

For B, let `w[j] = abs(q[j])`. A binary document's score is
`sum(w) - 2 * sum(w[j] for mismatching signs)`. Let `D_K` be the mismatch weight of the
existing exact K-th result. Even with that ideal threshold already known, a branch
cannot be rejected by this bound until its accumulated mismatch weight exceeds `D_K`
(with the implementation's rounding guard and tie rule also respected).

The next table finds the earliest depth where the sum of *all* inspected weights could
exceed `D_K`. It is an optimistic lower limit on possible pruning, not a count of useful
branches. B visits the largest weights first; the original order is a diagnostic control.

| K | Median `D_K` | Median earliest possible depth, largest weights first | Same calculation, original order |
|---:|---:|---:|---:|
| 1 | 1.04156 | 6 | 17 |
| 10 | 1.55627 | 10 | 25 |
| 100 | 1.91302 | 13 | 32 |

Actual thresholds may be weaker until good candidates are found, and an all-mismatch
branch may be empty. Small K therefore has a justified advantage for bound strength,
but still needs an end-to-end measurement. The first 16 original coordinates contain
8.57% of total query magnitude on average; the 16 largest contain 18.54%.

## Bounded parameter recommendations and ways to disprove them

| Parameter | Proposed comparison | Why it is plausible; evidence that would reject the hypothesis |
|---|---|---|
| K | 1, 10, 100; use the same reference array's first K columns | Smaller K can tighten B's threshold earlier. Reject a speed claim if its best recall-qualified result remains slower than A's own best result at that K. |
| Cluster count | Retain 1/256/4096/8192; fill gaps with 1024/2048; consider 16384 if practical | Larger clusters leave more scores to avoid but make B's masks wider. Smaller clusters reduce mask width but increase routing and per-cluster overhead. Extra clusters may improve A most. |
| Probes P | Shared geometric probe counts, full traversal, and existing recall-relevant counts for every method | More probes recover routing misses but create more roots/range lookups. Compare each method's best setting, and separately compare equal-cluster/equal-probe pairs. |
| B leaf size | 32, 128, 512, and no-split control | Small leaves allow more pruning; large leaves reduce traversal. At 8192 clusters the mean cluster size is only 122 rows, so leaf512 can act mostly as a scan. Count actual splits and scores avoided. |
| B node budget | 1024/4096, a probe-scaled budget such as `max(4096, 16P)`, and unlimited | Roots and leaves both consume budget. A budget below P can expire on root splits. Reject a setting that saves work by returning inadequate recall or too few results. |
| B random exploration | Probability 0, 0.05, 0.2 on a small promising subset of settings; use several predeclared seeds | Existing code can select a nonbest queued branch. It might find useful leaves sooner, but can also spend work on poorer branches. Require repeatable equal-recall gains; keep the deterministic control. |
| C starting depth | 8, 16, 24, 32 when supported; use 0 as a scan control if available | A deep start may avoid prematurely large initial ranges. Once two settings reach the same final depth, their candidate set is the same; extra empty levels only add lookup work. Log final depths. |
| C candidate target | K, 4K, 16K, 64K as a coarse start; include 0 exhaustive and higher targets up to the selected-row count when needed | Count actual distinct scores and overshoot after completing a depth across clusters. Small targets may demonstrate low recall. Only refine a promising interval; retain failures. |

These are an initial bounded grid, not a request to run every Cartesian product. Use
counts and recall to discard unpromising regions before repeating latency measurements.
Reference-derived probe thresholds are tuning information; results on this same 200-query
batch remain in-sample findings. Do not use exact reference answers inside the query algorithm.

## Lower recall and individual-query wins

It is informative to retain the full recall/latency curve and add a declared 50% target
alongside 80/90/95/99%. A speed advantage at 50% is a different operating point; it does
not satisfy an 80% requirement. Evaluate all three methods at every declared target.
The fixed-depth results above already caution that C may need many scores even around
50%, so a low target is not a reason to expect a win.

An aggregate loss can coexist with individual-query gains. For selected fixed settings,
report the number of all 200 queries where B or C is faster and meets the same per-query
recall requirement, plus the aggregate median, mean, and tail latency. Also report
queries that fail recall. For K=1, per-query recall is either zero or one; “80% recall”
is then a batch average and must not be described as an 80% success level for each query.

Choosing the best method or parameter separately for each query after reading its exact
answer is an oracle comparison: a best-case bound using information a real search lacks.
It can diagnose headroom but cannot establish a usable system. No favorable subset or
post-hoc query transformation should replace the full fixed-data comparison.

## RAM and completion criteria

All three methods must fit the same process budget, including build peaks, routing,
duplicate arrays, query buffers, and temporary structures. Approximate logical payloads
at one million 256-bit rows are 40 MB for packed codes plus IDs, about 72 MB with B's
bitplanes before padding, and 44 MB with C's four-byte prefix keys. These are not peak
RAM measurements. Merely raising the RAM limit does not make masks narrower, improve
prefix recall, or reduce scores. A 32-bit key in a sorted array does not require a dense
directory with 2^32 entries.

A credible result retains unchanged input hashes, the shared score/reference objective,
all low-recall cases, actual work counts, full search latency including routing, and
isolated repeated timing of finalists. Where timing excludes routing, label that scope
explicitly. B or C wins only if its qualifying configuration improves the stated metric
over A's separately tuned qualifying configuration. The previous fixed-data result
favored A at the tested 80–99% targets; new literature does not overturn those measurements.

## Reproducing the geometry checks

The counts require only NumPy and the three unchanged arrays. This diagnostic performs
no searches or timings and writes no files. It follows native packing: bit `j` is offset
`j % 64` in word `j // 64`, and `q[j] >= 0` selects sign one.

```python
from pathlib import Path
import numpy as np

p = Path("data/real-evaluation-2026-09-10/n1000000")
q = np.load(p / "queries.npy", mmap_mode="r")
r = np.load(p / "reference.npy", mmap_mode="r")
c = np.load(p / "codes.npy", mmap_mode="r")
j = np.arange(256)
selected = np.asarray(c[r])
signs = ((selected[:, :, j // 64] >>
          (j % 64).astype(np.uint64)) & np.uint64(1)).astype(bool)
wrong = signs != (q[:, None, :] >= 0)
weights = np.abs(q).astype(np.float64)
penalties = (wrong * weights[:, None, :]).sum(axis=2)
prefix_ok = np.logical_and.accumulate(~wrong, axis=2)
query_key = np.sum((q[:, :16] >= 0).astype(np.uint64) <<
                   np.arange(16, dtype=np.uint64), axis=1)

for depth in (1, 2, 4, 8, 12, 16):
    mask = np.uint64((1 << depth) - 1)
    counts = np.bincount((c[:, 0] & mask).astype(np.int64),
                         minlength=1 << depth)
    rows = counts[(query_key & mask).astype(np.int64)]
    survival = [prefix_ok[:, :k, depth - 1].mean() for k in (1, 10, 100)]
    print(depth, rows.mean(), np.median(rows), survival)

ordered = np.cumsum(np.sort(weights, axis=1)[:, ::-1], axis=1)
original = np.cumsum(weights, axis=1)
for k in (1, 10, 100):
    threshold = penalties[:, k - 1]
    first_sorted = (ordered <= threshold[:, None]).sum(axis=1) + 1
    first_original = (original <= threshold[:, None]).sum(axis=1) + 1
    print(k, np.median(threshold), np.median(first_sorted),
          np.median(first_original))
```
