# Controlled query-weight checks

The real-data comparison asks whether the current methods help on cached Nomic/MS MARCO
embeddings. A separate controlled experiment will check the conditions suggested by the
score formula. It must be labeled as constructed data, not as observed embedding quality.

## Setup, fixed before measuring

Use independent balanced document signs, dimensions 256, and fixed random seeds. Give each
query 13 coordinates with magnitude 1 and all other coordinates magnitude 0.0001. Randomize
query signs. Test two support patterns:

- Fixed: the 13 strong coordinates are the first 13.
- Query-adaptive: choose 13 coordinates uniformly without replacement for each query.

Use 100k and 1M documents initially, top 100, with the same full binary score and row-ID tie
rule for every method. Documents used by the router are the normalized sign vectors. Every
method receives the same declared router families and parameter-tuning opportunity. Retain
all generated queries, including those with too few perfect strong-bit matches.

## What the formula predicts

The total omitted weak magnitude is (256 - 13) × 0.0001 = 0.0243, smaller than one strong
magnitude. Therefore a document matching all 13 strong signs outranks every document that
misses a strong sign, regardless of the weak bits. If at least 100 documents match, the true
top 100 are entirely inside that set. Its expected population is N / 2^13; the actual count
is binomial and must be reported, not replaced by its expectation.

At 1M documents this expectation is about 122. Some queries may have fewer than 100 matches,
so the search must then explore additional strong-bit patterns. The existing exact stopping
rules and row-ID ties still apply. Weak bits are nonzero, so the baseline cannot simply skip
all remaining score terms as zero.

Bitplanes can follow a different strong-coordinate order for each query. Fixed short keys
can exploit the first setup directly but may capture only part of the important coordinates
in the second. Dense bitmap width, queue work, empty lookups and all full scores still count.
No speedup is assumed from the number of strong coordinates alone.

## Fair controls and limits

Include no-routing scan and tuned centroid/sign-prefix routing. Expand relevant cluster-count
boundaries before concluding that another method wins. For the fixed-prefix setup, also compare
a direct prefix-key routing control: otherwise a C advantage may just reflect unnecessarily
scanning centroid keys that can be addressed directly. If A and C reduce to the same candidate
selection, describe the overlap instead of claiming a fundamental algorithmic advantage.

Measure both latency and RAM, and compare only settings that meet the same recall target.
This experiment can establish behavior under its stated weight distribution. It cannot establish
that Matryoshka embeddings have that distribution, that one method always wins, or that a
billion-document projection has been measured.
