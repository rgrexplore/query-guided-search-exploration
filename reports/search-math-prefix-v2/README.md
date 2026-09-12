# Bitplanes and prefix Backward Walk

1. This edition defines Backward Walk as prefix relaxation: `101101 → 10110* → 1011**`.
2. It starts at the query's own prefix, removes one trailing bit at a time, and scores only newly added rows.
3. The introductory wording follows the project draft closely, with duplicate passages removed and technical corrections called out below.
4. The proposed prefix index has not been implemented or benchmarked. Its cost and memory sections describe a static sorted-key layout to check in the next experiment.

## Build

```bash
bash reports/search-math-prefix-v2/build.sh
```

1. Output: `output/pdf/search-methods-prefix-v2.pdf`.
2. The source archive includes the complete LaTeX source and the historical results table. Diagrams are drawn by LaTeX, and the prefix-growth chart is included as a PDF asset; no corpus or Python environment is needed to build the report.
3. Earlier PDFs remain under `output/pdf/`. Their contents are unchanged.

## Method versions

| Name | Meaning | Status |
|---|---|---|
| A | Select clusters, score all their rows | Existing code and measurements |
| B | Bitplane splits ordered by query magnitudes | Existing code and measurements |
| C_key | Enumerate distinct keys by mismatch penalty | Earlier `cpp/key_index.cpp`; historical results only |
| C, prefix Backward Walk | Widen a fixed prefix, scoring new ranges once | Proposal; new measurements pending |

## Reading the numbers

1. The three-row examples explain the operations; they are separate from the unchanged million-document benchmark.
2. The prefix population table assumes independent balanced bits in one global cluster. It estimates row counts, not recall.
3. The sorted-key memory calculation is a proposed field layout, not measured process RAM.
4. Section 4 labels the old C timings as `C_key`. They do not establish the speed or recall of prefix relaxation.
5. No data-generation, native search, or benchmark code was changed for this PDF revision.

## Technical clarifications

1. HNSW is a graph with layers, not a root-to-leaf tree, and its distinction from prefix lookup is not restricted to float versus binary data.
2. Matryoshka supports useful trained float-prefix lengths. It does not guarantee that every earlier coordinate is larger, or that every shortened sign prefix preserves recall.
3. Prefix counts can stay unchanged when a sibling is empty. New ranges must exclude rows already scored.
4. Stopping at a candidate count is approximate. The first proposed version completes each depth across all opened clusters before checking that count. A target of zero means continue to depth zero.
5. Sorted arrays may require row movement or rebuilding on insertion. Cheap online updates remain a hypothesis, outside the first static experiment.
