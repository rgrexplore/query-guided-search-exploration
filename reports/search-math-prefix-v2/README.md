# Bitplanes and Backward Walk

The report follows the query through clustered scanning, bitplane branching and
prefix Backward Walk. It includes Python examples, time and memory formulas,
recall calculations, and the measured parameter comparisons.

Backward Walk starts with the query's own prefix and removes one trailing bit at
a time. The sorted index locates each larger range; rows already scored are skipped.
A candidate target ends a depth after enough rows have been scored across the opened
clusters. A target of zero continues to the whole opened set.

## Read and run

- [PDF](../../output/pdf/search-methods-prefix-v2.pdf)
- [Three-document example](../../examples/prefix_search.py)
- [Native Backward Walk](../../cpp/prefix_index_v2.cpp)
- [Experiment configurations](../../configs/prefix-study-v2)
- [Saved study](../../results/prefix-study-2026-09-13)
- [Data preparation](../../docs/expanded-pool-preparation.md)

The root README maps the paper topics to their code files. The small example
prints exactly the three-row work counts used in the calculation section.

## Build the PDF

Run from the project root with a LaTeX installation on the executable path:

```sh
bash reports/search-math-prefix-v2/build.sh
```

Output: `output/pdf/search-methods-prefix-v2.pdf`. Diagrams are drawn by LaTeX;
charts are included as PDF assets. The source archive can rebuild the document
without downloading the corpus or installing the search library.

## Scope of the numbers

All methods in one comparison use the same document codes, query vectors and
binary-score references. A different model or code length is a separate comparison.
Recall measures recovery of those reference IDs, not text-answer relevance.

Stored field counts include packed-word padding in each cluster. They do not include
every allocator or runtime byte. Process RAM is measured separately. Likewise,
a split-word visit is a loop iteration containing several operations, not one CPU
instruction with a fixed universal duration.

The inline Python listings explain the algorithms. The experiments use the C++
implementation and its shared lookup-table scorer. The small numerical examples
explain the mechanics; the large-data timings come from the saved experiment records.
