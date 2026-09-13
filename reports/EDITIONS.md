# Report editions

All ten PDFs are retained under `output/pdf/`.

The current [prefix Backward Walk report](../output/pdf/search-methods-prefix-v3.pdf)
is the 29-page edition, with a linked table of contents. It goes directly from the methods and preliminary calculations
to **Section 3: Experiments**, without the separate parameter-derivation section.
Its [source guide](search-math-prefix-v3/README.md) links the implementation and saved
measurements. Use this edition for future report edits.

The [previous prefix report](../output/pdf/search-methods-prefix-v2.pdf) is preserved
as a 31-page edition with the derivation section, together with its source archive.
Earlier prefix drafts remain in Git. The algorithms and measurements are shared across
these two editions; creating the new report did not run another experiment.

The preceding [calculations and experiments report](../output/pdf/search-methods-experiments.pdf)
keeps both document and query vectors unchanged while varying index settings. It uses the
requested section order, with 16 pages of diagrams, short steps and measured results. Its
[source and run guide](search-math-experiments/README.md) describes the corrected comparison.
Earlier editions below remain available as historical reports.

The [13-page derivation and experiment paper](../output/pdf/search-methods-optima.pdf) adds a
fresh parameter study, diagram-led methods, revised time estimates and separate-query checks.
Its [source and reproduction guide](search-math-optima/README.md) records the experiment commands.
The five earlier editions below retain the preceding study and remain unchanged.

The [10-page brief formal paper](../output/pdf/search-methods-brief-formal.pdf) is the shortest
edition. It includes the three methods, inline pseudocode, one worked example, the main
formulas, and the measured conclusions. Its [LaTeX source](search-math-brief-formal/README.md)
builds separately from the editions below.

| Style | Concise | Detailed |
|---|---|---|
| Plain language | [21 pages](../output/pdf/search-methods-concise.pdf) | [74 pages](../output/pdf/search-methods-mathematics.pdf) |
| Formal research paper | [22 pages](../output/pdf/search-methods-concise-formal.pdf) | [74 pages](../output/pdf/search-methods-mathematics-formal.pdf) |

The formal editions use abstracts and formal prose while keeping definitions and worked
examples. Each edition has a separate source directory and build script. The source archives
are stored beside the PDFs; building them does not rerun the experiments.


## Recovered fixed-data study

The [fixed-data report](../output/pdf/search-methods-fixed-data.pdf) and its
[source](search-fixed-data/README.md) are retained from the earlier parallel study.
Its C method uses weighted key search, not the prefix Backward Walk in the current
v3 report. Its measurements remain separate under `results/fixed-data-2026-09-12/`.
