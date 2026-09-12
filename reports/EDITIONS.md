# Report editions

All six PDFs are retained under `output/pdf/`.

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
