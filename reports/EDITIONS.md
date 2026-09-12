# Report editions

All PDFs are retained under `output/pdf/`.

The [15-page fixed-dataset report](../output/pdf/search-methods-fixed-data.pdf) follows the
requested outline (pipeline, idea, two methods with index diagrams and pseudocode; time, memory
and recall calculations; per-method optima; one experiment on unchanged documents and queries
with only index parameters varied). Its [source and reproduction guide](search-fixed-data/README.md)
lists the commands. The earlier editions below are unchanged.

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
