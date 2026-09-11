# Three ways to search binary embeddings — concise edition

This 21-page report follows a query through the shared score, scan, bitplane branching,
weighted key lookup, recall, measured results, and conditional larger-corpus choices. It uses
12-point body text, six figures, three inline method listings, and one reproduction listing. The full derivations and
historical experiments remain in `../search-math/`.

## Build

From the repository root, with a TeX installation on PATH:

```bash
bash reports/search-math-concise/build.sh
```

The compiled report is written to `output/pdf/search-methods-concise.pdf`. Rendered figures are
included, so building the PDF requires no Python environment or dataset. The source ZIP
contains this report and its assets; running search experiments requires the full repository. To redraw the new
figures, use the project environment:

```bash
.venv/bin/python reports/search-math-concise/make_figures.py
```

The shared flow diagram and its Mermaid source are reused from the full report. The remaining
figures are generated from explicit example values, the saved real-target CSV, and the
projected-case JSON. Changes to
shared.mmd need Mermaid rendering; ordinary report builds use its included PDF.

## Reading and reproduction

`report.tex` loads six connected parts in `sections/`. Pseudocode appears in the method
explanations. The last section maps those explanations to the implementation and experiment
entry points. The example values come from `examples/small_search.py`.

The source selection begins at `f16ca66`. Evidence comes from the full report's frozen
`evidence/real-targets.csv` and `evidence/final-study.json`, which identify the original result
files and hashes. Final controlled comparisons use the 11 September boundary follow-up;
earlier controlled tables are not substituted for it. No new benchmark was run to write
this edition. The full report and unrelated calculator edits are preserved.

For the complete timing/recall reproduction, see `experiments/README.md` and `results/README.md`.
The small native example needs the installed extension but no corpus download. The arithmetic
check uses the original report's included evidence. To recompute the larger scenarios:

```bash
.venv/bin/python -m experiments.project_costs \
  --inputs results/projected-cases-2026-09-11/results.json \
  --output results/my-scenarios \
  --documents 1000000000 --ram-gb 32 64 1000
```

This reads frozen inputs; it does not build or query a billion-document index.

## Editing checklist

- [x] Draft the six parts from the approved 20-page plan.
- [x] Keep method pseudocode beside its explanation.
- [x] Verify the selected source results and basic example arithmetic.
- [x] Inspect the final layout and confirm the page count.
- [x] Rebuild from the source bundle and record final verification.

## Earlier prose revision

The original concise edition is preserved at commit `3d74cf4`. The revised prose keeps the
same methods, assumptions, equations, figures, pseudocode and results. The introduction
attributes the starting point to Exa's published description, and the explanations follow
the examples in a more conversational voice. The layout remains 20 pages.

## Plain-English edition

The latest pass uses short topic headings and explains unfamiliar terms through the existing
query and memory examples. Exa's shared representation and scoring choices are identified
where they are used. Methods, equations, pseudocode, measurements, and conclusions are unchanged.
The page count is allowed to grow when an example makes the explanation easier to follow.

## Plain-English rewrite

The latest edition uses shorter headings and explains unfamiliar terms with examples. It
attributes the shared representation and scoring techniques to Exa's published description.
The methods and measured results are unchanged. Earlier PDFs and sources remain in Git.
The concise edition is 21 pages and the detailed reference is 74 pages.
