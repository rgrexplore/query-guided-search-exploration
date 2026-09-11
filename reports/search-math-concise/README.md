# Three ways to search binary embeddings — concise edition

A connected explanation of the shared score, scan, bitplane branching, weighted key lookup,
recall, measured results, and conditional larger-corpus choices. The full derivations and
historical experiments remain in `../search-math/`.

## Build

From the repository root, with a TeX installation on PATH:

```bash
bash reports/search-math-concise/build.sh
```

The compiled report is written to `output/pdf/search-methods-concise.pdf`. Rendered figures are
included, so building the PDF requires no Python environment or dataset. To redraw the new
figures, use the project environment:

```bash
.venv/bin/python reports/search-math-concise/make_figures.py
```

The shared flow diagram and its Mermaid source are reused from the full report. The remaining
figures are generated from explicit example values and the saved real-target CSV. Changes to
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
- [ ] Inspect the final layout and confirm the page count.
- [ ] Rebuild from the source bundle and record final verification.
