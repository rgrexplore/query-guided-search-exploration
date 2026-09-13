# Searching binary embeddings: three methods, one dataset

A 15-page report that compares Exa's cluster-then-scan search (method A) with two additions,
Bitplanes (B) and Backward Walk (C), on one fixed dataset: one million MS MARCO passages as
256-bit codes, 200 queries, exact top-100 reference. Only the index parameters vary between
runs. The report derives time, memory and recall formulas, uses them to choose each method's
settings, checks every operation count against the code, and measures all 189 settings.

Result on this dataset: A is fastest at every recall target (80, 90, 95, 99%); B is 1-4%
slower with "no splits" as its best setting; C is 9-28% slower with its narrowest key. The
formulas say why (a sign split removes only 14-22% of real documents; no group or key ever
exceeds the gap that would let it be skipped) and state the conditions under which B or C win.

PDF: `output/pdf/search-methods-fixed-data.pdf`.

## Build the PDF

```bash
bash reports/search-fixed-data/build.sh
```

Figures and tables under `figures/` and `evidence/` are committed, so the PDF builds without
Python or the dataset.

## Reproduce the numbers

From the repository root, with the environment of the main README installed and the cached
dataset present under `data/`:

```bash
.venv/bin/python -m experiments.fixed_data_study run --output results/my-run     # about 40 minutes
.venv/bin/python reports/search-fixed-data/make_results.py facts --recompute    # dataset facts, Section 3
.venv/bin/python reports/search-fixed-data/make_results.py results --run results/my-run
bash reports/search-fixed-data/build.sh
```

`experiments.fixed_data_study analyze --output <run>` re-summarizes a finished run without
measuring again. The run used in the report is `results/fixed-data-2026-09-12/`; its raw
per-query records are archived in `runs/cases.zip` (unzip in place to re-run `analyze`).
