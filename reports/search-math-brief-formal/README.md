# Searching binary embeddings — brief formal edition

This 10-page paper presents the three methods, a shared worked example, the main cost and
recall formulas, and the measured conclusions. It summarizes the existing formal editions;
it does not introduce a new experiment or change the methods. Earlier editions remain available
through [the edition index](../EDITIONS.md).

## Build

With TeX Live on the command path, run from the repository root:

```bash
bash reports/search-math-brief-formal/build.sh
```

The output is `output/pdf/search-methods-brief-formal.pdf`. The source has no external figure
dependencies. Its archive includes the same folder structure, so the build command also works
after extracting it into an empty directory. Building the PDF does not run any experiments.

## Evidence

The source is a shortened presentation of `reports/search-math-concise-formal/` at checkpoint
`de32b5e`. Full derivations and additional experiments remain in `reports/search-math-formal/`.
The measurements originate from the saved study at checkpoint `f16ca66`.

| Paper content | Saved source in the full repository |
|---|---|
| Real-data table | `reports/search-math/evidence/independent-results-table.tex` |
| Further real-query control | `results/one-bit-evaluation-2026-09-11/report/targets.csv` |
| Fixed-position controlled comparison | `results/controlled-boundary-fixed-2026-09-11/evaluation/report/` |
| Changing-position controlled comparison | `results/controlled-boundary-adaptive-2026-09-11/evaluation/report/` |
| Frozen eight-million-row model check | `results/native-scale-confirmation-2026-09-11/frozen-model-check.json` |
| Complete text-query timings | `results/text-queries-2026-09-11/summary.json` |
| Billion-document forecasts | `results/projected-cases-2026-09-11/results.json` |

The archive rebuilds the explanation, not the full research environment. The main repository's
`experiments/README.md` and `results/README.md` describe data, setup, and experiment commands.
