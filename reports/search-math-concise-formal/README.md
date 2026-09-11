# Searching binary embeddings — formal edition

This formal edition presents the same methods, mathematical assumptions, measurements,
and worked examples as `../search-math-concise/`. The original plain-language PDF is preserved.
The shared representation and scoring choices are attributed to Exa's published description.

## Build

From the repository root, with TeX Live available:

```bash
bash reports/search-math-concise-formal/build.sh
```

The result is `output/pdf/search-methods-concise-formal.pdf`. Included figure PDFs allow a normal LaTeX build
without downloading data or installing Python packages. The section files contain the
editable text and inline pseudocode. Code and result paths in the paper refer to the full
repository; the source archive contains the files needed to rebuild the paper.

## Scope

The formal presentation removes personal narrative, contractions, and rhetorical prompts.
Technical terms retain their definitions and examples. The study's negative results, failed
approximations, and limits on large-scale forecasts remain explicit. No new experiment is
claimed. Original editions and subsequent revisions are preserved in Git.
