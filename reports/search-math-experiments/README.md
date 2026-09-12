# Calculations and experiments

This report uses one unchanged benchmark: one million 256-bit MS MARCO document codes,
200 existing Nomic query vectors, and their reference top-100 answers. Only index settings vary.
The source follows the introduction, per-method calculations, per-method optima, summary,
experiments, limitations and conclusion in that order.

Sections 2.1 and 2.2 use one query and the same three teaching documents for all methods.
Each method has numbered steps, short bullets, diagrams, loop counts and byte counts.
The small example is separate from the unchanged million-document benchmark.

## Build the PDF

With TeX Live available, run from the repository root:

```bash
bash reports/search-math-experiments/build.sh
```

Output: `output/pdf/search-methods-experiments.pdf`. The LaTeX source archive contains the
rendered plots, numerical tables and diagrams needed to rebuild the PDF without Python or
the embedding cache. Earlier PDFs remain available in `reports/EDITIONS.md`.

## Run the comparison

Use the installed project environment and the existing cached benchmark arrays. The paths
are explicit in `experiments/fixed_data_comparison.py`:

- Inputs: `data/real-evaluation-2026-09-10/n1000000/`.
- Prebuilt cluster layouts: `data/real-scale-2026-09-10/n1000000/routers/`.

The legacy directory name does not define a new data split. Every setting in this experiment
uses all 200 query rows. The document codes, floating-point source documents, query vectors
and reference answers are hash-checked before and after each stage.

```bash
.venv/bin/python -m experiments.fixed_data_comparison sweep \
  --output results/my-fixed-data
.venv/bin/python -m experiments.fixed_data_comparison extend \
  --output results/my-fixed-data
.venv/bin/python -m experiments.fixed_data_comparison repeat \
  --output results/my-fixed-data
```

The first command compares 128 settings: cluster counts 1, 256, 4096 and 8192; probe counts
at the recall boundaries; B's leaf sizes and branch limits; and C's short keys. The second
adds 128 settings with larger branch budgets and 16-, 20- and 24-bit keys with bounded
lookup work. The last repeats the 12 selected settings in three processes, using the same
queries and index parameters. No embedding is regenerated or modified.

All methods share the same score, reference answers, one CPU thread, and a 32 GB decimal
process-memory limit. Settings that miss a target remain in the records and cannot win
at that target. Results describe the tested range on this benchmark.

## Inputs and records

The saved run is `results/fixed-data-comparison-2026-09-12/`. Its `inputs-manifest.json`
contains array hashes and query IDs. The `inputs/` files show exactly how each cluster
count's probe cutoffs were calculated from those reference answers. `sweep/`, `extra/`,
and `repeats/` retain configurations, work counts, timings, memory and failures.

Each completed stage has a `cases.zip` and `archive.json`. Extract the ZIP in its containing
directory to restore the raw case tree. The embedding arrays are not included in the paper
archive. Reproducing the exact measurements requires the caches identified by the manifest;
the root README describes the data-preparation pipeline. A newly generated embedding set
must be treated as a separate benchmark if its hashes differ.

Regenerate plots and numerical tables from the saved run with:

```bash
.venv/bin/python reports/search-math-experiments/make_results.py
```

For another completed run, pass `--study results/my-fixed-data`. The command also checks
input-derived work counts and index-data sizes. It needs the original input arrays for
these calculations. The accompanying narrative describes the recorded run; review its
numbers before publishing a different run.

## Result and scope

A had the lowest repeated median at the 80%, 90%, 95% and 99% recall targets after each
method could choose its own settings. At 99%, the median times were 8.74 ms for A, 8.99 ms
for B, and 9.73 ms for C. All three scored about 295,858 documents per query; B added mask
reads and C added key lookups. Larger keys were also measured, without producing a faster
qualifying result.

The historical query-shape examples are separate experiments and are not used as evidence
in this report. No conclusion here relies on changing query values between methods or cases.

## Check the illustrations

```bash
.venv/bin/python reports/search-math-experiments/check_example.py
```

This compares direct five-value scores with all three C++ searches. It checks the returned
IDs, B's split trace and mask peak, C's lookups and key queue, and each index's stored fields.
The saved record is `evidence/three-document-example.json`. Single-query times are omitted;
these teaching rows do not provide performance evidence. Memory totals count named fields
and buffers, not full process RAM.
