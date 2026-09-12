# Searching binary embeddings: derivation and experiment

This 13-page paper connects three search methods to cost formulas, parameter choices and
measured results. It includes eight diagrams/plots and inline pseudocode. The earlier five
PDF editions remain unchanged; [the edition index](../EDITIONS.md) links them.

## Build the paper

With TeX Live available, run from the repository root:

```bash
bash reports/search-math-optima/build.sh
```

Output: `output/pdf/search-methods-optima.pdf`. The source archive includes rendered plots,
generated tables and the TikZ diagrams in the LaTeX source. It rebuilds the PDF without data
downloads or Python. The experiment requires the full repository and its installed environment.

## Reproduce the controlled experiment

The frozen run is under `results/optima-2026-09-12/`. For a new run, choose a different output
folder; existing measurements are not overwritten. The commands below reuse the original
balanced document cache after checking its hashes:

```bash
.venv/bin/python -m experiments.optima_study prepare --output results/my-optima
.venv/bin/python -m experiments.optima_study tune --output results/my-optima
.venv/bin/python -m experiments.optima_predictions fit --study results/my-optima
.venv/bin/python -m experiments.optima_study evaluate --output results/my-optima \
  --formula-choices results/my-optima/predictions/shortlist.json
.venv/bin/python -m experiments.optima_predictions check --study results/my-optima
.venv/bin/python -m experiments.check_controlled results/my-optima/fixed/evaluation
.venv/bin/python -m experiments.check_controlled results/my-optima/adaptive/evaluation
```

The `prepare` command creates the query/reference inputs and routing grid. `tune` uses only
the first 32 queries per condition. The prediction command freezes the cost model and its
choices before `evaluate` runs the remaining 64 queries. Each method may choose a different
router, cluster count, probe count and local setting. The native algorithms are unchanged.

On a checkout without the old document cache, create a source pool with the existing generator:

```bash
.venv/bin/python - <<'PY'
from experiments.controlled import prepare_controlled_pool

config = {
    "data": {
        "support": "fixed", "dimensions": 256, "queries": 1,
        "seed": 73, "strong_bits": 12, "weak_weight": 0.0001,
        "cache_dir": "data/optima-source",
    },
    "search": {"top_k": 100},
}
prepare_controlled_pool(config, 1_000_000)
PY
.venv/bin/python -m experiments.optima_study prepare --output results/my-optima \
  --source-pool data/optima-source/n1000000
```

Then run `tune`, `fit`, and the remaining commands above. Only the document arrays are reused
from that source. The study generates its own 96 queries. A fresh source needs router training;
this preparation time is excluded from retrieval timings. To change the size, distribution,
grid or repeats, supply a complete JSON configuration with `prepare --config PATH`; the defaults
are documented in `experiments/optima_study.py`. Its current query generator fixes the weak
weight at 0.0001 rather than silently claiming to test another value.

## Real-data replication

The exact cached Nomic/MS MARCO inputs and frozen original settings are specified by
`experiments/configs/optima-real-replication.toml`. Once those caches are available:

```bash
.venv/bin/python -m experiments.run \
  --config experiments/configs/optima-real-replication.toml \
  --output results/my-real-replication
.venv/bin/python -m experiments.evaluation_report results/my-real-replication
```

This repeats the same 200 earlier evaluation queries. It measures a new run of frozen
settings; it is not presented as new-query evidence. The root README describes obtaining
and encoding the real data. Cached corpus arrays are not included in the paper source ZIP.

## Read and regenerate the evidence

`results/optima-2026-09-12/predictions-v1/` preserves the first failed time model. The current
`predictions/` directory records the revised coefficients, 429 forecasts, six selected
settings and the final check. Both versions distinguish analytic work predictions from
empirical routing/work inputs. Remaining fit failures were retained.

Each completed measurement stage has `cases.zip` and `archive.json`. Extract the archive in
its containing folder before running analysis from a fresh checkout; this restores raw
query observations without rerunning the benchmark. The local case directories are retained
but ignored by Git. The `pool` directories are also ignored; configuration and hash records
identify their inputs, and `prepare` regenerates them for a new run.

After restoring the saved raw cases, regenerate this paper's evidence with:

```bash
.venv/bin/python reports/search-math-optima/make_results.py
```

For a new experiment, pass `--study results/my-optima` and optionally
`--real-folder results/my-real-replication`. The command rewrites the generated tables/plots
in this report directory. `evidence/results.json` retains source hashes, all pairwise
comparisons and work counts. Rebuild the PDF after changing those generated inputs.

## What the evidence supports

The real-data replication favors scanning. The constructed fixed-position case favors
backward walk by about 10% over a direct-prefix scan; both select the same group. The
changing-position case favors bitplanes by about 152 times over the best tested scan.
Every setting in the paper's 99% comparisons meets its recall and RAM requirement. The six model-selected
median-time predictions are within 10% on separate evaluation queries, but some other
settings remain outside the 20% model-error criterion. These are bounded experimental
conclusions, not a global optimum over all indexes or a billion-document measurement.
