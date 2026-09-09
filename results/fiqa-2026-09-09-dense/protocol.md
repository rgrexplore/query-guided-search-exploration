# A denser sweep

This protocol was written before the expanded run. It follows earlier exploratory FiQA runs,
so this is not an untouched test set or a claim of generalization to other collections.

## Fixed inputs

Use all 57,638 FiQA documents and all 648 test queries. Keep the cached Nomic vectors, 256-D
binary scoring, 768-D reranker, 64 IVF clusters, eight sign-routing bits, leaf size 32, candidate
limit 100, final top 10 and one CPU thread fixed. No index algorithm changes in this sweep.

## Operating points

`experiment-dense.toml` defines the grid:

- Probes: 1, 2, 4, 8, 16, 32, 64.
- Branch node budgets: 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, unlimited.
- Exploration probabilities: 0 and 0.1. Random runs use seeds 42, 43, 44; deterministic runs use 42.
- Three timing repetitions, with three warmup queries per condition.

The main grid has 581 seed-specific settings. Add an all-occupied sign endpoint with an exact
scan and unlimited deterministic branching. FiQA has 120 occupied sign buckets, so the expected
total is **583 settings × 648 queries × 3 repeats = 1,133,352 measured requests**.

Sixty-four probes covers every IVF centroid list but not every sign bucket. The endpoint handles
that distinction. IVF-Flat still selects 100 candidates under 256-D float scoring before reranking,
so its all-bucket result need not match the full 768-D reference exactly.

## Execution controls

Build every partition and reference result before timing. Shuffle all router/probe/search
conditions globally for each repetition. Shuffle query order once per repetition and use the
same order for every condition. Native random seeds still use the original query row, so changing
execution order does not change a query's search choices. Save the schedule.

Do not change power mode. Sample power settings before/during/after the run and keep the
observations. They cannot rule out all changes between samples or all thermal/background effects.
Do not run other heavy jobs concurrently. Timing includes routing, the call and float reranking;
query encoding remains outside it.

Keep the same selected buckets and candidate allowance for native scan/branch comparisons.
Faiss float-IVF remains a separate outside baseline. Equal probes across routers are not equal
work, so record routed-document counts and full-float-reference coverage before candidate scoring.

## Keep failures visible

Every query stays in the results, including empty returns. Record empty rate, fewer-than-ten
rate and candidate fill relative to min(100, routed documents). Fill is defined as 0 when the
selected scope is empty. A fast empty result is not a matched-quality speedup.

Unlimited branch candidates must exactly match the corresponding binary scan rows, including
ties. A mismatch fails verification; it is not another point on a quality curve. Keep the
unlimited endpoint separate from finite budgets instead of plotting numeric 0 on a logarithmic axis.

## Analysis

Timing repetitions are repeated measurements, not extra independent query outcomes. Confirm
quality/work are stable for each case/query across repeats, then retain one quality observation.
For exploration, average the three seeds within each query; keep the individual seed means too.

Bootstrap the 648 query IDs as paired units 1,000 times. Use the same query draws for compared
conditions. Report pointwise 95% intervals for branch-minus-scan and exploration-minus-deterministic
differences. These intervals are exploratory and conditional on this corpus and these seeds;
they are not adjusted for hundreds of comparisons. Seed min/max is a separate sensitivity measure.

Latency p50/p95 comes from the recorded requests and is descriptive. Do not call it a confidence
interval. Do not drop failures, choose the best random seed, force curves to be monotonic, or fit
smooth lines that pretend unmeasured settings were tested.

Plots separate routing coverage, scan/probe behavior and fixed-probe branch budget curves.
Detailed panels use probes 1, 4, 16, 64, selected before results; tables and overview retain every
setting. One collection and one layout cannot establish a universal winner. A new dataset or
held-out confirmation is needed before making broader claims.
