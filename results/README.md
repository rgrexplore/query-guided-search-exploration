# Experiment results

## Current three-method study

| Question | Result folder |
|---|---|
| What happened on real embeddings after tuning? | [Independent real queries](independent-evaluation-2026-09-10/report/README.md) |
| What if important coordinates stay fixed? | [Fixed-coordinate follow-up](controlled-boundary-fixed-2026-09-11/evaluation/report/README.md) |
| What if important coordinates change per query? | [Changing-coordinate follow-up](controlled-boundary-adaptive-2026-09-11/evaluation/report/README.md) |
| Does the smallest key improve the real baseline? | [One-bit control](one-bit-evaluation-2026-09-11/report/README.md) |
| Do the native cost predictions transfer to a larger corpus? | [Frozen 8M check](native-scale-confirmation-2026-09-11/frozen-model-check.json) |
| What changes when encoding is included? | [Actual text-query timings](text-queries-2026-09-11/summary.json) |
| What can be estimated at larger N and RAM? | [Conditional scenarios](projected-cases-2026-09-11/results.json) |

Each case archive retains raw measurements, exact settings and process status. Extract `cases.zip`
in its containing directory before re-running an aggregate analysis. The saved summaries and
paper can be read without downloading the corpus. Generating new measurements requires the
configured data and environment.

## Earlier studies


[Million-passage scaling](scaling-msmarco-1m/README.md): scan remained faster at both recall targets.
The saved study includes seven pool sizes, fixed budget curves and the complete raw results.

## FiQA runs

The [dense sweep](fiqa-2026-09-09-dense/README.md) is the expanded FiQA result: 583 settings and
1,133,352 requests, with fixed-probe curves, paired uncertainty and failure rates. It uses the
same data and index but expands the grid and shuffles conditions globally.

## Earlier power and readability checks

The three earlier runs use the same 57,638 documents, 648 queries, cached embeddings and 81 settings.
Each contains 157,464 measured requests. Every non-timing field in the request tables matches:
quality scores, candidate counts, branch counts, random choices and stopping reasons.

| Run | Code | Power observations | Use |
|---|---|---|---|
| [Original](fiqa-2026-09-09/README.md) | Initial version | Power mode changed; timings are uncontrolled | Historical record |
| [Power rerun](fiqa-2026-09-09-power-rerun/README.md) | Same as original | AC power, mode 2 at all three samples | Separates the rerun from code cleanup |
| [Cleaned version](fiqa-2026-09-09-clean/README.md) | Readability changes | AC power, mode 2 at all three samples | Final result for the original grid |

Power was observed before, during and after each rerun, at roughly 15-second intervals. No
changes were detected. These samples do not rule out every possible change between observations.

Across the 81 settings, the median ratio of cleaned-run p50 latency to original p50 latency was
**0.676**. Against the unchanged-code power rerun it was **0.968**. This is a comparison of
per-setting ratios, not a claim that the refactor made the algorithm faster. Power, temperature,
caches and ordinary run-to-run variation can affect timings.

[power-comparison.csv](power-comparison.csv) contains the three timings for every setting.
Each run folder preserves its own metadata and compressed request table. Use each run’s own metadata and scope when quoting latency; keep the original for its traceable history.
