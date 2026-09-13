# Comparing scan, Bitplanes and Backward Walk

**Completed on 13 September 2026.** The plan below records the original choices and
dated follow-ups. Final coverage, partial grids, timings and conclusions are in
[the results guide](../results/prefix-study-2026-09-13/README.md); the current
[TODO](prefix-study-todo.md) tracks artifact completion. No additional sweep is queued.

## Question

For a fixed collection and a required recall, which method gives the lowest query
time? Each method can choose its own cluster count and search settings, within the
same 32 GB process limit. A comparison uses identical documents, queries, binary
scores and reference IDs. A different model or collection is a separate comparison.

Section 1 of the paper is protected by `docs/section-one.sha256`.

## Inputs

| Collection | Model | Documents | Queries | Code lengths |
|---|---|---:|---:|---|
| MS MARCO passages | Nomic Embed Text v1.5 | 1,000,000 | 1,000 | 256 and 768 bits |
| BEIR Quora | Qwen3 Embedding 0.6B | 522,931 | 1,000 | 256 and 1024 bits |

The Nomic document vectors are already cached. Only the additional queries need
encoding. Quora uses the complete official corpus and a fixed random selection
from the official test queries. Selection happens before any search result is
known. Each code length gets its own exact binary-score references.

The experiment measures nearest-neighbor recall. For example, recovering 99 of
the exhaustive scorer's 100 results gives 99% recall. Human relevance labels remain
available for a separate question: whether those documents answer the query.

## First sweep

The four configurations live in `configs/prefix-study-v2/`.

| Setting | Values |
|---|---|
| Clusters | 1, 64, 1024, 4096 |
| Results returned | 1, 10, 100 |
| Recall targets | 50%, 80%, 90%, 95%, 99% |
| Opened clusters | Counts needed for the target recalls under exhaustive local scoring, plus all clusters |
| B: rows at which splitting stops | 128, 2048 |
| B: visited-set budget | 512, 8192 |
| B: random exploration | 0 for the first sweep |
| C: starting prefix | 16 bits |
| C: candidate target | 1,000; 10,000; 100,000; half the collection; 0 |

For C, target 0 continues to depth 0 and scores every document in the opened
clusters. For B, the work budget counts both split nodes and leaves. A budget can
run out before enough leaves are reached; retain that result with its low recall.

Opening more clusters can compensate for local misses. Every method receives the
same available probe counts for each layout and result count. Each then chooses
the fastest setting that actually reaches the required recall.

This is at most 570 settings per pool, or 2,280 across the four pools. Duplicate
probe counts reduce that number. The first pass uses every selected query once.
The runner saves completed settings, changes execution order using a fixed seed,
and runs one timing process at a time.

## Follow the evidence

After the first pass, inspect the whole recall-time curve and work counts. If B
or C is close to A, try nearby cluster counts and local limits. For C, also try
starting depths 1, 4 and 32 around useful candidate targets. For B, a small extra
run can compare exploration probabilities 0, 0.1 and 0.2 at the same budgets, with
seeds 7, 23 and 42. These are follow-up experiments, recorded separately from the
initial grid. Do not silently drop a losing setting or select favorable queries.

Repeat each method's selected settings in three fresh processes. Report the
median and the range of those process medians. A small timing difference that
changes direction across repetitions is not a stable advantage. These are the
best measured settings in the searched range; a finite sweep cannot prove a
global optimum over every possible index and parameter.

## Time and checkpoints

The first native pilot completed eight settings on 200 queries in about 43 seconds,
including process and index preparation. Its global 256-bit scan took about 25 ms
per query; the tested branch setting took about 61 ms. Clustered cases still need
measurement, so these values do not yet predict the whole sweep.

The Qwen pilot encoded representative Quora documents at about892 documents/sec
with batch 64. The full corpus plus 1,000 queries is expected to take about 15 minutes
with a 50% time margin. This estimate covers encoding only.

The first 32,768 full-run rows took 108.7 seconds, about 302 documents/sec. That
revises the encoding estimate to about 29 minutes. The full-run checkpoints take
precedence over the short pilot.

The working allocation is 60 minutes for encoding, exact references and routing;
up to five hours across the four search sweeps; 45 minutes for promising follow-ups
and repeat measurements; and 75 minutes for analysis and the paper. That totals
eight hours. Re-estimate after the first completed search group. If a sweep is
too expensive, save its partial evidence and reduce the next declared sweep;
do not report unrun settings as failures or winners.

Each pool has a 75-minute saved search budget covering its main run and repeats.
Each index job also has a 40-minute limit. These are upper bounds, not estimates
of how long every job will take. Encoding chunks and reference-query chunks have
their own saved progress. No embedding or reference work runs alongside timed
search queries on the same machine.

## Outputs

Save raw per-query records, every parameter setting, input hashes, power settings,
code revision, exact checks, stored bytes and peak process RAM. Produce simple
recall-time curves and tables that identify the chosen parameters. Update the
paper from those records, without changing Section 1 or inventing a favorable
outcome. Commit the completed stages and update project memory at the end.

## Broader follow-ups (13 September)

The completed encoding took 23.83 minutes. Preparing the four original reference
pools took 5.54 minutes. The Quora 256-bit first sweep completed all 570 settings
in 41.46 minutes. The original Nomic 256-bit allowance ended before all groups
finished; its attempted settings are retained, and the continuation is recorded
separately. The wider Qwen sweep uses results counts 1 and 100 to spend more time
on different settings instead of repeating the middle result count.

The next comparisons reuse the full embedding caches at two additional supported
widths: Qwen 32 and Nomic 64. The document IDs and query IDs stay fixed within each
collection; each width gets a fresh exact binary reference. These runs try smaller
leaves, a deeper-first order for equal branch penalties, extra routing coverage,
and C starting depths 0, 1, 4, 16 and 32. The full Nomic width also gets a focused
95%/99% comparison. The exact grids and time limits are saved in their JSON files.

`experiments/prefix_exploration_v2.py` divides the new schedules into small batches.
It retains every setting and query. Global settings run individually; clustered
settings run up to three at a time. This saves useful completed groups when a slow
setting reaches its time limit. A stopped setting is incomplete, not a loss.

An apparent speedup between two sparse A probe samples needs a direct comparison.
For example, B can score 512 opened clusters without making any split when its
leaf limit is large. A must also be allowed to scan those 512 clusters. Likewise,
C gets its depth-zero setting, which skips widening and scores the opened rows.
These controls prevent a parameter-grid gap from being mistaken for an algorithm
advantage. Small timing differences will be repeated; the full exploration grid
will not be run three times.
