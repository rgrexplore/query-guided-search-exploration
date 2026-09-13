# Prefix search results

A is cluster then scan, B is Bitplanes, and C is Backward Walk. This folder keeps the tested settings, query measurements, input hashes, comparisons, and work-count checks.

## Main findings

**B has a clear top-1 result on Qwen 32D.** With clusters trained on document sign bits, the repeated median query time at 99% required recall is **0.0809 ms for B**, **0.1948 ms for A**, and **0.1688 ms for C**. B reaches 99.2% recall. At the 95% target, B takes 0.0520 ms against A's 0.0805 ms. These compare each method's fastest qualifying tested setting on the same 1,000 queries. [Table](comparison-final/qwen-quora-d32/comparison-table.csv) · [Graph](comparison-final/qwen-quora-d32/figures/recall-latency-k1.pdf)

**C has a useful conditional case.** In the global, one-cluster experiment, 82 of 1,000 queries have an exact 32-bit document code. The groups below were defined by that existing code membership, not by observed speed. Settings were selected on all 1,000 queries and then held fixed.

| Query group | Queries | A mean (ms) | B mean (ms) | C mean (ms) |
|---|---:|---:|---:|---:|
| All queries | 1,000 | 2.4910 | 0.1346 | 1.3219 |
| Exact code present | 82 | 2.4896 | 0.1277 | 0.0052 |
| Exact code absent | 918 | 2.4911 | 0.1352 | 1.4395 |

Recall is 100% in every row. C is fastest for the 82-query group; B is fastest overall in this global setup. This does not compare separately optimized clusters for each group. [Full statistics](exact-code-groups/group-summary.csv) · [Group membership and selection](exact-code-groups/selection-provenance.json)

**Scans remain strong with wider vectors and top-100 retrieval.** For Qwen 256D at 99% recall@100, repeated medians are A 3.564 ms, B 3.721 ms, and C 4.039 ms. Several other wide-vector and top-100 choices are close. Some C settings score the same opened rows as A; their small timing differences should not be credited to skipped document scoring. The tables include `mean_opened_rows` and `mean_documents_scored` so this is visible.

**Short-code recall is not full-vector quality.** The exact Qwen 32D binary top-1 result agrees with the full 1024D float best score on only **140/1,000 queries (14%)**, using a score tolerance of 1e-5. The 256D and 1024D binary results reach 75.8% and 88.6%. These are score-agreement checks, not human relevance judgments. The 32D speed result is not evidence of a semantic retrieval improvement. [Representation checks](representation/qwen-quora/summary.csv) · [Definition](representation/qwen-quora/representation.json)

## Final comparisons

Each curve selects the fastest tested configuration that meets its recall target. The table attaches the repeat of that exact setting. It never substitutes a different setting's repeat. All 192 final comparison rows have qualified repeats.

| Fixed pool | Documents | Final table | Recall–latency graph |
|---|---:|---|---|
| Qwen Quora 32D | 522,931 | [CSV](comparison-final/qwen-quora-d32/comparison-table.csv) | [Top 1](comparison-final/qwen-quora-d32/figures/recall-latency-k1.pdf) / [Top 100](comparison-final/qwen-quora-d32/figures/recall-latency-k100.pdf) |
| Qwen Quora 256D | 522,931 | [CSV](comparison-final/qwen-quora-d256/comparison-table.csv) | [Top 1](comparison-final/qwen-quora-d256/figures/recall-latency-k1.pdf) / [Top 100](comparison-final/qwen-quora-d256/figures/recall-latency-k100.pdf) |
| Qwen Quora 1024D | 522,931 | [CSV](comparison-final/qwen-quora-d1024/comparison-table.csv) | [Top 1](comparison-final/qwen-quora-d1024/figures/recall-latency-k1.pdf) / [Top 100](comparison-final/qwen-quora-d1024/figures/recall-latency-k100.pdf) |
| Nomic MS MARCO 64D | 1,000,000 | [CSV](comparison-final/nomic-msmarco-d64/comparison-table.csv) | [Top 1](comparison-final/nomic-msmarco-d64/figures/recall-latency-k1.pdf) / [Top 100](comparison-final/nomic-msmarco-d64/figures/recall-latency-k100.pdf) |
| Nomic MS MARCO 256D | 1,000,000 | [CSV](comparison-final/nomic-msmarco-d256/comparison-table.csv) | [Top 1](comparison-final/nomic-msmarco-d256/figures/recall-latency-k1.pdf) / [Top 100](comparison-final/nomic-msmarco-d256/figures/recall-latency-k100.pdf) |
| Nomic MS MARCO 768D | 1,000,000 | [CSV](comparison-final/nomic-msmarco-d768/comparison-table.csv) | [Top 100](comparison-final/nomic-msmarco-d768/figures/recall-latency-k100.pdf) |

`comparison-final/<pool>/source-studies.json` lists the original studies and file hashes. Stamped setting IDs retain the study name and original setting ID. Cluster-count plots are beside the recall–latency plots.

## Setup and coverage

The machine is an **Apple M5 Max with 128 GiB RAM**. Queries run one at a time on one CPU thread. Each process has a **32 GB limit (32,000,000,000 bytes)**. Query embeddings are cached; query time includes cluster selection and native index search. MPS was used for embedding preparation, before query timing. [Hardware record](hardware.json)

Every qualified main setting uses all **1,000 queries**. Its qualified repeat uses the same queries in three fresh process blocks, giving 3,000 observations. “Qualified” requires complete query coverage, matching recorded power-state snapshots, and process peak memory within the limit.

Within a pool, A/B/C use identical document, query, and exact-reference arrays. Across vector widths, document/query IDs stay the same but the score changes, so each width has its own exact binary-score reference. The controller checks array and router hashes before and after stages; final merging requires identical input identities and array hashes.

The final A/B/C sources contain **4,024 scheduled setting IDs, of which 3,679 qualify**. Probability experiments add 72/72 qualified settings separately. Setting IDs count saved measurements: identical parameters measured in separate studies remain separate. The unused 380-setting Nomic 768D schedule is retained but excluded from these comparison totals.

| Study | Qualified / scheduled main settings | Qualified / selected repeat settings | Main status |
|---|---:|---:|---|
| [nomic-msmarco-d256](nomic-msmarco-d256/configuration.json) | 570/570 | 45/45 | complete |
| [nomic-msmarco-d256-followup](nomic-msmarco-d256-followup/configuration.json) | 42/42 | 20/20 | complete |
| [nomic-msmarco-d64](nomic-msmarco-d64/configuration.json) | 608/608 | 30/30 | complete |
| [nomic-msmarco-d64-binary](nomic-msmarco-d64-binary/configuration.json) | 301/560 | 30/30 | partial |
| [nomic-msmarco-d64-completion](nomic-msmarco-d64-completion/configuration.json) | 8/8 | 3/3 | complete |
| [nomic-msmarco-d64-probability](nomic-msmarco-d64-probability/configuration.json) | 36/36 | 3/3 | complete |
| [nomic-msmarco-d768](nomic-msmarco-d768/configuration.json) | 0/380 | 0/0 | not run |
| [nomic-msmarco-d768-focused](nomic-msmarco-d768-focused/configuration.json) | 27/35 | 6/6 | partial |
| [qwen-quora-d1024](qwen-quora-d1024/configuration.json) | 302/380 | 30/30 | partial |
| [qwen-quora-d1024-followup](qwen-quora-d1024-followup/configuration.json) | 27/27 | 14/14 | complete |
| [qwen-quora-d256](qwen-quora-d256/configuration.json) | 570/570 | 45/45 | complete |
| [qwen-quora-d256-followup](qwen-quora-d256-followup/configuration.json) | 50/50 | 21/21 | complete |
| [qwen-quora-d32](qwen-quora-d32/configuration.json) | 608/608 | 30/30 | complete |
| [qwen-quora-d32-binary](qwen-quora-d32-binary/configuration.json) | 560/560 | 34/34 | complete |
| [qwen-quora-d32-global-exact](qwen-quora-d32-global-exact/configuration.json) | 6/6 | 3/3 | complete |
| [qwen-quora-d32-probability](qwen-quora-d32-probability/configuration.json) | 36/36 | 3/3 | complete |

The three partial runs are Nomic 64D binary, Nomic 768D focused, and Qwen 1024D. An eight-setting Nomic 64D completion run fills selected missing choices, not the entire unfinished grid. A timed-out batch may contain partial observations; those settings do not qualify.

There are **four saved timeouts**: three in current study folders and one archived initial Nomic 256D attempt. Nine earlier completed repeat processes are also archived. These remain separate from the final setting counts. [Archived attempt](nomic-msmarco-d256/attempts/initial-time-allowance/continuation.json) · [Inventory](summary-inventory.json)

The saved work checks cover **890 completed index builds and 3,849,076 query-count checks, with zero mismatches**. These check stored fields and loop counts; they do not make latency a fixed cost per operation. Allocated process memory includes more than the stored-field totals in the comparison table.

## Time spent

- First query worker: **2026-09-13 01:46:11 Singapore time**. Last worker finished: **08:30:11**. The observed worker window is **6 h 44 min**.
- The 1,383 distinct worker processes total **22,350.25 seconds (6 h 12 min 30 s)**: 1,379 complete and four timed out. Each `(pid, started_at)` is counted once, including archived attempts; no duplicate process copies were found.
- Saved command logs contain **17,403.29 seconds (4 h 50 min 3 s)**. They omit some manual continuations, and overlap the worker times above. These numbers must not be added together.
- Qwen corpus/query encoding took 23.83 minutes; preparation of the initial four exact-reference pools took 5.54 minutes. These occurred before the main query-worker window. The first-to-last worker span is not a complete project runtime covering research, coding, encoding and report writing.
- Time limits and extensions are retained in each study's `configuration.json`, `time-budget.json`, and, where used, `stage-allowances.json`. Probability runs share a 300-second main-plus-repeat cap. The inventory preserves the current caps, stage allowances, command records, and process timestamps.

There is no single complete project-start-to-finish clock, so this record does not claim that all project work finished within eight hours. [Command logs](final-focused-execution.json) · [Full runtime inventory](summary-inventory.json)

## Probability within B

The two probability studies compare 0, 0.1, and 0.2 at seeds 7, 23, and 42, with the same selected router and queries. All 36 settings per dataset qualify. Interpret every seed, not only the fastest seed. They are excluded from the A/B/C comparison curves. Their all-seed grouped statistics are in `summary-inventory.json` under `probability`.

[Qwen 32D settings](qwen-quora-d32-probability/main/settings.csv) · [Nomic 64D settings](nomic-msmarco-d64-probability/main/settings.csv) · [Selection rule](qwen-quora-d32-probability/probability-selection.json)

## Reproduce and trace a number

Run commands from the project root with the project Python environment. Rebuilding search experiments needs the prepared arrays described in [data preparation](../../docs/expanded-pool-preparation.md). Reading saved tables and regenerating figures does not need the ignored data cache.

```sh
python -m experiments.prefix_figures_v2 results/prefix-study-2026-09-13/comparison-final/qwen-quora-d32

python -m experiments.prefix_exploration_v2 configs/prefix-study-v2/qwen-quora-d32-binary.json results/reproduce-qwen32
python -m experiments.prefix_study_v2 run results/reproduce-qwen32
python -m experiments.prefix_study_v2 summarize results/reproduce-qwen32
python -m experiments.prefix_study_v2 repeat results/reproduce-qwen32
```

The exact saved settings are in each study's `main/schedule.json`; its `configuration.json` records the time and RAM limits. [Source configs](../../configs/prefix-study-v2/qwen-quora-d32-binary.json) · [B's saved schedule](qwen-quora-d32-binary/repeat/schedule.json)

For the top-1 99% B result above, start with setting `qwen-quora-d32-binary-main-s00027`. The linked path is block `b0`; blocks `b1` and `b2` use the same job suffix. [Block 0 raw queries](qwen-quora-d32-binary/repeat/cases/repeat-b0-qwen-quora-d32-binary-main-j0001-part004/run/queries.jsonl) · [Block 0 run metadata](qwen-quora-d32-binary/repeat/cases/repeat-b0-qwen-quora-d32-binary-main-j0001-part004/run/result.json)

The conditional C result uses `qwen-quora-d32-global-exact-main-s00004`, selected before subgroup analysis. [Block 0 raw queries](qwen-quora-d32-global-exact/repeat/cases/repeat-b0-qwen-quora-d32-global-exact-main-j0002-part001/run/queries.jsonl) · [Selection and all source files](exact-code-groups/selection-provenance.json) · [Weighted-mean checks](exact-code-groups/mean-checks.json)
