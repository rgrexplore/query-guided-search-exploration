<!--
Reviewed source: /Users/vinroger/DATA/STUDY/latihan/docs/superpowers/plans/2026-09-09-bitplane-scaling-spec.md
Source SHA256: 9874eed1302644d2a77f3e35b625f3ec1554f553795d6534c55575d2f1307bcd
This is an adapted public copy with unchanged experimental requirements. The source path is provenance only.
-->

# Bitplane search scaling study

This earlier design is retained for reference. The current protocol is
[version 2](scaling-protocol-v2.md).

**Date:** 9 September 2026  
**Scope:** The existing packed scan and bitplane branching engines, on real text vectors through one million passages.  
**Code baseline:** `63ac9467d74e7c457eeb863f279fd337802f86d6` in `/Users/vinroger/DATA/STUDY/latihan/about/exa-interview/e2e-inverted-bitplane-project`.

This specification records the agreed direction: matched recall plus an exact reference, up to one million MS MARCO passages, with no semantic routing in the first experiment. GPU kernels, a replacement bitplane algorithm and deeper routing remain later decisions. This document specifies work; it does not report new measurements.

## 1. The question

Does the existing bitplane engine become faster than packed scanning as the document pool grows, while retaining useful recall? If it does, the next experiment can ask whether larger IVF buckets or fewer routing levels improve the whole pipeline.

The first graph has document count on X and fine-search latency on Y. Both methods receive exactly the same pool before pruning. The number of documents eventually scored is a separate output explaining the curve.

The four series are:

1. Packed scan: exact top 100 under the binary score.
2. Bitplanes: a setting chosen to reach 95% mean candidate recall on development queries.
3. Bitplanes: a setting chosen to reach 99% mean candidate recall on development queries.
4. Unlimited bitplanes: exact candidate reference check.

Finding no crossover is a valid outcome. Do not draw an assumed crossover outside the measured range.

## 2. Experimental objects

| Object | Definition |
|---|---|
| Corpus | One recorded selection of 1,000,000 unique MS MARCO passage IDs and texts. |
| Pool | A prefix of that selection, supplied to both engines. With no router, corpus size and search-pool size are the same for a case. |
| Query | A real provider development query, with its original string ID and one cached embedding. |
| Binary reference | The exact packed scan's ordered top 100 document rows for a query and pool. |
| Case | A pool size, method and node budget under the shared controls. |
| Attempt | One isolated worker running a case for one timing repetition. |
| Selection | A frozen mapping from pool and recall target to the chosen development case. |
| Run | Immutable inputs/protocol identity, reference results, attempts, selections and reports. |

Candidate recall is `size(returned ∩ reference) / 100`, ignoring padding and duplicates. The target is an average over queries, not a guarantee for every query. Preserve the query-level distribution. Reference arrays use local pool rows; exported ID lists map those rows back to original passage IDs.

## 3. Fixed first-stage configuration

| Setting | Value |
|---|---|
| Pool sizes | 1,000; 3,000; 10,000; 30,000; 100,000; 300,000; 1,000,000 |
| Router | Off; every document belongs to one native bucket with ID 0 |
| Model | `nomic-ai/nomic-embed-text-v1.5` |
| Model revision | `e9b6763023c676ca8431644204f50c2b100d9aab` |
| Recipe | Existing query/document prefixes; full-vector layer norm with epsilon 1e-5, prefix slicing, L2 normalization |
| Search representation | 256 sign bits/document; float32 query; existing C++ double score accumulation |
| Cached full representation | 768 float32 dimensions, reusable for later width/refinement studies |
| Text limit / encoding batch | 512 tokens / 32 texts |
| Encoding checkpoint chunk | 4,096 documents |
| Candidate count / leaf size | 100 / 32 |
| Exploration probability / native seed | 0 / 42 |
| CPU search | One thread, one query per call; no simultaneous benchmark workers |
| Development / evaluation queries | 100 / 200 disjoint queries |
| Repetitions / warmup | Three measured repetitions / first three queries of each attempt's order |
| Document / query / schedule / bootstrap seeds | 42 / 43 / 44 / 45 |
| Bootstrap draws | 1,000 paired query draws for descriptive intervals |

The representation and C++ algorithm stay fixed in stage one. Preparation may use the available MPS device; the measured search is CPU-only. Encoding device and dependency versions are recorded and cannot change within one embedding cache.

## 4. Real data and labels

Use Microsoft's `collectionandqueries.tar.gz` passage-v1 bundle, linked from the [official download table](https://microsoft.github.io/msmarco/Datasets):

`https://msmarco.z22.web.core.windows.net/msmarcoranking/collectionandqueries.tar.gz`

Consume the regular files `collection.tsv`, `queries.dev.small.tsv` and `qrels.dev.small.tsv`. These member names are confirmed by the maintained [ir_datasets MS MARCO loader](https://github.com/allenai/ir_datasets/blob/master/ir_datasets/datasets/msmarco_passage.py), which reads them from its `collectionandqueries` download. Stream archive members; do not extract arbitrary paths or fetch training triples. Verify 8,841,823 collection records, 6,980 unique small-dev query IDs, and the required TSV structures. A different archive layout/count is an input mismatch to investigate, not permission to silently switch datasets.

Generate a PCG64 permutation of the collection's source row positions with document seed 42; take the first million in that order. Stream the collection once to populate those selected slots. Save source positions, ordered IDs, selection hashes and actual archive SHA256. Assert selected IDs are unique. The saved selection, not a future library's RNG output, governs resume.

Sort small-dev queries by string ID, permute with PCG64/query seed 43, then allocate the first 100 to development and the next 200 to evaluation. Do not use qrels, model scores or evaluation outcomes to choose the passage pool or queries. Preserve the original qrels separately for provenance; they do not define stage-one candidate recall.

This is a custom one-million-passage scaling study. Its scores are not the official full-8.84M MS MARCO leaderboard scores. Do not mix FiQA points into a curve labeled as one MS MARCO dataset. Do not duplicate documents to manufacture larger pools.

### Two different kinds of ground truth

- **This stage:** deterministic nearest-neighbor answers under the same weighted binary scorer, recomputed for each pool. This isolates the index.
- **Later semantic evaluation:** provider human relevance judgments. MS MARCO labels are sparse; TREC DL judgments are deeper but not exhaustive. A full BEIR collection such as Quora is a useful independent check below one million documents.

For TREC passage binary metrics, NIST treats grades 2–3 as relevant; grade 1 is only related. Do not reuse the current FiQA helper as an assumed official graded evaluator. Changing SIFT/GIST vectors to a binary score likewise requires a new oracle; their provided Euclidean neighbors cannot silently become ground truth for another metric.

## 5. Resumable preparation

Write the selected texts/IDs in fixed row order. Load the encoder once per preparation process. Store full document vectors in an uncompressed `.npy` file suitable for memory mapping, checkpointing after every 4,096-document chunk. Flush data before atomically recording the completed chunk and its hash.

A restart validates manifest identity, array shape/dtype and all completed chunk hashes. It encodes only missing chunks. An unfinished last chunk is recomputed; a corrupt completed chunk is an error. A local file lock permits one writer per cache key. Partial files are never treated as a complete cache. Finalization renames arrays before marking the manifest complete; recovery handles interruption between those operations by checking the recorded hashes.

A `--stage prepare --prepare-check` invocation stops after the first document chunk and all 300 queries have been encoded and validated, leaving a resumable partial cache. It uses the same selection/recipe identity as full preparation; if those outputs already exist, it validates them without re-encoding. The ordinary prepare invocation continues the remaining chunks.

Encode the 300 queries with the same recipe. Derive normalized short queries and packed document signs from full cached vectors without running the model again. Pool sizes and requested short dimensions do not enter the full encoding identity: all pools and short widths reuse one million-vector cache. Store each derived code/query pair in its own dimension-keyed directory with immutable hashes, rather than mutating the full-vector manifest when adding another width. Raw texts and vector caches remain under ignored `data/`; they are not committed.

## 6. Reference and timing boundaries

Compute exact scan references for all 300 queries at every pool size before tuning. Reference jobs run in the same supervised workers under the setup/query/attempt/RSS limits in §9, with one repetition and no plotted timing. Their partial candidate rows and terminal status are preserved; incomplete reference attempts restart from the beginning in a new attempt directory. Save ordered candidate rows and scores. During reference-worker setup, validate the first min(1,000, pool size) real codes and first five available query rows using independently decoded signs and `math.fsum` accumulation. Build a small sample index for that check. Native returned scores must differ by no more than `1e-10 * (1 + abs(independent_score))`; no omitted sample row may exceed the returned cutoff by more than this tolerance. Record near-boundary ambiguities rather than silently treating approximate numerical ties as an exact ordering proof. Integer-valued fixtures separately verify the row-ID tie rule. Unlimited-versus-scan checks still require identical rows; this independent numerical tolerance does not weaken them. Any score/cutoff failure stops the run.

Each measured attempt gets a fresh child process, loads memory-mapped inputs, constructs the native index with zero assignments, and warms up. Build/loading time is recorded separately. Precreated query views and selected-bucket arrays are identical for both methods.

Start the outer timer immediately before `index.scan(...)` or `index.search(...)`; stop after the Python result object is returned. This includes NumPy boundary checks, C++ query sorting/table construction, allocation, traversal/scoring, top-100 selection and result conversion. Record the native timer separately. Reference comparisons, ID conversion, logging, progress messages and file writes occur after the timer stops.

No model encoding, index building, routing or final float reranking belongs in this graph. A later pipeline graph will include routing and reranking. No build, test, encoding or plotting job runs alongside measured attempts. Record power observations without modifying OS settings.

## 7. Development tuning

Initial finite budgets: 32, 128, 512, 2,048, 8,192 and 32,768, plus the unlimited endpoint. Every tested case uses the same 100 development queries and three timing repetitions.

For each pool, follow this fixed rule:

1. Run the initial cases in seeded shuffled order for each repetition.
2. If either target lacks a complete qualifying finite case, double the largest finite budget successively up to 262,144. These are the only expansion budgets: 65,536; 131,072; 262,144. Stop expansion once both targets qualify or the maximum is reached. A limited/failed attempt never counts as qualifying.
3. For targets in order 0.95 then 0.99, find the smallest tested complete finite budget meeting the target. Find the largest complete tested lower budget missing it. If both exist, test their integer midpoint, then recompute the bracket. Do at most three midpoint rounds per target. Do not retest an existing budget. If no bracket exists, skip refinement for that target.
4. Among all complete measured bitplane cases meeting the target, choose the lowest development request-p50 latency. Break equal-latency ties by smaller finite budget; unlimited sorts last for that tie. Unlimited may be selected when it is the only complete qualifying case. If nothing qualifies, record `no_feasible_case`.
5. Freeze the two chosen cases and all development observations before opening evaluation results.

Persist a per-pool round ledger (mode, target, budgets and saved schedules) before dispatch. Resume the same pending round; do not reset or spend extra midpoint rounds after an interruption. Terminal limited budgets count as already tested, but never as qualifying.

All three repetitions must be complete with stable quality/work for a case to qualify. A target label is not a claim of achieved evaluation quality. Selection is based on development data only; evaluation failures or missed targets do not trigger retuning.

## 8. Evaluation and uncertainty

For each pool, evaluate the exact scan, unlimited bitplanes and the selected target cases on the same 200 evaluation queries, with three repetitions. Deduplicate physical cases: if both targets select the same budget, measure it once and let both labels reference it.

Shuffle the global attempt order using seed 44 and a phase-specific offset (pilot 0, development 1, evaluation 2). Within each phase/repetition, use one query permutation shared by every case. Native seeds use original query row, not shuffled position. Save each schedule before dispatch; append any deterministically added tuning round's schedule before running it.

Keep mean recall, p10 per-query recall, empty-return rate and achieved-target status. Mean is over queries; timing repetitions are not extra quality examples. Plot raw request p50 and p95 separately. Bootstrap evaluation query IDs as paired units, keeping each query's three timing repeats together; label resulting quality/latency-ratio intervals as exploratory and conditional on this run. They do not model variability across hosts, power modes or unrelated workloads.

Unlimited branching must exactly match scan candidate rows, including ties. A mismatch is a correctness error, not an approximate point. Retain low-recall, empty, slow and resource-limited cases.

## 9. Resource limits and run lifecycle

These limits are fixed before the pilot and copied into every attempt:

| Limit | Value |
|---|---:|
| Worker setup deadline | 120 seconds |
| Warmup or measured query deadline | 10 seconds |
| Attempt deadline after worker ready | 1,800 seconds |
| Sampled worker RSS limit | 32 GiB |
| Parent monitoring interval | 100 milliseconds |

A parent supervises only the child process it launched. On a deadline or sampled RSS violation, terminate that child, wait at most two seconds, then kill it if necessary. Do not kill other Python processes. RSS is a sampled soft limit, not an OS-enforced hard ceiling; record both sampled RSS and the child's peak RSS when available.

A failed attempt keeps its completed request rows, explicit stop status and unexecuted IDs. It cannot supply a complete-case percentile or a chosen target point. Incomplete attempts are rerun from their start on explicit resume, in a new immutable attempt directory; never merge request rows from different attempts. Existing complete, validated attempts can be reused under identical code/config/input hashes. Failed or resource-limited terminal attempts are not automatically retried.

Status names: `complete`, `query_timeout`, `setup_timeout`, `attempt_timeout`, `memory_limit`, `worker_error`, `correctness_failure`, `interrupted`. Missing attempts are distinct from terminal limited attempts. A correctness failure halts the experiment. If exact scan references cannot complete, the affected pool has no usable oracle and is blocked, not scored as zero recall.

Observe active power source/mode at phase start, at least every 15 seconds, and phase end where supported. A detected change stops that phase and marks it interrupted. Preserve the attempt history and restart that phase under a new run ID; do not select timings from mixed power conditions. On unsupported hosts, record power data as unavailable. Normal temperature/background variation remains a limitation.

Protocol stages: `prepare` → `pilot` → `tune` → `evaluate` → `report`. Pilot first creates or validates supervised reference jobs for all seven pools (including their independent real-vector checks), then uses the first five development queries, pools 1k/100k/1M, and scan/branch-512/unlimited, one repetition. Pilot outcomes do not choose final budgets and are not evaluation points. Resource-limited pilot cases are recorded; reference/setup/correctness failures must be resolved before tuning. Changing a limit or protocol creates a new run identity and preserves old evidence.

## 10. Outputs and checks

A run saves copied configuration/spec, code/native/package/input hashes, query and passage selections, reference arrays, saved schedules, every attempt status, query measurements/candidate rows, frozen target choices and generated reports. The selected-ID lists are exportable without publishing passage text. Analysis records its own source/input hashes so plots can be fixed without rerunning search.

Required figures:

- Document count versus p50 and p95 latency, with four target series and actual recall annotations.
- Speedup over scan at the same pool size, including values below one.
- Pool size versus fraction fully scored, branch nodes, bitmap-word work and peak memory.
- Development budget versus recall/latency at each pool size.

Mark target misses distinctly. Break lines at unavailable/resource-limited points; display their status instead of invented latency. Do not label absent/unbounded runs as exact. Keep all raw rows and failed settings in tables.

Completion means valid data/cache provenance, a complete status inventory, correctness checks passing for completed exact cases, frozen development choices, evaluation with no retuning, inspected plots, preserved earlier results and an explanation of crossover or its absence under the stated caps. Success does not require bitplanes to win.

## 11. Follow-on checkpoints

After stage one, keep the same harness and make focused sweeps of representation width (128/256/512/768), exploration probability, leaf size and candidate count. Then compare one-level IVF plus scan against the same IVF selections plus bitplanes. Record actual selected populations and routing time. Only those results can motivate deeper-routing or coarser-routing comparisons.

For semantic confirmation, select a full BEIR component rather than presenting a random MS MARCO subset as an official full-corpus score. TREC DL can supplement MS MARCO when corpus coverage is explicit. SIFT1M/GIST1M are optional geometric checks, not text/Matryoshka substitutes. Each follow-on checkpoint needs its own frozen protocol; it is not hidden inside stage one's Cartesian grid.

## 12. Research references

- [MS MARCO data, schemas and terms](https://microsoft.github.io/msmarco/Datasets).
- [MS MARCO evaluation-methodology paper](https://arxiv.org/abs/2105.04021).
- [NIST TREC DL 2019 labels](https://trec.nist.gov/data/deep2019.html) and [2020 labels](https://trec.nist.gov/data/deep2020.html).
- [BEIR, NeurIPS 2021](https://arxiv.org/abs/2104.08663) and [official dataset table](https://github.com/beir-cellar/beir).
- [ColBERTv2 Figure 3 / Appendix C](https://aclanthology.org/2022.naacl-main.272.pdf#page=16): a relevant example of probes/bits/candidates versus latency and quality; its different model/hardware are not our results.
- [ANN-Benchmarks paper](https://arxiv.org/abs/1807.05614) and [provided vector/neighbor files](https://github.com/erikbern/ann-benchmarks/blob/main/README.md). The historical harness is no longer actively maintained; its data are optional references.

## 13. Implemented shared boundary

`scaling_config.py` is the shared reader for configuration and worker case files:

- `load_config(path)` returns the exact TOML sections `data`, `embedding`, `search`, `tuning`, `measurement` and `limits`.
- `initial_node_budgets(config)` returns the increasing finite budgets followed by `0`, the native unlimited value. It does not add a field to the loaded configuration.
- `identity_hash(value)` returns SHA256 of strict canonical JSON: sorted keys, separators `,` and `:`, and no NaN or infinity.
- `validate_case(value, expected_identity=None)` validates a case without opening files. When supplied, `expected_identity` has exactly `run_hash`, `input_hash`, `protocol_hash` and `code_hash`.

A case has exactly these top-level keys:

```text
schema_version
run_hash, input_hash, protocol_hash, code_hash
case_key, attempt_id
phase, pool_size, method, node_budget, repeat, attempt_number
query_rows, query_ids
candidate_limit, dimensions, leaf_size, explore_probability, seed
run_manifest_path, input_manifest_path, code_path, query_path, reference_path
warmup_queries, limits, output_path
```

`query_rows` and `query_ids` are ordered, unique lists with the same length. Every path is an absolute string. `reference_path` is null only while a `reference` phase case creates the reference; every other phase requires an absolute reference path. `limits` has exactly `setup_seconds`, `query_seconds`, `attempt_seconds`, `rss_gib` and `poll_seconds`.

The case key is the identity hash of `run_hash`, `pool_size`, `method`, `node_budget`, `dimensions`, `candidate_limit`, `leaf_size`, `explore_probability` and `seed`. The attempt ID is the identity hash of `case_key`, `phase`, `repeat` and `attempt_number`. Recomputing both prevents a changed job from keeping an old identity.
