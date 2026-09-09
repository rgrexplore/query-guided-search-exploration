# Scan versus bitplane scaling — v2

Protocol version 2. The study question and dataset stay the same; this version uses a fixed
budget list and ordinary per-case result files.

## Question

At what document count, if any, is the existing bitplane search faster than the existing packed
scan while keeping 95% or 99% of the same best candidates?

Start without routing or IVF. This isolates the work inside one pool. A result where bitplanes
never win is useful too. Do not change the C++ algorithm to improve this experiment's result.

## Inputs and controls

| Setting | First run |
|---|---|
| Passages | Fixed random sample of 1,000,000 real MS MARCO passages |
| Pool sizes | 1k, 3k, 10k, 30k, 100k, 300k, 1M; nested prefixes of the saved sample |
| Queries | 100 development, 200 separate evaluation queries |
| Encoder | Nomic v1.5, revision `e9b6763023c676ca8431644204f50c2b100d9aab` |
| Encoding | Existing prefixes and layernorm/prefix/L2 recipe; cache full 768-D float32 vectors |
| Search representation | 256 document sign bits; normalized 256-D float32 queries |
| Candidates | Best 100 under the existing weighted binary score |
| Branch leaf size | 32 documents |
| Random exploration | 0; test probabilities in a later focused experiment |
| Node budgets | 32, 128, 512, 2048, 8192, 32768, 65536, 131072, 262144, plus 0 for unlimited |
| Timing | One CPU thread, one query per call, three repetitions, three warmup queries |
| Seeds | Documents 42, query selection 43, schedule 44, bootstrap 45; native seed 42 + original query row |
| Encoding chunks | 4096 documents; model batch 32; max text length 512 tokens |
| Case limits | 1800 seconds including setup; sampled child RSS 32 GiB; poll every 0.25 seconds |

These are a fixed grid, not an automatic optimizer. The chosen setting is the fastest qualifying
setting we tested, not a claim that no other budget could be better. Changes require a new saved
configuration and run directory.

## Data and labels

Use the official `collectionandqueries.tar.gz` bundle:
[MS MARCO data, file formats and terms](https://microsoft.github.io/msmarco/Datasets).
The verified bundle contains 8,841,823 passages, 6,980 dev-small queries and 7,437 dev-small qrels.
The selected files, source SHA256, ordered IDs and seeds are already saved by `msmarco.py`.

Select documents without using qrels or model scores. Pools use prefixes of that same saved order.
Development and evaluation query IDs stay disjoint. Keep the provider qrels for later semantic
checks. This random subset is a custom scaling study, not an official full-corpus leaderboard run.

For this first graph, ground truth means the exact scan's best 100 candidates from the SAME pool
under the SAME score. Recall is the fraction of those candidates returned by branching. A mean
recall of 95% does not mean every query reaches 95%, nor that 95% of the documents are relevant.

Human-label evaluation follows separately with a full BEIR collection or the relevant MS MARCO
corpus coverage. Useful primary references are [BEIR](https://arxiv.org/abs/2104.08663),
[MS MARCO evaluation methodology](https://arxiv.org/abs/2105.04021), and
[NIST TREC DL labels](https://trec.nist.gov/data/deep2019.html).

## The end-to-end flow

1. **Prepare:** use the saved selection, encode each missing chunk, then create packed document
   codes and short query vectors. `--prepare-check` does the first chunk and all queries only.
2. **Pilot:** save the run settings and source identity. Create exact references for all pools.
   Try scan, branch-512 and unlimited branching on the first five development queries at 1k,
   100k and 1M. Check actual cost before starting the full grid.
3. **Tune:** run the fixed budget list plus scan on the 100 development queries. For each pool,
   choose the fastest complete bitplane setting reaching each target. Save those choices once.
4. **Evaluate:** run scan, unlimited branching and the selected settings on the 200 evaluation
   queries. Measure a shared setting only once when both targets choose it. Never retune here.
5. **Report:** plot latency against pool size, achieved recall, speedup and work counters. Show
   failed/limited cases and missed targets. Preserve raw results and settings with the report.

Public command stays `python scaling.py --config scaling.toml --stage <stage>`. Measured stages
also take `--run-dir`; explicit `--resume` can reuse finished cases from the same run. No daemon,
job database, plugin system or service is needed.

## Small, readable implementation

- `msmarco.py`: real text selection. Already implemented and checked.
- `embedding_model.py`: the existing model recipe, shared with `embeddings.py`.
- `scaling_cache.py`: chunk files and final arrays. A temporary file is renamed only after its
  write finishes. On restart, finished chunks are reused. If final concatenation stops, repeat
  that cheap copy from chunks. One preparation process writes a cache at a time.
- `scaling_config.py`: the settings this experiment actually consumes. Remove the unused v1
  case-identity schema when replacing its planned runner; retain useful config checks.
- `scaling_worker.py`: load arrays, build the existing index, execute one case, save rows.
- `scaling.py`: ordinary loops over phases, pool sizes, settings and repetitions; a small child
  process wrapper with a whole-case timeout and RSS sampling.
- `scaling_analysis.py`: choose settings from development results and draw the reports.

## Measurement rules that matter

Each case runs in one fresh child process. Setup and index building are outside the query timer.
Keep query arrays ready before timing. Start the timer immediately before the native Python call
and stop after its result object returns. This includes the binding, native work and returned
arrays. File writes, recall checks and conversion to JSON occur afterwards. Save the native timer
separately. Do not run encoding, tests or compilation alongside measured cases.

Use one saved shuffled global case order per phase/repetition, and the same query order for every
case in that phase/repetition. References contain all 300 original query rows and IDs. The worker
uses the original row for its random seed, never its shuffled position.

The parent stops only the child it launched. Save a case as complete only after every expected
query is present. Keep partial rows and mark a timeout, memory limit or error explicitly. Do not
include partial cases in percentiles or target selection. On explicit resume, keep interrupted
output in a separate folder and rerun that case; do not combine two partial attempts. Finished
limited cases are retained rather than silently retried.

Save one run record with configuration, input hashes, source/native identity, package versions and
hardware. Check it once when resuming a phase. No nested hash identities or worker message ledger.
Observe active power source/mode at phase boundaries and every 15 seconds during cases. A detected
change invalidates that phase and stops it; use a new run for clean timing. Unsupported telemetry
is recorded as unavailable. This check exists because power settings changed during earlier runs.

## Correctness and unbiased reporting

- Before large runs, compare the native scorer with independently decoded signs and `math.fsum`
  on up to 1000 real documents and five queries. Score/cutoff tolerance is
  `1e-10 * (1 + abs(independent_score))`; record near-boundary ambiguities.
- Scan and unlimited branching must return identical ordered candidate rows. Integer fixtures
  check exact ties. A correctness failure stops the study.
- Trim native padding before writing JSON: valid candidate rows/scores only, including empty lists
  for an empty result. Never serialize `-inf` padding.
- A candidate setting needs three complete repetitions with stable candidates and work counters.
  Pick the lowest development p50 among qualifying bitplane settings. Scan is a reference, never
  an eligible bitplane choice. Equal times prefer the smaller finite budget, then unlimited.
- Evaluation reports achieved mean recall, p10 recall and empty-result rate. A missed 95%/99%
  target is marked, not repaired by choosing another setting.
- Plot p50 and p95 separately. Use paired query resampling for uncertainty: keep a query's three
  repeats together, using the same query draw for both methods. These intervals describe this
  run, not all hardware. Timing repeats do not create new independent relevance examples.

Required figures: pool size versus p50/p95; speedup over scan; documents fully scored, branch nodes,
bitmap words and sampled peak memory; development budget versus recall/latency. Missing points
break lines. Speedup below one remains visible. No crossover is a valid outcome.

## Scope boundary

After these graphs, consider one focused width, probability, leaf-size or candidate-count sweep,
then one-level IVF with identical selected pools. GPU operations and a new data layout come after
measured counters identify a useful change. Do not build those extensions in this first run.
