# Qwen and Quora: measured preparation cost

2026-09-13, Apple M5 Max, 128 GiB RAM, macOS 26.3.2.

**The first full-run checkpoints revise the encoding estimate to about 29 minutes.**
The initial short pilot projected 9.79 minutes, or about 15 minutes with 50% extra time.
The actual run saved its first 32,768 rows in 108.7 cumulative chunk seconds, about
302 questions per second. Checkpoints now supersede that short-pilot forecast. This
covers encoding only; routing,
exact reference scores, search experiments, and reporting need separate estimates.
An eight-hour study is not established by encoding throughput alone.

## Model and official data

The model is the ungated, Apache-2.0
[Qwen/Qwen3-Embedding-0.6B](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B), pinned to
revision `97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3`. Its full output has 1,024 values;
the model supports Matryoshka prefixes. The downloaded weights occupy 1,191,586,416 bytes.
The official instructions use the stored `query` prompt for queries and no prompt for
documents, with last-token pooling and normalization. The experiment follows that
recipe and does not use Nomic's layer normalization.

The dataset is the complete official BEIR Quora archive from the
[BEIR dataset catalog](https://github.com/beir-cellar/beir) and its
[published download](https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/quora.zip).
The downloaded archive matches BEIR's published MD5:
`18fb154900ba42a600f84b839c173167`.
Its SHA-256 is `56aacd9dcc4d9c093b63f175afdda0e21cbc8442ecf6c717d09de7b358d77531`.

Direct file checks found 522,931 corpus questions with unique IDs and empty titles,
15,000 query records, and 10,000 official test-query IDs with 15,675 relevance rows.
Every test relevance row points to an existing corpus ID. Mean corpus question length
is 62.16 characters; the median is 53, the 99th percentile is 173, and the maximum is
1,169 UTF-8 bytes.

## Fixed pilot setup

The pilot sampled 1,024 corpus rows uniformly without replacement using NumPy seed
20260913, preserving their original row order. Its mean question length was 62.81
characters. A fixed shuffled selection from the official test-query IDs supplied
2,000 candidate queries; the first 256 were used for the pilot. The first 1,000 are
the authorized full study's query set. Selection did not use search results or scores.

| Setting | Recorded value |
|---|---|
| Runtime | PyTorch 2.14.0, Transformers 5.16.1, SentenceTransformers 6.0.1, NumPy 2.5.3 |
| Device and weights | MPS, FP16 |
| Attention | PyTorch SDPA |
| Token limit | 8,192; left padding, right truncation |
| Output | Full 1,024 values; model's Normalize module |
| CPU threads | 4; tokenizer parallelism disabled |
| Document token lengths | Mean 15.36; median 13; maximum 60 |
| Prompted query token lengths | Mean 31.54; median 31; maximum 57 |

The exact query prefix was:

```text
Instruct: Given a web search query, retrieve relevant passages that answer the query
Query:
```

Documents received their unchanged question text. No pilot text reached the token limit.
The model loaded from the local pinned snapshot in 0.68 seconds. A separate 32-document
warm-up preceded measurement. Each measured call synchronized MPS immediately before
starting the clock and after encoding, and included tokenization, encoding and returning
the vectors to NumPy. No index search timings overlapped this pilot.

## Observed throughput and forecast

Each document row below summarizes two passes over the same 1,024 questions. The query
measurement is one pass over the same 256 prompted queries. Raw measurements and output
hashes are retained; batch size 64 is the selected encoding configuration.

| Batch size | Document seconds, passes 1 / 2 | Median documents/second | Query seconds | Queries/second |
|---:|---:|---:|---:|---:|
| 32 | 1.342 / 1.221 | 800.87 | 0.445 | 575.82 |
| 64 | 1.173 / 1.123 | 892.50 | 0.452 | 566.76 |
| 128 | 1.198 / 1.199 | 854.17 | 0.501 | 511.31 |

Using the batch-64 median document rate and measured query rate:

| Encoding job | Direct projection | With 50% additional time |
|---|---:|---:|
| 522,931 documents + 1,000 queries | 9.79 minutes | 14.69 minutes |
| 522,931 documents + 2,000 queries | 9.82 minutes | 14.74 minutes |

The full run may differ because the pilot calls were short, long questions are rare,
and the full run also saves files and concatenates completed chunks. Its checkpoints
provide a better remaining-time estimate once it starts. The 1,000-query job is the
authorized full encoding; the 2,000-query row is a forecast only.

Memory was sampled every 50 ms. Across the pilot, the largest observed MPS driver
allocation was 3.47 GB decimal; process lifetime peak RSS was 2.83 GB. Batch-64 document
passes observed about 2.39 GB of MPS driver allocation. These are different memory
measurements with potentially overlapping backing memory, so they must not be added as
independent allocations. Sampled peaks can miss a brief higher peak, and a full-corpus
run can encounter longer batches.

All outputs had 1,024 finite values. FP16 normalization produced norms between about
0.99964 and 1.00037. Output hashes were identical between repeated document passes at
the same batch size, but differed across batch sizes. Therefore one fixed encoding is
cached for all A/B/C comparisons; changing the encoding batch is not part of the index
parameter sweep. The full-run writer converts outputs to float32 and applies final L2
normalization, preserving signs while producing unit vectors for downstream routing.

## Saved evidence and the authorized full run

The ignored cache is `data/prefix-study-2026-09-13/`, shared by the worktree and main
checkout. It contains:

- `qwen-quora-pilot.json`: nine measurements, runtime/model settings, source hashes,
  sampled memory, and the projections above.
- `qwen-quora-pilot-inputs.json`: all sampled text IDs, original corpus positions and
  exact texts. Canonical content hash:
  `b9143c5b36e16f5ee02a279cf2b7b0e0baa9eb60820cddc1d08e44fbf5797760`.
- `quora-test-queries-1000.jsonl`: the fixed study queries, SHA-256
  `76be1ca191ca07399c80ae39b6c02d9de9637e94152914cab55d34f813a7f7ba`.
- `quora-test-queries-2000.jsonl`: the larger unused selection, SHA-256
  `2ecb73df9ced7f4b5f2e6432a44a718a4419a87fc9ef687008cc64bd7c6c49be`.
- `quora-overlap-audit.json`: zero ID or exact-text matches between the selected 1,000
  queries and the corpus, and no repeated exact text among those queries. No rows were
  removed to obtain this result.

The full encoder is `experiments/encode_quora_v2.py`. It uses batch size 64 and saves
complete 8,192-row chunks atomically. A checkpoint records input/recipe identity,
completed rows, chunk hashes, wall time, and process lifetime peak RSS. An interrupted
run reuses finished chunks. Changing texts, recipe, or chunk size refuses reuse.

The command is started under `caffeinate -i` to prevent idle sleep, without changing
power mode. Progress and MPS allocations are retained in `qwen-full-encode.log`.
Six focused tests cover interruption and resume, changed-input rejection, invalid
vectors, and preserving source IDs, text order and labels, including an exact-text
match fixture.

The output directory is `quora-qwen-full/` within that cache. Expected final artifacts
are `full-documents.npy` (522,931 × 1,024 float32), `full-queries.npy` (1,000 × 1,024
float32), `corpus-ids.json`, `query-ids.json`, the unchanged selected `queries.jsonl`,
raw `qrels-test.tsv`, and `manifest.json`. Only a manifest with `state: complete`
establishes completion. Its array hashes and row IDs identify the input used to derive
the separate 256- and 1,024-dimensional search pools. The full-source check found 568
extra corpus rows with repeated text. All are preserved.

Full encoding is authorized and running; this note does not yet claim it has finished.
