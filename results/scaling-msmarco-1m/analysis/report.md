# Fixed-grid scan versus bitplane scaling

This report uses 200 evaluation queries with three measured repeats per query. Development choices were fixed before the test run. Each choice is the fastest tested complete setting that reached its development target; it is not a global optimum.

## Results

| Pool | Target | Budget | Validation recall | Test recall | p50 ms | p95 ms | Achieved | Speedup |
|---:|---:|---:|---:|---:|---:|---:|:---:|---:|
| 1,000 | 95% | 262144 | 1.000 | 1.000 | 0.059 | 0.065 | yes | 0.76 |
| 1,000 | 99% | 262144 | 1.000 | 1.000 | 0.059 | 0.065 | yes | 0.76 |
| 3,000 | 95% | 512 | 1.000 | 1.000 | 0.149 | 0.162 | yes | 0.68 |
| 3,000 | 99% | 512 | 1.000 | 1.000 | 0.149 | 0.162 | yes | 0.68 |
| 10,000 | 95% | 8192 | 1.000 | 1.000 | 0.522 | 0.585 | yes | 0.55 |
| 10,000 | 99% | 8192 | 1.000 | 1.000 | 0.522 | 0.585 | yes | 0.55 |
| 30,000 | 95% | 2048 | 0.961 | 0.962 | 1.032 | 1.225 | yes | 0.79 |
| 30,000 | 99% | 65536 | 1.000 | 1.000 | 2.415 | 3.223 | yes | 0.34 |
| 100,000 | 95% | 8192 | 0.995 | 0.994 | 10.514 | 12.375 | yes | 0.25 |
| 100,000 | 99% | 8192 | 0.995 | 0.994 | 10.514 | 12.375 | yes | 0.25 |
| 300,000 | 95% | 8192 | 0.959 | 0.961 | 28.545 | 30.043 | yes | 0.27 |
| 300,000 | 99% | 32768 | 1.000 | 1.000 | 117.026 | 129.614 | yes | 0.06 |
| 1,000,000 | 95% | 32768 | 0.993 | 0.993 | 244.985 | 261.524 | yes | 0.10 |
| 1,000,000 | 99% | 32768 | 0.993 | 0.993 | 244.985 | 261.524 | yes | 0.10 |

There are 0 incomplete or failed scheduled cases. They remain in [failures.csv](failures.csv) and [settings.csv](settings.csv), and they do not contribute full-case percentiles or choices.

No observed median crossover among target cases that achieved their evaluation target.
No observed p95 crossover among target cases that achieved their evaluation target.

The speedup intervals use paired query bootstrap draws. Each draw samples original query IDs and keeps all three times for each sampled ID. The quality interval uses one recall observation per query. These intervals are exploratory and conditional on this run; they do not include hardware, population, or development-selection uncertainty.

When two target labels choose the same budget, they point to the same physical measurement. [query-summary.csv.gz](query-summary.csv.gz) records that measurement once and lists both labels.

## Measurement limits

Worker peak memory includes Python, input arrays, the shared Index and temporary search buffers. The worker's own peak reading captures short peaks that periodic sampling can miss. Both methods build packed rows and bitplanes. The `bitplane_words` counter reports split bitmap words only; leaf bitmap reads are not counted. The plots do not claim to measure all memory traffic.

Candidate recall compares each result with the exact scan from the same saved pool and score. A target miss on evaluation remains a miss; the report does not retune it.

## Saved evidence

- [settings.csv](settings.csv): every summarized development, pilot, and evaluation setting, including limited cases.
- [target-points.csv](target-points.csv): frozen choices, achieved recall, latency, and paired intervals.
- [failures.csv](failures.csv): incomplete and failed scheduled cases.
- [query-summary.csv.gz](query-summary.csv.gz): one row per physical evaluation setting and original query.
- [scaling.png](scaling.png), [speedup.png](speedup.png), [work.png](work.png), and [budgets.png](budgets.png): saved figures.
- [metadata.json](metadata.json): analysis source hash and input file hashes.
