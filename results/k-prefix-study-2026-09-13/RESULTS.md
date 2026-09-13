# K and prefix experiment results

Same Quora documents, queries and Qwen 32-bit binary score. These results are separate from the paper.

## Required average recall: 80%

Each cell is median ms / achieved recall. Times include routing. Selected configurations were repeated three times.

| K | A: scan | B: Bitplanes | C: Backward Walk |
|---:|---:|---:|---:|
| 1 | 0.0221 / 80.30% | 0.0213 / 81.90% | 0.0225 / 80.30% |
| 2 | 0.0224 / 80.00% | 0.0215 / 81.40% | 0.0218 / 80.00% |
| 3 | 0.0226 / 80.50% | 0.0224 / 80.57% | 0.0234 / 80.50% |
| 5 | 0.0234 / 80.20% | 0.0243 / 80.30% | 0.0232 / 80.20% |
| 7 | 0.0244 / 80.27% | 0.0282 / 81.70% | 0.0247 / 80.27% |
| 10 | 0.0253 / 80.10% | 0.0315 / 80.89% | 0.0257 / 80.10% |
| 15 | 0.0279 / 80.05% | 0.0312 / 80.05% | 0.0280 / 80.05% |
| 20 | 0.0301 / 80.44% | 0.0335 / 80.44% | 0.0293 / 80.44% |
| 30 | 0.0326 / 80.15% | 0.0358 / 80.15% | 0.0327 / 80.15% |
| 50 | 0.0403 / 80.31% | 0.0426 / 80.31% | 0.0387 / 80.31% |
| 75 | 0.0466 / 80.23% | 0.0494 / 80.23% | 0.0447 / 80.23% |
| 100 | 0.0531 / 80.11% | 0.0546 / 80.11% | 0.0512 / 80.11% |

## Required average recall: 90%

Each cell is median ms / achieved recall. Times include routing. Selected configurations were repeated three times.

| K | A: scan | B: Bitplanes | C: Backward Walk |
|---:|---:|---:|---:|
| 1 | 0.0353 / 90.10% | 0.0316 / 91.30% | 0.0360 / 90.10% |
| 2 | 0.0365 / 90.25% | 0.0320 / 90.90% | 0.0361 / 90.25% |
| 3 | 0.0362 / 90.00% | 0.0311 / 90.03% | 0.0369 / 90.00% |
| 5 | 0.0383 / 90.10% | 0.0354 / 90.78% | 0.0384 / 90.10% |
| 7 | 0.0412 / 90.09% | 0.0395 / 90.80% | 0.0414 / 90.09% |
| 10 | 0.0430 / 90.09% | 0.0419 / 90.12% | 0.0432 / 90.09% |
| 15 | 0.0477 / 90.13% | 0.0540 / 90.60% | 0.0468 / 90.13% |
| 20 | 0.0507 / 90.05% | 0.0564 / 90.05% | 0.0492 / 90.05% |
| 30 | 0.0554 / 90.13% | 0.0615 / 90.13% | 0.0546 / 90.13% |
| 50 | 0.0665 / 90.09% | 0.0702 / 90.09% | 0.0643 / 90.09% |
| 75 | 0.0759 / 90.17% | 0.0808 / 90.17% | 0.0731 / 90.17% |
| 100 | 0.0863 / 90.09% | 0.0900 / 90.09% | 0.0821 / 90.09% |

## Required average recall: 95%

Each cell is median ms / achieved recall. Times include routing. Selected configurations were repeated three times.

| K | A: scan | B: Bitplanes | C: Backward Walk |
|---:|---:|---:|---:|
| 1 | 0.0576 / 95.00% | 0.0368 / 95.30% | 0.0570 / 95.00% |
| 2 | 0.0578 / 95.00% | 0.0427 / 96.60% | 0.0595 / 95.00% |
| 3 | 0.0592 / 95.00% | 0.0393 / 95.17% | 0.0585 / 95.00% |
| 5 | 0.0609 / 95.02% | 0.0476 / 96.72% | 0.0640 / 95.02% |
| 7 | 0.0674 / 95.07% | 0.0537 / 95.40% | 0.0672 / 95.07% |
| 10 | 0.0704 / 95.00% | 0.0575 / 95.03% | 0.0702 / 95.00% |
| 15 | 0.0758 / 95.01% | 0.0736 / 95.77% | 0.0737 / 95.01% |
| 20 | 0.0814 / 95.02% | 0.0850 / 95.21% | 0.0792 / 95.02% |
| 30 | 0.0921 / 95.03% | 0.1029 / 95.03% | 0.0913 / 95.03% |
| 50 | 0.1101 / 95.04% | 0.1174 / 95.04% | 0.1066 / 95.04% |
| 75 | 0.1235 / 95.01% | 0.1347 / 95.01% | 0.1178 / 95.01% |
| 100 | 0.1370 / 95.03% | 0.1452 / 95.03% | 0.1309 / 95.03% |

## Required average recall: 99%

Each cell is median ms / achieved recall. Times include routing. Selected configurations were repeated three times.

| K | A: scan | B: Bitplanes | C: Backward Walk |
|---:|---:|---:|---:|
| 1 | 0.1366 / 99.00% | 0.0575 / 99.20% | 0.1180 / 99.00% |
| 2 | 0.1579 / 99.00% | 0.0581 / 99.30% | 0.1646 / 99.00% |
| 3 | 0.1565 / 99.00% | 0.0600 / 99.13% | 0.1510 / 99.00% |
| 5 | 0.1613 / 99.00% | 0.0633 / 99.20% | 0.1633 / 99.00% |
| 7 | 0.1700 / 99.00% | 0.0665 / 99.10% | 0.1671 / 99.00% |
| 10 | 0.1812 / 99.00% | 0.0775 / 99.13% | 0.1781 / 99.00% |
| 15 | 0.1993 / 99.00% | 0.0995 / 99.12% | 0.1935 / 99.00% |
| 20 | 0.2088 / 99.00% | 0.1193 / 99.05% | 0.2033 / 99.00% |
| 30 | 0.2110 / 99.00% | 0.1590 / 99.40% | 0.2135 / 99.00% |
| 50 | 0.2452 / 99.00% | 0.2172 / 99.33% | 0.2336 / 99.00% |
| 75 | 0.2651 / 99.00% | 0.2764 / 99.27% | 0.2600 / 99.00% |
| 100 | 0.2865 / 99.00% | 0.3097 / 99.21% | 0.2761 / 99.00% |

## Files

- `comparison.json`: selected parameters, achieved recall, times, repeat spread and memory.
- `k-latency.png`: the K comparison.
- `depth/`: same-cluster traces, normal local timings and plots.

A small difference between overlapping timing repeats is not a clear win. A short-code binary recall result is not a claim about full-embedding or human relevance quality.


## Compressed query logs

The three `depth/timing-prefix-*/queries.jsonl` logs exceed GitHub's per-file limit.
Their `.jsonl.gz` copies contain the same records without any rounding or filtering.
To restore the original filenames after cloning:

```sh
gzip -dk results/k-prefix-study-2026-09-13/depth/timing-prefix-*/queries.jsonl.gz
```

The decompressed SHA-256 values are:

| Log | SHA-256 |
|---|---|
| `depth/timing-prefix-0/queries.jsonl` | `c1a9fd541c8bb4b25a83d0c2291ee6327d4dd00cb7cde902885edc16e33fd8a1` |
| `depth/timing-prefix-1/queries.jsonl` | `17e14196538132670ce33b47888c639e1f9782a7a5237c96f8ca46119acdd0e4` |
| `depth/timing-prefix-2/queries.jsonl` | `d22955b121243309a37b65c49f7a66fe7a9ae8cf8a1b68215f85c49ade41f076` |
