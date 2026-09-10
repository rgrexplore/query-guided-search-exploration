# One text query through the complete pipeline

20 query IDs, three blocks and two repeats per method. Each request includes single-query MPS encoding, normalization, routing and native search.
The settings were copied from the earlier frozen 99% choices. References were computed afterward from the actual encoded vectors.

| Method | Encode median ms | Retrieval median ms | Total median ms | Total p95 ms | Recall |
|---|---:|---:|---:|---:|---:|
| scan | 6.488 | 9.406 | 16.313 | 19.201 | 99.45% |
| branch | 6.500 | 9.651 | 16.303 | 19.217 | 99.45% |
| keys | 7.543 | 11.201 | 18.912 | 21.898 | 99.45% |

Totals were measured per request; their median is not the sum of the stage medians. The small A/B difference does not establish a speedup.
Model loading, index construction and reference computation are excluded from request latency.
MPS allocations may overlap RSS; the environment record keeps both rather than adding them.

Reproduce on the configured MPS environment:
```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python -m experiments.text_queries --output results/my-text-queries
```
