# Independent evaluation of frozen settings

100 disjoint query IDs. Parameters were frozen before this evaluation.
Mean and conservative training selections are reported separately. Target misses remain visible.

| Documents | Target | Selection | Method | Recall | p50 ms | Meets target |
|---:|---:|---|---|---:|---:|---|
| 1,000,000 | 99% | conservative | scan | 98.95% | 9.5409 | False |
| 1,000,000 | 99% | conservative | branch | 98.95% | 9.8548 | False |
| 1,000,000 | 99% | conservative | keys | 98.95% | 10.5455 | False |

A speedup is marked supported only when both settings meet the target and environment checks, and the paired 95% interval for scan time / alternative time lies above one.
See comparisons.csv for every ratio and interval. This is a comparison of the frozen shortlist, not a search for new parameters on the evaluation queries.
