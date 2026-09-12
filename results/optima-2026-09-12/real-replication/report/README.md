# Independent evaluation of frozen settings

200 disjoint query IDs. Parameters were frozen before this evaluation.
Mean and conservative training selections are reported separately. Target misses remain visible.

| Documents | Target | Selection | Method | Recall | p50 ms | Meets target |
|---:|---:|---|---|---:|---:|---|
| 1,000,000 | 99% | conservative | scan | 99.25% | 10.1050 | True |
| 1,000,000 | 99% | conservative | branch | 99.25% | 10.4267 | True |
| 1,000,000 | 99% | conservative | keys | 99.25% | 12.5883 | True |

A speedup is marked supported only when both settings meet the target and environment checks, and the paired 95% interval for scan time / alternative time lies above one.
See comparisons.csv for every ratio and interval. This is a comparison of the frozen shortlist, not a search for new parameters on the evaluation queries.
