# Independent evaluation of frozen settings

4 disjoint query IDs. Parameters were frozen before this evaluation.
Mean and conservative training selections are reported separately. Target misses remain visible.

| Documents | Target | Selection | Method | Recall | p50 ms | Meets target |
|---:|---:|---|---|---:|---:|---|
| 1,024 | 90% | mean | branch | 100.00% | 0.0069 | True |
| 1,024 | 90% | conservative | branch | 100.00% | 0.0069 | True |
| 1,024 | 99% | mean | branch | 100.00% | 0.0069 | True |
| 1,024 | 99% | conservative | branch | 100.00% | 0.0069 | True |
| 1,024 | 90% | mean | keys | 100.00% | 0.0055 | True |
| 1,024 | 90% | conservative | keys | 100.00% | 0.0055 | True |
| 1,024 | 99% | mean | keys | 100.00% | 0.0055 | True |
| 1,024 | 99% | conservative | keys | 100.00% | 0.0055 | True |
| 1,024 | 90% | mean | scan | 100.00% | 0.0082 | True |
| 1,024 | 90% | conservative | scan | 100.00% | 0.0082 | True |
| 1,024 | 99% | mean | scan | 100.00% | 0.0082 | True |
| 1,024 | 99% | conservative | scan | 100.00% | 0.0082 | True |

A speedup is marked supported only when both settings meet the target and environment checks, and the paired 95% interval for scan time / alternative time lies above one.
See comparisons.csv for every ratio and interval. This is a comparison of the frozen shortlist, not a search for new parameters on the evaluation queries.
