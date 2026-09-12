# Independent evaluation of frozen settings

64 disjoint query IDs. Parameters were frozen before this evaluation.
Mean and conservative training selections are reported separately. Target misses remain visible.

| Documents | Target | Selection | Method | Recall | p50 ms | Meets target |
|---:|---:|---|---|---:|---:|---|
| 1,000,000 | 80% | mean | branch | 100.00% | 0.1649 | True |
| 1,000,000 | 80% | conservative | branch | 100.00% | 0.1649 | True |
| 1,000,000 | 90% | mean | branch | 100.00% | 0.1649 | True |
| 1,000,000 | 90% | conservative | branch | 100.00% | 0.1649 | True |
| 1,000,000 | 95% | mean | branch | 100.00% | 0.1649 | True |
| 1,000,000 | 95% | conservative | branch | 100.00% | 0.1649 | True |
| 1,000,000 | 99% | mean | branch | 100.00% | 0.1649 | True |
| 1,000,000 | 99% | conservative | branch | 100.00% | 0.1649 | True |
| 1,000,000 | 90% | conservative | scan | 94.47% | 20.1759 | True |
| 1,000,000 | 95% | mean | scan | 94.47% | 20.1759 | False |
| 1,000,000 | 95% | conservative | keys | 98.88% | 26.2204 | True |
| 1,000,000 | 99% | mean | keys | 98.88% | 26.2204 | False |
| 1,000,000 | 80% | conservative | scan | 89.47% | 17.2138 | True |
| 1,000,000 | 90% | mean | scan | 89.47% | 17.2138 | False |
| 1,000,000 | 90% | conservative | keys | 94.47% | 22.3910 | True |
| 1,000,000 | 95% | mean | keys | 94.47% | 22.3910 | False |
| 1,000,000 | 99% | conservative | keys | 100.00% | 26.7614 | True |
| 1,000,000 | 80% | mean | scan | 78.44% | 12.3458 | False |
| 1,000,000 | 95% | conservative | scan | 99.16% | 23.8939 | True |
| 1,000,000 | 99% | mean | scan | 99.16% | 23.8939 | True |
| 1,000,000 | 80% | mean | keys | 78.44% | 13.7027 | False |
| 1,000,000 | 80% | conservative | keys | 89.47% | 18.8995 | True |
| 1,000,000 | 90% | mean | keys | 89.47% | 18.8995 | False |
| 1,000,000 | 99% | conservative | scan | 100.00% | 25.0578 | True |
| 1,000,000 | 99% | formula | scan | 100.00% | 26.3998 | True |
| 1,000,000 | 99% | formula | branch | 100.00% | 0.1661 | True |
| 1,000,000 | 99% | formula | keys | 100.00% | 26.7609 | True |

A speedup is marked supported only when both settings meet the target and environment checks, and the paired 95% interval for scan time / alternative time lies above one.
See comparisons.csv for every ratio and interval. This is a comparison of the frozen shortlist, not a search for new parameters on the evaluation queries.
