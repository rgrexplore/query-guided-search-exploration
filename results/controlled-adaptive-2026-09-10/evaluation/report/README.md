# Independent evaluation of frozen settings

20 disjoint query IDs. Parameters were frozen before this evaluation.
Mean and conservative training selections are reported separately. Target misses remain visible.

| Documents | Target | Selection | Method | Recall | p50 ms | Meets target |
|---:|---:|---|---|---:|---:|---|
| 100,000 | 90% | conservative | keys | 96.20% | 2.4288 | True |
| 100,000 | 95% | mean | keys | 96.20% | 2.4288 | True |
| 100,000 | 80% | conservative | keys | 90.75% | 2.1198 | True |
| 100,000 | 90% | mean | keys | 90.75% | 2.1198 | True |
| 1,000,000 | 80% | mean | branch | 100.00% | 0.1496 | True |
| 1,000,000 | 80% | conservative | branch | 100.00% | 0.1496 | True |
| 1,000,000 | 90% | mean | branch | 100.00% | 0.1496 | True |
| 1,000,000 | 90% | conservative | branch | 100.00% | 0.1496 | True |
| 1,000,000 | 95% | mean | branch | 100.00% | 0.1496 | True |
| 1,000,000 | 95% | conservative | branch | 100.00% | 0.1496 | True |
| 1,000,000 | 99% | mean | branch | 100.00% | 0.1496 | True |
| 1,000,000 | 99% | conservative | branch | 100.00% | 0.1496 | True |
| 100,000 | 80% | mean | keys | 66.75% | 1.5443 | False |
| 100,000 | 99% | mean | scan | 98.10% | 2.4783 | False |
| 1,000,000 | 95% | conservative | scan | 97.00% | 23.2989 | True |
| 100,000 | 90% | conservative | scan | 96.20% | 2.2563 | True |
| 100,000 | 95% | mean | scan | 96.20% | 2.2563 | True |
| 100,000 | 95% | conservative | scan | 97.20% | 2.3697 | True |
| 1,000,000 | 90% | conservative | scan | 95.30% | 18.9841 | True |
| 1,000,000 | 95% | mean | scan | 95.30% | 18.9841 | True |
| 100,000 | 80% | mean | branch | 100.00% | 0.1375 | True |
| 100,000 | 80% | conservative | branch | 100.00% | 0.1375 | True |
| 100,000 | 90% | mean | branch | 100.00% | 0.1375 | True |
| 100,000 | 90% | conservative | branch | 100.00% | 0.1375 | True |
| 100,000 | 95% | mean | branch | 100.00% | 0.1375 | True |
| 100,000 | 95% | conservative | branch | 100.00% | 0.1375 | True |
| 100,000 | 99% | mean | branch | 100.00% | 0.1375 | True |
| 100,000 | 99% | conservative | branch | 100.00% | 0.1375 | True |
| 1,000,000 | 99% | mean | scan | 98.90% | 23.5153 | False |
| 1,000,000 | 80% | conservative | scan | 89.85% | 16.1645 | True |
| 1,000,000 | 90% | mean | scan | 89.85% | 16.1645 | False |
| 100,000 | 95% | conservative | keys | 97.20% | 2.5442 | True |
| 1,000,000 | 99% | conservative | scan | 99.95% | 24.9377 | True |
| 100,000 | 80% | conservative | scan | 87.90% | 1.8772 | True |
| 100,000 | 90% | mean | scan | 87.90% | 1.8772 | False |
| 100,000 | 80% | mean | scan | 77.15% | 1.3477 | False |
| 100,000 | 99% | mean | keys | 100.00% | 2.7170 | True |
| 100,000 | 99% | conservative | keys | 100.00% | 2.7170 | True |
| 1,000,000 | 80% | mean | keys | 100.00% | 26.8429 | True |
| 1,000,000 | 80% | conservative | keys | 100.00% | 26.8429 | True |
| 1,000,000 | 90% | mean | keys | 100.00% | 26.8429 | True |
| 1,000,000 | 90% | conservative | keys | 100.00% | 26.8429 | True |
| 1,000,000 | 95% | mean | keys | 100.00% | 26.8429 | True |
| 1,000,000 | 95% | conservative | keys | 100.00% | 26.8429 | True |
| 1,000,000 | 99% | mean | keys | 100.00% | 26.8429 | True |
| 1,000,000 | 99% | conservative | keys | 100.00% | 26.8429 | True |
| 1,000,000 | 80% | mean | scan | 79.90% | 11.6465 | False |
| 100,000 | 99% | conservative | scan | 100.00% | 2.5438 | True |

A speedup is marked supported only when both settings meet the target and environment checks, and the paired 95% interval for scan time / alternative time lies above one.
See comparisons.csv for every ratio and interval. This is a comparison of the frozen shortlist, not a search for new parameters on the evaluation queries.
