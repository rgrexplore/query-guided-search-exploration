# Independent evaluation of frozen settings

20 disjoint query IDs. Parameters were frozen before this evaluation.
Mean and conservative training selections are reported separately. Target misses remain visible.

| Documents | Target | Selection | Method | Recall | p50 ms | Meets target |
|---:|---:|---|---|---:|---:|---|
| 1,000,000 | 99% | mean | scan | 99.95% | 24.9235 | True |
| 1,000,000 | 99% | conservative | scan | 99.95% | 24.9235 | True |
| 1,000,000 | 80% | conservative | scan | 89.65% | 16.2865 | True |
| 100,000 | 80% | conservative | scan | 89.35% | 1.8251 | True |
| 1,000,000 | 80% | mean | scan | 79.05% | 11.5922 | False |
| 1,000,000 | 90% | mean | scan | 89.85% | 16.1179 | False |
| 1,000,000 | 95% | conservative | scan | 97.00% | 23.2754 | True |
| 100,000 | 80% | mean | scan | 81.35% | 1.5418 | True |
| 1,000,000 | 80% | mean | branch | 100.00% | 0.1446 | True |
| 1,000,000 | 80% | conservative | branch | 100.00% | 0.1446 | True |
| 1,000,000 | 90% | mean | branch | 100.00% | 0.1446 | True |
| 1,000,000 | 90% | conservative | branch | 100.00% | 0.1446 | True |
| 1,000,000 | 95% | mean | branch | 100.00% | 0.1446 | True |
| 1,000,000 | 95% | conservative | branch | 100.00% | 0.1446 | True |
| 1,000,000 | 99% | mean | branch | 100.00% | 0.1446 | True |
| 1,000,000 | 99% | conservative | branch | 100.00% | 0.1446 | True |
| 100,000 | 80% | mean | keys | 90.40% | 2.1027 | True |
| 100,000 | 80% | conservative | keys | 90.40% | 2.1027 | True |
| 100,000 | 90% | mean | keys | 90.40% | 2.1027 | True |
| 100,000 | 99% | mean | scan | 100.00% | 2.5455 | True |
| 100,000 | 99% | conservative | scan | 100.00% | 2.5455 | True |
| 1,000,000 | 80% | mean | keys | 100.00% | 14.6301 | True |
| 1,000,000 | 80% | conservative | keys | 100.00% | 14.6301 | True |
| 1,000,000 | 90% | mean | keys | 100.00% | 14.6301 | True |
| 1,000,000 | 90% | conservative | keys | 100.00% | 14.6301 | True |
| 1,000,000 | 95% | mean | keys | 100.00% | 14.6301 | True |
| 1,000,000 | 95% | conservative | keys | 100.00% | 14.6301 | True |
| 1,000,000 | 99% | mean | keys | 100.00% | 14.6301 | True |
| 1,000,000 | 99% | conservative | keys | 100.00% | 14.6301 | True |
| 100,000 | 95% | conservative | keys | 98.05% | 2.5501 | True |
| 100,000 | 99% | mean | keys | 100.00% | 2.7329 | True |
| 100,000 | 99% | conservative | keys | 100.00% | 2.7329 | True |
| 100,000 | 90% | mean | scan | 96.15% | 2.2395 | True |
| 100,000 | 90% | conservative | scan | 96.15% | 2.2395 | True |
| 100,000 | 95% | mean | scan | 96.15% | 2.2395 | True |
| 100,000 | 95% | conservative | scan | 96.15% | 2.2395 | True |
| 100,000 | 80% | mean | branch | 100.00% | 0.1327 | True |
| 100,000 | 80% | conservative | branch | 100.00% | 0.1327 | True |
| 100,000 | 90% | mean | branch | 100.00% | 0.1327 | True |
| 100,000 | 90% | conservative | branch | 100.00% | 0.1327 | True |
| 100,000 | 95% | mean | branch | 100.00% | 0.1327 | True |
| 100,000 | 95% | conservative | branch | 100.00% | 0.1327 | True |
| 100,000 | 99% | mean | branch | 100.00% | 0.1327 | True |
| 100,000 | 99% | conservative | branch | 100.00% | 0.1327 | True |
| 100,000 | 90% | conservative | keys | 96.15% | 2.4127 | True |
| 100,000 | 95% | mean | keys | 96.15% | 2.4127 | True |
| 1,000,000 | 90% | conservative | scan | 94.60% | 19.9126 | True |
| 1,000,000 | 95% | mean | scan | 94.60% | 19.9126 | False |

A speedup is marked supported only when both settings meet the target and environment checks, and the paired 95% interval for scan time / alternative time lies above one.
See comparisons.csv for every ratio and interval. This is a comparison of the frozen shortlist, not a search for new parameters on the evaluation queries.
