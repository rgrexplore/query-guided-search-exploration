# Independent evaluation of frozen settings

20 disjoint query IDs. Parameters were frozen before this evaluation.
Mean and conservative training selections are reported separately. Target misses remain visible.

| Documents | Target | Selection | Method | Recall | p50 ms | Meets target |
|---:|---:|---|---|---:|---:|---|
| 100,000 | 80% | mean | keys | 100.00% | 0.0160 | True |
| 100,000 | 80% | conservative | keys | 100.00% | 0.0160 | True |
| 100,000 | 90% | mean | keys | 100.00% | 0.0160 | True |
| 100,000 | 90% | conservative | keys | 100.00% | 0.0160 | True |
| 100,000 | 95% | mean | keys | 100.00% | 0.0160 | True |
| 100,000 | 95% | conservative | keys | 100.00% | 0.0160 | True |
| 100,000 | 99% | mean | keys | 100.00% | 0.0160 | True |
| 100,000 | 99% | conservative | keys | 100.00% | 0.0160 | True |
| 100,000 | 90% | mean | scan | 100.00% | 0.0189 | True |
| 100,000 | 90% | conservative | scan | 100.00% | 0.0189 | True |
| 100,000 | 95% | mean | scan | 100.00% | 0.0189 | True |
| 100,000 | 95% | conservative | scan | 100.00% | 0.0189 | True |
| 100,000 | 99% | mean | scan | 100.00% | 0.0189 | True |
| 100,000 | 99% | conservative | scan | 100.00% | 0.0189 | True |
| 100,000 | 80% | mean | scan | 86.65% | 0.0163 | True |
| 100,000 | 80% | conservative | scan | 86.65% | 0.0163 | True |
| 100,000 | 90% | mean | branch | 90.80% | 0.0221 | True |
| 100,000 | 90% | conservative | branch | 90.80% | 0.0221 | True |
| 100,000 | 95% | mean | branch | 100.00% | 0.0212 | True |
| 100,000 | 95% | conservative | branch | 100.00% | 0.0212 | True |
| 100,000 | 99% | mean | branch | 100.00% | 0.0212 | True |
| 100,000 | 99% | conservative | branch | 100.00% | 0.0212 | True |
| 1,000,000 | 80% | mean | scan | 100.00% | 0.0140 | True |
| 1,000,000 | 80% | conservative | scan | 100.00% | 0.0140 | True |
| 1,000,000 | 90% | mean | scan | 100.00% | 0.0140 | True |
| 1,000,000 | 90% | conservative | scan | 100.00% | 0.0140 | True |
| 1,000,000 | 95% | mean | scan | 100.00% | 0.0140 | True |
| 1,000,000 | 95% | conservative | scan | 100.00% | 0.0140 | True |
| 1,000,000 | 99% | mean | scan | 100.00% | 0.0140 | True |
| 1,000,000 | 99% | conservative | scan | 100.00% | 0.0140 | True |
| 1,000,000 | 80% | mean | keys | 100.00% | 0.0128 | True |
| 1,000,000 | 80% | conservative | keys | 100.00% | 0.0128 | True |
| 1,000,000 | 90% | mean | keys | 100.00% | 0.0128 | True |
| 1,000,000 | 90% | conservative | keys | 100.00% | 0.0128 | True |
| 1,000,000 | 95% | mean | keys | 100.00% | 0.0128 | True |
| 1,000,000 | 95% | conservative | keys | 100.00% | 0.0128 | True |
| 1,000,000 | 99% | mean | keys | 100.00% | 0.0128 | True |
| 1,000,000 | 99% | conservative | keys | 100.00% | 0.0128 | True |
| 100,000 | 80% | mean | branch | 86.65% | 0.0196 | True |
| 100,000 | 80% | conservative | branch | 86.65% | 0.0196 | True |
| 1,000,000 | 80% | mean | branch | 100.00% | 0.0160 | True |
| 1,000,000 | 80% | conservative | branch | 100.00% | 0.0160 | True |
| 1,000,000 | 90% | mean | branch | 100.00% | 0.0160 | True |
| 1,000,000 | 90% | conservative | branch | 100.00% | 0.0160 | True |
| 1,000,000 | 95% | mean | branch | 100.00% | 0.0160 | True |
| 1,000,000 | 95% | conservative | branch | 100.00% | 0.0160 | True |
| 1,000,000 | 99% | mean | branch | 100.00% | 0.0160 | True |
| 1,000,000 | 99% | conservative | branch | 100.00% | 0.0160 | True |

A speedup is marked supported only when both settings meet the target and environment checks, and the paired 95% interval for scan time / alternative time lies above one.
See comparisons.csv for every ratio and interval. This is a comparison of the frozen shortlist, not a search for new parameters on the evaluation queries.
