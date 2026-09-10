# Independent evaluation of frozen settings

200 disjoint query IDs. Parameters were frozen before this evaluation.
Mean and conservative training selections are reported separately. Target misses remain visible.

| Documents | Target | Selection | Method | Recall | p50 ms | Meets target |
|---:|---:|---|---|---:|---:|---|
| 100,000 | 99% | mean | scan | 98.82% | 1.3592 | False |
| 100,000 | 80% | mean | keys | 77.95% | 0.2242 | False |
| 100,000 | 80% | conservative | keys | 80.55% | 0.2698 | True |
| 100,000 | 95% | conservative | branch | 95.47% | 0.7976 | True |
| 100,000 | 90% | conservative | branch | 91.29% | 0.5293 | True |
| 100,000 | 95% | conservative | keys | 95.57% | 0.9308 | True |
| 100,000 | 95% | conservative | scan | 95.47% | 0.7527 | True |
| 100,000 | 90% | conservative | keys | 91.29% | 0.5918 | True |
| 100,000 | 99% | conservative | scan | 99.44% | 1.6219 | True |
| 100,000 | 90% | conservative | scan | 90.98% | 0.5034 | True |
| 100,000 | 80% | conservative | scan | 80.55% | 0.2345 | True |
| 100,000 | 99% | conservative | keys | 99.44% | 1.8109 | True |
| 100,000 | 95% | mean | scan | 94.46% | 0.6686 | False |
| 100,000 | 99% | conservative | branch | 99.44% | 1.6779 | True |
| 100,000 | 80% | conservative | branch | 80.55% | 0.2454 | True |
| 10,000 | 95% | conservative | scan | 95.84% | 0.1683 | True |
| 10,000 | 95% | conservative | branch | 95.84% | 0.1818 | True |
| 10,000 | 80% | conservative | keys | 84.91% | 0.1129 | True |
| 10,000 | 99% | conservative | scan | 99.27% | 0.2330 | True |
| 10,000 | 90% | conservative | branch | 91.79% | 0.1474 | True |
| 10,000 | 99% | conservative | branch | 99.27% | 0.2542 | True |
| 10,000 | 80% | conservative | branch | 84.91% | 0.1052 | True |
| 10,000 | 95% | conservative | keys | 95.84% | 0.1968 | True |
| 10,000 | 80% | conservative | scan | 84.91% | 0.0979 | True |
| 10,000 | 90% | conservative | scan | 91.79% | 0.1339 | True |
| 10,000 | 99% | conservative | keys | 99.27% | 0.2812 | True |
| 10,000 | 90% | conservative | keys | 91.79% | 0.1560 | True |
| 1,000,000 | 95% | conservative | branch | 95.47% | 3.4295 | True |
| 1,000,000 | 99% | conservative | scan | 99.25% | 9.5796 | True |
| 1,000,000 | 99% | conservative | branch | 99.25% | 9.9645 | True |
| 1,000,000 | 95% | conservative | scan | 95.47% | 3.3600 | True |
| 1,000,000 | 95% | conservative | keys | 95.47% | 3.7055 | True |
| 1,000,000 | 99% | conservative | keys | 99.25% | 11.0127 | True |
| 1,000,000 | 90% | conservative | branch | 91.73% | 1.6856 | True |
| 1,000,000 | 90% | conservative | scan | 91.73% | 1.6341 | True |
| 1,000,000 | 90% | conservative | keys | 91.73% | 1.8802 | True |
| 1,000,000 | 80% | conservative | branch | 82.53% | 0.6985 | True |
| 1,000,000 | 80% | conservative | scan | 82.53% | 0.6749 | True |
| 1,000,000 | 80% | conservative | keys | 82.53% | 0.7506 | True |
| 10,000 | 95% | mean | keys | 95.17% | 0.1833 | True |
| 100,000 | 99% | mean | keys | 98.73% | 1.4939 | False |
| 100,000 | 90% | mean | branch | 88.34% | 0.4145 | False |
| 1,000,000 | 80% | mean | keys | 78.84% | 0.5414 | False |
| 100,000 | 90% | mean | scan | 88.34% | 0.3854 | False |
| 1,000,000 | 80% | mean | scan | 78.69% | 0.5088 | False |
| 1,000,000 | 99% | mean | branch | 98.73% | 7.3017 | False |
| 10,000 | 95% | mean | scan | 94.90% | 0.1595 | False |
| 10,000 | 99% | mean | scan | 99.06% | 0.2273 | True |
| 10,000 | 99% | mean | branch | 99.06% | 0.2481 | True |
| 1,000,000 | 95% | mean | keys | 94.26% | 2.7799 | False |
| 10,000 | 99% | mean | keys | 99.09% | 0.2504 | True |
| 1,000,000 | 99% | mean | keys | 98.73% | 8.1677 | False |
| 1,000,000 | 90% | mean | scan | 89.02% | 1.1739 | False |
| 1,000,000 | 99% | mean | scan | 98.73% | 6.8553 | False |
| 10,000 | 90% | mean | scan | 90.03% | 0.1207 | True |
| 100,000 | 80% | mean | branch | 77.20% | 0.2067 | False |
| 10,000 | 80% | mean | branch | 79.73% | 0.0897 | False |
| 10,000 | 80% | mean | keys | 79.73% | 0.0929 | False |
| 10,000 | 80% | mean | scan | 79.73% | 0.0808 | False |
| 10,000 | 90% | mean | keys | 90.12% | 0.1412 | True |
| 1,000,000 | 95% | mean | scan | 94.26% | 2.3952 | False |
| 100,000 | 80% | mean | scan | 77.20% | 0.1941 | False |
| 1,000,000 | 90% | mean | keys | 89.02% | 1.3463 | False |
| 100,000 | 95% | mean | keys | 94.41% | 0.7621 | False |
| 100,000 | 90% | mean | keys | 88.34% | 0.4450 | False |
| 1,000,000 | 90% | mean | branch | 89.02% | 1.2322 | False |
| 1,000,000 | 80% | mean | branch | 78.84% | 0.5130 | False |
| 100,000 | 95% | mean | branch | 94.43% | 0.7058 | False |
| 10,000 | 95% | mean | branch | 94.90% | 0.1739 | False |
| 1,000,000 | 95% | mean | branch | 94.26% | 2.4799 | False |
| 100,000 | 99% | mean | branch | 98.73% | 1.3710 | False |
| 10,000 | 90% | mean | branch | 90.03% | 0.1328 | True |

A speedup is marked supported only when both settings meet the target and environment checks, and the paired 95% interval for scan time / alternative time lies above one.
See comparisons.csv for every ratio and interval. This is a comparison of the frozen shortlist, not a search for new parameters on the evaluation queries.
