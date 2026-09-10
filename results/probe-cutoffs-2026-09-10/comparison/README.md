# Real-data tuning

640 complete cases; 0 failed or limited cases.
Choices below use tuning-query mean recall and measured p50 latency. They are not a final independent result.
Every local search recall was checked against scan under the same routing choice.

| Documents | Target | Method | Router | Clusters / probes | Recall | p50 ms | Case |
|---:|---:|---|---|---|---:|---:|---:|
| 10,000 | 80% | scan | ivf | 256 / 49 | 80.17% | 0.0763 | 328 |
| 10,000 | 80% | branch | ivf | 256 / 49 | 80.17% | 0.0889 | 322 |
| 10,000 | 80% | keys | ivf | 256 / 49 | 80.17% | 0.0892 | 324 |
| 10,000 | 90% | scan | ivf | 256 / 85 | 90.07% | 0.1235 | 260 |
| 10,000 | 90% | branch | ivf | 256 / 85 | 90.07% | 0.1345 | 192 |
| 10,000 | 90% | keys | ivf | 64 / 26 | 90.16% | 0.1361 | 402 |
| 10,000 | 95% | scan | ivf | 256 / 119 | 95.01% | 0.1600 | 58 |
| 10,000 | 95% | branch | ivf | 256 / 119 | 95.01% | 0.1665 | 343 |
| 10,000 | 95% | keys | ivf | 64 / 36 | 95.42% | 0.1791 | 24 |
| 10,000 | 99% | scan | ivf | 256 / 186 | 99.02% | 0.2288 | 77 |
| 10,000 | 99% | branch | ivf | 256 / 186 | 99.02% | 0.2385 | 514 |
| 10,000 | 99% | keys | ivf | 64 / 51 | 99.03% | 0.2446 | 101 |
| 100,000 | 80% | scan | ivf | 1024 / 61 | 80.25% | 0.1939 | 506 |
| 100,000 | 80% | branch | ivf | 1024 / 61 | 80.25% | 0.2099 | 277 |
| 100,000 | 80% | keys | ivf | 1024 / 61 | 80.25% | 0.2254 | 182 |
| 100,000 | 90% | scan | ivf | 1024 / 138 | 90.01% | 0.3854 | 32 |
| 100,000 | 90% | branch | ivf | 1024 / 138 | 90.01% | 0.4202 | 28 |
| 100,000 | 90% | keys | ivf | 1024 / 138 | 90.01% | 0.4350 | 614 |
| 100,000 | 95% | scan | ivf | 2048 / 447 | 95.02% | 0.6437 | 82 |
| 100,000 | 95% | branch | ivf | 2048 / 447 | 95.02% | 0.7037 | 78 |
| 100,000 | 95% | keys | ivf | 1024 / 248 | 95.01% | 0.7560 | 550 |
| 100,000 | 99% | scan | ivf | 1024 / 499 | 99.00% | 1.2882 | 106 |
| 100,000 | 99% | branch | ivf | 1024 / 499 | 99.00% | 1.3735 | 461 |
| 100,000 | 99% | keys | ivf | 1024 / 499 | 99.00% | 1.4285 | 26 |
| 1,000,000 | 80% | scan | ivf | 8192 / 112 | 80.03% | 0.4963 | 39 |
| 1,000,000 | 80% | branch | ivf | 4096 / 64 | 80.00% | 0.5310 | 231 |
| 1,000,000 | 80% | keys | ivf | 4096 / 64 | 80.00% | 0.5385 | 31 |
| 1,000,000 | 90% | scan | ivf | 8192 / 313 | 90.01% | 1.1795 | 162 |
| 1,000,000 | 90% | branch | ivf | 8192 / 313 | 90.01% | 1.2409 | 456 |
| 1,000,000 | 90% | keys | ivf | 8192 / 313 | 90.01% | 1.3394 | 535 |
| 1,000,000 | 95% | scan | ivf | 8192 / 679 | 95.01% | 2.3706 | 500 |
| 1,000,000 | 95% | branch | ivf | 8192 / 679 | 95.01% | 2.4873 | 503 |
| 1,000,000 | 95% | keys | ivf | 8192 / 679 | 95.01% | 2.6931 | 98 |
| 1,000,000 | 99% | scan | ivf | 8192 / 2097 | 99.00% | 6.9202 | 176 |
| 1,000,000 | 99% | branch | ivf | 8192 / 2097 | 99.00% | 7.1908 | 45 |
| 1,000,000 | 99% | keys | ivf | 8192 / 2097 | 99.00% | 8.3158 | 128 |

The complete settings table retains losing configurations. Each selected case links back through selected.json to its exact input and raw query observations.
Selection is limited to the declared settings and tuning queries. No universal best method or billion-document result is claimed.
