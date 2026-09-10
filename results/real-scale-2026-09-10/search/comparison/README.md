# Real-data tuning

2938 complete cases; 0 failed or limited cases.
Choices below use tuning-query mean recall and measured p50 latency. They are not a final independent result.
Every local search recall was checked against scan under the same routing choice.

| Documents | Target | Method | Router | Clusters / probes | Recall | p50 ms | Case |
|---:|---:|---|---|---|---:|---:|---:|
| 10,000 | 80% | scan | ivf | 64 / 16 | 80.62% | 0.0843 | 524 |
| 10,000 | 80% | branch | ivf | 64 / 16 | 80.62% | 0.0930 | 822 |
| 10,000 | 80% | keys | ivf | 64 / 16 | 80.62% | 0.0953 | 338 |
| 10,000 | 90% | scan | ivf | 256 / 96 | 91.89% | 0.1299 | 757 |
| 10,000 | 90% | branch | ivf | 256 / 96 | 91.89% | 0.1387 | 261 |
| 10,000 | 90% | keys | ivf | 256 / 96 | 91.89% | 0.1566 | 1080 |
| 10,000 | 95% | scan | ivf | 256 / 128 | 95.74% | 0.1652 | 14 |
| 10,000 | 95% | branch | ivf | 256 / 128 | 95.74% | 0.1750 | 23 |
| 10,000 | 95% | keys | ivf | 256 / 128 | 95.74% | 0.1929 | 601 |
| 10,000 | 99% | scan | ivf | 256 / 192 | 99.26% | 0.2350 | 205 |
| 10,000 | 99% | branch | ivf | 256 / 192 | 99.26% | 0.2439 | 287 |
| 10,000 | 99% | keys | ivf | 256 / 192 | 99.26% | 0.2732 | 850 |
| 1,000,000 | 80% | scan | ivf | 4096 / 64 | 80.00% | 0.5180 | 1403 |
| 1,000,000 | 80% | branch | ivf | 4096 / 64 | 80.00% | 0.5164 | 2228 |
| 1,000,000 | 80% | keys | ivf | 4096 / 64 | 80.00% | 0.5682 | 2025 |
| 1,000,000 | 90% | scan | ivf | 4096 / 192 | 90.12% | 1.3420 | 2184 |
| 1,000,000 | 90% | branch | ivf | 4096 / 192 | 90.12% | 1.3633 | 2546 |
| 1,000,000 | 90% | keys | ivf | 4096 / 192 | 90.10% | 1.4600 | 1723 |
| 1,000,000 | 95% | scan | ivf | 4096 / 512 | 96.25% | 3.3440 | 2327 |
| 1,000,000 | 95% | branch | ivf | 4096 / 512 | 96.25% | 3.4304 | 1443 |
| 1,000,000 | 95% | keys | ivf | 4096 / 512 | 96.25% | 3.7895 | 2385 |
| 1,000,000 | 99% | scan | ivf | 1024 / 384 | 99.12% | 9.4567 | 2239 |
| 1,000,000 | 99% | branch | ivf | 4096 / 1536 | 99.42% | 10.0621 | 2262 |
| 1,000,000 | 99% | keys | ivf | 1024 / 384 | 99.12% | 10.2298 | 2354 |

The complete settings table retains losing configurations. Each selected case links back through selected.json to its exact input and raw query observations.
Selection is limited to the declared settings and tuning queries. No universal best method or billion-document result is claimed.
