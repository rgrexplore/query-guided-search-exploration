# Real-data tuning

2002 complete cases; 0 failed or limited cases.
Choices below use tuning-query mean recall and measured p50 latency. They are not a final independent result.
Every local search recall was checked against scan under the same routing choice.

| Documents | Target | Method | Router | Clusters / probes | Recall | p50 ms | Case |
|---:|---:|---|---|---|---:|---:|---:|
| 100,000 | 80% | scan | ivf | 2048 / 128 | 82.70% | 0.2299 | 1005 |
| 100,000 | 80% | branch | ivf | 2048 / 128 | 82.70% | 0.2435 | 1945 |
| 100,000 | 80% | keys | ivf | 2048 / 128 | 82.70% | 0.2575 | 19 |
| 100,000 | 90% | scan | ivf | 2048 / 256 | 90.17% | 0.4274 | 1877 |
| 100,000 | 90% | branch | ivf | 2048 / 256 | 90.17% | 0.4289 | 1195 |
| 100,000 | 90% | keys | ivf | 2048 / 256 | 90.17% | 0.4680 | 376 |
| 100,000 | 95% | scan | ivf | 2048 / 448 | 95.06% | 0.6579 | 1345 |
| 100,000 | 95% | branch | ivf | 2048 / 448 | 95.06% | 0.7127 | 1082 |
| 100,000 | 95% | keys | ivf | 2048 / 448 | 95.06% | 0.8192 | 590 |
| 100,000 | 99% | scan | ivf | 2048 / 1024 | 99.17% | 1.4418 | 567 |
| 100,000 | 99% | branch | ivf | 2048 / 1024 | 99.17% | 1.4988 | 1525 |
| 100,000 | 99% | keys | ivf | 2048 / 1024 | 99.11% | 1.6857 | 1924 |

The complete settings table retains losing configurations. Each selected case links back through selected.json to its exact input and raw query observations.
Selection is limited to the declared settings and tuning queries. No universal best method or billion-document result is claimed.
