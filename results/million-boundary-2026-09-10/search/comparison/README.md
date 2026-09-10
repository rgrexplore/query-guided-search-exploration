# Real-data tuning

832 complete cases; 0 failed or limited cases.
Choices below use tuning-query mean recall and measured p50 latency. They are not a final independent result.
Every local search recall was checked against scan under the same routing choice.

| Documents | Target | Method | Router | Clusters / probes | Recall | p50 ms | Case |
|---:|---:|---|---|---|---:|---:|---:|
| 1,000,000 | 80% | scan | ivf | 8192 / 128 | 81.51% | 0.5696 | 646 |
| 1,000,000 | 80% | branch | ivf | 8192 / 128 | 81.51% | 0.5962 | 642 |
| 1,000,000 | 80% | keys | ivf | 8192 / 128 | 81.51% | 0.6530 | 598 |
| 1,000,000 | 90% | scan | ivf | 8192 / 320 | 90.24% | 1.2082 | 831 |
| 1,000,000 | 90% | branch | ivf | 8192 / 320 | 90.24% | 1.2415 | 613 |
| 1,000,000 | 90% | keys | ivf | 8192 / 320 | 90.24% | 1.3877 | 467 |
| 1,000,000 | 95% | scan | ivf | 8192 / 768 | 95.55% | 2.6647 | 550 |
| 1,000,000 | 95% | branch | ivf | 8192 / 768 | 95.55% | 2.7644 | 629 |
| 1,000,000 | 95% | keys | ivf | 8192 / 768 | 95.55% | 3.1402 | 291 |
| 1,000,000 | 99% | scan | ivf | 8192 / 3072 | 99.55% | 10.2352 | 77 |
| 1,000,000 | 99% | branch | ivf | 8192 / 3072 | 99.55% | 10.5601 | 553 |
| 1,000,000 | 99% | keys | ivf | 8192 / 3072 | 99.55% | 12.5221 | 271 |

The complete settings table retains losing configurations. Each selected case links back through selected.json to its exact input and raw query observations.
Selection is limited to the declared settings and tuning queries. No universal best method or billion-document result is claimed.
