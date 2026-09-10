# Real-data tuning

1378 complete cases; 0 failed or limited cases.
Choices below use tuning-query mean recall and measured p50 latency. They are not a final independent result.
Every local search recall was checked against scan under the same routing choice.

| Documents | Target | Method | Router | Clusters / probes | Recall | p50 ms | Case |
|---:|---:|---|---|---|---:|---:|---:|
| 100,000 | 80% | scan | ivf | 1024 / 64 | 80.94% | 0.2094 | 869 |
| 100,000 | 80% | branch | ivf | 1024 / 64 | 80.94% | 0.2122 | 1293 |
| 100,000 | 80% | keys | ivf | 1024 / 64 | 80.94% | 0.2218 | 619 |
| 100,000 | 90% | scan | ivf | 256 / 48 | 90.82% | 0.5042 | 354 |
| 100,000 | 90% | branch | ivf | 256 / 48 | 90.82% | 0.5152 | 699 |
| 100,000 | 90% | keys | ivf | 256 / 48 | 90.82% | 0.5402 | 717 |
| 100,000 | 95% | scan | ivf | 1024 / 256 | 95.20% | 0.7023 | 411 |
| 100,000 | 95% | branch | ivf | 1024 / 256 | 95.20% | 0.7282 | 553 |
| 100,000 | 95% | keys | ivf | 1024 / 256 | 95.20% | 0.7951 | 859 |
| 100,000 | 99% | scan | ivf | 1024 / 512 | 99.06% | 1.3160 | 46 |
| 100,000 | 99% | branch | ivf | 1024 / 512 | 99.06% | 1.3875 | 626 |
| 100,000 | 99% | keys | ivf | 1024 / 512 | 99.02% | 1.4790 | 7 |

The complete settings table retains losing configurations. Each selected case links back through selected.json to its exact input and raw query observations.
Selection is limited to the declared settings and tuning queries. No universal best method or billion-document result is claimed.
