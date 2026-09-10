# Coarse real-data tuning

500 complete cases; 0 failed or limited cases.
Choices below use tuning-query mean recall and measured p50 latency. They are not a final independent result.
Every local search recall was checked against scan under the same routing choice.

| Target | Method | Router | Clusters / probes | Recall | p50 ms | Case |
|---:|---|---|---|---:|---:|---:|
| 80% | scan | ivf | 256 / 64 | 92.76% | 0.6543 | 482 |
| 80% | branch | ivf | 64 / 16 | 81.16% | 0.6701 | 319 |
| 80% | keys | ivf | 16 / 4 | 83.60% | 0.7128 | 56 |
| 90% | scan | ivf | 256 / 64 | 92.76% | 0.6543 | 482 |
| 90% | branch | ivf | 256 / 64 | 92.76% | 0.6981 | 238 |
| 90% | keys | ivf | 256 / 64 | 92.76% | 0.8638 | 337 |
| 95% | scan | sign | 64 / 64 | 100.00% | 2.5037 | 204 |
| 95% | branch | sign | 16 / 16 | 96.00% | 1.6721 | 346 |
| 95% | keys | sign | 16 / 16 | 100.00% | 2.6030 | 2 |
| 99% | scan | sign | 64 / 64 | 100.00% | 2.5037 | 204 |
| 99% | branch | ivf | 256 / 256 | 100.00% | 2.6616 | 326 |
| 99% | keys | sign | 16 / 16 | 100.00% | 2.6030 | 2 |

The complete settings table retains losing configurations. Each selected case links back through selected.json to its exact input and raw query observations.
This coarse grid needs refinement where useful settings lie at its boundaries. No universal best method or billion-document result is claimed.
