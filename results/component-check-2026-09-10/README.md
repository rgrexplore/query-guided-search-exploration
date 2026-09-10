# Component checks

Storage equations matched native logical byte counts in every case.
These timings are exploratory component observations on random signs, not a tuned comparison.

| Documents | Method | Recall | p50 ms | Scored rows |
|---:|---|---:|---:|---:|
| 64 | scan | 1.000 | 0.0070 | 64.0 |
| 64 | branch | 1.000 | 0.0129 | 64.0 |
| 64 | keys | 1.000 | 0.0137 | 64.0 |
| 65 | scan | 1.000 | 0.0068 | 65.0 |
| 65 | branch | 1.000 | 0.0126 | 65.0 |
| 65 | keys | 1.000 | 0.0135 | 65.0 |
| 1024 | scan | 1.000 | 0.0443 | 1024.0 |
| 1024 | branch | 1.000 | 0.0515 | 1024.0 |
| 1024 | keys | 1.000 | 0.0498 | 1024.0 |
| 16384 | scan | 1.000 | 0.4512 | 16384.0 |
| 16384 | branch | 1.000 | 1.0361 | 16384.0 |
| 16384 | keys | 1.000 | 0.4720 | 16384.0 |
