# Search results

57,638 documents, 648 queries.

The full selected dataset split was used.
Query embeddings are cached. Times include routing, the search call and float reranking.

| Method | Probes | Node budget | Explore | Seed | p50 ms | Float recall | nDCG |
|---|---:|---:|---:|---:|---:|---:|---:|
| ivf/float | 16 | 0 | 0 | 42 | 0.372 | 0.964 | 0.365 |
| ivf/float | 4 | 0 | 0 | 42 | 0.153 | 0.817 | 0.317 |
| ivf/scan | 16 | 0 | 0 | 42 | 0.678 | 0.793 | 0.354 |
| ivf/scan | 4 | 0 | 0 | 42 | 0.247 | 0.712 | 0.314 |
| sign/scan | 16 | 0 | 0 | 42 | 1.101 | 0.678 | 0.334 |
| ivf/branch | 4 | 512 | 0.1 | 44 | 0.264 | 0.656 | 0.306 |
| ivf/branch | 4 | 512 | 0.1 | 43 | 0.261 | 0.656 | 0.304 |
| ivf/branch | 4 | 512 | 0.1 | 42 | 0.257 | 0.655 | 0.304 |
| ivf/branch | 4 | 512 | 0 | 42 | 0.257 | 0.655 | 0.304 |
| ivf/float | 1 | 0 | 0 | 42 | 0.090 | 0.488 | 0.216 |
| ivf/scan | 1 | 0 | 0 | 42 | 0.118 | 0.459 | 0.215 |
| ivf/branch | 1 | 512 | 0 | 42 | 0.155 | 0.459 | 0.215 |
| ivf/branch | 1 | 512 | 0.1 | 42 | 0.157 | 0.459 | 0.215 |
| ivf/branch | 1 | 512 | 0.1 | 44 | 0.157 | 0.459 | 0.215 |
| ivf/branch | 1 | 512 | 0.1 | 43 | 0.158 | 0.459 | 0.215 |
| sign/branch | 16 | 512 | 0 | 42 | 0.261 | 0.422 | 0.249 |
| ivf/branch | 1 | 128 | 0 | 42 | 0.117 | 0.397 | 0.203 |
| ivf/branch | 1 | 128 | 0.1 | 42 | 0.117 | 0.396 | 0.205 |
| ivf/branch | 1 | 128 | 0.1 | 43 | 0.119 | 0.396 | 0.203 |
| ivf/branch | 1 | 128 | 0.1 | 44 | 0.119 | 0.395 | 0.204 |
| sign/scan | 4 | 0 | 0 | 42 | 0.384 | 0.391 | 0.220 |
| sign/branch | 16 | 512 | 0.1 | 42 | 0.269 | 0.367 | 0.227 |
| sign/branch | 16 | 512 | 0.1 | 44 | 0.279 | 0.363 | 0.228 |
| sign/branch | 16 | 512 | 0.1 | 43 | 0.270 | 0.362 | 0.224 |
| ivf/branch | 16 | 512 | 0 | 42 | 0.228 | 0.361 | 0.223 |
| sign/branch | 4 | 512 | 0.1 | 44 | 0.275 | 0.355 | 0.211 |
| sign/branch | 4 | 512 | 0.1 | 42 | 0.279 | 0.355 | 0.212 |
| sign/branch | 4 | 512 | 0 | 42 | 0.279 | 0.355 | 0.210 |
| sign/branch | 4 | 512 | 0.1 | 43 | 0.280 | 0.354 | 0.210 |
| ivf/branch | 16 | 512 | 0.1 | 43 | 0.241 | 0.277 | 0.171 |
| ivf/branch | 16 | 512 | 0.1 | 42 | 0.232 | 0.276 | 0.175 |
| ivf/branch | 16 | 512 | 0.1 | 44 | 0.235 | 0.274 | 0.168 |
| ivf/branch | 4 | 128 | 0 | 42 | 0.078 | 0.170 | 0.128 |
| sign/scan | 1 | 0 | 0 | 42 | 0.145 | 0.151 | 0.090 |
| sign/branch | 1 | 512 | 0.1 | 43 | 0.201 | 0.149 | 0.091 |
| sign/branch | 1 | 512 | 0 | 42 | 0.198 | 0.149 | 0.091 |
| sign/branch | 1 | 512 | 0.1 | 44 | 0.200 | 0.149 | 0.091 |
| sign/branch | 1 | 512 | 0.1 | 42 | 0.202 | 0.149 | 0.091 |
| sign/branch | 4 | 128 | 0 | 42 | 0.113 | 0.144 | 0.116 |
| sign/branch | 1 | 128 | 0 | 42 | 0.121 | 0.127 | 0.085 |
| sign/branch | 1 | 128 | 0.1 | 44 | 0.125 | 0.127 | 0.085 |
| sign/branch | 1 | 128 | 0.1 | 43 | 0.122 | 0.126 | 0.085 |
| ivf/branch | 4 | 128 | 0.1 | 43 | 0.103 | 0.126 | 0.091 |
| ivf/branch | 4 | 128 | 0.1 | 44 | 0.105 | 0.126 | 0.095 |
| sign/branch | 1 | 128 | 0.1 | 42 | 0.122 | 0.126 | 0.084 |
| ivf/branch | 4 | 128 | 0.1 | 42 | 0.103 | 0.124 | 0.092 |
| sign/branch | 4 | 128 | 0.1 | 42 | 0.114 | 0.113 | 0.091 |
| sign/branch | 4 | 128 | 0.1 | 43 | 0.113 | 0.113 | 0.090 |
| sign/branch | 4 | 128 | 0.1 | 44 | 0.115 | 0.111 | 0.088 |
| ivf/branch | 1 | 32 | 0 | 42 | 0.027 | 0.051 | 0.053 |
| ivf/branch | 1 | 32 | 0.1 | 42 | 0.044 | 0.037 | 0.038 |
| ivf/branch | 1 | 32 | 0.1 | 44 | 0.044 | 0.036 | 0.039 |
| ivf/branch | 1 | 32 | 0.1 | 43 | 0.042 | 0.035 | 0.029 |
| sign/branch | 1 | 32 | 0 | 42 | 0.041 | 0.029 | 0.027 |
| sign/branch | 1 | 32 | 0.1 | 42 | 0.049 | 0.022 | 0.020 |
| sign/branch | 1 | 32 | 0.1 | 43 | 0.049 | 0.021 | 0.017 |
| sign/branch | 1 | 32 | 0.1 | 44 | 0.047 | 0.020 | 0.018 |
| sign/branch | 16 | 128 | 0 | 42 | 0.066 | 0.012 | 0.013 |
| sign/branch | 16 | 128 | 0.1 | 43 | 0.090 | 0.010 | 0.011 |
| sign/branch | 16 | 128 | 0.1 | 44 | 0.089 | 0.010 | 0.012 |
| sign/branch | 16 | 128 | 0.1 | 42 | 0.090 | 0.008 | 0.010 |
| sign/branch | 4 | 32 | 0 | 42 | 0.029 | 0.003 | 0.003 |
| sign/branch | 4 | 32 | 0.1 | 42 | 0.038 | 0.003 | 0.003 |
| sign/branch | 4 | 32 | 0.1 | 44 | 0.037 | 0.003 | 0.003 |
| sign/branch | 16 | 32 | 0 | 42 | 0.033 | 0.003 | 0.005 |
| sign/branch | 16 | 32 | 0.1 | 43 | 0.037 | 0.003 | 0.005 |
| sign/branch | 16 | 32 | 0.1 | 44 | 0.037 | 0.003 | 0.005 |
| sign/branch | 16 | 32 | 0.1 | 42 | 0.039 | 0.003 | 0.005 |
| sign/branch | 4 | 32 | 0.1 | 43 | 0.037 | 0.002 | 0.002 |
| ivf/branch | 16 | 128 | 0.1 | 44 | 0.073 | 0.001 | 0.000 |
| ivf/branch | 4 | 32 | 0.1 | 42 | 0.036 | 0.000 | 0.000 |
| ivf/branch | 16 | 128 | 0.1 | 42 | 0.072 | 0.000 | 0.000 |
| ivf/branch | 16 | 128 | 0.1 | 43 | 0.074 | 0.000 | 0.000 |
| ivf/branch | 4 | 32 | 0.1 | 44 | 0.036 | 0.000 | 0.000 |
| ivf/branch | 16 | 128 | 0 | 42 | 0.047 | 0.000 | 0.000 |
| ivf/branch | 4 | 32 | 0 | 42 | 0.028 | 0.000 | 0.000 |
| ivf/branch | 16 | 32 | 0 | 42 | 0.029 | 0.000 | 0.000 |
| ivf/branch | 16 | 32 | 0.1 | 44 | 0.029 | 0.000 | 0.000 |
| ivf/branch | 16 | 32 | 0.1 | 43 | 0.030 | 0.000 | 0.000 |
| ivf/branch | 16 | 32 | 0.1 | 42 | 0.030 | 0.000 | 0.000 |
| ivf/branch | 4 | 32 | 0.1 | 43 | 0.036 | 0.000 | 0.001 |

![Latency and quality](tradeoffs.png)

For branch search, node budget 0 means unlimited. Scans do not use a node budget.
`queries.csv` has every measured request. `summary.csv` keeps random seeds separate.
`metadata.json` records data/model identity, settings, hardware and build/storage counts.
Binary recall uses the exact scan in the same buckets; float recall uses the whole corpus.
Faiss float-IVF includes routing in its core time and uses float scores to choose candidates.
It is an outside baseline, not the controlled binary-scan comparison.
