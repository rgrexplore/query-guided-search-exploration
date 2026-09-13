# Same-cluster local search

This follow-up holds the selected clusters fixed for A, B and C. It tests the three
routing layouts used by the Qwen32 top-1 headline comparison, plus the selected
Qwen256 and Nomic256 top-100 layouts.

## What was measured

- All 1,000 queries from each existing pool are retained, with unchanged code/query/global-reference arrays.
- Cluster IDs are computed once and saved before timing. Every method in a layout reads that same array.
- The local reference scans exactly those selected clusters using the shared full binary scorer and ID tie rule.
- The timer measures the local native call and returned arrays. It excludes routing, reference preparation and index construction.
- Global recall still compares with the full-collection binary reference, separately from local recall.
- Three fresh-process repeats rotate method order. B and C each test two local settings; their variant order is reversed in the middle repeat.
- The common memory cap is 64 GB. Peak worker memory was 229.7 MB.

B tests leaf128/budget8192/deeper-first ties and an immediate-scan control with a leaf
size equal to the whole collection. C tests depth4/exact-stop and depth0. Both C
settings use candidate target zero. This is a small mechanism comparison, not a
new exhaustive search for every method's best local parameters.

## Qwen32, top 1

The table shows median local time over all 3,000 repeated query observations for
the filtering settings. All entries reached 100% local recall.

| Fixed routing | Opened rows, mean | A scan (ms) | B leaf128 (ms) | C depth4 (ms) | Global recall |
|---|---:|---:|---:|---:|---:|
| 4096 clusters, 180 probes | 24,181 | 0.0978 | 0.1297 | 0.1004 | 99.0% |
| 16 clusters, 9 probes | 295,031 | 0.9402 | 0.0565 | 0.4806 | 99.2% |
| 1024 clusters, 78 probes | 41,147 | 0.1379 | 0.1099 | 0.1103 | 99.0% |

B's large-cluster advantage remains when A scans the same selected rows. C also
helps there, but B is faster. On the intermediate layout B/C are close and their
repeat ranges overlap. On the many-small-cluster layout B is slower, while C's
small difference from A is within overlapping repeat ranges.

The immediate-scan controls remain in summary.json. A small timing difference on
those paths is not a pruning gain: they score every opened row.

## 256-bit codes, top 100

| Fixed routing | Method | Local recall | Median local ms |
|---|---|---:|---:|
| Qwen, 4096 clusters / 668 probes | A | 100% | 2.3576 |
| Same | B leaf128 | 100% | 2.7690 |
| Same | C depth4 | 100% | 2.7649 |
| Nomic, 4096 clusters / 1414 probes | A | 100% | 8.9999 |
| Same | B leaf128 | 84.135% | 4.6308 |
| Same | C depth4 | 100% | 10.5210 |

Nomic's faster B setting does not meet 99% local recall. Its global recall is 83.338%.
The immediate-scan controls recover all local reference IDs: B/C take 2.4086/2.5096 ms
on Qwen and 9.3839/9.8842 ms on Nomic. These tested settings do not establish a
high-recall pruning advantage at 256 bits.

## Files and checks

- `configuration.json`: layouts, parameters and timing boundary.
- `*-routing.json`: source setting IDs and SHA-256 hashes of actual inputs and saved selections.
- `cases/<layout>-<method>-r<repeat>/job.json`: the exact worker request.
- `cases/.../queries.jsonl`: every returned ID, local/global recall, timing and work count.
- `cases/.../result.json`: index fields, process peak, power snapshots and native/worker hashes.
- `summary.json`: all 25 settings, medians, 95th percentiles, repeat medians and counts.

All 25 settings completed three repeats: 75,000 query records. All 60,000 calls in
exact modes matched the local reference IDs. B's finite-budget mode is measured
rather than assumed exact. All workers used the same recorded power configuration,
and hashes of every input and selected-cluster array matched after the run.

Prepared selections/local references are in the ignored
`data/same-clusters-2026-09-13/` folder. They can be regenerated from the existing
embedding pools. The command is:

```sh
.venv/bin/python -m experiments.same_clusters
```

Completed cases are skipped on a repeat invocation. The source runner is
`experiments/same_clusters.py` and its recorded SHA identifies the measured version.
No native search algorithm changed for this follow-up.

These times are separate measurements from the earlier complete-query sweep.
Do not subtract them from the earlier whole-query medians to infer routing costs.
