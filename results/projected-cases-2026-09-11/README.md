# Conditional larger-corpus scenarios

These are forecasts, not measured 1B query timings or a global optimum. See the paper for assumptions and the 8M model check.
The fixed-prefix rows assume 13 known important coordinates. B also gets the option to use that route and a scan-like leaf.

| 1B-row path | Modeled ms | Payload lower bound GB | Planning RAM GB |
|---|---:|---:|---:|
| scan_all | 24726.94 | 40.0 | 49.0 |
| scan_prefix | 3.35 | 40.0 | 49.0 |
| keys | 3.34 | 40.0 | 49.0 |
| branch | 101.59 | 72.0 | 82.9 |
| branch_prefix | 3.26 | 72.0 | 81.0 |

At 32 GB all current layouts are rejected by payload alone. At 64 GB, B is rejected by its 72 GB minimum.
With a good fixed prefix, the modeled A/C/B-scan timings are too close to rank confidently; A/C use less storage.
For the changing-coordinate global B point, scan must score less than about 0.41% of the corpus to undercut it before routing costs. Required routing recall must still be checked.
Doubling B variable cost is sensitivity, not a confidence interval. Planning RAM includes a 1 GB allowance; a modeled fit is not measured resident memory.

Recalculate from the saved input bundle, without the corpus or native extension:
```bash
python -m experiments.project_costs --inputs results/projected-cases-2026-09-11/results.json --output results/my-scenarios --documents 1000000000 --ram-gb 32 64 1000
```
