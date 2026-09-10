# Confirmation of the timing model

This run applies frozen coefficients from
`../isolated-calibration-2026-09-10/analysis-relative/models.json` to four new document sizes
and a new random seed. The coefficients were not refitted on these observations.

| Method | Settings checked | Within 20% p50 error | Worst absolute error |
|---|---:|---:|---:|
| Scan | 4 | 4 | 3.7% |
| Bitplanes | 24 | 21 | 27.2% |
| Key lookup | 24 | 24 | 7.2% |

All 52 configurations completed. There were 1,248 saved query observations and 672 exact ordered
reference checks. Power configuration was unchanged in every case. The largest OS-reported
process peak was 252,887,040 bytes, including construction and warmup.

These are random-sign experiments with one cluster, 256 dimensions and top 100 results.
The model uses measured work counts. It does not predict recall, the number of branches a real
query will require, or the cost of a learned cluster router. The global routing step merely
selects the one cluster. CPU, allocator and cache effects are combined in the fitted coefficients.

The scan and key models met the declared 20% criterion in this confirmation range. The bitplane
model did not meet it everywhere. Close method comparisons must use direct measured times;
these estimates alone do not establish the fastest configuration or a billion-document result.

## Files

- `config.toml`: dimensions, sizes, random seed and parameter grid.
- `schedule.json`: randomized execution order.
- `cases/`: each input, process status, query rows and memory observation.
- `analysis-frozen/settings.csv`: measured and predicted values, including the failing settings.
- `analysis-frozen/predicted-versus-measured.png`: the check plot.
- `analysis-frozen/models.json`: unchanged copy of the frozen model.

## What changed after the first calibration

The first model minimized squared error in milliseconds. Slow cases dominated the fit, even
though the stated acceptance criterion was percentage error. Fitting squared percentage error
improved the diagnostic checks. A cache-size indicator did not resolve the remaining bitplane
error and was not added. The original absolute-error fit remains in the calibration run's
`analysis/` directory, and the revised fit in `analysis-relative/`.

This confirmation was run after that revision with the model fixed. It supports the narrower
claims above, not a universal constant cost for every bitmap word or key lookup.
