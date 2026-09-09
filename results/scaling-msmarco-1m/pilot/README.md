# Scaling pilot

All seven exact-reference jobs and nine pilot cases completed. References cover all 300 queries
at every pool size. Independent score checks passed, and every completed unlimited branch result
matched the exact scan's ordered candidates.

The pilot uses five development queries at 1k, 100k and 1M documents. It checks correctness and
run cost; the final comparisons use the separate evaluation queries after development choices
are frozen. No timeout or memory limit was reached in this pilot.

[pilot.json](pilot.json) preserves process timings, memory observations, power readings and
verification results. These five-query timings are not the final scaling results.
