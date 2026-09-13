# K and prefix experiment progress

- [x] Save the previous report state: `639f672`.
- [x] Add optional prefix-depth snapshots and pass 77 focused checks: `92273e5`.
- [x] Prepare unchanged inputs and 3,120 settings: `d8aa78c`.
- [x] Run the 24-setting pilot on all 1,000 queries.
- [x] Complete the main sweep for K = 1, 2, 3, 5, 10, 20, 50, 100.
- [x] Repeat selected configurations three times.
- [x] Check the depth diagnostic script and record all 33 depths.
- [x] Time A, B and C on identical selected clusters.
- [x] Compare deeper C starts and approximate count stopping on the fixed layout.
- [x] Run the extra K = 7, 15, 30, 75 sweep and repeats.
- [ ] Refine competitive global settings and any crossover interval.
- [ ] Extend the common clustering grid around current winners, including a finer 8,192-cluster scan baseline.
- [ ] Run a targeted 256-bit comparison if time permits.
- [ ] Verify counts, recall, unchanged inputs, power and repeat spread.
- [ ] Inspect the final graphs and explain the measured results.
- [ ] Save code/results checkpoints and update memory.

The report remains unchanged. The complete plan is in the workspace's
`docs/superpowers/plans/2026-09-13-search-k-and-prefix-experiments.md`.

Use `status.json` and completed `process.json` files for live progress. Timing workers
run serially. Do not run the depth timings or a second benchmark alongside the sweep.
