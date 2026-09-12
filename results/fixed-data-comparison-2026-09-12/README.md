# Unchanged-input parameter comparison

One million document codes, 200 query vectors and the same top-100 reference IDs are used
throughout. Only index parameters change. See `inputs-manifest.json` for the exact files,
hashes and query IDs, and each stage's `inputs-after.json` for the final identity check.

- `sweep/`: 128 parameter settings on all 200 queries, two repetitions each.
- `extra/`: 128 additional branch/key settings on those same queries.
- `repeats/`: 12 selected settings measured in three processes, still on all 200 queries.
- `results.csv`: repeated results at 80%, 90%, 95% and 99% recall requirements.
- `summary.json`: results, raw setting descriptions, process ranges and any failures.
- `inputs/`: per-cluster-count reference ranks and probe cutoffs.

The first selection is retained in `initial-selected-settings.json`. The extension adds
meaningful longer-key cases because short keys cannot rule out any pattern on this input.
No data array is changed to obtain a favorable case. All failed and below-target settings
remain in their stage records. Native and worker hashes identify the implementation used.

Restore a stage's raw cases by extracting `cases.zip` in that directory. The archive's file
hashes are recorded in `archive.json`. Code and commands are documented in
`reports/search-math-experiments/README.md`.
