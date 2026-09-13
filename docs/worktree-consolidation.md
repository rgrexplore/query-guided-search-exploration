# Worktree consolidation — 13 September 2026

Development now uses `main` in the main checkout. All former branch tips are part
of main's history. No commits were discarded or pushed to a remote.

## Merge order

1. `07f09c7` saves the uncommitted fixed-data report updates and run log.
2. `4cd8a2c` merges that study. The report-list conflict keeps prefix v3 as the
   default and adds the older weighted-key report as a separate historical edition.
3. `0a21bb5` connects the branch-order experiment history. Its patch was already applied.
4. `cb4b59f` connects the exact-stop experiment history. Its patch was already applied.

All other branches were already ancestors of main. Their labels and the seven extra
worktrees were removed only after checking ancestry and preserving local files.

| Former worktree | Branch | Preserved tip |
|---|---|---|
| bitplane-deeper-ties | `bitplane-deeper-ties-experiment` | `74a69c1` |
| claude-report | `claude-fixed-data-report` | `07f09c7` |
| component-validation | `component-validation` | `1dbd1d4` |
| concise-report | `brief-formal-report` | `3cc58a0` |
| optima-study | `prefix-experiments-v2` | `d466e8d` |
| prefix-exact-stop | `prefix-exact-stop-experiment` | `d5157f4` |
| scaling-study | `scaling-study` | `723e7ab` |

## Local files

Ignored raw experiment directories were moved into the corresponding `results/`
paths in main, without replacing existing files. Build outputs and caches are preserved
under `build/worktree-archive-2026-09-13/`. Its manifest records all 63 moved paths and
seven removed aliases to the shared main data, environment and runs directories.
The shared directories themselves were not removed.

The previous native library is retained in the archive. The development package is
installed from the main checkout, so its source location no longer refers to a deleted
worktree. Historical run metadata still records its original execution paths.

## Checks

The original and merged trees each passed all 400 tests. The recovered study's command
line imports successfully. Current native source, tests and prefix worker match the
pre-merge main versions. Prefix v3 remains the current PDF and is unchanged by the merge.
