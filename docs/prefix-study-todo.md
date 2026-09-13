# Prefix Backward Walk study TODO

## Completed study: 13 September 2026

The native implementation, broad parameter runs, repeated comparisons and mathematical
checks are complete. The final evidence is in
[the dated results](../results/prefix-study-2026-09-13/README.md). Earlier schedules and
partial runs remain there. The complete TODO history is preserved in Git.

## Checkpoints

- [x] Preserve all project files before changes: `570b51e`, dated 12 September.
- [x] Restore the supplied Section 1 and protect its exact text with `docs/section-one.sha256`.
- [x] Implement prefix Backward Walk in C++ with Python bindings and a direct Python reference.
- [x] Verify prefix ranges, no rescoring, whole-depth stopping, candidate targets and ID ties.
- [x] Add the optional exact-stop bound without changing the default index or query behavior.
- [x] Run the appropriate checks once per implementation change: 380 tests for the branch-order build; 147 focused tests for the final exact-stop build.
- [x] Prepare six fixed pools using Nomic/MS MARCO and Qwen/Quora, with 1,000 queries each.
- [x] Keep document, query and reference arrays identical across A/B/C within each pool.
- [x] Use one CPU query at a time and the same 32 GB process cap; record input hashes and power state.
- [x] Run the declared sweeps and follow-ups within recorded time limits. 3,679 of 4,024 A/B/C settings qualify; three grids remain partial and are explicitly labeled.
- [x] Try smaller leaves, deeper-first ties, wider routing coverage, short prefixes and binary-trained clusters.
- [x] Give A direct probe-count controls at apparent B/C advantages, and give C its depth-zero scan.
- [x] Run all 72 probability settings, retaining all three seeds.
- [x] Compare short binary codes with the full float rankings separately from index recall.
- [x] Repeat every final selected comparison in three fresh processes; no final table row lacks its own qualified repeat.
- [x] Check 890 index builds and 3,849,076 query-count checks: zero mismatches.
- [x] Match all four exact-stop predictions against 3,000 global Qwen32 query records.
- [x] Save recall–latency, cluster-count and probability graphs, with their source tables.
- [x] Report the aggregate B win, the conditional C case and the wider-code scan results without changing the data to obtain a winner.
- [x] Rewrite later paper sections with definitions, worked examples and measured results. Leave Section 1 unchanged.
- [x] Inspect the final 26-page PDF and independently rebuild its 14-file source archive with identical extracted text.
- [x] Save the final dated artifact checkpoint. Main receives it by fast-forward after the commit.

Memory handoff: `AGENT_MEMORY/exa-interview/prefix-backward-walk-v2.md` at the
workspace root records completion after the artifact commit.

## Result boundaries

B is about 2.4 times faster than the best tested A at the Qwen32 top-1 99% target.
C is fastest on the 82 queries whose complete code exists in the global exact-search
setup, but B remains fastest across all 1,000 queries. At the top-100 99% target,
the selected B and C settings perform scans across all six pools.

Short-code speed is not equal full-embedding quality: the exact Qwen32 binary top 1
agrees with the full-float best score on 14% of queries. The study does not establish
a universal optimum, a semantic retrieval improvement or billion-document timings.

## Scope

This run is closed to new datasets, algorithms and large sweeps. Complete the PDF,
source bundle, checkpoint and memory only. No new UI, GPU search, update system or
indexing framework is needed. Unfinished grid cells remain recorded rather than
silently counted as losses.
