# Prefix Backward Walk study TODO

## Current checkpoint

- [x] Commit all current project files before changes.
  Commit: `570b51e` — `2026-09-12: checkpoint all project files before prefix experiments and paper updates`.
- [x] Work on branch `prefix-experiments-v2` in the existing isolated worktree.
- [x] Save this TODO before continuing the remaining work.

## 1. Restore and protect Section 1

- [x] Restore the supplied wording from attachment `91b13429-39e0-4962-af98-a9f40ec08eef/pasted-text.txt`.
- [x] Choose the complete draft where the attachment contains duplicates; remove only duplicate pasted equations and editorial directions.
- [x] Keep the supplied paragraph structure and personal voice. Do not turn it into bullets or rewrite surrounding sentences.
- [x] Apply only the specifically requested local edits in Section 1:
  - The short “Note that…” explanation and growth graph after Figure 6.
  - Python versions of the algorithm listings; no unrelated prose changes during that conversion.
  - Three short bold labels after Figure 5, with the paragraph wording preserved.
- [x] Save the restored Section 1 and record its hash. Later tasks must leave it unchanged unless a new request names a specific edit.

## 2. Research the experiment before choosing the grid

- [x] Collect primary papers and documentation relevant to prefix relaxation, bitplane pruning, and clustered scans.
- [x] Check existing query/reference geometry to identify promising settings and reasons a method may fail.
- [x] Declare the recall targets, cluster counts, result counts K, prefix settings and branch settings before the full run. See `configs/prefix-study-v2/` and `docs/prefix-study-experiment-plan.md`.
- [ ] Keep the same document arrays and every selected query vector for every method in a comparison.
- [ ] If K changes, compare all methods at that same K using the first K IDs from the unchanged exact reference array.
- [ ] Treat RAM as a common feasibility limit. Do not claim a speed improvement merely by giving methods different hardware or memory limits.

## 3. Implement the intended Backward Walk

- [x] Add a direct Python reference calculation and meaningful small-case tests.
- [x] Implement a separate `PrefixIndexV2` in C++ with Python bindings.
- [x] Start at the query's own prefix, remove one trailing bit per level, and score only newly added rows.
- [x] Finish each depth across all selected clusters before checking the candidate target.
- [x] Keep the full score and document-ID tie rule unchanged.
- [x] Check empty prefixes, no rescoring, whole-depth stopping, and target zero matching a full scan over opened clusters.
- [x] Record prefix depths, binary boundary searches, rows scored, stored bytes and peak process RAM.
- [x] Run the appropriate tests and save a dated implementation checkpoint.

## 4. Run fair, extensive experiments

Current run: `results/prefix-study-2026-09-13/`.
Quora256 completed570settings and the selected repeats. Quora1024 has302completed,
qualified settings from380scheduled; its selected repeats are complete. The time-limited
prefix1024 batch and unstarted global alternatives remain recorded, not counted as losses.
Nomic256 is completing its missing groups in small batches; its earlier interrupted
attempt and repeat records are preserved under `attempts/initial-time-allowance/`.

The old outer sequencer has stopped. `checkpoint-continuation.json` records the live
Nomic continuation. Once it ends, the native agent builds the optional deeper-first
order once, then the pool agent prepares Qwen32 and Nomic64. The serial follow-up is
queued in `continue-exploration.py`: short widths, focused Nomic768, and direct A/C
controls. Each stage has a saved time allowance; original schedules remain beside changes.

- [x] Connect the new method to the existing isolated runner.
- [ ] Verify document/query/reference hashes before and after every stage.
- [x] Run a small set of configurations on all 200 original queries to check the complete run path. Saved in `results/prefix-study-pilot-2026-09-13/`.
- [ ] Run the declared parameter grid, including previously missing cluster sizes and useful branch budgets/leaf sizes.
- [ ] Try the documented shorter float prefixes: Qwen 32 dimensions and Nomic 64 dimensions, using the same full-vector caches, document IDs and query IDs. Recompute exact references for each width.
- [ ] Follow up on B with smaller leaves (8 and 32) and larger budgets where the first sweep runs out before reaching useful leaves. Include probe choices above 99% routing recall when local search needs that extra coverage.
- [ ] Give C its depth-zero full-scan setting at A's chosen routing settings, and try shorter starting prefixes around useful C settings. Do not make C pay for 16 widening steps when it needs the whole opened set.
- [ ] Test the optional deeper-first order when branch penalties tie. Patch prepared on isolated branch `bitplane-deeper-ties-experiment`, commit `74a69c1`; do not merge/build it until the current native timing run finishes. Then compare both orders with the same data and budgets.
- [ ] Include deterministic branching and a bounded probability/seed comparison where relevant.
- [ ] Before recommending a shorter representation because it is faster, check equal-score ties and agreement with the full float representation. Keep binary-index recall separate from semantic retrieval quality.
- [ ] Let A, B and C each choose their fastest tested settings that meet the same recall target and RAM limit.
- [ ] Fill A probe-count gaps at any observed B/C advantage between the declared recall targets. Quora256 has B points at98.529% and99.79% with zero splits: budget512 simply scans512 selected clusters. Compare A at512 probes and derive A cutoffs at the other apparent-win recalls before claiming an advantage. C has an apparent58.586% point between sparse A probe samples as well.
- [ ] Retain every setting, including recall misses and resource-limit failures, in the saved data.
- [ ] Repeat the selected settings with timing order varied and power state recorded.
- [ ] Report a B/C win only when it survives the fair comparison. Do not change documents, queries or labels to produce one.
- [ ] Distinguish aggregate results from query-by-query results; do not present selected favorable queries as the whole dataset.
- [ ] Save a dated results checkpoint.

## 5. Update the mathematics from measured work

- [ ] Begin time, memory and recall sections with symbol definitions in plain English.
- [ ] Explain symbols as concrete counts: “how many clusters we open,” “how many rows we score,” and “how many prefix levels we visit.”
- [ ] Begin recall with a small example and then derive the equation.
- [ ] Check predicted work counts against actual ranges, masks and scored rows; investigate a disagreement in either code or math.
- [ ] Update each method's parameter derivation separately, then compare them at the end.
- [ ] Put symbol definitions beside the summary tables. Remove the generic summary-introduction paragraph requested by the user.
- [ ] Keep measured timings distinct from estimated costs; preserve model error in the supporting evidence.

## 6. Rewrite the remaining paper sections

- [ ] Leave the restored Section 1 unchanged.
- [ ] Use the same personal, explanatory voice elsewhere: “In this experiment, I…,” “If we…,” and “So the next question is…”.
- [ ] Use past tense only for experiments actually completed.
- [ ] Keep the useful step-by-step explanations in Section 2; change only the language and required symbol definitions.
- [ ] Rewrite the experiment section around the current A/B/C setup and new measured results.
- [ ] Describe the intended prefix method directly. Remove revision history and prior misunderstandings from the reader-facing paper.
- [ ] Keep historical methods and failed internal attempts in repository history/supporting records, not in the main story.
- [ ] Include scientific limits needed to interpret results, without adding unrelated caution paragraphs.

## 7. Produce the final artifacts

- [ ] Generate clear recall–latency, cluster-size and work-count graphs.
- [ ] State where each method is best within the tested settings; do not claim a global optimum or invent a winner.
- [ ] Check tables against saved measurements and confirm Section 1's protected text is unchanged.
- [ ] Build and inspect the full PDF, keeping Python listings and readable page flow.
- [ ] Rebuild from the source archive independently.
- [ ] Commit with a date, preserve previous PDFs/results, and update memory.

## Expanded experiment: 2026-09-13

- [x] Confirm the first eight pilot settings finish in about43seconds, including process and index preparation.
- [x] Locate the cached one-million-document Nomic embeddings at all768 dimensions.
- [x] Prepare 1,000 fixed MS MARCO queries, then derive 256- and 768-dimensional pools from the same full embeddings.
- [x] Measure Qwen3-Embedding-0.6B encoding on representative Quora questions before choosing a full run.
- [x] Use the measured encoding, exact-reference and search costs to allocate an eight-hour run budget.
- [x] Encode the official Quora corpus and 1,000 test queries; prepare 256- and 1024-dimensional comparisons. Full encoding took 23.83 minutes; all four exact-reference pools took 5.54 minutes.
- [x] Save each dataset/model combination separately. Within that comparison, A/B/C receive identical arrays and exact references.
- [ ] Save progress after encoding chunks, completed parameter settings and experiment stages. Record any timeout or memory stop.

## Scope boundary

Build the intended prefix index, a small experiment runner, real-data results and the paper.
New models and datasets are now authorized as separate comparisons. Reuse the existing full
Nomic document cache. Choose added work using measured time rather than model size alone.
Do not add a new UI, GPU search implementation, online-update system or generic indexing
framework. Repairs must address a demonstrated failure or required current behavior.

Section1 is protected by `docs/section-one.sha256`. Update the abstract and later sections only.

## Fast exploration follow-up

- [x] Keep the next runs in small independent batches, preserving every setting and query.
- [x] Declare Qwen32, Nomic64 and the focused Nomic768 comparisons in configuration files.
- [ ] Finish the current old-build timing sequence, then build the optional B visit order once.
- [ ] Prepare the two short pools and run the next comparisons serially.
- [ ] Fill missing A probe points and measure C at depth zero before calling an apparent win.
- [ ] Repeat promising choices, without repeating the whole grid or adding review rounds.

### Paper setup update

1. Build: describe the actual pools, parameter families and timing limits in the experiment section.
2. Leave out: changes to Section 1, new methods, and results that have not completed.
3. Out of scope: rewriting the whole paper again or adding a separate experiment framework.
4. Necessary: show which inputs stay fixed, distinguish code lengths, and identify actual tested coverage.

## Evidence-led final controls

- [x] Qwen32 completed608settings and selected repeats. B's observed top-one time at the99% target was0.125ms versus A0.760ms with float-trained IVF; B recovered all1000reference IDs in that setting. This is provisional until the binary-trained router comparison.
- [x] Add binary-code training as a router option, shared by A/B/C, keeping every pool array unchanged. Commit `a45a311`.
- [x] Prepare optional C exact stopping in isolated commit `d5157f4`; do not change the active native build during timings.
- [x] Declare focused binary-router comparisons and a global exact-search control in the configuration folder.
- [x] Prepare probability/seed comparisons within B, using the same selected routers and query arrays.
- [ ] After `exploration-finished.json`, integrate C's exact-stop option and run its focused checks once.
- [ ] Run `final-focused-runs.py` after the exact-stop build marker. Keep all parameter choices and full-query records.
- [ ] Check predicted prefix end depths and row counts against the global exact-stop run.
- [ ] Compare all tested router choices within each unchanged pool before writing the final conclusions.

The optional C rule leaves its default behavior unchanged. After a complete prefix
level, any unseen row must differ in at least one retained sign. Its score is at
most Q minus twice the smallest absolute query value in that prefix. If the worst
kept score is above that bound, C can stop without losing any result in the opened
clusters. This adds a stopping check, not a new index layout. The quick reference
probe found possible positive-depth stopping for665/1000Qwen32 top-one queries,
75/1000Nomic64 top-one queries, and none in either256-bit pool. Actual timing and
returned-ID checks remain necessary.

### Finish this run

The remaining scope is fixed: finish the declared routing/exact-stop/probability
controls, check representation agreement, generate the comparison tables and plots,
finish the PDF, and update memory. Do not add another dataset, index layout or search
method to this run. Missing repeated selections, if any, need only a small direct
confirmation; do not repeat the full grid.
