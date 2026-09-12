# Prefix Backward Walk study TODO

## Current checkpoint

- [x] Commit all current project files before changes.
  Commit: `570b51e` — `2026-09-12: checkpoint all project files before prefix experiments and paper updates`.
- [x] Work on branch `prefix-experiments-v2` in the existing isolated worktree.
- [x] Save this TODO before continuing the remaining work.

## 1. Restore and protect Section 1

- [ ] Restore the supplied wording from attachment `91b13429-39e0-4962-af98-a9f40ec08eef/pasted-text.txt`.
- [ ] Choose the complete draft where the attachment contains duplicates; remove only duplicate pasted equations and editorial directions.
- [ ] Keep the supplied paragraph structure and personal voice. Do not turn it into bullets or rewrite surrounding sentences.
- [ ] Apply only the specifically requested local edits in Section 1:
  - The short “Note that…” explanation and growth graph after Figure 6.
  - Python versions of the algorithm listings; no unrelated prose changes during that conversion.
  - Three short bold labels after Figure 5, with the paragraph wording preserved.
- [ ] Save the restored Section 1 and record its hash. Later tasks must leave it unchanged unless a new request names a specific edit.

## 2. Research the experiment before choosing the grid

- [ ] Collect primary papers and documentation relevant to prefix relaxation, bitplane pruning, and clustered scans.
- [ ] Check existing query/reference geometry to identify promising settings and reasons a method may fail.
- [ ] Declare the recall targets, cluster counts, result counts K, prefix settings and branch settings before the full run.
- [ ] Keep the same document arrays and all 200 query vectors for every method in a comparison.
- [ ] If K changes, compare all methods at that same K using the first K IDs from the unchanged exact reference array.
- [ ] Treat RAM as a common feasibility limit. Do not claim a speed improvement merely by giving methods different hardware or memory limits.

## 3. Implement the intended Backward Walk

- [ ] Add a direct Python reference calculation and meaningful small-case tests. (Started.)
- [ ] Implement a separate `PrefixIndexV2` in C++ with Python bindings. (In progress.)
- [ ] Start at the query's own prefix, remove one trailing bit per level, and score only newly added rows.
- [ ] Finish each depth across all selected clusters before checking the candidate target.
- [ ] Keep the full score and document-ID tie rule unchanged.
- [ ] Check empty prefixes, no rescoring, whole-depth stopping, and target zero matching a full scan over opened clusters.
- [ ] Record prefix depths, binary boundary searches, rows scored, stored bytes and peak process RAM.
- [ ] Run the appropriate tests and save a dated implementation checkpoint.

## 4. Run fair, extensive experiments

- [ ] Connect the new method to the existing isolated runner.
- [ ] Verify document/query/reference hashes before and after every stage.
- [ ] Run a small set of configurations on all 200 queries to check the complete run path.
- [ ] Run the declared parameter grid, including previously missing cluster sizes and useful branch budgets/leaf sizes.
- [ ] Include deterministic branching and a bounded probability/seed comparison where relevant.
- [ ] Let A, B and C each choose their fastest tested settings that meet the same recall target and RAM limit.
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

## Scope boundary

Build the intended prefix index, tests, experiment results and paper. Do not add a new UI,
embedding model, GPU path, online-update system or generic indexing framework. Any code repair
must address an observed failure or a required behavior in the current experiment.
