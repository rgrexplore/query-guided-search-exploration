# End-to-end mathematical report

Deliverable: a rendered LaTeX PDF, editable LaTeX sources, Mermaid diagrams and a small
arithmetic-verification script. The report owns the mathematical model; the unfinished
calculator is not a source of validated predictions.

## Reader's path

1. Define the retrieval task and show one complete query in a small example.
2. Introduce the score, parameters, units and the difference between stored data and query RAM.
3. Explain the shared preparation, embedding, routing, filtering, score tables and reranking.
4. Derive cluster + full scan, with pseudocode, per-step work, storage and query RAM.
5. Derive cluster + bitplanes, including real bitmap widths, both branches, leaves and frontier.
6. Derive cluster + weighted key enumeration, with actual hash/posting layout and subset algorithm.
7. Derive exact recall definitions, prefix error bounds, binomial/weighted distributions and
   conditional-normal approximations; separate full-key exactness from candidate shortcuts.
8. Check the mathematics using worked examples and archived/controlled measurements.
9. Summarize the three end-to-end latency, index-storage and query-memory formulas and explain
   how to search parameters under a fixed RAM and recall target.

All cost formulas use named variables. Numerical settings appear only in labeled examples
and measured-result tables. A/B/C use the same binary-score ground truth when compared.
The primary third method starts with clusters; a global hash is the C=1 special case.

## Scope and checks

- [x] Write and mathematically check the report.
- [x] Render Mermaid diagrams and include them in LaTeX.
- [x] Verify small score, key-order, memory and bitmap examples independently.
- [x] Compile the PDF and resolve layout/cross-reference warnings.
- [x] Inspect rendered pages, including every diagram and long table.
- [x] Save sources, PDF, reproduction instructions and a memory pointer.

The original math-first edition did not include a new retrieval implementation, optimizer
calibration, embedding run or billion-document benchmark. The report explicitly identifies what remains empirical.

Original edition: 43 pages, six figures, twelve pseudocode blocks; preserved at `67bdfc7`.

## Recall extension

- [x] Derive the nearest-distance distribution and conditional prefix survival.
- [x] Derive finite top-K recall, including score ties and duplicate codes.
- [x] Explain how routing, branching and key stopping rules change the candidate event.
- [x] Check weighted and dependent-bit distributions with 10,000 corpora per case.
- [x] Compare native exact scan and unlimited branching against independent references.
- [x] Reanalyse saved prediction errors with query-level intervals.
- [x] Rewrite dense prose, add examples, and render the updated report.

The extension has 55 pages, nine figures and fourteen pseudocode listings. A new real-data
predictor is not claimed: the saved normal approximation fails the proposed two-point error
criterion on the shown Nomic cases. A fresh calibration/test study remains future work, with
its own query provenance required; previously inspected queries are not relabeled as fresh.


## Implemented comparison extension

- [x] Add the implemented directory layout and global key-enumeration policy.
- [x] Report isolated calibration and its remaining prediction errors.
- [x] Report frozen choices on 200 independent real queries, keeping target misses.
- [x] Run fixed and changing strong-coordinate cases with direct-routing scan controls.
- [x] Compare 447 applicable work predictions with native counters.
- [x] Add a small runnable query and a paper-to-code map.
- [x] Compile and inspect the expanded 65-page report.
- [x] Check relevant parameter boundaries and larger working-set costs before extending timing claims.
- [x] Update conditional larger-corpus numerical projections.
- [x] Finish repository integration and final reproduction audit.


Final evidence edition:72pages,11figures,15listings. It includes the explicit limits of the1B
forecasts rather than claiming a measured1B run or a global optimum. Original and intermediate
PDF/source versions remain available in Git.
