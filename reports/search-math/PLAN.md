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

No new retrieval implementation, optimizer calibration, embedding run or billion-document
benchmark is part of this deliverable. The report explicitly identifies what remains empirical.

Completed: 43-page PDF, six figures, twelve complete pseudocode blocks. All arithmetic checks pass; final compilation has no warnings or unresolved references. Every page was visually reviewed.
