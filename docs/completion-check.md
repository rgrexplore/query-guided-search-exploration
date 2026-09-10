# Research deliverables and evidence

The study compares the best tested settings under declared constraints. It does not prove a
global optimum over every index or certify a billion-document latency prediction.

| Requirement | Evidence |
|---|---|
| Three readable native methods with one score/reference | `cpp/index.cpp`, `cpp/key_index.cpp`, `cpp/score.hpp`; correctness tests |
| Explain the stored data, query steps, work and memory | LaTeX method sections, six Mermaid diagrams, pseudocode and the final formula table |
| Check formulas independently | `verify_math.py`, finite-corpus recall checks, 447 applicable native work predictions |
| Measure physical memory costs | Isolated workers, buffer profiles through 125 MB bitmaps, logical/capacity/RSS separation |
| Investigate failed estimates | Preserved warm/fresh leaf fits, query-weighted fit, median correction and selection-work term |
| Check complete native cost predictions | Frozen 8M check: six methods/support combinations, errors within 7.8% |
| Fair parameter comparisons | Shared scorer/routers, real grids, rank-derived probes, exploration seeds, 16,384-cluster follow-ups and one-bit control |
| Avoid selecting on the final query outcomes | Saved frozen choices; disjoint real-query IDs; constructed follow-up IDs 40–59 |
| Keep losing settings and target misses | Raw case archives and reports, including 98.95% in the one-bit control |
| Establish cases where the methods help | Real Nomic result; small fixed-prefix C advantage; independently checked changing-coordinate B advantage |
| Include actual query encoding | `text_queries.py`,360 complete text requests with references from actual vectors |
| Give larger-N/RAM calculations | `project_costs.py`, explicit assumptions, 32/64GB rejection checks, prefix fallback and scan-fraction crossing |
| Paper -> code -> run navigation | Root README, experiment guide, result index and `examples/small_search.py` |
| Preserve the research history | Local Git checkpoints, earlier PDFs/models/results and verified case ZIPs |

Final integration and artifact verification are recorded below after the checks run.
