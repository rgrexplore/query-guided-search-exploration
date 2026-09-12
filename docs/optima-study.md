# Derivation and experiment protocol

Status: completed. The protocol below was written before the new timings. This study extends
the saved experiment; it does not overwrite its results or assume that each method must win.
Final evidence is in `results/optima-2026-09-12/`; the paper is
`output/pdf/search-methods-optima.pdf`.

## Question

For one million documents, a 32 GB decimal RAM limit and 99% binary-score Recall@100, which
settings minimize query time for each method? The available family includes global search,
direct sign-prefix groups and learned IVF groups. Each method chooses its own routing and local
settings. The score, tie rule, queries and reference remain common.

The benchmark compares the available C++ implementations, including the time to select clusters
and cross the Python/C++ boundary. It is not a benchmark of Exa's production system.

## Known work and predicted work

Let N be documents, d code bits, C clusters, P opened clusters, K results. Let n_c be the actual
size of cluster c and W_c=ceil(n_c/64) its dense-mask width. Logical score work has G=ceil(d/4)
table terms per scored document. Coefficients below include the specified work and memory access;
they are milliseconds per row/word/lookup, not CPU instruction counts.

| Method | Local latency components |
|---|---|
| A: scan | table preparation + F_A × contiguous score cost + result selection |
| B: bitplanes | table + query ordering + sum(J_c W_c) × split-word cost + sum(E_c W_c) × leaf-word cost + F_B × gathered score cost + queue + result selection |
| C: backward walk | table + key preparation/enumeration + H × directory lookup cost + F_C × posting-score cost + result selection |

F_A=sum n_c over opened clusters. J_c counts splits, E_c enumerated leaves, H all directory
attempts including empty keys. One shared key queue serves all selected clusters. Add routing
to each local time. Do not add a selection/leaf term again if a fitted combined coefficient
already includes it. Work formulas require predictions for F/J/E/H, not just coefficients.

## Controlled distribution

The documents retain the earlier balanced random-sign corpus (seed73), so document layouts
can be reused after hash comparison. Fresh queries use seed20260912:32 tuning and64 evaluation
queries. Both fixed and independently changing supports have twelve strong coordinates of
magnitude1 and244 weak coordinates of magnitude0.0001. No query is discarded.

The strong weight exceeds the entire weak tail: 1 > 244×0.0001. Therefore all documents that
match every strong sign outrank those with a strong mismatch. Normalizing the query scales
all weights equally and preserves the ordering.

Z, the number of strong matches, follows Binomial(N,2^-12): mean244.140625, sd15.6231. Searching
only that group has expected recall E[min(Z,K)]/K = sum(P(Z≥i), i=1..K)/K. The probability
of fewer than100 matches is approximately3.95e-26, but the actual count is checked rather than
assumed. An experiment with insufficient matches remains present and outside the one-path formula.

### Fixed important positions

The first m=12 positions are strong. With r≤m direct routing bits and P=1, A scores about N/2^r
rows. With disjoint h=m-r local key bits, C scores about N/2^m regardless of how those m bits
are divided between routing and local lookup. Its expected occupied directory entries are
2^(r+h) × [1-(1-2^-(r+h))^N]. Routing/lookup allocation and call cost decide the best division.

A at r=m and C at r=0,h=m retrieve the same ideal group. This is an explicit fairness control:
the formula predicts equivalent candidate work. Any supported timing difference is due to the
implemented organization, not an asymptotic advantage.

B at r=0 needs m full-width splits to reach that group, then one full-width leaf enumeration.
At N=1M, W=15625: split visits187500, leaf visits15625, about244 scored rows. Its conditional
peak masks occupy (m+2)×W×8=1.75 MB. At nonzero r, current C++ still visits routing-fixed bits.
If a leaf is reached after all m strong positions, depth is m, not m-r. An initial cluster
already below the leaf threshold instead causes no splits. The native test demonstrates this.

### Changing important positions

For a fixed h-coordinate key, captured strong coordinates S follow Hypergeometric(d,m,h).
If all weak-key patterns compatible with the captured strong signs must be checked, expected
scored fraction is E[2^-S] and expected attempted patterns is E[2^(h-S)]. At h=m=12 this gives
about75% of documents and3072 of4096 patterns; the probability of zero strong overlap is55.48%.
These are policy-specific work predictions, not a general recall estimate. Actual upper bounds
can stop earlier; compare the stated count conditions with the native trace.

B can select the important coordinates separately for each query. The same global one-path
formula applies when enough strong matches exist and the leaf threshold stops by depth m.
For arbitrary learned clusters, the required probes and selected-row counts are empirical.

## Deriving settings

For a global balanced B path stopped at depth j≤m:

T_B(j) ≈ T0 + j W c_split + W c_leaf + N 2^-j c_score.

Its stationary depth is log2(N ln(2) c_score / (W c_split)). Evaluate adjacent integers,
zero and the valid maximum. A larger N increases both potential savings and full-mask width.
For a prefix cluster, charge the redundant fixed-coordinate splits before useful halving.
This derivative does not model arbitrary branch expansion or establish recall by itself.

Expected final occupancy plus3 standard deviations is approximately291, suggesting nearby
leaf sizes288 and320. The preceding strong group averages488 rows. Include128/256/384 and
earlier stopping depths as checks, plus a scan-sized leaf. Do not use the final queries to
choose these neighbors.

For A, T(C)=aC+bNP/C has stationary C=sqrt(bNP/a) only if P is held fixed. At fixed recall,
P generally changes with C. The correct substitution is P=P_rho(C), obtained from a specified
distribution or tuning-only neighbor ranks. The same router family remains available to B/C.
For fixed strong positions, the analytical group widths above identify candidates around r=m.

## Candidate family and measurements

- Direct prefix widths:4,8,10,11,12,13,14; no-routing control; IVF counts256 and4096.
- Probes: tuning-only recall cutoffs for99% and100%, plus analytical fixed-prefix choices.
- Global B leaves:128,256,288,320,384,640,1280,4096,N; routed B:128,320,N. Node budget0 (unlimited).
- Global C widths:1,8,11,12,13,14; routed C:1,4,max(1,m-r), using coordinates after routing.
  Candidate and lookup limits0 (unlimited). Every attempted empty key is charged.
- A receives every layout/probe pair. All methods can independently choose any available layout.
- Two repetitions during tuning; three fresh process blocks and two repetitions per query during
  evaluation. Worker order is shuffled. One CPU thread, batch size1; query encoding excluded.
- Every worker records power before/after, peak process RAM, native binary/source provenance,
  reference overlap, and work counters. Failures/timeouts remain in the result.

The formula layer will select candidate settings after calibration and before evaluation.
Its choices and the fastest measured tuning choices are saved separately. For predictions
requiring measured work or routing, label those empirical inputs explicitly. A numerical
minimum over a finite family is not a universal best index.

## Validation and failure handling

Exact conditional count predictions must equal native counts when their conditions hold.
Distribution expectations are compared with the sample distribution, not asserted as exact
per-query counts. Small exhaustive examples verify the formulas without using the C++ scorer.

The initial time-prediction threshold is20% relative error in median complete retrieval time.
Coefficients are fixed before final evaluation. If the model fails, preserve the original
prediction, diagnose using tuning/component data, and label reused final observations diagnostic.
A revised model needs a separately frozen follow-up check. A failure is not repaired by
relaxing its test threshold or deleting an inconvenient query.

Both compared methods must meet99% measured recall and32GB process budget. Paired95% time-ratio
intervals must lie above1 for a supported speedup. Repeated timings are not new query samples.
The real-data run repeats frozen historical choices on cached real queries and is explicitly
a replication, not a fresh generalization claim. Its actual work counts explain scan fallbacks.

## Paper and checkpoints

New output: `output/pdf/search-methods-optima.pdf`, editable source in `reports/search-math-optima/`.
Target10–12 pages, hard maximum15, with abstract; Exa pipeline/background/method figures;
time/memory/recall derivations and three summary tables; constrained optima; experiments with
predictions and chosen parameters; limitations/next steps; conclusion; references.
Code/evidence entry points are adjacent to their claims. Earlier five editions remain unchanged.

Source attribution: [Exa's vector database](https://exa.ai/blog/building-web-scale-vector-db),
[Matryoshka Representation Learning](https://arxiv.org/abs/2205.13147). Both inspected2026-09-12.
Exa's public article describes learned similarity clusters; sign-prefix routing is a control
implemented in this project, not an attributed detail of Exa's system.

## Completion checks

| Requirement | Verified evidence |
|---|---|
| Independently chosen layouts/local settings | 130 fixed and 299 changing-position tuning cases; `layouts.json`, `tuning-cases.json`, and frozen shortlists |
| Same score, reference, budget and implementation | All 498 worker records completed with one common native hash and worker hash, unchanged recorded power per case, and peak RAM below 32 GB |
| Separate tuning and final queries | Controlled schedules use rows 0–31 for tuning and 32–95 for evaluation; source hashes match the frozen choices |
| Explain and test exact work formulas | 1,395 applicable conditional count checks match native counters; other observations retained |
| Debug failed estimates | First fit retained in `predictions-v1/`; v2 adds expected nonempty leaf words and successful posting starts, tested with independent small distributions and the reproduced mismatch |
| Check revised predictions without refitting | Six frozen model choices meet recall; maximum absolute final p50 error 9.66%, below the unchanged 20% threshold |
| Compare every method pair fairly | `reports/search-math-optima/evidence/results.json` contains paired A/B, A/C and B/C query/process-block intervals |
| Preserve limits and losing settings | Broad-grid residual failures remain; constructed cases are separated from the repeated real-data evaluation |
| Concise paper with requested content | 13 pages including references, eight diagrams/plots, three inline algorithms, time/memory/recall tables, optima, experiments, limits and code pointers |
| Runnable source and previous versions | 18-file LaTeX archive independently rebuilt with identical page text; all five earlier PDF bytes unchanged |
| Code verification | Full current suite: 287 tests pass |

There were 38,736 timed observations across the 498 workers. Repeated observations and repeated
conditional checks are not independent query populations. The claims remain bounded to the
declared family and recorded hardware state; no global optimum or billion-row measurement is claimed.
