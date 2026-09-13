how about the experiment, do you think thisis unclear as well?
actually i dont know what is the experimet doing, before rewriting give ur critic whether this experiment makes snese or not
 \section{Experiments}
The previous section describes the work each method performs. Here I vary the index and search settings, compare measured query time at the same recall requirement, and inspect the work counts to understand the results.
Each method can choose its own parameters. Within a comparison, the document codes, queries, score, reference IDs, recall requirement and RAM allowance stay fixed. The question is: at that same recall requirement, what is the fastest configuration we tested for each method?
\subsection{Setup}
I use two collections and several supported code lengths from their cached embeddings. Every method sees all 1,000 fixed queries in a pool. A different model or code length is a separate comparison:
\begin{center}\small
\begin{tabular}{@{}llrrl@{}}\toprule
Collection & Embedding model & Documents & Queries & Bits\\\midrule
MS MARCO & Nomic v1.5 & 1,000,000 & 1,000 & 64, 256, 768\\
Quora (BEIR) & Qwen3 0.6B & 522,931 & 1,000 & 32, 256, 1024\\\bottomrule
\end{tabular}
\end{center}
I reused the cached Nomic document embeddings and selected 1,000 queries from MS MARCO's official \texttt{dev.small} file~\cite{msmarco,nomic}. For Quora, I encoded the full BEIR corpus and selected 1,000 official test queries~\cite{beir,qwen}. Both random selections were fixed before search. Corpus duplicates remain; none of the selected Quora queries has identical text or ID in its corpus.
Each model produces a full vector once. I take supported prefixes, normalize them, and store their signs. Each length has its own reference: score every document and keep the best 100, resolving ties by document ID. The first one or ten IDs give the smaller result sets. Recall here measures recovery of these binary-score references; human relevance labels are a separate measure.
The machine is an Apple M5 Max with 128 GiB of memory. I use one native search query at a time and one Faiss search thread. The completed runs used a common 32 GB process cap. This is a recorded experiment setting, not the budget required by the formulas; we can evaluate other budgets such as 64 or 128 GB. The largest peak among the selected repeated runs was 505.6 MB, so none of those selections was close to the 32 GB cap. Reported query time starts with a cached query vector and ends with the returned IDs. It includes selecting clusters and searching their documents. Embedding generation and text reranking are outside this timer. Encoding and reference preparation finish before timed searches begin.
\begin{figure}[htbp]\centering
\begin{tikzpicture}[node distance=7mm]
\node[box,text width=124mm] (input) {One fixed collection and code length\\Same document codes and 1,000 floating-point queries};
\node[scan,text width=33mm,below left=9mm and -34mm of input] (a) {A: scan\\Own index settings};
\node[branch,text width=33mm,below=9mm of input] (b) {B: Bitplanes\\Own index settings};
\node[keys,text width=33mm,below right=9mm and -34mm of input] (c) {C: Backward Walk\\Own index settings};
\node[box,text width=124mm,below=9mm of b] (check) {Same exact binary top-$K$ reference\\Compare whole-query time, recall and process RAM};
\draw[arr](input)--(a);\draw[arr](input)--(b);\draw[arr](input)--(c);
\draw[arr](a)--(check);\draw[arr](b)--(check);\draw[arr](c)--(check);
\end{tikzpicture}
\caption{Only index and search settings differ within a pool. Changing the code length creates a new pool with its own reference ranking.}
\end{figure}
\subsection{How recall is calculated}
The recall in these experiments is measured from returned document IDs. It is not estimated by assuming a normal distribution or independent bits.
For each query, I first score every document in the collection using the floating-point query and binary document codes. I keep the best $K$ IDs as the reference, resolving equal scores by document ID. A, B and C are then compared with that same reference.
For example, suppose we ask for five results:
\begin{center}
\begin{tabular}{@{}ll@{}}\toprule
Exact scan's top 5 & \texttt{[12, 7, 91, 4, 30]}\\
Method returns & \texttt{[12, 7, 91, 4, 85]}\\
Reference IDs recovered & \texttt{[12, 7, 91, 4]}\\\bottomrule
\end{tabular}
\end{center}
We recovered four of the five reference IDs, so
\[
 \mathrm{Recall@}K=\frac{\text{reference IDs recovered}}{K},
 \qquad \mathrm{Recall@}5=\frac45=80\%.
\]
The table values average this calculation over all 1,000 queries. For $K=1$, recovering the exact best ID on 990 of those queries gives 99\% recall. Result order does not affect this overlap count.
\textbf{What does that percentage mean?} A value of 99\% means recovering 99\% of the exact binary top-$K$ IDs on average. It does not mean finding 99\% of all human-relevant documents, and it does not establish agreement with the original full-float embedding. Those are separate evaluations. Each code length has its own binary reference.
\textbf{Does it measure only search inside a cluster?} The current reference covers the whole collection. A reference document counts as missed whether its cluster was never opened or the local search skipped it after opening that cluster. The reported recall therefore includes both sources of loss.
To isolate search inside selected clusters, we would instead obtain the reference by scanning those same selected clusters completely. B and C could then be compared against that local reference. The existing tables use the whole-collection reference, not this isolated measurement.
\subsection{Parameter search}
For A, I vary the cluster layout, cluster count $C$ and probes $P$. The reference-cluster ranks give probe choices at the required recall targets. B receives those choices along with leaf sizes and node budgets. C receives them along with starting depths, candidate targets and the optional exact-stop rule.
\begin{center}\small
\begin{tabularx}{\linewidth}{@{}lY@{}}\toprule
Settings & Tested ranges and follow-ups\\\midrule
Routing & Float-trained clusters; binary-trained layouts also available to all methods in the short-code follow-ups. Cluster counts include 1, 16, 64, 256, 1024 and 4096.\\
Result count and recall & $K=1,10,100$; targets 50\%, 80\%, 90\%, 95\%, 99\%, with additional probes near high recall and whole-collection controls.\\
B & Initial leaves 128/2048 and budgets 512/8192; shorter codes also use leaves 8/32 and deeper-first ties. Probability 0/0.1/0.2 is a separate three-seed comparison.\\
C & Starting depths 0/1/4/16/32; candidate targets 1,000/10,000/100,000/half the collection/0. Exact-stop runs use target zero.\\\bottomrule
\end{tabularx}
\end{center}
This is a summary of the settings tried, not a full cross-product at every width. The two 256-bit grids each have 570 settings; wider Qwen keeps $K=1,100$, and full-width Nomic focuses on $K=100$ at 95\% and 99\% recall. Across the runs, 3,679 of 4,024 A/B/C settings qualify, plus all 72 probability settings. Three grids remain partial under their recorded time limits. The saved schedules list every attempted and missing setting.
I record complete-query time, recall, process RAM and the method's work counts. For each target, I select the fastest qualifying setting in the main run, then repeat that exact choice in three fresh processes with varied run order. The tables report those repeated observations. Failed recall targets and incomplete runs stay in the saved data.
I also check direct alternatives. If B visits 512 clusters and scores all their rows without splitting, A must be allowed to use 512 probes too. C gets a depth-zero scan at A's selected routing settings. These controls prevent a gap in the parameter grid from looking like an algorithmic advantage.
The equations identify useful quantities to inspect, such as the scored fraction and mask width. They do not turn the broad grid into a proven optimum, or supply the measured time constants in advance.
\subsection{Which method is fastest?}
\input{evidence/study-results}
\subsection{Checking the calculations}
\input{evidence/work-checks}
\subsection{Code and reproduction}
The C++ scorer is shared by all three methods. A and B are in \path{cpp/index.cpp}; C is in \path{cpp/prefix_index_v2.cpp}. The Python worker in \path{experiments/prefix_batch_worker_v2.py} times routing and search, records the work counts, and compares returned IDs with the exact references.
After preparing the arrays using \path{experiments/prepare_prefix_pools_v2.py}, we can run one complete setup with this Python flow:
\begin{lstlisting}
import json
from pathlib import Path
from experiments import prefix_study_v2 as study
path = Path("configs/prefix-study-v2/qwen-quora-d256.json")
config = json.loads(path.read_text())
output = Path("results/my-quora-run")
study.prepare(config, output)  # Build clusters and the schedule.
study.run(output)             # Measure the declared settings.
study.summarize(output)       # Select by recall and time.
study.repeat(output)          # Repeat those choices three times.
\end{lstlisting}
Full commands, input hashes, selected settings and raw query records are linked from \path{results/prefix-study-2026-09-13/README.md}. The data and encoding recipe is in \path{docs/expanded-pool-preparation.md}. The root \path{README.md} maps the paper to its code, including the exact-stop prediction and full-float agreement checks.
\section{Limitations and next steps}
The timings describe this CPU implementation. All three methods reuse the same C++ lookup-table scorer, which adds entries using float64. Exa's production system and its SIMD kernels are not reproduced here. A faster scoring kernel can change the balance between scoring documents and maintaining an index search.
I hold the collection, model and code length fixed when choosing each method's parameters. Shorter codes change the reference ranking, so their binary recall does not establish the same retrieval quality as a longer representation. The source datasets provide human relevance labels for that separate evaluation.
These are static indexes built before querying. The experiment does not measure document updates or concurrent requests. In particular, keeping rows sorted still costs work when new documents are added. The memory formulas count the binary search index and its query buffers; an application that keeps uncompressed vectors or document text in RAM must add that storage.
Prefix expansion also appears in LSH Forest~\cite{lshforest} and PUFFINN~\cite{puffinn}. Those methods use multiple hash indexes and different stopping rules. Their recall guarantees do not transfer to a single prefix in the embedding's original coordinate order.
\section{Conclusion}
\input{evidence/study-conclusion}
\begin{thebibliography}{9}\small\raggedright
\addcontentsline{toc}{section}{References}
\bibitem{exa} The Exa Team. \emph{How we built a web-scale vector database}. 17 December 2024.
\url{}.
\bibitem{hnsw} Y. Malkov and D. Yashunin. \emph{Efficient and robust approximate nearest neighbor search using Hierarchical Navigable Small World graphs}. 2016; revised 2018.
\url{}.
\bibitem{mrl} A. Kusupati et al. \emph{Matryoshka Representation Learning}. NeurIPS, 2022.
\url{}.
\bibitem{nomic} Nomic AI. \emph{nomic-embed-text-v1.5}. Model card, accessed 13 September 2026. The saved manifest identifies the exact model revision.
\url{}.
\bibitem{qwen} Y. Zhang et al. \emph{Qwen3 Embedding: Advancing Text Embedding and Reranking Through Foundation Models}. 2025.
\url{}. Encoding recipe: \url{}.
\bibitem{msmarco} P. Bajaj et al. \emph{MS MARCO: A Human Generated MAchine Reading COmprehension Dataset}. 2016; revised 2018.
\url{}. Passage-ranking data: \url{}.
\bibitem{beir} N. Thakur et al. \emph{BEIR: A Heterogeneous Benchmark for Zero-shot Evaluation of Information Retrieval Models}. NeurIPS Datasets and Benchmarks, 2021.
\url{}. Dataset catalog: \url{}.
\bibitem{lshforest} M. Bawa, T. Condie and P. Ganesan. \emph{LSH Forest: Self-Tuning Indexes for Similarity Search}. WWW, 2005, pp. 651--660.
\url{}.
\bibitem{puffinn} M. Aum\"uller, T. Christiani, R. Pagh and M. Vesterli. \emph{PUFFINN: Parameterless and Universally Fast FInding of Nearest Neighbors}. ESA, 2019.
\url{}.
\end{thebibliography}





Show more







Yes — the experiment mostly makes sense technically, but I think the reason you feel lost is real: right now it is trying to answer too many different questions at once, and one of them does not fully match the question stated in your introduction.
The strongest part is this core comparison:
Fix one dataset, one embedding model, one code length, one \(K\), one set of queries, and one exact binary reference. Let A, B, and C each tune their parameters. At a given recall target, compare the fastest valid configuration.

That is a legitimate experiment. Your Qwen-32 example does exactly that: at 99% binary recall for \(K=1\), A, B, and C are compared against the same 32-bit binary ranking, and B is fastest. search-methods-prefix-v3.pdfPDF
The problem is that the report currently mixes that core experiment with several other experiments:
- different code lengths: 32 / 256 / 1024;
- different models and datasets;
- \(K=1,10,100\);
- float-trained vs binary-trained routers;
- approximate vs exact stopping;
- random Bitplane exploration;
- exact-code-present vs absent queries;
- binary ranking vs full-float ranking;
- memory checks;
- formula-validation checks.
All of these can be useful, but together they make it hard to answer the basic question: “What experiment am I looking at right now?”
There is also one more important conceptual issue.
Your introduction says:
B and C replace the search stage after cluster selection.

But your main experiment allows each method to choose a different router, cluster count, and probe count. For example, at Qwen32 top-1 / 99% recall:
- A: 4096 clusters, 180 probes
- B: 16 clusters, 9 probes
- C: 1024 clusters, 78 probes. search-methods-prefix-v3.pdfPDF
That comparison is valid if your question is:
“What is the fastest complete retrieval system I can build using A, B, or C?”

But it does not isolate:
“Is Bitplanes/Backward Walk a better replacement for the local search step after the same clusters have already been selected?”

Those are two different research questions.
In fact, your own results section already notices this for C: C scores more rows than A's selected configuration but is still faster, while also using completely different routing. You correctly say that you cannot attribute the difference entirely to prefix pruning. search-methods-prefix-v3.pdfPDF
So I think your experiment needs two levels.
First should be the clean mechanism experiment:
Fix the routing configuration \(C,P\) and therefore the opened clusters for all three methods. Compare:
A = scan those rows
B = Bitplanes over those exact rows
C = Backward Walk over those exact rows

Now you can directly answer:
- Did B reduce enough scores to repay bitmap work?
- Did C reduce enough scores to repay prefix lookups?
- At the same opened clusters, what recall did local approximation lose?
This connects directly to the complexity section you just wrote.
Then separately do the practical optimization experiment:
Allow A, B, and C to independently choose \(C,P\) and their local parameters. Ask which complete system is fastest at the same global recall.

That tells you whether B's unusual behavior — for example, opening a much larger candidate pool and aggressively pruning it — creates a better overall operating point. And your current Qwen32 result is actually interesting precisely because B does this: it opens about 295k rows but only scores 818, while A opens/scans about 24k. search-methods-prefix-v3.pdfPDF
That is a cool result. Right now, however, it appears before the reader has been shown the simpler fixed-routing comparison, so it is difficult to interpret.
A second thing I would change is how you handle code length.
When you compare:
- Qwen 32-bit,
- Qwen 256-bit,
- Qwen 1024-bit,
those are not simply three sizes of the same search problem, because you recompute the ground truth for each code length. A 99%-recall 32-bit system means:
99% agreement with the exact 32-bit binary ranking.

It does not mean 99% agreement with the 1024-dimensional embedding. You do state this, and your full-float agreement experiment is actually useful: Qwen32 binary top-1 agrees with the full-float best on only 14% of queries. search-methods-prefix-v3.pdfPDF
But structurally, I would treat code length as a separate sensitivity experiment, not mix it into the core A-vs-B-vs-C answer.
Similarly, \(K=1\) versus \(K=100\) is important, because your result genuinely changes:
- with Qwen32, \(K=1\): B has a strong advantage;
- with \(K=100\): the selected B/C configurations become essentially scan-like and the advantage disappears. search-methods-prefix-v3.pdfPDF
That should be a deliberate second research question:
How does result-set size \(K\) change which method is favorable?

Right now it appears under “Where does the result change?”, which is correct evidence but not a very clear experimental plan.
There are also two smaller issues.
Your 32 GB RAM requirement currently doesn't really constrain anything: the largest selected repeated-process peak is only about 505.6 MB. search-methods-prefix-v3.pdfPDF So it is fine to keep 32 GB as a feasibility constraint, but don't make it sound like the experiments are exploring a latency-vs-memory frontier. In this dataset size, they're not. All selected configurations are comfortably feasible.
And the “parameter search” wording is a little confusing. It says B “receives” A's routing choices, but the final selected B configuration clearly chooses a very different routing layout. I would explicitly say:
Each method can use every eligible routing configuration in the declared routing grid; A, B, and C then add their own method-specific parameters.

That is much clearer.
So I think the experiment itself is good enough to support interesting conclusions, but I would reorganize it around explicit questions:
1. Local-search experiment: with the same opened clusters, does B or C beat scanning?
2. End-to-end optimized experiment: after allowing each method to choose its own routing + local parameters, which is fastest at the same global binary recall?
3. Operating-regime experiments: why does the answer change with \(K\), code length, exact-prefix availability, etc.?
4. Validation/ablations: memory formulas, score-bound prediction, randomness, full-float agreement.
Then the reader will understand immediately why your headline Qwen32 result matters:
B is not merely a faster way to scan the same cluster. Its best complete configuration chooses a very different routing/local-search balance: open a much broader pool, then use query-dependent Bitplanes to avoid almost all of the full scores.

That, to me, is actually the most interesting finding in the current report. The experimental design can expose it well; the current organization just makes the reader work too hard to discover it.




 Sources





ok then should we run more experiment ore restart it which one we shoudl run again adn how do we struture our report so that the reader understand and we focus on what the reader actually care













I would not restart everything. Most of the work you already ran is useful. The problem is mainly that the experiments were accumulated over time, so the story is harder to see than the evidence itself.
I’d keep the existing runs as supporting evidence, then run 2–3 targeted experiments that answer the questions cleanly.
The reader really cares about only two things:
1. Does B or C actually make the search stage better than A when everything before it is held fixed?
2. If we let each method optimize its whole retrieval configuration, is the final system actually faster at the same recall?
Right now your experiment mostly answers #2, but your introduction sounds like you are answering #1. That mismatch is the main issue. search-methods-prefix-v3.pdfPDF
Experiment 1 — Same routing, compare only the local search
This is the one I think you should definitely run.
Fix exactly the same:
- collection
- code length
- queries
- cluster layout \(C\)
- probes \(P\)
- exact opened clusters for every query
Then run:
- A: scan all those rows
- B: Bitplanes on those rows
- C: Backward Walk on those rows
Now there is zero ambiguity.
For every run record:
Method	Global recall	Local recall	Rows opened	Rows scored	Extra work	Latency
A		100%			—	
B			same		split/leaf words	
C			same		depths/lookups	


Here local recall means: compare against the exact top-K obtained by fully scanning those same opened clusters.
That lets you directly say:
Routing gave us these 50,000 rows. Does Bitplanes actually search those rows faster than scanning them?

This connects perfectly to Section 2.
Even better: use your already selected routing configurations
You don't need hundreds of new configurations.
For your headline Qwen32, \(K=1\), 99% case, you currently have:
- A routing: \(C=4096,P=180\)
- B routing: \(C=16,P=9\)
- C routing: \(C=1024,P=78\). search-methods-prefix-v3.pdfPDF
Take each routing choice and run all three local methods on it.
Then make a 3×3 matrix:
Routing originally selected by	Scan A	Bitplanes B	Backward Walk C
A routing: 4096 / 180			
B routing: 16 / 9			
C routing: 1024 / 78			


This would be extremely informative.
For example, maybe you discover:
On B's 16-cluster routing, A takes 2.0 ms because it scans 295k rows, while B takes 0.08 ms because it scores only 818.

Then the reader immediately understands the innovation:
B makes it possible to use a much coarser router and compensate with cheap query-dependent pruning.

That is far more interesting than merely saying “B = 0.0809 ms, A = 0.1948 ms.”
Experiment 2 — Best complete system
Then keep what you already basically have.
Allow every method to independently choose:
A
\[
C,\ P
\]B
\[
C,\ P,\ \text{leaf size},\ \text{node budget}
\]C
\[
C,\ P,\ x,\ \text{stopping rule}
\]and ask:
At 80%, 90%, 95%, 99% global recall, what is the fastest valid configuration for each method?

This is your practical system result.
Your existing Qwen32 result is already interesting:
Required recall	A	B	C
80%	0.0308	0.0295	0.0318
90%	0.0498	0.0432	0.0497
95%	0.0805	0.0520	0.0799
99%	0.1948	0.0809	0.1688


That says B has a real operating regime where it is substantially faster. search-methods-prefix-v3.pdfPDF
I would not rerun every old grid.
I would only rerun the headline pools if the routing options were not symmetric. For the final paper, make sure:
A, B and C all have access to exactly the same router families and cluster layouts.

Otherwise someone can fairly ask whether B won simply because A wasn't offered its best router.
Experiment 3 — When does the conclusion change?
This should be a much smaller sensitivity study.
You already have evidence that the answer changes dramatically with \(K\).
Positive regime
Qwen32, \(K=1\):
B strongly wins.

Negative / scan-like regime
For \(K=100\), your selected B and C settings often stop pruning and essentially turn back into scans. For example, at 99% recall on Qwen256:
- A: 3.564 ms
- B: 3.721 ms
- C: 4.039 ms. search-methods-prefix-v3.pdfPDF
That is a very useful contrast.
So your third question can simply be:
What workload properties determine whether pruning is useful?

Test just:
- \(K=1\)
- \(K=10\)
- \(K=100\)
and perhaps:
- short code
- medium code
- full code
Don't turn it into another giant parameter sweep.
What I would not rerun
Keep these existing experiments, but move them to supporting/ablation sections:
- random branch exploration;
- exact code present vs absent;
- memory formula verification;
- exact-stop prediction;
- full-float agreement;
- independent-bit approximation checks.
They're useful, but they are not the main experiment.
For example, your exact-code-present result is a nice explanation of C:
when the full query code exists, C can be extremely fast — 4.875 µs in that subgroup — but those queries are only 82/1000, so it does not dominate overall. search-methods-prefix-v3.pdfPDF

That's a great mechanism analysis, not the headline result.
The report structure I would use
This would make the entire report much easier to follow.
1. Motivation and methods
What is A?
Observation:
\[
S(q,b)=Q-2p.
\]Then:
- B = broad → narrow, query-dependent
- C = narrow → broad, fixed Matryoshka prefix
No experiments yet.
2. What should make B or C faster?
Much simpler than the current huge calculation section.
A
\[
T_A
=
T_{\text{route}}
+
\text{full scoring}.
\]B
\[
T_B
=
T_{\text{route}}
+
\text{bitmap work}
+
\text{remaining scoring}.
\]Therefore:
B wins when scores avoided > bitmap work added.

C
\[
T_C
=
T_{\text{route}}
+
\text{prefix lookup}
+
\text{remaining scoring}.
\]Therefore:
C wins when scores avoided > prefix work added.

Then memory and recall.
That's all the reader needs before seeing data.
3. Experiments
Start this section with the research questions explicitly.
The experiments answer three questions:
Q1. Holding routing fixed, can B or C search the same candidate pool faster than A?
Q2. When each method chooses its own routing and local settings, which complete system is fastest at the same global recall?
Q3. Under what workloads does the answer change?

That one paragraph will make the whole report much easier to understand.
3.1 Fixed data and measurement
Dataset, embeddings, queries, reference, hardware.
Keep this concise.
3.2 Experiment 1 — Same routing
This is the clean algorithmic test.
First table:
Same routing, same candidate pool
Routing	Method	Global recall	Local recall	Rows opened	Rows scored	Extra work	ms


Then immediately explain:
Here every method receives exactly the same rows. Any timing difference therefore comes from the local search strategy.

This is the missing experiment in the current report.
3.3 Experiment 2 — Best complete configuration
Then your current optimized comparison.
Main headline table
Recall	A	B	C	Winner
80				
90				
95				
99				


Then latency-vs-recall graph.
Then the parameter table:
Method	C	P	Local settings	Rows opened	Rows scored	Recall	ms


This is where your current Qwen32 result belongs.
3.4 Why did the winner win?
Now inspect the work counts.
For B:
\[
\text{opened rows}
\rightarrow
\text{split work}
\rightarrow
\text{rows scored}.
\]For your current Qwen32 result:
- A opened/scored ~24k;
- B opened ~295k;
- B scored only ~818;
- B paid ~53.7k split-mask words + ~5.8k leaf words. search-methods-prefix-v3.pdfPDF
That is a fantastic figure.
I would visualize it like:
A
24k opened
████████████████████
24k scored


B
295k opened
████████████████████████████████████████████████████████
                   ↓ Bitplanes
                   818 scored
                   █
Then put the bitmap-work number beside B.
The reader instantly sees what's happening.
3.5 When does it stop working?
Now show \(K=1\) vs \(K=100\).
Something like:
Workload	A	B	C	Interpretation
Qwen32, K=1	0.195	0.081	0.169	B prunes heavily
Qwen32, K=100	0.400	0.439	0.404	B/C become scan-like
Qwen256, K=100	3.564	3.721	4.039	scoring dominates / pruning not useful


This answers where the method is useful, which is much more valuable than reporting dozens of parameter grids.
3.6 Representation quality
Only here discuss 32 vs 256 vs 1024.
Because otherwise readers can misread:
“32-bit B is faster”

as
“B retrieves the same thing as the 1024-dimensional model faster.”

Your current result already shows that's false:
- Qwen32 binary top-1 agrees with full-float best only 14%;
- Qwen256: 75.8%;
- Qwen1024: 88.6%. search-methods-prefix-v3.pdfPDF
So frame this as:
Search speed and representation quality are two separate axes.

Very important.
3.7 Mechanism / ablations
Put here:
- exact prefix present vs absent;
- exact stopping;
- random branch order;
- memory validation.
They support the explanation but aren't needed to understand the main result.
4 Discussion
This is where you can say something much more meaningful than “B wins.”
Potential conclusion from your existing evidence looks like:
Bitplanes are most useful when the result set is small and the query weights permit aggressive pruning. Their advantage is not that they search a smaller routed pool; in the strongest result, B deliberately opens a much larger pool than A and uses cheap bitwise filtering to score only a tiny fraction of it.
As \(K\) increases, the search needs many more candidates. B can no longer prune enough to repay the bitmap work, and its selected configuration collapses toward an ordinary scan.
Backward Walk can be extremely fast when a specific query prefix exists, but this condition is uncommon enough in the current data that it does not win overall.

That's an actual research finding.
So: do we need more experiments?
Yes, but not many.
I would run exactly these first:
1. Matched-routing cross-over experiment on Qwen32 \(K=1,99\%\).
   - Three existing winning routers × A/B/C local methods.
2. Same cross-over experiment on one negative case, probably Qwen256 \(K=100,99\%\).
3. Rerun the headline optimized comparison only if necessary so A/B/C all receive the exact same router family/grid.
Then stop.
Those experiments would close the biggest interpretability gap in the current report without throwing away the thousands of runs you already have.