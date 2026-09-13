I’d use **“I”** consistently for the experiment section because the rest of the report is already written as a single-author investigation: “I study…”, “I am trying to improve…”. “We” can sound like a different paper suddenly appeared. You can still use “let’s” occasionally when walking the reader through an example, but for methodology use **I**.

I’d also stick closely to terms already familiar from the Exa pipeline:

- **cluster selection / routing** = first select nearby clusters to reduce how many documents need to be searched;
- **opened clusters** = the clusters selected for this query;
- **documents inside the opened clusters** instead of “candidate pool”;
- **full score** = the float-query × binary-document score;
- **local search** = what A/B/C do *after* the clusters have been selected.

That makes the experiment map directly onto Figure 1.

Below is how I would write the **full experiment section**. I’ll distinguish results you already have from the one new matched-routing experiment that still needs to be run. 

---

# 3 Experiments

The previous section tells us what each method spends time on.

Method A mainly pays for fully scoring the documents inside the opened clusters. Method B adds Bitplane traversal in the hope of avoiding some of those scores. Method C adds prefix lookups in the hope of starting from a much smaller set of documents.

Now I want to answer three practical questions:

1. **If A, B and C are given exactly the same opened clusters, can B or C search those documents faster than simply scanning them?**
2. **If each method is allowed to choose its own cluster and search settings, which complete retrieval setup is fastest at the same recall?**
3. **When does the answer change?** For example, does Bitplanes only help when returning one result, or does the advantage remain when we need 100?

The first question isolates the new search methods themselves.

The second is closer to how the methods would actually be deployed: B may prefer a completely different number of clusters from A because it can afford to open more documents and then prune them locally.

The third helps us understand *why* a method works in one setting but not another.

---

# 3.1 Setup

I use two document collections and two embedding models:

| Collection | Embedding model | Documents | Queries | Tested binary lengths |
|---|---|---:|---:|---|
| MS MARCO | Nomic v1.5 | 1,000,000 | 1,000 | 64, 256, 768 |
| Quora (BEIR) | Qwen3 0.6B | 522,931 | 1,000 | 32, 256, 1024 |

For MS MARCO, I reuse the cached Nomic embeddings and select 1,000 queries from the official `dev.small` set.

For Quora, I encode the full BEIR corpus using Qwen3-Embedding-0.6B and select 1,000 queries from its official test set.

The same fixed queries are used by A, B and C.

## What exactly stays fixed?

For one comparison, I fix:

- the document collection;
- embedding model;
- binary code length;
- query vectors;
- result count \(K\);
- scoring function;
- exact reference results.

Only the **index/search settings** are allowed to change.

For example, when comparing A, B and C on Qwen with 32-bit codes, all three methods search the exact same 522,931 binary documents using the same 1,000 floating-point queries.

Changing from 32 bits to 256 bits is a **different comparison**, because it also changes the binary ranking.

### Diagram

I would keep this diagram very simple:

```text
                Fixed experiment

      Same binary documents + same queries
                       |
             cluster selection
                       |
          -------------------------
          |           |           |
       A: scan    B: Bitplanes   C: Backward Walk
          |           |           |
          -------------------------
                       |
             same exact binary
                top-K reference
                       |
          compare recall + latency
```

Use:

- blue for A;
- orange for B;
- purple for C;
- gray for everything shared.

Caption:

> **Figure X:** Within one experiment, A, B and C use the same documents, queries, score and binary reference. Only their index and search settings change.

The purpose of this figure is just to make the fairness condition obvious.

---

## What time do I measure?

The timer starts from an already-computed query embedding.

It includes:

```text
select nearby clusters
        ↓
search documents inside those clusters
        ↓
return top-K IDs
```

So it includes both **cluster selection** and the local A/B/C search.

It does not include:

- generating document embeddings;
- generating the query embedding;
- building the index;
- final text or full-vector reranking.

Those are separate stages from the search mechanism studied here.

The machine is an Apple M5 Max with 128 GiB physical memory. Searches use one native query at a time and one Faiss search thread.

I keep a 32 GB process-memory limit for all methods. In the completed runs, the selected configurations are far below this limit, so memory is a feasibility check rather than the main bottleneck.

---

# 3.2 Reference results and recall

Before timing approximate search, I need to know the correct answer under the binary score.

For every query, I therefore score **every document in the collection** using the same float-query × binary-document score used by A, B and C.

I then keep the exact top \(K\) document IDs.

These IDs form the reference.

For example, suppose:

```text
Exact top 5:
[12, 7, 91, 4, 30]

Method returns:
[12, 7, 91, 4, 85]
```

Four of the five reference IDs were recovered, so:

\[
Recall@5=4/5=80\%.
\]

The reported recall averages this overlap across all 1,000 queries.

For \(K=1\), 99% recall means returning the exact best binary document for 990 of the 1,000 queries.

## What does 99% recall mean here?

It means:

> **99% agreement with the exact top-\(K\) ranking under this binary representation.**

It does **not** mean:

- 99% human relevance;
- 99% agreement with the original uncompressed embedding;
- or 99% semantic accuracy.

Those are different questions.

This distinction becomes especially important when comparing different code lengths.

---

# 3.3 Experiment 1 — same opened clusters

This is the cleanest way to test the new search algorithms themselves.

Recall the pipeline:

```text
query
 ↓
select nearby clusters
 ↓
A / B / C search inside them
```

For this experiment, I hold the first stage fixed.

For each query, A, B and C receive **exactly the same opened clusters**.

This means they receive exactly the same documents before their local search begins.

The only difference is what happens next:

### Method A

Score every document in those clusters.

### Method B

Use Bitplanes to avoid some full scores.

### Method C

Use prefix relaxation to avoid some full scores.

This lets us answer:

> **Given the exact same set of documents, is the extra work performed by B or C actually cheaper than scanning them?**

---

## 3.3.1 Routing settings to test

I would not run another huge grid here.

Use the three routing configurations already selected by the headline Qwen32, \(K=1\), 99% experiment:

| Routing source | Clusters \(C\) | Probes \(P\) |
|---|---:|---:|
| A's selected routing | 4096 | 180 |
| B's selected routing | 16 | 9 |
| C's selected routing | 1024 | 78 |

Then run **all three local search methods** on every routing configuration.

This produces the most useful table in the paper:

### Table X — Same routing, different local search

| Opened clusters come from | Local search | Opened rows | Rows fully scored | Local recall | Global recall | Extra work | Query time |
|---|---|---:|---:|---:|---:|---|---:|
| A routing | A scan | … | … | 100% | … | — | … |
| A routing | B Bitplanes | same | … | … | … | split/leaf words | … |
| A routing | C Backward Walk | same | … | … | … | prefix depths/lookups | … |
| B routing | A scan | … | … | 100% | … | — | … |
| B routing | B Bitplanes | same | … | … | … | … | … |
| B routing | C Backward Walk | same | … | … | … | … | … |
| C routing | A scan | … | … | 100% | … | — | … |
| C routing | B Bitplanes | same | … | … | … | … | … |
| C routing | C Backward Walk | same | … | … | … | … | … |

The word **same** in the opened-row column should be visually highlighted within each three-row block.

### Why this table matters

Suppose B is still much faster than A under B's 16-cluster / 9-probe routing.

Then we can say:

> B's advantage really comes from the Bitplane local search.

If instead A scanning the exact same opened clusters is equally fast, then B's overall advantage came mainly from choosing a different routing layout.

That distinction is currently missing from the report.

---

## Local recall

For this experiment I would report a second recall number:

**local recall**.

Instead of comparing with the whole collection, first fully scan the exact same opened clusters and obtain their exact local top \(K\).

Then ask:

> How many of those local top-\(K\) results did B or C recover?

For A this is always 100%, because A scans every opened document.

This separates:

```text
recall lost because the correct cluster was never opened
```

from:

```text
recall lost because B/C stopped searching too early
```

That makes the behavior much easier to interpret.

---

# 3.4 Experiment 2 — fastest complete configuration

The previous experiment holds routing fixed.

But that is not necessarily how we would actually deploy these methods.

Bitplanes may work best with **larger clusters and fewer probes**, because it can open many documents and then remove most of them cheaply.

Backward Walk may prefer a different cluster size again.

So in the second experiment, I allow every method to choose its own complete configuration.

For A, I vary:

```text
cluster layout
C
P
```

For B:

```text
cluster layout
C
P
leaf size
node budget
```

For C:

```text
cluster layout
C
P
starting prefix depth
stopping rule / candidate target
```

Every method has access to the same available routing layouts.

For each recall target, I select the fastest configuration whose measured recall meets the requirement.

Then I repeat that exact configuration in three fresh processes.

This answers the practical question:

> **If I were actually choosing the fastest implementation of each method, which one would I use?**

---

# 3.5 Main result

I would make the first headline result one setting only:

> **Qwen, Quora, 32-bit codes, return one result.**

Why this one first?

Because this is currently the clearest case where the new method actually changes the result.

The measured repeated medians are: 

| Required binary recall | A: scan | B: Bitplanes | C: Backward Walk | Fastest |
|---:|---:|---:|---:|---|
| 80% | 0.0308 ms | **0.0295 ms** | 0.0318 ms | B |
| 90% | 0.0498 ms | **0.0432 ms** | 0.0497 ms | B |
| 95% | 0.0805 ms | **0.0520 ms** | 0.0799 ms | B |
| 99% | 0.1948 ms | **0.0809 ms** | 0.1688 ms | B |

At the 99% target, B is approximately:

\[
0.1948/0.0809\approx2.4\times
\]

faster than A.

This is the first result the reader should see.

---

## Diagram — latency versus recall

Use the plot you already have:

- x-axis: required binary recall;
- y-axis: median query time;
- blue: A;
- orange: B;
- purple: C.

But initially show **only the K=1 plot**.

Move K=100 to the later “when does this stop working?” subsection.

Otherwise the reader has to interpret two different regimes at once.

Caption:

> **Figure X:** Qwen32, \(K=1\). B has the lowest measured latency at all four high-recall operating points.

---

# 3.6 Why is B faster here?

Now explain the headline result.

At the 99% target, the selected settings are: 

| Method | Clusters | Probes | Local setting | Recall | Median |
|---|---:|---:|---|---:|---:|
| A | 4096 | 180 | scan | 99.00% | 0.1948 ms |
| B | 16 | 9 | leaf 128, budget 8192 | 99.20% | 0.0809 ms |
| C | 1024 | 78 | start 4, exact stop | 99.00% | 0.1688 ms |

The interesting part is that B is **not** opening fewer documents.

Its selected routing opens far more:

| Method | Opened rows | Rows fully scored |
|---|---:|---:|
| A | 24,181 | 24,181 |
| B | 295,031 | 818 |
| C | 41,147 | 26,153 |

B opens about 295 thousand documents, but fully scores only about 818 of them.

That is roughly:

\[
818/295031\approx0.28\%.
\]

So the selected B configuration follows a very different strategy from A:

### A

```text
route very aggressively
      ↓
only ~24k documents remain
      ↓
score all ~24k
```

### B

```text
use much coarser routing
      ↓
~295k documents remain
      ↓
Bitplanes prune them
      ↓
score only ~818
```

That is probably the most important finding in the report.

---

## Diagram — what A and B are doing differently

I would absolutely make a dedicated figure.

### A

Blue:

```text
24,181 opened
████████████████████████

24,181 fully scored
████████████████████████
```

### B

Orange:

```text
295,031 opened
████████████████████████████████████████████████████████████

                     ↓ Bitplanes

818 fully scored
█
```

Beside B:

```text
53,693 split-mask word visits
5,838 leaf-mask word visits
```

Caption:

> **Figure X:** At the selected 99% setting, B deliberately opens a much larger set of documents than A, but uses Bitplanes to reduce the number receiving a full score from about 295k to 818.

This is much easier to understand than presenting the work-counter table first.

---

## Does this match the complexity section?

Yes.

Section 2 said B only helps when:

> **the cost of the scores avoided is greater than the bitmap and branch work added.**

Here B avoids approximately:

\[
295031-818
\]

full scores.

It pays instead for approximately:

- 53,693 split-mask word visits;
- 5,838 leaf-mask word visits;
- queue/branch management.

The complete measured query is still faster.

So in this operating regime, the break-even condition is crossed.

We do not need to pretend that one mask-word visit has the same cost as one full score. The whole-query timing already gives the final answer; the counters explain *why*.

---

# 3.7 What about Backward Walk?

At the same 99% operating point, C uses:

```text
1024 clusters
78 probes
start depth 4
exact stopping
```

It opens around:

\[
41,147
\]

documents and scores about:

\[
26,153.
\]

So C does remove some work, but much less aggressively than B.

It also performs around:

- 3.5 prefix depths per query;
- 495 boundary searches per query. 

C is faster than A's independently selected complete configuration:

\[
0.1688\text{ ms}
<
0.1948\text{ ms},
\]

but because A and C use different routing layouts, this alone does not prove that prefix relaxation is the reason.

That is exactly why Experiment 1 uses the same opened clusters.

Once those matched-routing results are available, this paragraph can say clearly whether C's local search itself helps.

---

# 3.8 When does the advantage disappear?

The Qwen32 top-1 result is useful, but it is not the whole story.

The result changes when we ask for many more documents.

At \(K=100\) and 99% binary recall, the selected settings are: 

| Collection / representation | A | B | C |
|---|---:|---:|---:|
| MS MARCO / Nomic 64 | 1.915 ms | 2.071 ms | **1.875 ms** |
| MS MARCO / Nomic 256 | **13.453 ms** | 13.866 ms | 14.645 ms |
| MS MARCO / Nomic 768 | **45.088 ms** | 45.217 ms | 45.196 ms |
| Quora / Qwen 32 | **0.400 ms** | 0.439 ms | 0.404 ms |
| Quora / Qwen 256 | **3.564 ms** | 3.721 ms | 4.039 ms |
| Quora / Qwen 1024 | **22.863 ms** | 23.098 ms | 22.930 ms |

In these selected \(K=100\) settings, B performs no useful splitting and C starts from depth zero.

In other words, both methods effectively fall back toward scanning.

This gives us an important limitation:

> **The Bitplane advantage is strongest when only a small number of final results are required. As \(K\) increases, many more branches or prefix ranges have to remain possible, so the methods lose their ability to prune aggressively.**

That is a much more useful conclusion than simply saying “A wins on another table.”

---

# 3.9 Code length changes the search problem

There is another important issue.

A 32-bit binary representation is much easier to search than a 1024-dimensional one, but they do not produce the same ranking.

So a 99%-recall Qwen32 result means:

> 99% recall against the **exact Qwen32 binary ranking**.

It does not mean:

> 99% recall against the original full Qwen embedding.

To measure this difference, I also compare each exact binary ranking with the best full-vector score.

The existing result is: 

| Representation | Exact binary top-1 agrees with full-float best | Full-float best appears within binary top-100 |
|---|---:|---:|
| Qwen 32 | 14.0% | 73.0% |
| Qwen 256 | 75.8% | 100.0% |
| Qwen 1024 | 88.6% | 100.0% |
| Nomic 64 | 17.6% | 65.6% |
| Nomic 256 | 48.0% | 97.2% |
| Nomic 768 | 70.2% | 99.9% |

This is a **representation-quality tradeoff**, not a property of A, B or C.

I would keep it separate from the main algorithm comparison.

The takeaway is simply:

> Short codes can make binary search much cheaper, but a faster search over a different binary ranking is not automatically a better retrieval system.

---

# 3.10 When is Backward Walk especially good?

Your existing exact-code experiment is actually useful here.

In the global Qwen32 experiment, the complete 32-bit query pattern exists in the collection for 82 of the 1,000 queries.

For those 82 queries, C takes only:

\[
4.875\ \mu s
\]

and scores about 1.68 rows on average. 

That is exactly the situation Backward Walk was designed for:

```text
query prefix
     ↓
exact matching range exists
     ↓
very few rows
     ↓
stop almost immediately
```

But for the other 918 queries, the exact pattern is absent.

C has to broaden the prefix and ends up scoring roughly 303k rows on average.

So:

> **Backward Walk can be extremely fast when the query lands near an occupied deep prefix, but this condition is not common enough in the current data for C to win overall.**

This is a much cleaner place for that experiment than mixing it into the main result.

---

# 3.11 Memory

The selected configurations all fit easily inside the stated 32 GB limit.

For the headline Qwen32 top-1 settings: 

| Method | Native index + router fields | Measured process peak |
|---|---:|---:|
| A | 8.924 MB | 77.971 MB |
| B | 10.463 MB | 72.139 MB |
| C | 10.598 MB | 77.529 MB |

The largest selected repeated-process peak across the completed study is around 505.6 MB.

So for these collection sizes:

> **memory does not determine the winner.**

The 32 GB requirement is still useful as a common feasibility constraint, but the latency differences above are not caused by one method running out of memory.

---

# 3.12 Additional checks

I would move the more technical experiments here.

### Exact-stop prediction for C

Check that the score-bound formula predicts the stopping depth and number of scored rows.

Your current saved experiment already matches all 3,000 checked query records. 

### Bitplane work counters

Check that measured split-word and leaf-word counts match the implementation formulas.

### Random branch exploration

Keep this as an ablation.

Your current result shows no improvement in the tested settings, so it is useful evidence but not central to the story.

### Storage calculations

Verify the calculated index bytes against the actual C++ structures.

These checks establish that the explanation matches the implementation; they do not need to interrupt the main result.

---

# 3.13 What the experiments show

I would end the experiment section with a short synthesis:

> The experiments show that the new methods are not uniformly better or worse than scanning.
>
> The clearest advantage appears for Qwen32 with \(K=1\). Here Bitplanes can use much coarser cluster selection, open around 295k documents, and still fully score only around 818 of them. The saved scoring work is large enough to repay the additional bitmap traversal, giving about a 2.4× improvement over the fastest tested A configuration at 99% binary recall.
>
> As \(K\) increases to 100, that pruning advantage largely disappears. The selected B and C configurations become scan-like, and A is generally as fast or faster.
>
> Backward Walk shows a narrower advantage. It is extremely cheap when a deep query prefix already exists in the collection, but many queries require substantial prefix relaxation, which reduces its overall benefit.
>
> The matched-routing experiment separately checks whether these gains come from the local search method itself or from the different routing configuration that each method prefers.

---

## Why I think this structure is much better

The reader now gets answers in the order they naturally care about:

**First:**  
> “Does the new algorithm itself work if I hold Exa's cluster selection fixed?”

**Second:**  
> “Okay, if I let it choose its own cluster setup, can the whole retrieval system actually be faster?”

**Third:**  
> “Why?”

**Fourth:**  
> “When does that advantage disappear?”

**Finally:**  
> “Does using a shorter binary representation change what I'm retrieving?”

And the terminology stays very close to the pipeline you introduced at the start:

> **select clusters → search documents inside those clusters → return top K**

instead of suddenly introducing “candidate pool,” “pre-routing,” “ANN stage,” etc.

That consistency will make the paper much easier to read.