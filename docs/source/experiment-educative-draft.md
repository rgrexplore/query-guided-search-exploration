Yeah, I get what you mean. The issue is not mainly the experiment design; it’s that the writing is **describing the experiment from the author’s perspective instead of walking the reader through why each thing is being done**.

And yes: at the start of the experiment section, I would write it in **future tense** — “I will first…”, “then I will…” — because we are telling the reader what is about to happen. Once we get to the results, switch to past tense: “B was faster…”.

I’d write it much more like this:

---

# 3 Experiments

So far, we have only looked at the methods mathematically.

The calculations tell us what *could* make Bitplanes or Backward Walk faster, but they cannot tell us whether that actually happens on real embeddings.

In this section, I will test the three methods directly.

There are two things I want to understand.

First, I want to compare the search methods under the **same cluster selection**. In other words, if A, B and C are all given exactly the same documents to search, does Bitplanes or Backward Walk actually search those documents faster than simply scanning them?

After that, I will allow each method to choose its own cluster settings and search parameters. This answers the more practical question: if I were actually building the fastest version of each method, which one would be faster at the same recall?

Finally, I will look at cases where the result changes, such as returning one document versus returning 100 documents. This helps us understand not only **which method wins**, but also **when and why**.

---

## 3.1 Experiment setup

I use two document collections:

| Collection | Embedding model | Documents | Queries | Binary dimensions tested |
|---|---|---:|---:|---|
| MS MARCO | Nomic v1.5 | 1,000,000 | 1,000 | 64, 256, 768 |
| Quora | Qwen3 0.6B | 522,931 | 1,000 | 32, 256, 1024 |

For each collection, I first generate the document and query embeddings once. The experiments then start from those already-generated vectors.

This is important because the thing I am trying to compare is the **search stage**, not how fast Nomic or Qwen can generate an embedding.

For one comparison, I keep the following fixed:

- the same documents;
- the same 1,000 queries;
- the same binary dimension;
- the same scoring function;
- the same number of requested results \(K\);
- the same exact reference answers.

The idea is simple:

> **A, B and C should solve exactly the same search problem.**

If B is faster than A, I want that difference to come from the search method itself, not because B was given easier queries or a different definition of the correct result.

Changing the binary dimension, for example from 32 bits to 256 bits, is treated as a separate experiment because it also changes the ranking that we are searching for.

### Figure: what stays fixed

I would show:

```text
             Same documents
             Same 1,000 queries
             Same binary score
                   |
            ----------------
            |      |       |
            A      B       C
          Scan  Bitplanes  Backward Walk
            |      |       |
            ----------------
                   |
           Same exact top-K
              reference
```

A in blue, B in orange, C in purple.

The point of this figure is simply:

> **same problem, different way of searching it.**

---

## 3.2 What do I measure?

For every query, the timer starts after the query embedding has already been produced.

I measure:

```text
select the clusters
        ↓
search the documents inside them
        ↓
return the top-K IDs
```

So the reported latency includes both:

1. selecting which clusters to open; and
2. searching the documents inside those clusters.

It does not include embedding generation or the later reranking stage.

I also record:

- recall;
- number of documents inside the opened clusters;
- number of documents that receive the full score;
- Bitplane word visits for B;
- prefix depths and prefix lookups for C;
- peak process memory.

This lets us answer not only:

> **Which method is faster?**

but also:

> **Why is it faster?**

---

## 3.3 How do I know whether a result is correct?

Before testing A, B or C, I first calculate the exact binary result for every query.

For one query, I score **every document in the collection** using the same binary score described earlier.

Suppose the exact top five are:

```text
[12, 7, 91, 4, 30]
```

and one method returns:

```text
[12, 7, 91, 4, 85]
```

It recovered four of the five correct IDs, so:

\[
Recall@5 = 4/5 = 80\%.
\]

I repeat this over all 1,000 queries and report the average recall.

So when I say:

> **99% recall**

I specifically mean:

> the method recovers 99% of the exact top-\(K\) document IDs under the binary score being tested.

It does **not** mean 99% human relevance, and it does not necessarily mean 99% agreement with the original full embedding. I will look at that distinction separately later.

---

# 3.4 Experiment 1: give every method the same clusters

The first experiment asks the simplest possible question:

> **If A, B and C are given the exact same documents, which search method is faster?**

Normally, the pipeline first selects some clusters:

```text
Query
  ↓
Select nearby clusters
  ↓
Search documents inside them
```

For this experiment, I freeze the first part.

A, B and C will receive exactly the same opened clusters for every query.

So if those clusters contain 50,000 documents:

- A may score all 50,000;
- B may use Bitplanes and score only some of them;
- C may start from a prefix and expose only some of them.

But all three started from the same 50,000 documents.

### Figure: fixed cluster experiment

```text
                Same opened clusters
               same documents inside
                        |
          -----------------------------
          |             |             |
       A: Scan      B: Bitplanes   C: Backward Walk
          |             |             |
      score all      prune first      prefix first
```

The reason for doing this experiment is straightforward.

Suppose B is faster than A here.

Then we know the improvement is actually coming from the Bitplane search.

But suppose B is only faster when it also uses a completely different cluster layout.

Then the story is different: the advantage may come from a combination of **cluster selection + Bitplanes**, rather than Bitplanes alone.

Both results are useful, but I want to separate them.

---

## 3.4.1 Local recall

In this experiment, I also calculate a second type of recall.

Imagine the selected clusters contain 50,000 documents.

I can fully scan those 50,000 documents and find their exact top-\(K\). That becomes the **local reference**.

Then:

- A always has 100% local recall because it scans all 50,000;
- B may lose some local results if it stops branching early;
- C may lose some if it stops at a prefix before reaching them.

This lets us distinguish two different ways of missing a result:

```text
correct document was never inside the opened clusters
```

versus

```text
correct document was inside the opened clusters,
but the local search failed to reach it
```

The normal global recall still matters for the final system, but local recall makes this particular experiment much easier to understand.

---

# 3.5 Experiment 2: let each method choose its own settings

The first experiment intentionally fixes the clusters.

But this may not be the best way to actually use B or C.

For example, Bitplanes may work better if I use **larger clusters**.

A larger cluster means more documents arrive at the local search stage, which would normally make A slower.

But B may be able to cheaply remove most of those documents before fully scoring them.

So for the second experiment, I let each method choose the configuration that works best for it.

For A, I vary:

- number of clusters \(C\);
- number of opened clusters \(P\).

For B, I vary:

- \(C\);
- \(P\);
- leaf size;
- node budget.

For C, I vary:

- \(C\);
- \(P\);
- starting prefix depth;
- stopping rule.

For each target recall, such as

```text
80%
90%
95%
99%
```

I choose the fastest tested configuration that reaches at least that recall.

This gives us the practical comparison:

> **If I were actually deploying the fastest tested version of A, B and C, which one would be fastest at the same recall?**

---

# 3.6 Main result: Qwen 32-bit, return one document

I will first look at Qwen with 32-bit binary documents and \(K=1\), because this is the clearest case where the three methods behave differently.

The fastest qualifying configurations are:

| Required recall | A: Scan | B: Bitplanes | C: Backward Walk |
|---:|---:|---:|---:|
| 80% | 0.0308 ms | **0.0295 ms** | 0.0318 ms |
| 90% | 0.0498 ms | **0.0432 ms** | 0.0497 ms |
| 95% | 0.0805 ms | **0.0520 ms** | 0.0799 ms |
| 99% | 0.1948 ms | **0.0809 ms** | 0.1688 ms |

At the 99% recall target, B takes:

\[
0.0809\text{ ms}
\]

compared with

\[
0.1948\text{ ms}
\]

for A.

So B is about:

\[
2.4\times
\]

faster in this setting.

### Figure: latency versus recall

Plot:

- x-axis: required recall;
- y-axis: median query latency;
- A: blue;
- B: orange;
- C: purple.

The main thing the reader should see is that the gap becomes larger as we move toward higher recall.

---

# 3.7 Why is Bitplanes faster here?

The selected 99% configurations are:

| Method | Clusters | Opened clusters | Local search |
|---|---:|---:|---|
| A | 4096 | 180 | scan |
| B | 16 | 9 | leaf 128, budget 8192 |
| C | 1024 | 78 | start depth 4, exact stop |

Now look at how many documents each method actually touches:

| Method | Documents inside opened clusters | Documents fully scored |
|---|---:|---:|
| A | 24,181 | 24,181 |
| B | 295,031 | 818 |
| C | 41,147 | 26,153 |

This is the most important part of the result.

A tries to make the cluster selection very selective:

```text
A
select small clusters
      ↓
24,181 documents
      ↓
score all 24,181
```

B does almost the opposite:

```text
B
select much larger clusters
      ↓
295,031 documents
      ↓
Bitplanes
      ↓
score only 818
```

So B is not faster because fewer documents reach the opened clusters.

It actually opens **more than ten times as many documents as A**.

The difference is that Bitplanes remove almost all of them before the expensive full-score calculation.

B scores only around:

\[
818/295031 \approx 0.28\%
\]

of its opened documents.

### Figure: documents opened versus documents scored

Show A:

```text
A

Opened
████████████████████████    24,181

Scored
████████████████████████    24,181
```

Show B:

```text
B

Opened
██████████████████████████████████████████████████  295,031

                    Bitplanes
                        ↓

Scored
█                                                     818
```

Beside B:

```text
53,693 split-mask word visits
5,838 leaf-mask word visits
```

This connects directly back to the calculation in Section 2.

Bitplanes only make sense if:

> **the full scores we avoid are more expensive than the bitmap work we add.**

In this setting, B avoids roughly 294,000 full scores per query while performing around 59,000 bitmap-word visits.

The measured end-to-end query time shows that this tradeoff is favorable.

---

# 3.8 What happens to Backward Walk?

C also avoids some full scores.

It opens around:

\[
41,147
\]

documents but only fully scores around:

\[
26,153.
\]

It therefore removes roughly 15,000 full scores.

However, it also performs prefix searches and range expansion.

At the selected setting, it visits around 3.5 prefix depths and performs around 495 boundary searches per query.

Its final time is:

\[
0.1688\text{ ms},
\]

which is lower than A's independently selected configuration, but still much slower than B.

Because A and C use different cluster layouts here, this result alone cannot tell us whether the gain comes from Backward Walk itself or from C's preferred routing configuration.

That is what the fixed-cluster experiment in Section 3.4 will clarify.

---

# 3.9 When does Bitplanes stop helping?

The previous result uses:

\[
K=1.
\]

Now consider the same type of comparison when we ask for:

\[
K=100.
\]

At 99% recall:

| Representation | A | B | C |
|---|---:|---:|---:|
| Nomic 64 | 1.915 ms | 2.071 ms | 1.875 ms |
| Nomic 256 | **13.453 ms** | 13.866 ms | 14.645 ms |
| Nomic 768 | **45.088 ms** | 45.217 ms | 45.196 ms |
| Qwen 32 | **0.400 ms** | 0.439 ms | 0.404 ms |
| Qwen 256 | **3.564 ms** | 3.721 ms | 4.039 ms |
| Qwen 1024 | **22.863 ms** | 23.098 ms | 22.930 ms |

The behavior is now very different.

At these selected settings:

- B performs almost no useful splitting;
- C starts at depth zero;
- all three methods end up scoring essentially the same documents.

So the advantage disappears.

This makes intuitive sense.

When we only need one result, B can aggressively search for one very promising region and prune many alternatives.

When we need 100 results, many more branches remain relevant. B must keep exploring until enough good documents have been found, which removes much of its pruning advantage.

So one important conclusion is:

> **Bitplanes are most useful when the number of requested results is small enough that many branches can be safely ignored.**

---

# 3.10 Binary length is a separate tradeoff

The 32-bit result above is fast, but a 32-bit binary vector does not produce the same ranking as the full embedding.

To measure this difference, I compare the exact binary results with the original full-vector score.

| Representation | Binary top-1 matches full-vector best | Full-vector best appears in binary top-100 |
|---|---:|---:|
| Qwen 32 | 14.0% | 73.0% |
| Qwen 256 | 75.8% | 100.0% |
| Qwen 1024 | 88.6% | 100.0% |
| Nomic 64 | 17.6% | 65.6% |
| Nomic 256 | 48.0% | 97.2% |
| Nomic 768 | 70.2% | 99.9% |

This result is not comparing A, B and C.

It answers a different question:

> **How much does the binary representation itself change the ranking?**

For example, Qwen32 may be extremely cheap to search, but its exact binary top-1 agrees with the full-vector best on only 14% of the queries.

So search speed and representation quality should be treated separately.

---

# 3.11 When does Backward Walk work especially well?

Backward Walk is designed to be very cheap when the query's specific binary prefix already corresponds to stored documents.

I test this directly using Qwen32 with no cluster routing.

Among the 1,000 queries, 82 have their complete 32-bit query pattern present in the collection.

Using the same C configuration for all queries:

| Query group | Queries | A | B | C |
|---|---:|---:|---:|---:|
| All queries | 1000 | 2481.7 µs | 124.1 µs | 1163.5 µs |
| Exact 32-bit code exists | 82 | 2481.3 µs | 128.5 µs | **4.9 µs** |
| Exact code absent | 918 | 2481.7 µs | 123.8 µs | 1170.7 µs |

When the exact code exists, C finds a tiny range immediately and scores only around 1.7 documents on average.

Its search therefore looks approximately like:

```text
query binary code
       ↓
exact prefix range exists
       ↓
1–2 documents
       ↓
done
```

In this case, Backward Walk is extremely fast.

However, this only happens for 82 of the 1,000 queries.

For the other 918 queries, C must progressively relax the prefix and eventually scores hundreds of thousands of documents.

So the current data suggests:

> **Backward Walk works very well when a deep query prefix is occupied, but often loses its advantage when it needs to broaden the prefix substantially.**

---

# 3.12 Memory

All selected configurations remain far below the 32 GB process limit.

For the Qwen32 \(K=1\) 99% settings:

| Method | Index + router fields | Measured process peak |
|---|---:|---:|
| A | 8.924 MB | 77.971 MB |
| B | 10.463 MB | 72.139 MB |
| C | 10.598 MB | 77.529 MB |

The largest selected process peak across the completed experiments is around 505.6 MB.

So at the collection sizes tested here, memory does not determine the winner.

The main difference between the methods is query-time work.

---

# 3.13 Additional checks

I also verify that the work counters match the calculations from Section 2.

For Bitplanes, I record:

- split-mask words;
- leaf-mask words;
- number of visited branches;
- rows fully scored.

For Backward Walk, I record:

- starting depth;
- final depth;
- number of prefix depths visited;
- number of boundary searches;
- rows fully scored.

The stored index-size formulas are also checked against the actual index fields.

For Backward Walk's exact stopping rule, the predicted stopping depth and number of scored rows match the saved query records in the checked runs.

These checks do not determine which method is fastest; the measured whole-query time does that. Their purpose is to verify that the explanation of the work matches what the implementation actually performs.

---

# 3.14 Experiment summary

The experiments show that there is no single method that is best in every setting.

For Qwen32 with \(K=1\), Bitplanes are substantially faster. At 99% binary recall, B takes 0.0809 ms compared with 0.1948 ms for A.

The reason is not that B opens fewer documents. B actually opens a much larger set of documents, but its query-dependent Bitplane traversal reduces roughly 295,000 opened documents to only around 818 full scores.

When \(K\) increases to 100, this behavior disappears. More branches need to remain active, B can no longer prune aggressively, and its selected configuration falls back toward ordinary scanning.

Backward Walk has a more specific advantage. When the query lands directly on an occupied deep prefix, it can return results extremely cheaply. When that prefix is absent, however, it often needs to broaden the search substantially.

The remaining fixed-cluster comparison will determine how much of each speedup comes from the local search method itself and how much comes from the different cluster configuration that each method prefers.