## 2.1 Time complexity

Alright, so far we have three methods. We can first break down the time complexity of each method to clearly see how the different parameters, or hyperparameters, can influence the overall latency.

The goal is to find a mathematically optimized set of parameters for each method. In the end, we will run experiments to verify these predictions and determine whether Method B and Method C can actually outperform Method A in some, or possibly all, cases.

To make the comparison easier to follow, let’s use the same small example for all three methods.

Suppose we open one cluster containing three documents and want to return the best two results:

$$
N=3,\qquad C=P=1,\qquad d=5,\qquad K=2.
$$

The query is

$$
q=[0.7,\ 0.5,\ -0.4,\ -0.6,\ 0.3],
$$

so its ideal binary signs are

$$
11001.
$$

The three documents are:

| ID | Document bits | Full score |
| -: | :-----------: | ---------: |
|  0 |     10100     |        0.1 |
|  1 |     11001     |        2.5 |
|  2 |     11000     |        1.9 |

These are only small teaching examples. The actual experiments later use the full benchmark dataset.

For the larger formulas, when the clusters are approximately balanced, let

$$
n=\frac{N}{C}
$$

be the average number of documents in one cluster.

---

## 2.1.1 Original method

The original method is relatively straightforward. After selecting the clusters, every document inside those clusters is fully scored.

### Step 1: open the clusters and prepare the query

For the example above, we open cluster 0:

$$
[\text{ID 0},\text{ ID 1},\text{ ID 2}].
$$

Following the four-sign lookup-table scoring described in [1], the query is divided into groups of four values.

For our five-dimensional example:

$$
[0.7,\ 0.5,\ -0.4,\ -0.6]
$$

and

$$
[0.3,\ 0,\ 0,\ 0].
$$

Each group has \(2^4=16\) possible binary sign patterns, so two groups contain

$$
2\times16=32
$$

table entries.

In general, with dimension \(d\) and group size \(g=4\), one full document score requires

$$
G=\left\lceil\frac{d}{4}\right\rceil
$$

table lookups.

**Diagram.** I would keep this diagram very simple and use **blue**, since this is the baseline method. Show:

```text
Query
  ↓
Select P of C clusters
  ↓
[ selected documents ]
  ↓
Score every document
  ↓
Top K
```

Above the selected documents, add:

$$
F_A \approx \frac{PN}{C}
$$

documents.

The purpose of the diagram is just to make one thing very clear: **after routing, Method A performs no additional filtering. Every selected document reaches the full scorer.**

---

### Step 2: score each document

For the small example, each document reads two score-table entries:

$$
\text{ID 0}\rightarrow0.1
$$

$$
\text{ID 1}\rightarrow2.5
$$

$$
\text{ID 2}\rightarrow1.9.
$$

Three documents therefore require

$$
3\times2=6
$$

score-table terms.

If \(F_A\) documents reach the scorer, the scoring work is approximately

$$
F_A\left\lceil\frac{d}{4}\right\rceil.
$$

With balanced clusters,

$$
F_A\approx\frac{PN}{C}.
$$

Therefore,

$$
T_{\text{score},A}
=
O\left(
\frac{PN}{C}d
\right)
$$

when the group width is treated as constant.

For fixed \(d\), this is effectively linear in the number of selected documents.

---

### Step 3: keep the best \(K\)

While the documents are being scored, we keep only the best \(K\).

In our example:

```text
[0: 0.1]
      ↓
[1: 2.5, 0: 0.1]
      ↓
[1: 2.5, 2: 1.9]
```

Using a bounded heap, result maintenance is bounded by

$$
O(F_A\log K),
$$

and final sorting of the retained results costs

$$
O(K\log K).
$$

---

### Total

Let \(T_0\) represent the shared work such as routing and query-table preparation, and let \(a_A\) represent the average time required to fully process one selected document, including scoring and updating the result set.

Then, for approximately balanced clusters,

$$
\boxed{
T_A
\approx
T_0+
\frac{PN}{C}a_A.
}
$$

So the main parameters for Method A are \(C\) and \(P\).

Increasing \(C\) reduces the average number of documents in each cluster, but may require opening more clusters to maintain the same recall. Increasing \(P\) directly increases the number of documents that need to be scored. The actual experiments therefore need to find the \(C,P\) combination that gives the lowest latency while still meeting the target recall. 

---

# 2.1.2 Bitplanes

Bitplanes try to reduce the number of documents that reach the full scorer.

The tradeoff is that we now need to spend additional work traversing the bitplanes and maintaining alternative branches.

### Step 1: open, prepare, and order the bits

We begin with exactly the same routing and score-table setup as Method A.

The additional step is to sort the query coordinates by decreasing absolute value.

For

$$
q=[0.7,\ 0.5,\ -0.4,\ -0.6,\ 0.3],
$$

the order is

$$
1\;(0.7)
\rightarrow
4\;(0.6)
\rightarrow
2\;(0.5)
\rightarrow
3\;(0.4)
\rightarrow
5\;(0.3).
$$

Sorting the coordinates costs

$$
O(d\log d).
$$

We also calculate

$$
Q=\sum_i |q_i|,
$$

which costs \(O(d)\). In this example,

$$
Q=2.5.
$$

**Diagram.** Start with a **blue** box for the shared stage and then switch to **orange** for Bitplanes:

```text
[ Select P clusters ] → [ Sort bits by |qᵢ| ]
       blue                     orange
```

Below the orange box, show the actual order:

```text
Bit 1 → Bit 4 → Bit 2 → Bit 3 → Bit 5
```

The purpose is to show that Bitplanes do not replace the coarse routing step. They add another search stage **inside the already-selected clusters**.

---

### Step 2: split the candidate masks

Suppose one cluster contains \(n\) documents.

With 64-bit machine words, one candidate mask has

$$
W=
\left\lceil\frac{n}{64}\right\rceil
$$

words.

For the small example, all three documents fit inside one machine word, so

$$
W=1.
$$

We initially have

```text
111
```

meaning all three documents are active.

The first preferred bit is bit 1. All documents have bit 1 equal to `1`, so:

```text
111 AND 111 = 111
```

Nothing is removed.

The next preferred bit is bit 4. All documents also match the preferred value there, so again nothing is removed.

At bit 2:

```text
bitplane = 011
```

and therefore:

```text
111 AND 011 = 011
```

The preferred branch contains IDs 1 and 2.

The alternative branch contains:

```text
100
```

or ID 0, with mismatch penalty

$$
p=|q_2|=0.5.
$$

Three splits have therefore been performed.

For the example:

$$
3\times1=3
$$

word positions are visited.

In general, if an average of \(S\) splits are performed for each of \(P\) opened clusters,

$$
T_{\text{split}}
\propto
PS
\left\lceil
\frac{N}{64C}
\right\rceil.
$$

**Diagram.** Use an **orange branching tree**:

```text
                  111 : IDs 0,1,2
                         |
                 Bit 1 prefers 1
                         |
                        111
                         |
                 Bit 4 prefers 0
                         |
                        111
                       /   \
              Bit 2=1       Bit 2=0
                011            100
              IDs 1,2         ID 0
              p = 0           p = 0.5
```

The currently explored path should have an **orange border**.

The alternative branch should have a **light gray border** while it is waiting in the priority queue.

Beside every node, also write its physical mask width:

```text
mask width = 1 word
```

For a larger illustrative cluster, add a small callout:

```text
6,400 documents → 100 words
80 surviving documents → still 100 words
```

This is important because otherwise it is easy to assume that a smaller logical candidate set automatically means a smaller bitmap operation. It does not.

---

### Step 3: score the surviving rows

Suppose we choose a leaf size of two.

The preferred mask

```text
011
```

already contains only two documents, so we stop splitting it and fully score IDs 1 and 2.

Their scores are

$$
2.5,\qquad1.9.
$$

Let

$$
F_B
$$

be the number of rows that Bitplanes eventually fully score.

Alternatively, define

$$
f_B=
\frac{F_B}{F_A}
$$

as the fraction of Method A's selected documents that still reach the scorer.

Then approximately

$$
F_B
=
f_B\frac{PN}{C}.
$$

The scoring term is therefore

$$
f_B\frac{PN}{C}a_B.
$$

The goal is for

$$
f_B<1,
$$

meaning Bitplanes avoid some of the full scoring work performed by Method A.

---

### Step 4: check the waiting branches

We still have the alternative branch containing ID 0.

Its accumulated penalty is

$$
p=0.5.
$$

The best possible score of anything inside that branch is therefore

$$
Q-2p
=
2.5-2(0.5)
=
1.5.
$$

Our current top-two results are:

$$
2.5,\qquad1.9.
$$

Since

$$
1.5<1.9,
$$

the waiting branch cannot improve the current top two.

We can discard it without scoring ID 0.

**Diagram.** Keep the same tree, but now turn the alternative branch **gray with a dashed border**:

```text
                    alternative
                      p = 0.5
                         |
                best possible = 1.5
                         |
                current Kth = 1.9
                         |
                       PRUNE
```

The purpose is to show that branching does **not** mean that every branch eventually gets fully explored. The score bound is what allows us to stop.

---

### Total

Let:

* \(S\) = average number of split nodes per opened cluster,
* \(b\) = average time per split-word visit,
* \(f_B\) = fraction of selected documents that are fully scored,
* \(T_{\text{other},B}\) = sorting, queue, mask, and leaf-processing overhead.

Then a simplified model is

$$
\boxed{
T_B
\approx
T_0
+
PS
\left\lceil
\frac{N}{64C}
\right\rceil b
+
f_B\frac{PN}{C}a_B
+
T_{\text{other},B}.
}
$$

The important comparison is therefore:

$$
\boxed{
\text{cost of scores avoided}
>
\text{additional bitmap and branching cost}.
}
$$

A smaller leaf size can reduce \(f_B\), but usually increases \(S\). Larger clusters reduce the amount of coarse routing, but make every bitmap wider.

This is why parameters such as cluster count, probe count, leaf size, and node budget need to be tuned together. 

---

# 2.1.3 Backward Walk: prefix relaxation

Backward Walk changes the direction of the search.

Instead of starting from many documents and progressively removing them, we start from a very specific binary prefix and progressively broaden the candidate set.

For this section, we use the prefix-relaxation version of Backward Walk.

### Step 1: look up the full prefix

The query signs are

```text
11001
```

Suppose the documents are stored in sorted binary-key order:

```text
10100
11000
11001
```

We first look for the complete five-bit query prefix:

```text
11001
```

Using two binary searches, one for the start and one for the end of the prefix range, we find:

```text
[2, 3) → ID 1
```

We fully score ID 1:

$$
S_1=2.5.
$$

Only one candidate has been found, while

$$
K=2,
$$

so we continue.

**Diagram.** Use **purple** throughout Backward Walk.

Start with a narrow purple box:

```text
11001
[ ID 1 ]
```

Add:

```text
depth = 5
candidates = 1
```

The purpose is to visually contrast this with Bitplanes. Bitplanes begin with all three documents; Backward Walk begins with only the most specific matching range.

---

### Step 2: remove one prefix constraint

We now remove the last constraint:

```text
11001
  ↓
1100*
```

The new prefix contains:

```text
11000
11001
```

or IDs 2 and 1.

Importantly, ID 1 has already been scored.

So we do **not** score both rows again.

We only score the newly exposed slice:

```text
ID 2
```

which has score

$$
1.9.
$$

We have now scored two distinct candidates:

$$
2.5,\qquad1.9.
$$

For this teaching example, the candidate target has been reached, so the search stops.

This happens to recover the true top two in this particular example. Simply collecting \(K\) candidates does **not** guarantee exact top-\(K\) recall on arbitrary data; this stopping rule is approximate and will later be evaluated experimentally.

**Diagram.** I would show the sorted document array horizontally:

```text
ID 0        ID 2        ID 1
10100       11000       11001
 gray      light purple  dark purple
```

Then draw two brackets:

```text
                  └──── 11001 ────┘

           └────────── 1100* ──────────┘
```

Use:

* **dark purple** for the range already visited,
* **light purple** for the newly exposed rows,
* **gray** for rows still outside the prefix.

Point from the light-purple section to:

```text
Score only the new slice
```

The purpose of this diagram is important: when the prefix becomes broader, the old range is contained inside the new range. We do not need to rescore documents we already processed. The long study describes this same nested-range property for prefix relaxation. 

---

### Step 3: count the prefix lookup work

Suppose we start at prefix depth \(x\) and eventually stop at depth \(\ell\).

The number of visited depths is

$$
L=x-\ell+1.
$$

For a sorted-key implementation, each prefix range can be found using two binary searches.

If one opened cluster contains approximately

$$
n=\frac NC
$$

rows, one range lookup costs approximately

$$
2\log_2(n+1)
$$

key comparisons.

Across \(P\) clusters and \(L\) visited depths:

$$
T_{\text{lookup}}
\propto
2PL\log_2(n+1).
$$

For the teaching example:

* \(P=1\),
* \(L=2\),
* two binary searches per prefix,

so four binary-search calls are performed.

---

### Step 4: count the documents scored

Let

$$
F_C
$$

be the total number of **distinct** documents exposed by the time the walk stops.

Because each wider range contains the previous range, documents already scored at a deeper level are not counted again.

Therefore, the scoring work is proportional to

$$
F_Ca_C.
$$

The main unknown is therefore how quickly \(F_C\) grows as the prefix becomes shorter.

Under a simple model where binary bits are independent and equally likely to be `0` or `1`, an \(\ell\)-bit prefix has probability

$$
2^{-\ell}.
$$

For one cluster of size \(n\),

$$
E[F_\ell]
=
\frac{n}{2^\ell}.
$$

Across \(P\) equally sized opened clusters,

$$
\boxed{
E[F_\ell]
=
\frac{PN}{C2^\ell}.
}
$$

For example, in a flat collection containing one million documents:

| Prefix depth | Expected documents |
| -----------: | -----------------: |
|           16 |      \(\approx15\) |
|           15 |      \(\approx30\) |
|           14 |      \(\approx61\) |
|           13 |     \(\approx122\) |

So although we only remove one prefix bit at a time, the expected number of candidates approximately doubles at each step.

**Diagram.** Show a vertical series of increasingly wide **purple boxes**:

```text
16 bits       ~15 docs
   ↓
15 bits       ~30 docs
   ↓
14 bits       ~61 docs
   ↓
13 bits      ~122 docs
```

Make the physical width of the boxes increase as the prefix gets shorter.

The purpose is to show that the walk looks gradual in terms of prefix length, but the candidate set can grow very quickly.

Add a small note underneath:

> Expected counts assume independent, balanced bits. Real embedding bits may be correlated or unevenly distributed.

That qualification matters because the detailed analysis already shows that the independent-bit assumption can give inaccurate quality predictions on real embeddings. 

---

### Total

Let:

* \(x\) = starting prefix depth,
* \(\ell\) = final prefix depth,
* \(L=x-\ell+1\) = number of visited depths,
* \(n=N/C\) = average documents per cluster,
* \(t_b\) = average time per binary-search key comparison,
* \(F_C\) = number of distinct documents fully scored,
* \(a_C\) = average cost per scored document,
* \(T_{\text{ranges}}\) = range-update and new-slice bookkeeping.

Then for the sorted-key implementation,

$$
\boxed{
T_C
\approx
T_0
+
2PL\log_2(n+1)t_b
+
F_Ca_C
+
T_{\text{ranges}}.
}
$$

Substituting \(n=N/C\),

$$
\boxed{
T_C
\approx
T_0
+
2PL
\log_2\left(\frac NC+1\right)t_b
+
F_Ca_C
+
T_{\text{ranges}}.
}
$$

The prefix lookup itself grows only logarithmically with the number of rows in the cluster.

The more important quantity is \(F_C\).

If a relatively deep prefix already contains enough useful candidates,

$$
F_C\ll\frac{PN}{C},
$$

Backward Walk may avoid most of the scoring performed by Method A.

In the worst case, however, the prefix may need to be relaxed until almost every selected document is exposed:

$$
F_C\rightarrow\frac{PN}{C}.
$$

At that point, Backward Walk approaches the scoring work of the original method.

A trie implementation could reduce or change the prefix-lookup term, but the main tradeoff remains the same: **how far do we need to walk backward before enough useful documents are found?** 

---

## 2.1.4 Summary

At this point, the main parameters we want to optimize are:

| Method            | Main parameters                  | Main latency tradeoff                   |
| ----------------- | -------------------------------- | --------------------------------------- |
| A — Original      | \(C,P\)                          | number of documents scored vs recall    |
| B — Bitplanes     | \(C,P,\) leaf size, node budget  | bitmap traversal vs scores avoided      |
| C — Backward Walk | \(C,P,x,\) stopping depth / rule | prefix selectivity vs documents exposed |

The three methods are therefore paying for different kinds of work.

Method A mainly pays for full document scores.

Bitplanes add bitmap traversal in an attempt to reduce the number of those scores.

Backward Walk adds prefix-range lookups in an attempt to start from a much smaller group of documents.

The next step is to use these expressions to choose sensible parameter ranges for each method, and then check experimentally whether those choices actually reduce latency at the same recall and memory requirement.
