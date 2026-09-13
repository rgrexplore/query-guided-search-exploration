Yes. I think the current section is **mostly mathematically sound**, but it is difficult to read because it mixes four things too early:

- the small example,
- asymptotic complexity,
- implementation-specific byte counts,
- and the final cost model.

The reader has to understand all four at once.

I would rewrite the whole section around one simple question:

> **For each method, what extra work do we do for one query, and what are we hoping to save by doing it?**

I would also make two technical changes to the current version:

1. Don’t say the goal is to “find the mathematically optimized parameters.” The math cannot determine \(F_B\), \(F_C\), branch counts, prefix occupancy, cache effects, etc. by itself. It gives us **cost relationships and sensible parameter ranges**, which the experiment then checks.
2. For Bitplanes, keep **split-mask reads and leaf-mask reads separate**. Your experiments measure them separately, and both can be significant. Hiding leaf reads inside `T_other,B` makes the model harder to connect to the results. 

Below is how I would write the complete section.

---

# 2 Preliminary calculations

Before running the experiments, let’s first understand what each method actually spends time and memory on.

The goal here is not to predict the exact latency from equations alone. A full document score, a bitmap operation and a binary search do not cost the same amount of CPU time, and their actual cost also depends on memory access and caching.

Instead, we want to answer three simpler questions:

1. **What work grows when the collection becomes larger?**
2. **Which parameters control that work?**
3. **What does Method B or C need to save in order to make its additional index work worthwhile?**

We will later measure the actual latency, but these calculations give us a clearer idea of what to look for in the experiments.

Throughout this section:

- \(N\): total number of documents.
- \(C\): number of clusters.
- \(P\): number of clusters opened for one query.
- \(d\): number of binary dimensions.
- \(K\): number of requested results.
- \(n_c\): number of rows in cluster \(c\).
- \(F_A,F_B,F_C\): number of documents fully scored by A, B and C respectively.

When clusters are approximately balanced,

\[
n \approx N/C
\]

and opening \(P\) clusters exposes roughly

\[
Pn = PN/C
\]

documents.

This is only an approximation. In the experiments we use the actual cluster sizes.

---

# 2.1 Query-time work

To make the three methods easier to compare, let’s use the same small example throughout.

Suppose we have one cluster containing three documents:

| ID | Binary document | Full score |
|---|---|---:|
| 0 | `10100` | 0.1 |
| 1 | `11001` | 2.5 |
| 2 | `11000` | 1.9 |

and the query is

\[
q=[0.7,\ 0.5,\ -0.4,\ -0.6,\ 0.3].
\]

The ideal binary signs are

`11001`

and we want the best

\[
K=2
\]

documents.

For this example:

\[
N=3,\qquad C=P=1,\qquad d=5.
\]

These three rows are only for explaining the algorithms. The experiments later use the full collections.

---

## 2.1.1 Method A: scan the selected clusters

Method A is the simplest.

Once routing has decided which clusters to open, every document inside those clusters receives the complete score.

### Step 1: select the documents that must be searched

For our example, we open the only cluster:

`[ID 0, ID 1, ID 2]`

so

\[
F_A=3.
\]

In general,

\[
\boxed{
F_A=\sum_{c\in O}n_c
}
\]

where \(O\) is the set of opened clusters.

If the clusters are roughly balanced,

\[
F_A\approx\frac{PN}{C}.
\]

This number is the main source of work for A.

### Diagram 1 — Method A

Use a simple **blue horizontal pipeline**:

`Query → Select P clusters → FA documents → Full score every document → Top K`

Inside the `FA documents` box, show three tiny rows:

`10100`
`11001`
`11000`

Caption:

> **Figure X:** Method A performs no additional filtering after routing. Every row in the opened clusters reaches the full scorer.

**Purpose of the diagram:** make it immediately clear that routing is the only candidate reduction step for A.

---

### Step 2: prepare the score table

The scorer groups four document signs together.

For \(d=5\), the query becomes two groups:

`[0.7, 0.5, -0.4, -0.6]`

and

`[0.3, 0, 0, 0]`.

Each four-bit group has

\[
2^4=16
\]

possible sign patterns.

So our example builds

\[
2\times16=32
\]

table entries.

In general, let

\[
G=\left\lceil\frac d4\right\rceil.
\]

Building the table costs \(O(d)\) for a fixed group width.

One document score then reads \(G\) entries.

---

### Step 3: score every selected row

Our three documents require:

\[
3\times2=6
\]

score-table lookups.

More generally, the scoring work is proportional to

\[
F_AG.
\]

Since \(G\) is proportional to \(d\),

\[
T_{\text{score},A}=O(F_Ad).
\]

For fixed \(d\), this is simply linear in \(F_A\).

---

### Step 4: retain the best K results

As each score is produced, we maintain a bounded top-\(K\) heap.

For our example:

`[ID 0: 0.1]`

→

`[ID 1: 2.5, ID 0: 0.1]`

→

`[ID 1: 2.5, ID 2: 1.9]`

Heap maintenance is bounded by

\[
O(F_A\log(K+1)).
\]

---

### Total cost for A

Let:

- \(T_{\text{common}}\) contain routing, score-table preparation and other work shared by a particular configuration;
- \(a_A\) be the average measured time to fully process one document, including its score and top-\(K\) update.

Then the useful cost model is

\[
\boxed{
T_A\approx T_{\text{common}}+F_Aa_A.
}
\]

For balanced clusters,

\[
\boxed{
T_A\approx
T_{\text{common}}
+
\frac{PN}{C}a_A.
}
\]

The main question for A is therefore simple:

> **How few documents can routing send to the scorer while still meeting the required recall?**

Increasing \(C\) usually makes each cluster smaller, but it may require a larger \(P\) to recover the same neighbors.

---

# 2.1.2 Method B: Bitplanes

Bitplanes try to reduce \(F_A\).

Instead of immediately scoring every selected document, B first spends some time filtering documents using the query bits.

So B has two main costs:

\[
\boxed{
\text{bitmap traversal}
+
\text{full scoring of the survivors}.
}
\]

The method is useful only if the scoring it avoids is worth more than the bitmap work it adds.

---

## Step 1: order the query bits

B uses the same routing and score table as A.

It additionally sorts the query coordinates by decreasing magnitude.

For

\[
q=[0.7,\ 0.5,\ -0.4,\ -0.6,\ 0.3],
\]

the order is

`Bit 1 (0.7) → Bit 4 (0.6) → Bit 2 (0.5) → Bit 3 (0.4) → Bit 5 (0.3)`.

Sorting costs

\[
O(d\log d).
\]

We also calculate

\[
Q=\sum_i|q_i|=2.5.
\]

### Diagram 2 — where B starts

Show:

**blue box:** `Select P clusters`

→

**orange box:** `Order bits by |qi|`

Under the orange box:

`1 → 4 → 2 → 3 → 5`

Caption:

> **Figure X:** B keeps the same routing stage as A, then adds query-dependent filtering inside the opened clusters.

---

## Step 2: split a candidate mask

A candidate mask has one bit for every document position in the cluster.

If cluster \(c\) contains \(n_c\) documents, its mask contains

\[
\boxed{
W_c=\left\lceil\frac{n_c}{64}\right\rceil
}
\]

64-bit machine words.

Our three-document example has only one word.

The initial mask is:

`111`

meaning that all three rows are active.

### First split

Bit 1 prefers `1`.

All documents match:

`111 AND 111 = 111`.

Nothing is removed.

### Second split

Bit 4 prefers `0`.

Again all three documents match:

`111 AND 111 = 111`.

Nothing is removed.

### Third split

Bit 2 prefers `1`.

Its bitplane is:

`011`.

So:

`111 AND 011 = 011`.

The preferred branch contains IDs 1 and 2.

The alternative branch is:

`100`

containing ID 0.

Because ID 0 disagrees with the query on bit 2, that branch has penalty

\[
p=|q_2|=0.5.
\]

---

### Diagram 3 — Bitplane search tree

Use **orange** for the currently explored branch and **light gray** for waiting branches.

Structure:

`111: IDs 0,1,2`
`mask width = 1 word`

↓ Bit 1

`111`

↓ Bit 4

`111`

then split:

left/orange:

`011`
`IDs 1,2`
`penalty = 0`

right/gray:

`100`
`ID 0`
`penalty = 0.5`

Beside every node write:

`mask width = 1 word`.

Caption:

> **Figure X:** The first two bit tests remove nothing. The third separates ID 0 from the preferred branch, but both branches are retained until a score bound proves one unnecessary.

---

## Step 3: understand what one split actually costs

A common source of confusion is to think:

> “A bitwise AND is one CPU operation, so this should be almost free.”

That is only true for one machine word.

If a cluster contains 6,400 documents,

\[
W=6400/64=100
\]

machine words.

One split must therefore iterate across roughly 100 word positions.

More importantly, the physical bitmap width does **not** shrink when the logical candidate count shrinks.

A node may contain only 80 active documents while still occupying 100 words.

### Diagram 4 — logical size versus physical width

Two equal-width horizontal bars.

Top:

`Initial mask`
`6,400 active candidates`
`100 words wide`

Bottom:

`Later mask`
`80 active candidates`
`still 100 words wide`

Make most positions in the lower mask gray/empty, but keep the physical box the same width.

Caption:

> **Figure X:** Removing candidates does not shorten a dense bitmap. Later splits can still scan the full physical mask.

This is the main reason Bitplanes can become expensive.

---

## Step 4: count split-mask work

Let \(J_c\) be the number of split nodes visited in cluster \(c\).

Each split scans \(W_c\) words.

Therefore the exact split-word count is

\[
\boxed{
V_{\text{split}}
=
\sum_{c\in O}J_cW_c.
}
\]

This is better than immediately writing a balanced-cluster approximation because it also works for uneven clusters.

If clusters are approximately balanced and B performs an average of \(S\) splits per opened cluster,

\[
V_{\text{split}}
\approx
PS
\left\lceil
\frac{N}{64C}
\right\rceil.
\]

---

## Step 5: score the leaf nodes

Eventually, B stops splitting a candidate set and fully scores its remaining rows.

In our example, we choose leaf size 2.

The preferred branch contains exactly two rows:

`011 → IDs 1 and 2`.

So B scores:

- ID 1 → 2.5
- ID 2 → 1.9

and therefore

\[
F_B=2.
\]

A would have scored all three.

---

### Leaf masks still cost work

Before extracting the surviving IDs, the implementation also scans the leaf bitmap.

Let \(H_c\) be the number of leaf masks read in cluster \(c\).

Then

\[
\boxed{
V_{\text{leaf}}
=
\sum_{c\in O}H_cW_c.
}
\]

This is worth keeping separate from split-mask work because the experiments record the two counters separately.

---

## Step 6: prune the waiting branch

ID 0 is still waiting.

Its accumulated mismatch penalty is

\[
p=0.5.
\]

From the earlier score identity,

\[
S=Q-2p.
\]

So the best possible score anywhere inside that branch is

\[
2.5-2(0.5)=1.5.
\]

Our current top two are:

\[
2.5,\qquad1.9.
\]

Because

\[
1.5<1.9,
\]

ID 0's branch cannot enter the top two.

We skip it.

### Diagram 5 — pruning

Left gray dashed box:

`Waiting branch`
`penalty = 0.5`
`best possible = 1.5`

Arrow:

`1.5 < 1.9`

Right orange box:

`Current top-2`
`2.5`
`1.9`

Below:

`PRUNE`

Caption:

> **Figure X:** Once a branch's best possible score is below the current top-\(K\) threshold, the entire branch can be skipped.

---

## Total cost for B

B therefore pays for:

1. sorting the query bits;
2. split-mask scans;
3. leaf-mask scans;
4. priority-queue and branch management;
5. full scores for \(F_B\) surviving rows.

Let:

- \(b_s\): average time for one split-word visit;
- \(b_l\): average time for one leaf-word visit;
- \(T_{\text{queue}}\): queue, allocation and branch bookkeeping;
- \(T_{\text{order}}\): sorting and query-weight preparation.

Then:

\[
\boxed{
T_B
\approx
T_{\text{common}}
+
T_{\text{order}}
+
b_sV_{\text{split}}
+
b_lV_{\text{leaf}}
+
F_Ba_B
+
T_{\text{queue}}.
}
\]

For approximately balanced clusters,

\[
T_B
\approx
T_{\text{common}}
+
T_{\text{order}}
+
PS
\left\lceil\frac{N}{64C}\right\rceil b_s
+
PH
\left\lceil\frac{N}{64C}\right\rceil b_l
+
F_Ba_B
+
T_{\text{queue}},
\]

where \(H\) is the average number of leaf masks per opened cluster.

This is the main tradeoff:

\[
\boxed{
\text{time saved from avoided scores}
>
\text{bitmap + branch overhead}.
}
\]

A smaller leaf size may reduce \(F_B\), but usually increases \(V_{\text{split}}\).

That is why “B scores fewer documents” by itself does not prove that B is faster.

---

# 2.1.3 Method C: Backward Walk

Backward Walk approaches the problem from the opposite direction.

A starts with every row in the opened clusters.

B also starts broad, then narrows the candidate set.

C starts with a **very specific binary prefix** and progressively makes it broader.

So its two main costs are:

\[
\boxed{
\text{prefix navigation}
+
\text{full scoring of the exposed rows}.
}
\]

---

## Step 1: store rows in prefix order

For the same three documents, suppose the stored key width is five bits.

We sort them:

| Position | Key | ID |
|---:|:---:|---:|
| 0 | `10100` | 0 |
| 1 | `11000` | 2 |
| 2 | `11001` | 1 |

The query's ideal key is:

`11001`.

---

## Step 2: look up the deepest prefix

Suppose we start at depth

\[
x=5.
\]

We search for:

`11001`.

Because the keys are sorted, two binary searches find the start and end of its range.

The result is:

`[2,3)`

containing only ID 1.

We score ID 1:

\[
2.5.
\]

But we need two results, so we continue.

---

## Step 3: remove one prefix constraint

We move from:

`11001`

to:

`1100*`.

Its range is:

`[1,3)`

containing IDs 2 and 1.

ID 1 was already processed.

Therefore we only score the **new part** of the range:

`[1,2)` → ID 2.

ID 2 scores:

\[
1.9.
\]

Now we have two candidates.

---

### Diagram 6 — nested prefix ranges

Use one horizontal array:

**gray**
`ID 0: 10100`

**light purple**
`ID 2: 11000`

**dark purple**
`ID 1: 11001`

Below ID 1 draw a small bracket:

`11001`

Below IDs 2+1 draw a wider bracket:

`1100*`

Arrow to the light-purple section:

`new rows only`

Caption:

> **Figure X:** Relaxing a prefix enlarges the previous range. Already-scored rows stay inside the new range, so C only scores the newly exposed rows.

This picture is much more important than showing the binary-search calls themselves.

---

## Step 4: count how many prefix levels are visited

Let:

- \(x\): starting prefix depth;
- \(\ell\): final prefix depth.

Then:

\[
L=x-\ell+1
\]

prefix levels are visited.

Depth zero represents the whole cluster and needs no binary search.

Let \(L_+\) be the number of visited depths greater than zero.

For each positive depth and each selected cluster, we perform two boundary searches.

So the number of binary-search calls is:

\[
\boxed{
2PL_+.
}
\]

If cluster \(c\) contains \(n_c\) rows, one binary search takes

\[
O(\log(n_c+1))
\]

key comparisons.

Therefore the lookup work is approximately

\[
\boxed{
2L_+
\sum_{c\in O}
\log_2(n_c+1).
}
\]

For balanced clusters:

\[
\boxed{
2PL_+
\log_2\left(\frac NC+1\right).
}
\]

---

## Step 5: count the distinct rows scored

Let \(F_C\) be the total number of distinct rows exposed before Backward Walk stops.

Because every wider prefix contains the previous prefix, each document only needs to be scored once.

So the full-scoring term is simply:

\[
F_Ca_C.
\]

The critical question for C is therefore not really the binary search.

It is:

> **How large does the prefix range become before we can stop?**

---

## Step 6: simple intuition for prefix size

To understand this, suppose temporarily that the bits are:

- independent;
- balanced;
- equally likely to be `0` or `1`.

Then an \(\ell\)-bit prefix has probability

\[
2^{-\ell}.
\]

In a cluster containing \(n\) documents, the expected range size is

\[
\frac{n}{2^\ell}.
\]

Across \(P\) balanced clusters:

\[
\boxed{
E[F_\ell]
\approx
\frac{PN}{C2^\ell}.
}
\]

For one global collection containing one million rows:

| Prefix depth | Expected matching rows |
|---:|---:|
| 16 | about 15 |
| 15 | about 31 |
| 14 | about 61 |
| 13 | about 122 |
| 12 | about 244 |

So dropping one bit approximately doubles the candidate set.

### Diagram 7 — prefix growth

Use **purple boxes that double in width**:

`16 bits → ~15 rows`

↓ broader

`15 bits → ~31 rows`

↓ broader

`14 bits → ~61 rows`

↓ broader

`13 bits → ~122 rows`

Add a small note:

> Illustrative only: assumes independent, balanced bits.

Caption:

> **Figure X:** Prefix relaxation changes only one bit at a time, but under the balanced-bit model the expected candidate count approximately doubles at every step.

Do **not** present this as a recall estimate.

Real embedding bits can be correlated, and clustering can make their distribution even less uniform.

---

## Step 7: approximate stopping versus exact stopping

There are two different ways Backward Walk can stop.

### Approximate stopping

Stop after at least \(M\) candidates have been found.

This is cheap and simple, but:

> finding enough candidates does not prove that the best unseen document is worse.

So recall has to be measured.

### Exact local stopping

At prefix depth \(\ell>0\), every unseen row must disagree with the query in at least one of the first \(\ell\) bits.

Therefore its mismatch penalty is at least

\[
m_\ell
=
\min_{1\le i\le\ell}|q_i|.
\]

Its best possible score is therefore:

\[
\boxed{
U_\ell
=
Q-2m_\ell.
}
\]

Let \(\tau\) be the worst score currently retained in the full top-\(K\) heap.

If:

\[
\boxed{
\tau>U_\ell,
}
\]

with the implementation's tie and rounding rules handled correctly, no unseen row in the opened clusters can improve the answer.

The search can stop safely.

This distinction should be stated clearly because a candidate target and a mathematical score bound provide very different guarantees.

---

## Total cost for C

Let:

- \(T_{\text{key}}\): build the query key and any bound-related query data;
- \(t_b\): average time per binary-search key comparison;
- \(T_{\text{ranges}}\): range updates and new-slice bookkeeping.

Then:

\[
\boxed{
T_C
\approx
T_{\text{common}}
+
T_{\text{key}}
+
2L_+t_b
\sum_{c\in O}\log_2(n_c+1)
+
F_Ca_C
+
T_{\text{ranges}}.
}
\]

For balanced clusters:

\[
\boxed{
T_C
\approx
T_{\text{common}}
+
T_{\text{key}}
+
2PL_+
\log_2\left(\frac NC+1\right)t_b
+
F_Ca_C
+
T_{\text{ranges}}.
}
\]

The binary-search part grows slowly with cluster size.

The main question is \(F_C\).

If C stops at a deep prefix:

\[
F_C\ll F_A,
\]

and it may save substantial scoring work.

If C has to widen all the way to depth zero:

\[
F_C=F_A,
\]

so it performs the same full scoring as A **plus** its prefix-search overhead.

---

# 2.1.4 Query-time comparison

At this point the three methods are easier to compare:

| Method | What it mainly pays for | What it tries to reduce |
|---|---|---|
| A | Full scores for all opened rows | Nothing after routing |
| B | Bitmap traversal + surviving full scores | Number of full scores |
| C | Prefix navigation + exposed full scores | Number of full scores |

A useful compact formula table is:

| Method | Query-time model |
|---|---|
| A | \(T_common + F_A a_A\) |
| B | \(T_common + T_order + b_sV_split + b_lV_leaf + F_Ba_B + T_queue\) |
| C | \(T_common + T_key + T_lookup + F_Ca_C + T_ranges\) |

where

\[
T_{\text{lookup}}
=
2L_+t_b
\sum_{c\in O}\log_2(n_c+1).
\]

The main conditions are therefore:

### Method B

\[
\boxed{
\text{scoring avoided}
>
\text{bitmap and branch work added}.
}
\]

### Method C

\[
\boxed{
\text{scoring avoided}
>
\text{prefix navigation and range work added}.
}
\]

The experiment later checks whether either condition actually occurs.

---

# 2.2 Memory complexity

For memory, I would **not** immediately start with byte arithmetic.

First make one distinction clear:

> There are two different kinds of memory:
>
> **stored index memory**, which remains allocated between queries, and  
> **temporary query memory**, which only exists while a query is running.

This makes the rest much easier to follow.

---

## 2.2.1 Method A

### Stored index

Every document stores:

- its packed binary code;
- its document ID.

With 64-bit words, one code occupies:

\[
8\left\lceil\frac d{64}\right\rceil
\]

bytes.

An ID occupies 8 bytes.

So:

\[
\boxed{
M_{A,\text{rows}}
=
N
\left(
8\left\lceil\frac d{64}\right\rceil+8
\right).
}
\]

Routing structures are additional.

For float32 centroids with routing dimension \(d_R\):

\[
M_{\text{routing}}
\approx
4Cd_R+8C,
\]

plus cluster offsets and implementation overhead.

---

### Query memory

A needs:

- floating-point query;
- score table;
- top-\(K\) results;
- small routing/search state.

For the current field layout:

\[
Q_0
=
4d
+
128\left\lceil\frac d4\right\rceil
+
16K
\]

bytes for the named query, table and result fields.

### Small example

With three five-bit documents:

- codes: \(3\times8=24\) bytes;
- IDs: \(3\times8=24\) bytes.

So the named document fields use:

\[
48\text{ bytes}.
\]

The exact number is not important; it simply gives us a baseline for comparing B and C.

---

## 2.2.2 Method B

B keeps everything A stores.

It adds two things:

1. stored bitplanes;
2. temporary branch masks.

### Stored bitplanes

For coordinate \(i\), cluster \(c\) needs \(W_c\) words.

There are \(d\) coordinates.

Therefore:

\[
\boxed{
M_{\text{planes}}
=
8d
\sum_{c=1}^{C}W_c.
}
\]

with

\[
W_c=\left\lceil\frac{n_c}{64}\right\rceil.
\]

Notice the per-cluster padding.

---

### Temporary masks

One live mask for cluster \(c\) uses:

\[
8W_c
\]

bytes.

If several branches are waiting, several full-width masks may be alive at once.

If \(A_c(t)\) masks from cluster \(c\) are live at time \(t\),

\[
\boxed{
Q_{\text{masks}}
=
8
\max_t
\sum_c A_c(t)W_c.
}
\]

The query also keeps:

- sorted bit positions;
- priority-queue records;
- node metadata.

### Diagram 8 — B memory

Three columns:

**Stored once**
`Packed rows`
`IDs`
`Bitplanes`

**Per query**
`Query`
`Score table`
`Top K`

**B-specific temporary**
`Mask`
`Mask`
`Mask`
`...`

Make the repeated masks orange and the same physical width.

Caption:

> **Figure X:** B's main temporary-memory risk comes from multiple full-width masks being alive at the same time.

---

## 2.2.3 Method C

C also keeps A's packed documents and IDs.

In the current sorted-key implementation it adds one 32-bit prefix key per row.

So:

\[
\boxed{
M_{\text{prefix}}=4N
}
\]

bytes.

Documents are sorted once by this key inside each cluster.

We do **not** store a separate posting list for every prefix depth.

The same sorted rows support:

`11001`

`1100*`

`110**`

and so on.

This is important because otherwise prefix relaxation could duplicate a large amount of storage.

---

### Query memory

For every opened cluster, C only needs to remember the previously visited range:

`[left, right)`.

In the current 64-bit implementation, a selected-cluster record contains approximately:

- 8-byte cluster pointer;
- 8-byte left boundary;
- 8-byte right boundary.

So about:

\[
24P
\]

bytes.

The current query key uses another four bytes.

Optional exact-stop data adds small \(O(x)\) query state.

### Diagram 9 — C memory

Show one sorted document array:

`10100 | 11000 | 11001 | ...`

Above it draw multiple nested brackets:

`11001`

`1100*`

`110**`

But all brackets point to the **same underlying array**.

Caption:

> **Figure X:** Backward Walk does not store a new document list for every prefix depth. Different prefix ranges refer to the same sorted rows.

---

# 2.2.4 Memory comparison

| Method | Persistent index | Main additional query memory |
|---|---|---|
| A | Packed rows + IDs + router | Query/table/top-K |
| B | A + bitplanes | Multiple live full-width masks |
| C | A + 4-byte prefix keys | One previous range per opened cluster |

The field formulas describe the logical data.

Actual process RAM can be larger because of:

- vector capacity;
- allocator overhead;
- temporary construction buffers;
- Python/C++ interface buffers;
- runtime state.

Therefore the experiment should report both:

1. calculated/index field bytes;
2. measured process peak RAM.

---

# 2.3 Recall

The complexity formulas tell us how much work a method performs.

They do **not** tell us whether the correct documents survive.

So recall is measured directly from document IDs.

For every query \(q\), let:

- \(G_q\): exact reference top-\(K\) IDs;
- \(O_{a,q}\): IDs returned by method \(a\).

Then:

\[
\boxed{
R_a
=
\frac{
\sum_q |G_q\cap O_{a,q}|
}{
n_qK
}.
}
\]

For example:

reference:

`[1,2]`

returned:

`[1,0]`

gives

\[
1/2=50\%.
\]

Returning:

`[1,2]`

gives 100%.

---

## Diagram 10 — where recall can be lost

I think one diagram can make the three recall formulas much easier.

Start with:

**Global exact top-K**

`Gq`

↓ routing

**Reference IDs whose clusters were opened**

`gq`

then branch into:

### A

`score every opened row`

→ all \(g_q\) survive.

### B

`branch / prune`

→ only fraction \(r_{B,q}\) survive.

### C

`prefix stopping depth ℓ`

→ only reference IDs inside \(D_{q,\ell}\) survive.

Use:

- blue for A,
- orange for B,
- purple for C.

Caption:

> **Figure X:** Recall can be lost during routing, and B or C may introduce an additional local-search loss if they stop early.

This one diagram replaces a lot of abstract explanation.

---

## 2.3.1 Method A

Let

\[
g_q
\]

be the number of global reference IDs whose clusters are opened.

A fully scores every opened row.

Therefore every one of those \(g_q\) reference documents is recovered locally.

So:

\[
\boxed{
R_{A,q}
=
\frac{g_q}{K}.
}
\]

A can lose recall through routing, but not through its local scan.

---

## 2.3.2 Method B

Among the \(g_q\) reference IDs that reached the opened clusters, suppose B recovers fraction

\[
r_{B,q}.
\]

Then:

\[
\boxed{
R_{B,q}
=
\frac{g_q}{K}r_{B,q}.
}
\]

Example:

- 8 of 10 reference IDs reach the opened clusters;
- B returns 6 of those 8.

Then:

\[
(8/10)(6/8)=60\%.
\]

If B runs its complete branch-and-bound traversal using safe score bounds and correct tie handling,

\[
r_{B,q}=1.
\]

A finite node budget can make

\[
r_{B,q}<1.
\]

---

## 2.3.3 Method C

Let

\[
D_{q,\ell}
\]

be the set of rows exposed when Backward Walk stops at prefix depth \(\ell\).

Every row in \(D_{q,\ell}\) has received the complete score.

Therefore:

\[
\boxed{
R_{C,q}
=
\frac{|G_q\cap D_{q,\ell}|}{K}.
}
\]

The important point is:

> The number of candidates in the prefix does not tell us its recall.

A prefix may contain many documents but still miss the best document.

---

### Small counterexample

Let:

\[
q=[0.1,\ 0.9].
\]

The ideal signs are:

`11`.

Suppose the only documents are:

`10`

and

`01`.

Their scores are:

`10 → 0.1 - 0.9 = -0.8`

`01 → -0.1 + 0.9 = 0.8`.

If C relaxes the query to:

`1*`

it finds `10`.

But `01`, which lies outside that prefix, is actually better.

So:

> **finding one candidate is not a proof that the best candidate has been found.**

This is why candidate-target stopping is approximate.

---

## When is C exact inside the opened clusters?

There are two ways.

### 1. Reach depth zero

At depth zero, every row in every opened cluster has been exposed.

Therefore C becomes equivalent to A inside those clusters.

### 2. Prove that unseen rows cannot win

At depth \(\ell>0\), every unseen row differs from the query in at least one of the first \(\ell\) signs.

The cheapest possible such mismatch is:

\[
m_\ell
=
\min_{1\le i\le\ell}|q_i|.
\]

So every unseen row has score at most:

\[
U_\ell
=
Q-2m_\ell.
\]

If the current worst top-\(K\) score is \(\tau\) and:

\[
\tau>U_\ell,
\]

with tie and floating-point rounding handled safely, C can stop exactly within the opened clusters.

This is a much stronger condition than simply finding \(K\) candidates.

---

## Optional rough prefix model

If you still want the independence approximation in the paper, I would move it into a clearly labeled optional box.

Something like:

> **Rough intuition only.** Suppose \(p_i\) is the probability that an opened reference document agrees with the query at prefix coordinate \(i\). If these agreement events were independent, the fraction agreeing with all first \(\ell\) positions would be approximately
>
> \[
> \prod_{i=1}^{\ell}p_i.
> \]
>
> This is not used as the reported recall. Real coordinates can be dependent, so the experiment measures recall directly from returned IDs.

I would **not** leave this mixed into the main recall derivation. It makes the section feel much more complicated than necessary.

---

# 2.4 Summary

I would keep this section almost entirely tables.

## Query-time work

| Method | Approximate cost |
|---|---|
| Original | \(T_common + F_A a_A\) |
| Bitplanes | \(T_common + T_order + b_sV_split + b_lV_leaf + F_Ba_B + T_queue\) |
| Backward Walk | \(T_common + T_key + T_lookup + F_Ca_C + T_ranges\) |

where:

\[
F_A=\sum_{c\in O}n_c,
\]

\[
V_{\text{split}}
=
\sum_{c\in O}J_cW_c,
\]

\[
V_{\text{leaf}}
=
\sum_{c\in O}H_cW_c,
\]

and

\[
T_{\text{lookup}}
=
2L_+t_b
\sum_{c\in O}
\log_2(n_c+1).
\]

---

## Memory

| Method | Stored index | Temporary query memory |
|---|---|---|
| A | \(M_0\) | \(Q_0\) |
| B | \(M_0+M_planes\) | \(Q_0+Q_B\) |
| C | \(M_0+M_prefix\) | \(Q_0+Q_C\) |

---

## Recall

| Method | Per-query recall | Exact local-search condition |
|---|---|---|
| A | \(g_q/K\) | Score all opened rows |
| B | \((g_q/K)r_{B,q}\) | Complete safe branch-and-bound search |
| C | \(|G_q\cap D_{q,\ell}|/K\) | Reach depth zero or safely bound all unseen rows |

---

# Variable definitions

I would split this into **three smaller tables** instead of the huge long table you currently have.

### Shared

| Symbol | Meaning |
|---|---|
| \(N\) | Total number of documents |
| \(C\) | Number of clusters |
| \(P\) | Number of opened clusters |
| \(n_c\) | Rows in cluster \(c\) |
| \(d\) | Binary dimensions |
| \(K\) | Number of results |
| \(O\) | Set of opened clusters |
| \(F_A\) | Rows fully scored by A |
| \(T_common\) | Routing + shared query preparation for that configuration |

### Bitplanes

| Symbol | Meaning |
|---|---|
| \(W_c\) | Machine words in one mask for cluster \(c\) |
| \(J_c\) | Split nodes visited in cluster \(c\) |
| \(H_c\) | Leaf masks read in cluster \(c\) |
| \(V_split\) | Total split-word visits |
| \(V_leaf\) | Total leaf-word visits |
| \(F_B\) | Documents fully scored by B |
| \(Q_B\) | Additional live masks and branch queue memory |

### Backward Walk

| Symbol | Meaning |
|---|---|
| \(h\) | Stored prefix-key width |
| \(x\) | Starting prefix depth |
| \(\ell\) | Final prefix depth |
| \(L\) | Number of depths visited |
| \(L_+\) | Number of positive depths requiring lookup |
| \(F_C\) | Distinct documents fully scored by C |
| \(D_{q,\ell}\) | Documents exposed at final prefix depth |
| \(Q_C\) | Prefix-range state for the query |

---

I think this version would be **much easier to understand** because each method now has the same rhythm:

> **What does it do? → what work does that create? → formula → what parameter controls it?**

And the formulas are now much closer to the counters you actually report later. In particular, exposing \(V_{\text{split}}\) and \(V_{\text{leaf}}\) separately makes the theory line up directly with your experiment table that reports split-mask and leaf-mask words. 