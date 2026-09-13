Yes — I’d keep the same mathematics, but make the prose feel much more like the Step 7 example you wrote: **“remember…”, “however…”, “meanwhile…”, “the good news is…”, “the problem is…”**.

Below is the text I’d use for the other parts of the calculation section. I’m keeping the terminology aligned with your current report: **cluster, probe, Bitplane, mask, branch, leaf, prefix, binary search, top-K, recall**. 

---

# Preliminary calculations

Before running the experiments, let’s first understand how much work each method actually needs to do.

To recap what we have so far:

- **Method A** is the original approach: after we open the clusters, every document inside those clusters gets a document scoring operation.
- **Method B, Bitplanes**, tries to reduce the number of document scoring operations by filtering documents first.
- **Method C, Backward Walk**, tries to avoid starting with a large number of documents in the first place, by starting from a specific prefix and broadening it only when necessary.

So the main question in this section is actually quite simple:

> **Methods B and C both add some extra search work. Is the amount of document scoring they save large enough to make that extra work worth it?**

To make the calculations easier to follow, I will use the same small example throughout this section.

Suppose we have one opened cluster containing three documents:

| ID | Binary document | Full score |
|---|---|---:|
| 0 | `10100` | 0.1 |
| 1 | `11001` | 2.5 |
| 2 | `11000` | 1.9 |

The query is:

\[
q=[0.7,\ 0.5,\ -0.4,\ -0.6,\ 0.3].
\]

From the previous section, we already know that the ideal binary document is:

`11001`

and suppose we want the best two results, so:

\[
K=2.
\]

For this small example:

\[
N=3,\qquad C=P=1,\qquad d=5.
\]

Of course, the real experiments later contain hundreds of thousands or millions of documents. This example is only here so that we can actually see what every operation is doing.

---

# Method A: scan the opened clusters

Method A is the easiest one to reason about.

Remember the original pipeline: first we select which clusters to open, and after that there is no additional filtering. Every document inside those clusters gets a scoring operation.

## Step 1: how many scoring operations do we need?

In our example, the only opened cluster contains:

`[ID 0, ID 1, ID 2]`

so we need three scoring operations:

\[
F_A=3.
\]

More generally, if \(O\) is the set of clusters we opened, and cluster \(c\) contains \(n_c\) documents, then:

\[
F_A=\sum_{c\in O}n_c.
\]

So this is already the most important number for Method A:

> **How many documents did cluster selection leave for us to score?**

If the clusters are roughly the same size, then one cluster contains approximately:

\[
\frac{N}{C}
\]

documents.

If we open \(P\) clusters, then the number of documents we expect to score is roughly:

\[
F_A\approx\frac{PN}{C}.
\]

### Figure

The diagram should simply show:

`Query → Select P clusters → FA documents → scoring operation for every document → Top K`

Use blue for the whole path.

Inside the \(F_A\) box, show the three example rows:

`10100`  
`11001`  
`11000`

The purpose of this figure is to make one thing obvious: **after cluster selection, Method A does not try to reduce the document set anymore.**

---

## Step 2: what does one scoring operation actually do?

Now we know how many documents need to be scored, but we still need to understand the cost of scoring one document.

The implementation follows the four-sign lookup-table approach described in Exa’s article.

Instead of calculating every query value against every binary sign one by one, we group four positions together.

For our five-dimensional query, we get two groups:

`[0.7, 0.5, -0.4, -0.6]`

and:

`[0.3, 0, 0, 0]`

A group of four binary signs can have:

\[
2^4=16
\]

possible patterns.

So for our example, we prepare:

\[
2\times16=32
\]

lookup-table values.

After this table is prepared, scoring one document becomes much simpler: we only need to look up the corresponding value for each group and add them together.

If the document has \(d\) bits, then the number of groups is:

\[
G=\left\lceil\frac d4\right\rceil.
\]

So one document scoring operation needs \(G\) table lookups.

In our example:

\[
G=2.
\]

We have three documents, so altogether we perform:

\[
3\times2=6
\]

table lookups.

In general, the scoring work grows with:

\[
F_AG.
\]

Since \(G\) grows with \(d\), we can write this as:

\[
O(F_Ad).
\]

For a fixed binary dimension, the important part is simply \(F_A\): the more documents we open, the more scoring work we do.

---

## Step 3: keep only the best K

Of course, after calculating the scores, we do not need to keep every result.

We only need the best \(K\).

In our example, the results arrive like this:

`ID 0 → 0.1`

Then:

`ID 1 → 2.5`

So our current best two are:

`[ID 1: 2.5, ID 0: 0.1]`

Then ID 2 arrives with score 1.9, which is better than ID 0:

`[ID 1: 2.5, ID 2: 1.9]`

Using a bounded heap, maintaining these top-\(K\) results takes at most approximately:

\[
O(F_A\log(K+1)).
\]

For our later experiments, \(K\) is usually much smaller than the number of documents, so the dominant part is still normally the document scoring itself.

---

## Total cost for A

Putting everything together, let \(T_{\text{base}}\) contain the work before document scoring, such as selecting the clusters and preparing the lookup table.

Let \(a_A\) be the average time required to fully process one document, including its scoring operation and the top-\(K\) update.

Then:

\[
T_A\approx T_{\text{base}}+F_Aa_A.
\]

If the clusters are approximately balanced:

\[
T_A\approx T_{\text{base}}+\frac{PN}{C}a_A.
\]

So there is a fairly straightforward tradeoff here.

If we increase the number of clusters \(C\), each cluster becomes smaller, which is good because every opened cluster contains fewer documents.

However, the downside is that we may need to open more clusters \(P\) to maintain the same recall.

> **TL;DR — Method A:**  
> The main cost of A is simply how many documents are left after cluster selection. If fewer documents are opened, we do fewer scoring operations. However, making cluster selection too selective can cause us to miss the correct results.

---

# Method B: Bitplanes

Now let’s look at Method B.

The idea of Bitplanes is to reduce the number of documents that reach the scoring operation.

Instead of immediately scoring every document inside the opened clusters, we first spend some work filtering them using Bitplanes.

So compared with A, B has an extra cost.

However, the hope is that this extra filtering is cheaper than the document scoring operations that it allows us to skip.

---

## Step 1: inspect the most important query bits first

Remember from the previous section that not every mismatch has the same cost.

For our query:

\[
q=[0.7,\ 0.5,\ -0.4,\ -0.6,\ 0.3],
\]

a mismatch on the first position costs more than a mismatch on the fifth position.

So before traversing the Bitplanes, we order the query positions by decreasing \(|q_i|\):

`Bit 1 (0.7) → Bit 4 (0.6) → Bit 2 (0.5) → Bit 3 (0.4) → Bit 5 (0.3)`

The intuition here is simple: **if we are going to split the documents, we should first use the bits that can change the final score the most.**

Sorting these positions costs:

\[
O(d\log d).
\]

We also calculate:

\[
Q=\sum_i|q_i|=2.5,
\]

which, as we saw earlier, is the maximum possible score.

### Figure

Show:

`Select P clusters → Order bits by |qi| → Bitplane search`

Use blue for `Select P clusters`, then orange for the rest.

This helps remind the reader that Bitplanes do not replace cluster selection; they happen **after** the clusters have already been opened.

---

## Step 2: represent the documents using a mask

Inside one cluster, Bitplanes keep a mask telling us which document positions are still active.

For our three documents, the initial mask is:

`111`

which simply means:

- first `1` → ID 0 is active;
- second `1` → ID 1 is active;
- third `1` → ID 2 is active.

Now, a computer does not normally store an arbitrarily large bitmap as one gigantic number. It stores it in machine words.

Using 64-bit words, if cluster \(c\) contains \(n_c\) documents, one mask needs:

\[
W_c=\left\lceil\frac{n_c}{64}\right\rceil
\]

machine words.

Our tiny three-document example fits into one word.

However, later we will see that this becomes important when one cluster contains thousands or millions of documents.

---

## Step 3: start splitting the mask

Now we start from the most important query bit.

The first one is bit 1, and the query prefers `1`.

All three documents have bit 1 equal to `1`, so:

`111 AND 111 = 111`

Nothing changes.

Next we check bit 4.

The query prefers `0`, and again all three documents match, so the mask remains:

`111`

So far, the first two splits have cost us work, but they have not removed a single document.

Now we reach bit 2.

Its Bitplane is:

`011`

and the query prefers `1`, so:

`111 AND 011 = 011`

This time, the split is useful.

The preferred branch now contains:

`011 → IDs 1 and 2`

while the other branch contains:

`100 → ID 0`

Because ID 0 disagrees with the query on bit 2, this branch has a known mismatch penalty:

\[
p=|q_2|=0.5.
\]

### Figure

Use a tree:

`111: IDs 0,1,2`

↓ Bit 1

`111`

↓ Bit 4

`111`

Then branch:

left/orange:

`011: IDs 1,2, penalty 0`

right/gray:

`100: ID 0, penalty 0.5`

Also write the mask width beside each node.

The reader should see two things from this diagram:

1. some splits may remove nothing;
2. the alternative branch is kept instead of immediately thrown away.

---

## Step 4: one important problem — Bitplane filtering is not free

At first, it is very tempting to think:

> “This is just a bitwise AND, so this should basically be O(1).”

However, that is only true if the entire mask fits inside one machine word.

Imagine one cluster contains:

\[
6,400
\]

documents.

Using 64-bit words, one mask needs:

\[
6400/64=100
\]

machine words.

So one split does not perform one AND operation.

Instead, we need to loop through roughly 100 word positions and perform the operation on each one.

And there is another slightly annoying detail.

Suppose after several splits we only have 80 active documents left.

You might expect the next split to become much cheaper.

Unfortunately, with this dense bitmap representation, the physical mask can still be 100 words wide.

So even though only 80 documents are logically active, the next split may still have to scan those same 100 machine words.

### Figure

Show two masks with exactly the same physical width.

Top:

`6,400 active documents — 100 words`

Bottom:

`80 active documents — still 100 words`

Most of the second mask should look empty, but its box should still have exactly the same width.

> **This is one of the main costs of Bitplanes: fewer surviving documents do not automatically mean cheaper future splits.**

---

## Step 5: how much Bitplane work do we actually perform?

Now we can put a simple formula around this.

Suppose cluster \(c\) has a mask width of \(W_c\) words, and we perform \(J_c\) splits inside that cluster.

Each split needs to scan those \(W_c\) words.

So the total number of split-mask word operations is:

\[
V_{\text{split}}
=
\sum_{c\in O}J_cW_c.
\]

For roughly balanced clusters, if we perform an average of \(S\) splits per opened cluster, this becomes approximately:

\[
V_{\text{split}}
\approx
PS
\left\lceil
\frac{N}{64C}
\right\rceil.
\]

The formula looks slightly complicated, but the intuition is actually simple:

> **Bitplane cost = how wide the masks are × how many times we split them.**

---

## Step 6: eventually, we stop splitting

Of course, we do not want to keep splitting forever.

Once a branch contains few enough documents, we treat it as a leaf and simply score the documents inside it.

Suppose our leaf size is two.

The preferred branch:

`011`

contains exactly two documents:

- ID 1;
- ID 2.

So we stop splitting and perform their document scoring operations:

`ID 1 → 2.5`

`ID 2 → 1.9`

Therefore:

\[
F_B=2.
\]

Meanwhile, Method A would have scored all three documents.

So, at least in this tiny example, Bitplanes successfully saved one scoring operation.

However, remember that we had to perform three Bitplane splits to achieve that saving.

This is exactly the tradeoff that we care about.

---

## Another small cost: reading the leaf mask

There is also one more cost that is easy to forget.

Before we can score the documents inside a leaf, we still have to read its bitmap and identify which document positions are active.

If cluster \(c\) has \(H_c\) leaves that we eventually score, then the amount of leaf-mask work is:

\[
V_{\text{leaf}}
=
\sum_{c\in O}H_cW_c.
\]

So B actually has two different kinds of bitmap work:

- split masks;
- leaf masks.

We keep them separate because the experiments later measure them separately.

---

## Step 7: what about the branch that we did not explore?

[Keep your current Step 7 almost exactly as written.]

I would only slightly clean the opening to:

> Remember why we introduced branching in the first place: if we only followed the preferred bit every time, we could accidentally throw away the true best document. Because of that, ID 0 is still waiting in the queue as an alternative branch.
>
> The good news is that we do not necessarily need to explore it.

Then continue with your existing \(p=0.5\), \(Q-2p=1.5\), current top two 2.5 and 1.9, and “good news is that we can safely skip it!” wording.

---

## Total cost for Bitplanes

At this point, we can see where B spends its time.

Compared with A, B additionally has to:

- order the query bits;
- read split masks;
- read leaf masks;
- keep alternative branches in a priority queue.

However, in return, it hopefully reduces the number of documents that need a scoring operation.

So we can summarize the time as:

\[
T_B
\approx
T_{\text{base}}
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
\]

The individual symbols are not the main takeaway.

The important idea is:

\[
\boxed{
\text{document scoring work avoided}
>
\text{Bitplane work added}
}
\]

If this inequality is true, B has a chance to be faster than A.

If it is false, then we spent more time filtering than we saved.

This is also why the Bitplane parameters matter.

For example, a smaller leaf size means we continue filtering for longer, which can reduce \(F_B\).

However, the downside is that we now perform more splits.

Likewise, a larger node budget lets us explore more branches and potentially improve recall, but once again, that means more Bitplane work.

> **TL;DR — Bitplanes:**  
> Bitplanes are useful when a relatively small amount of bitmap work can eliminate a large number of document scoring operations. The fact that B scores fewer documents is not enough by itself; the filtering used to remove those documents also has to be cheap.

---

# Method C: Backward Walk

Now let’s look at the second idea.

Backward Walk approaches the problem from almost the opposite direction.

Method A starts with every document inside the opened clusters.

Method B also starts with all of them, but progressively removes documents.

Backward Walk instead asks:

> **What if we start from the most specific binary prefix first, and only broaden the search when we need more documents?**

So C mainly pays for two things:

- looking up different prefixes;
- performing scoring operations for the documents exposed by those prefixes.

---

## Step 1: store the documents in prefix order

Using the same example, let’s sort the three documents by their binary key:

| Position | Binary key | ID |
|---:|:---:|---:|
| 0 | `10100` | 0 |
| 1 | `11000` | 2 |
| 2 | `11001` | 1 |

Meanwhile, the ideal query pattern is:

`11001`.

Because the rows are sorted, documents with the same prefix will appear next to each other.

That means we can use binary search to quickly find the range corresponding to a particular prefix.

---

## Step 2: start from the most specific prefix

Let’s first use all five bits:

`11001`.

Two binary searches find the beginning and the end of this range.

In our example, the result is:

`[2,3)`

which contains only ID 1.

So we perform one scoring operation:

`ID 1 → 2.5`

However, remember that we want the top two results.

Right now we only have one document.

So unfortunately, we cannot stop yet.

We need to broaden the prefix.

---

## Step 3: remove one constraint

We now go from:

`11001`

to:

`1100*`

The `*` means that we no longer care whether the last bit is `0` or `1`.

The new range becomes:

`[1,3)`

which contains:

- ID 2: `11000`;
- ID 1: `11001`.

Now, ID 1 was already scored at the previous depth.

So the good news is that we do **not** need to score it again.

The only new document is ID 2:

`ID 2 → 1.9`

Now we have two results.

### Figure

Show the three sorted rows horizontally.

ID 0 in gray:

`10100`

ID 2 in light purple:

`11000`

ID 1 in darker purple:

`11001`

Then show a small bracket around ID 1 labeled:

`11001`

and a larger bracket around ID 2 + ID 1 labeled:

`1100*`

Point to ID 2 with:

`new document`

The purpose is to show visually that the new prefix contains the old prefix, so we only need to process the new part.

---

## Step 4: how much prefix lookup work do we need?

Suppose we start at prefix depth \(x\), and eventually stop at depth \(\ell\).

Then the number of depths we visit is:

\[
L=x-\ell+1.
\]

At every positive depth, we need two binary searches for every opened cluster:

- one to find where the prefix starts;
- one to find where it ends.

So if we open \(P\) clusters and visit \(L_+\) positive prefix depths, we perform:

\[
2PL_+
\]

binary searches.

Each binary search over cluster \(c\) costs roughly:

\[
O(\log(n_c+1)).
\]

So the total lookup work is approximately:

\[
2L_+
\sum_{c\in O}
\log_2(n_c+1).
\]

Again, the formula looks more complicated than the idea.

The idea is simply:

> **Backward Walk cost grows with how many prefix depths we visit and how many clusters we search at each depth.**

---

## Step 5: how many documents do we eventually score?

Let \(F_C\) be the total number of distinct documents that become visible before we stop.

Because each broader prefix contains the previous prefix, a document that was already scored does not need another scoring operation.

So C's document-scoring work is simply:

\[
F_Ca_C.
\]

This means the main question for Backward Walk is actually:

> **How broad does the prefix need to become before we have enough useful documents?**

If we can stop while the prefix is still very specific, \(F_C\) can be very small.

If we need to keep broadening all the way to the root, \(F_C\) eventually becomes all documents in the opened clusters.

---

## Step 6: how quickly does the prefix grow?

Let’s use a simple model just to build intuition.

Assume, temporarily, that every bit is independent and has a 50% chance of being `0` or `1`.

Then the probability that a random document matches an \(\ell\)-bit prefix is:

\[
2^{-\ell}.
\]

So if we have \(n\) documents, the expected number matching that prefix is:

\[
\frac{n}{2^\ell}.
\]

For a collection of one million documents:

| Prefix depth | Expected documents |
|---:|---:|
| 16 | about 15 |
| 15 | about 31 |
| 14 | about 61 |
| 13 | about 122 |
| 12 | about 244 |

So something interesting happens here.

We only remove **one bit** each time, but the number of matching documents approximately **doubles**.

### Figure

Use purple horizontal bars that approximately double in width:

`16 bits → ~15`

`15 bits → ~31`

`14 bits → ~61`

`13 bits → ~122`

`12 bits → ~244`

This makes the exponential growth obvious.

However, we should remember that this is only an intuition.

Real embedding bits can be correlated, and once the documents have already been divided into clusters, the distribution can be even further from a perfect 50/50 split.

Most importantly:

> **This only estimates how many documents we might find. It does not tell us whether those documents are actually the correct top results.**

---

## Step 7: when should Backward Walk stop?

There are two ways to stop.

### Approximate stopping

The simplest option is to stop after we have found enough documents.

For example:

`16 bits → 12 documents`

`15 bits → 31 documents`

`14 bits → 66 documents`

`13 bits → 136 documents`

If our target is 100 documents, we could stop at 13 bits.

This is simple and cheap.

However, there is one problem.

Finding 100 documents does not prove that the best unseen document is worse than the ones we found.

So this stopping rule is approximate, and we need to measure its recall experimentally.

---

### Exact stopping

Fortunately, we can also use the scoring equation to derive a safe bound.

Suppose we have already searched all documents matching the first \(\ell\) query signs.

Any unseen document must disagree with the query on at least one of those first \(\ell\) positions.

The cheapest possible mismatch is:

\[
m_\ell
=
\min_{1\le i\le\ell}|q_i|.
\]

Therefore, the best possible score of an unseen document is:

\[
U_\ell
=
Q-2m_\ell.
\]

Meanwhile, suppose the worst document currently inside our top-\(K\) has score:

\[
\tau.
\]

If:

\[
\tau>U_\ell,
\]

then even the **best possible unseen document** cannot enter our top-\(K\).

So the good news is that we can stop safely.

If this never happens, we can keep broadening the prefix until depth zero.

At depth zero, we have exposed every document inside the opened clusters, so Backward Walk becomes equivalent to Method A locally.

---

## Total cost for Backward Walk

Backward Walk therefore pays for:

- building the query prefix;
- performing prefix lookups;
- updating the current prefix ranges;
- performing scoring operations for \(F_C\) exposed documents.

We can summarize this as:

\[
T_C
\approx
T_{\text{base}}
+
T_{\text{key}}
+
2L_+t_b
\sum_{c\in O}
\log_2(n_c+1)
+
F_Ca_C
+
T_{\text{ranges}}.
\]

Again, the exact symbols are not the most important part.

The important condition is:

\[
\boxed{
\text{document scoring work avoided}
>
\text{prefix lookup work added}
}
\]

If C can stop at a fairly deep prefix, then:

\[
F_C\ll F_A,
\]

and we may save a lot of scoring work.

However, if C has to broaden the prefix all the way to depth zero:

\[
F_C=F_A,
\]

then C ends up doing all of A's scoring operations **plus** the additional prefix lookups.

> **TL;DR — Backward Walk:**  
> Backward Walk works best when a fairly specific prefix already contains the documents we need. If we have to keep broadening until almost the whole cluster becomes visible, then the prefix lookup has not saved us anything.

---

# Memory complexity

So far, we have only talked about query time.

However, B and C also need additional index structures, so we should also check how much memory they require.

There are two kinds of memory that matter here:

- **persistent index memory**, which remains loaded between queries;
- **temporary query memory**, which only exists while one query is running.

---

# Method A memory

Method A stores:

- the packed binary document;
- the document ID;
- the cluster/routing information.

Using 64-bit words, one binary document needs:

\[
8\left\lceil\frac d{64}\right\rceil
\]

bytes.

The ID adds another 8 bytes.

So the document storage is:

\[
M_{A,\text{rows}}
=
N
\left(
8\left\lceil\frac d{64}\right\rceil+8
\right).
\]

During a query, A also keeps the query values, the lookup table and the current top-\(K\).

This is our baseline.

---

# Method B memory

B keeps everything A needs.

However, it adds two things.

First, it stores the Bitplanes.

For each cluster, every binary dimension needs one bitmap spanning all document positions in that cluster.

Therefore:

\[
M_{\text{planes}}
=
8d
\sum_{c=1}^{C}W_c.
\]

So in a rough sense, B stores the document bits twice:

- once arranged by document;
- once arranged as Bitplanes.

The second extra cost is temporary query memory.

Remember that branching means we can have several masks waiting in the queue at the same time.

Each one still has the full physical width of its cluster.

If \(A_c(t)\) masks belonging to cluster \(c\) are alive at time \(t\), then the raw mask memory is:

\[
Q_{\text{masks}}
=
8
\max_t
\sum_cA_c(t)W_c.
\]

### Figure

Show three columns:

`Stored once | Common query | B-specific query`

Under stored once:

`Packed documents`
`IDs`
`Bitplanes`

Under common query:

`Query`
`Lookup table`
`Top K`

Under B-specific:

`Mask`
`Mask`
`Mask`
`...`

Use orange for the masks.

> **TL;DR — B memory:**  
> Bitplanes need extra persistent memory because we store another arrangement of the document bits. During search, B can also temporarily hold several full-width branch masks at the same time.

---

# Method C memory

C also keeps the normal packed documents and IDs.

The current implementation additionally stores one 32-bit prefix key for every document.

So:

\[
M_{\text{prefix}}=4N.
\]

The important part is that we do **not** create a new copy of the documents for every prefix depth.

For example:

`11001`

`1100*`

`110**`

all refer to different ranges of the **same sorted document array**.

### Figure

Show:

`10100 | 11000 | 11001 | ...`

and then nested brackets for:

`11001`

`1100*`

`110**`

The reader should visually see that we have one array, not three copies.

During the query, C mainly needs to remember the previous range for every opened cluster.

So compared with B, its temporary search state is relatively small.

> **TL;DR — C memory:**  
> Backward Walk adds a small prefix key per document, but it does not need full-width masks for many waiting branches.

---

# Recall

Now we know roughly how much work each method performs.

However, none of those savings matter if we stop too early and return the wrong documents.

So we also need to understand where recall can be lost.

For query \(q\):

- \(G_q\) is the exact global top-\(K\);
- \(O_{a,q}\) is the result returned by method \(a\).

Average recall is:

\[
R_a
=
\frac{
\sum_q|G_q\cap O_{a,q}|
}{
n_qK
}.
\]

For example:

`Reference: [1,2]`

`Returned: [1,0]`

means we recovered one of two:

\[
50\%.
\]

---

## Where can the correct document disappear?

There are actually two places.

First, cluster selection can fail to open the cluster containing the correct document.

Second, even if the correct document is inside an opened cluster, B or C may stop their local search before reaching it.

### Figure

`Global exact top-K`

↓ `cluster selection`

`Correct documents that reached the opened clusters`

Then branch into:

A → `scan everything → no additional local loss`

B → `Bitplane search → may stop early`

C → `prefix search → may stop early`

This figure is important because it separates **routing loss** from **local search loss**.

---

# Recall for Method A

Let \(g_q\) be the number of global top-\(K\) documents whose clusters were opened.

Because A scores every document inside those clusters, all \(g_q\) of those documents are recovered locally.

So:

\[
R_{A,q}
=
\frac{g_q}{K}.
\]

In other words:

> A can lose a correct document because cluster selection missed it, but once the document is inside an opened cluster, A will not lose it during local search.

---

# Recall for Bitplanes

Suppose \(g_q\) correct documents reached the opened clusters, but B only recovers some fraction \(r_{B,q}\) of them.

Then:

\[
R_{B,q}
=
\frac{g_q}{K}r_{B,q}.
\]

For example, imagine:

- 8 of the global top 10 documents are inside the opened clusters;
- B recovers 6 of those 8.

Then:

\[
Recall
=
\frac{8}{10}
\times
\frac{6}{8}
=
60\%.
\]

If B performs a complete branch-and-bound search using valid score bounds, then it eventually recovers all of the correct documents inside the opened clusters.

However, if we impose a finite node budget, B may stop early.

> **TL;DR — B recall:**  
> B can lose recall in two places: cluster selection can miss the document before B starts, and a limited Bitplane search can miss a document even after its cluster has been opened.

---

# Recall for Backward Walk

For Backward Walk, let \(D_{q,\ell}\) be all documents exposed when the search stops at prefix depth \(\ell\).

Then:

\[
R_{C,q}
=
\frac{|G_q\cap D_{q,\ell}|}{K}.
\]

The important thing here is that **candidate count and recall are not the same thing**.

Finding 100 documents does not mean that we found the correct 100 documents.

Consider:

\[
q=[0.1,\ 0.9].
\]

The ideal binary pattern is:

`11`

and suppose the only documents are:

`10`

and:

`01`.

Their scores are:

\[
S(10)=-0.8,
\]

\[
S(01)=0.8.
\]

If we search prefix:

`1*`

we find `10`.

However, the better document is `01`, which is outside that prefix.

So even though we found one candidate, we did not find the best result.

That is why stopping after “enough documents” is only an approximate rule.

C is exact locally only when:

1. we broaden all the way to depth zero and expose every document inside the opened clusters; or
2. the score bound proves that none of the unseen documents can beat the current top-\(K\).

> **TL;DR — C recall:**  
> A large prefix range does not automatically mean high recall. Backward Walk can safely stop only when it has either searched every document or mathematically proved that the unseen documents cannot win.

---

# Summary

The whole calculation section can be summarized with one simple idea:

| Method | What it does |
|---|---|
| A | Score every document inside the opened clusters |
| B | Spend Bitplane work to avoid some document scoring |
| C | Spend prefix-lookup work to avoid some document scoring |

So:

> **A just pays for document scoring. B and C deliberately add extra search work, hoping that this lets them avoid enough document scoring to make the total query faster.**

The experiment in the next section will tell us whether that actually happens on real embeddings.