Yes. I’d make the experiment section feel like the natural continuation of Section 3:

* Section 2: **what does each method cost?**
* Section 3: **what needs to happen for B/C to beat A?**
* Section 4: **now measure whether those conditions actually happen.**

I would also avoid making it sound like we already expect B or C to win. The experiment should allow outcomes like “B wins only at 80% recall,” “C wins only for large clusters,” or “neither wins.”

Here’s how I’d draft it.

---

# 4 Experiments

From the previous section, we now have a rough idea of what needs to happen for each new method to outperform the original method.

For Bitplanes, the number of full scores avoided needs to be large enough to compensate for the additional bitmap and branching work.

For Backward Walk, the prefix should narrow the candidate set enough that the saved document scoring is larger than the cost of walking through and expanding the prefix ranges.

The purpose of the experiments is therefore to check whether these conditions actually happen on real embeddings.

Rather than forcing all three methods to use the same parameters, each method is allowed to choose its own best configuration. The only things that remain fixed are the documents, queries, score definition, reference results, recall requirement and memory limit.

---

## 4.1 Experimental setup

We use the same collection of

$$
N=1,000,000
$$

MS MARCO document embeddings and the same 200 query embeddings throughout the experiment.

The embeddings are produced using Nomic, and the documents are represented using

$$
d=256
$$

binary sign bits. The query remains floating point and uses the same 256 dimensions for the final binary score. 

For every query, we first compute the exact top

$$
K=100
$$

documents using the complete 256-bit score.

These IDs form the fixed reference result:

$$
G_q.
$$

All three methods are evaluated against exactly the same \(G_q\).

For example, if a method returns 99 of the 100 reference IDs,

$$
Recall@100=99\%.
$$

This means that a lower query time is only considered useful if the method still reaches the required recall.

The experiments use cached embeddings, so the measured retrieval time begins from the search stage rather than including neural-network encoding. This keeps the comparison focused on the index structures being studied.

---

### Diagram

I would put one simple diagram here that makes the fairness of the experiment obvious:

```text
                       SAME INPUT

               1M binary documents
                    200 queries
                         |
          --------------------------------
          |              |               |
          A              B               C
       Original       Bitplanes      Backward Walk
          |              |               |
          --------------------------------
                         |
               SAME reference top-100
                         |
             Compare latency / recall
```

Use:

* **blue** for A,
* **orange** for B,
* **purple** for C,
* black/gray for the shared input and evaluation.

Underneath, write:

```text
Same documents
Same queries
Same 256-bit score
Same Top-100 reference
Same RAM limit
```

The purpose of this diagram is to make clear that we are **not changing the representation or the evaluation target to make one method look better**. Only the index/search strategy changes.

---

## 4.2 What parameters do we vary?

Each method has different parameters that control its tradeoff between latency and recall.

We therefore tune each method separately.

### Method A — Original

For Method A, the main parameters are:

$$
C=\text{number of clusters}
$$

and

$$
P=\text{number of clusters probed}.
$$

For each \(C\), we find probe counts corresponding approximately to the required recall targets.

For example:

$$
80\%,\ 90\%,\ 95\%,\ 99\%.
$$

This gives us the fastest tested baseline for each recall level.

---

### Method B — Bitplanes

Bitplanes use the same routing parameters:

$$
C,\ P,
$$

but introduce additional search parameters such as:

$$
L_B=\text{leaf size}
$$

and

$$
B_{\text{node}}=\text{maximum number of visited branch nodes}.
$$

A smaller leaf size means we try to remove more documents before falling back to full scoring.

However, this usually requires more bitmap traversal.

A larger node budget allows more complete search, but again increases the amount of work.

So for B we vary:

$$
(C,P,L_B,B_{\text{node}}).
$$

For every configuration we record not only latency and recall, but also:

* number of split nodes;
* split-word visits;
* leaf-word visits;
* number of documents fully scored.

These are the quantities that appear in the break-even condition from Section 3.

---

### Method C — Backward Walk

For Backward Walk, the main parameters are:

$$
C,\ P,
$$

the starting prefix depth

$$
x,
$$

and the stopping rule.

For example, we may begin at

$$
x=16
$$

and progressively relax:

$$
16\rightarrow15\rightarrow14\rightarrow13\rightarrow\cdots
$$

until a candidate target is reached.

If we use a candidate target \(M\), then the tested configuration can be represented as

$$
(C,P,x,M).
$$

We should test several values of \(x\) around the range suggested by

$$
x\approx
\log_2\left(\frac{Pn}{M}\right).
$$

For each query, we record:

* starting depth;
* final depth;
* number of prefix levels visited \(L\);
* number of distinct documents exposed \(F_C\);
* number of range lookups;
* final recall.

This lets us check whether the mathematical intuition from Section 3 actually holds.

---

## 4.3 Parameter ranges

The math from Section 3 is useful here because it prevents us from testing completely arbitrary settings.

For example, suppose the selected pool contains approximately one million documents and we want roughly 100 initial candidates.

The balanced-bit model gives

$$
x\approx
\log_2\left(\frac{1,000,000}{100}\right)
\approx13.3.
$$

So instead of testing every prefix depth from 1 to 256, we might start around:

$$
x\in\{12,13,14,16,20\}.
$$

Likewise, for Bitplanes, the approximate optimal split depth

$$
j^*
=
\log_2
\left(
\frac{64a\ln2}{b}
\right)
$$

can suggest a reasonable region for the leaf sizes and node budgets.

These calculations do not choose the final configuration. They simply tell us where it is most useful to look.

---

## 4.4 What do we measure?

For every configuration, we record three main things:

### 1. Recall

For query \(q\),

$$
R_q
=
\frac{|G_q\cap O_q|}{K},
$$

where \(G_q\) is the reference top-\(K\) and \(O_q\) is the returned result.

Batch recall is then

$$
R=
\frac{1}{n_q}
\sum_qR_q.
$$

A configuration only qualifies for a target \(\rho\) when

$$
R\ge\rho.
$$

---

### 2. Retrieval latency

We measure the complete retrieval time for that configuration, including routing and its local search stage.

For example:

```text
A:
route → scan → top K

B:
route → bitplane traversal → score leaves → top K

C:
route → prefix lookup / relaxation → score exposed rows → top K
```

The full request should be timed rather than adding the medians of individual stages afterwards.

This is consistent with the measurement approach already used in the current experiments. 

---

### 3. Work counters

Latency tells us who wins.

The work counters tell us **why**.

For A:

$$
F_A=\text{documents scored}.
$$

For B:

$$
S,\quad
F_B,\quad
\text{split-word visits},\quad
\text{leaf-word visits}.
$$

For C:

$$
L,\quad
F_C,\quad
\text{range lookups},\quad
\text{final prefix depth}.
$$

This lets us directly compare the measured result with the break-even equations derived earlier.

---

## 4.5 Choosing the winner

For each recall requirement \(\rho\), we first find the fastest qualifying configuration for each method:

$$
\theta_A^*(\rho),
\qquad
\theta_B^*(\rho),
\qquad
\theta_C^*(\rho).
$$

Then we compare:

$$
T_A^*(\rho),
\qquad
T_B^*(\rho),
\qquad
T_C^*(\rho).
$$

For example, the final result might look something like this:

| Required recall |           A |           B |           C | Fastest |
| --------------- | ----------: | ----------: | ----------: | ------- |
| 80%             |     0.60 ms | **0.48 ms** |     0.55 ms | B       |
| 90%             |     1.40 ms | **1.20 ms** |     1.30 ms | B       |
| 95%             | **3.00 ms** |     3.10 ms |     3.20 ms | A       |
| 99%             |     8.70 ms |     9.00 ms | **7.90 ms** | C       |

**These numbers are only an example of how the result should be interpreted, not measured results.**

A result like this would be much more interesting than simply saying “B is faster” or “A is faster.”

It would tell us:

> Bitplanes may be useful when the recall requirement is lower and aggressive pruning is allowed, while Backward Walk may become useful at another operating point.

Or the experiment may show:

```text
A wins at 80%
A wins at 90%
A wins at 95%
A wins at 99%
```

which would tell us that neither additional index structure pays for its overhead on this dataset.

Both are valid outcomes.

---

## 4.6 Connecting the experiments back to the math

After identifying the fastest configurations, we should check **why** they won or lost.

For Bitplanes, Section 3 predicted that B should win only when

$$
(1-f_B)a
>
\frac{Sb}{64}
+
\frac{T_{\mathrm{other},B}}{Pn}.
$$

So if B loses, we can inspect whether:

* \(f_B\) remained too high, meaning not enough scores were avoided; or
* \(S\) became too large, meaning too much bitmap work was required.

For Backward Walk, the condition was

$$
(1-f_C)na
>
2L\log_2(n+1)t_b
+
\frac{T_{\mathrm{ranges}}}{P}.
$$

So if C wins, we should expect to see a relatively small \(F_C\) and a small number of prefix relaxations.

If C loses, we may find that the prefix had to become too shallow:

```text
16 → 15 → 14 → 13 → ... → 5
```

causing \(F_C\) to approach the number of documents Method A would have scanned anyway.

This is the part that really connects Section 3 and Section 4:

> **The experiment is not only checking which method is faster. It is checking whether the break-even condition predicted for each method actually occurs on the real data.**

---

## 4.7 Final comparison

I would end the setup section with something simple:

> For every recall target, each method is allowed to choose its own best tested parameters. A method is considered better only when it achieves lower measured retrieval latency while meeting the same recall requirement and staying within the same 32 GB RAM limit.
>
> The following results therefore answer two questions:
>
> 1. **Which method is fastest at each recall target?**
> 2. **Do the measured work counts explain the result in the way predicted by Section 3?**

That sets up the actual results section cleanly.

One thing I would definitely preserve from your current experiment design is the use of the **same document/query/reference arrays for every method**. That is one of the strongest parts of the current report because it prevents a speedup from coming from accidentally changing the task rather than changing the search algorithm. 
