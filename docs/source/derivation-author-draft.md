I’d make Section 3 answer the question more directly: **what has to be true for B or C to beat A?** The math then naturally tells us what parameter ranges to test.

Here’s a draft in the same style as the earlier sections.

---

# 3 When can the new methods outperform the original?

From the previous section, we already have an approximate time model for each method.

Instead of optimizing each formula separately, we can now ask a more useful question:

> **Under what conditions can Method B or Method C actually be faster than Method A?**

For the comparison to be meaningful, the methods still need to satisfy the same recall requirement and fit within the same 32 GB RAM limit.

The math in this section will not directly give us the final parameters. Some quantities, such as how many Bitplane branches are visited or how many good documents exist inside a prefix, depend on the actual embeddings and queries. The goal here is instead to derive the **break-even conditions** and use them to decide which settings are worth testing experimentally. This follows the same principle as the existing report: operation counts can guide the search, but actual recall and latency still need to be measured. 

For the derivations below, assume approximately balanced clusters and define

$$
n=\frac{N}{C}
$$

as the average number of documents in one cluster.

Since all three methods eventually use the same complete binary score, we will use

$$
a
$$

as the approximate cost of fully scoring one document. If the actual implementations have different row-access costs, the experiments can keep separate measured values \(a_A,a_B,a_C\).

---

## 3.1 First, find the best original method

Method A gives us the baseline that B and C need to beat.

From Section 2,

$$
T_A
\approx
T_0+
\frac{PN}{C}a.
$$

For a fixed cluster count \(C\), opening fewer clusters is always cheaper, but it may reduce recall.

Therefore, for a target recall \(\rho\), the first thing we want is the smallest probe count that reaches that recall:

$$
\boxed{
P_A(C,\rho)
=
\min\{P:R_A(C,P)\ge\rho\}.
}
$$

For example, if a reference result belongs to the fifth cluster in the query's routing order, it is only recovered when

$$
P\ge5.
$$

By recording this cluster rank for every reference result, we can calculate the smallest \(P\) required for 80%, 90%, 95%, 99%, or any other target before timing the search. This is also how the current experiment report chooses probe cutoffs. 

We then compare

$$
T_A(C,P_A(C,\rho))
$$

across different values of \(C\).

### A simple intuition for \(C\)

Suppose routing over \(C\) clusters costs approximately

$$
\alpha C,
$$

and scoring each selected row costs \(\beta\).

If \(P\) were fixed,

$$
T_A(C)
\approx
\alpha C
+
\beta\frac{NP}{C}.
$$

Differentiating,

$$
\frac{dT_A}{dC}
=
\alpha-
\frac{\beta NP}{C^2}.
$$

Setting this to zero gives

$$
\boxed{
C^*
=
\sqrt{\frac{\beta NP}{\alpha}}.
}
$$

This gives the intuitive balance:

* too few clusters \(\rightarrow\) each cluster contains too many documents;
* too many clusters \(\rightarrow\) routing itself becomes more expensive.

However, this is only a guide. In reality, \(P\) is not fixed: increasing \(C\) may require probing more clusters to maintain the same recall.

So the final baseline is simply:

$$
\boxed{
T_A^*(\rho)
=
\min_{C,P}
T_A(C,P)
\quad
\text{subject to }
R_A(C,P)\ge\rho.
}
$$

This is the number that B and C ultimately need to beat.

### Takeaway

For Method A, the main problem is:

> **Find the smallest set of routed documents that still contains enough of the correct results.**

---

# 3.2 When can Bitplanes beat the original method?

To first understand the local search itself, assume A and B use the same \(C\) and \(P\).

Method A scores approximately

$$
Pn
$$

documents, so ignoring shared work,

$$
T_A-T_0
\approx
Pna.
$$

For Bitplanes, let

* \(S\) = average number of splits per opened cluster;
* \(b\) = time for one split-word visit;
* \(f_B\) = fraction of Method A's selected documents that B eventually fully scores.

The bitmap width of one cluster is approximately

$$
W
\approx
\frac{n}{64}.
$$

Therefore,

$$
T_B-T_0
\approx
PS\frac{n}{64}b
+
f_BPna
+
T_{\text{other},B}.
$$

Bitplanes are faster when

$$
T_B<T_A.
$$

Substituting the two expressions gives

$$
PS\frac{n}{64}b
+
f_BPna
+
T_{\text{other},B}
<
Pna.
$$

Move the scoring term to the other side:

$$
(1-f_B)Pna
>
PS\frac{n}{64}b
+
T_{\text{other},B}.
$$

Dividing by \(Pn\),

$$
\boxed{
(1-f_B)a
>
\frac{Sb}{64}
+
\frac{T_{\text{other},B}}{Pn}.
}
$$

This is the main break-even condition for Bitplanes.

The left-hand side is the **scoring time that B saves**.

The right-hand side is the **additional cost introduced by Bitplanes**.

Equivalently, B needs its surviving fraction to satisfy

$$
\boxed{
f_B
<
1-
\frac{Sb}{64a}
-
\frac{T_{\text{other},B}}{Pna}.
}
$$

So if the right-hand side evaluates to \(0.6\), for example, B would need to fully score less than roughly 60% of A's documents before the filtering starts paying for itself.

This is why simply showing that B scores fewer documents is not enough. The detailed study already found cases where scoring dropped substantially, but hundreds of millions of bitmap-word visits made the overall search slower. 

---

## 3.2.1 How many splits should we make?

We can get another useful estimate by making one simple assumption:

> each successful split halves the remaining candidate set.

After \(j\) splits,

$$
f_B(j)
\approx
2^{-j}.
$$

Ignoring queue and other branch costs for the moment,

$$
T_B(j)-T_0
\approx
P
\left(
j\frac{n}{64}b
+
n2^{-j}a
\right).
$$

Factor out \(Pn\):

$$
T_B(j)-T_0
\approx
Pn
\left(
\frac{jb}{64}
+
2^{-j}a
\right).
$$

Differentiate with respect to \(j\):

$$
\frac{dT_B}{dj}
=
Pn
\left(
\frac{b}{64}
-
a\ln(2)2^{-j}
\right).
$$

Set the derivative to zero:

$$
\frac{b}{64}
=
a\ln(2)2^{-j}.
$$

Therefore,

$$
\boxed{
j^*
=
\log_2
\left(
\frac{64a\ln(2)}{b}
\right).
}
$$

This is quite interesting because \(n\) disappears.

Under this simplified model, the ideal split depth is mainly controlled by the ratio

$$
\frac{\text{cost of fully scoring one document}}
{\text{cost of processing one bitmap word}}.
$$

If document scoring is expensive relative to bit operations, more splitting can be worthwhile.

If scoring is already extremely cheap, splitting quickly becomes unnecessary overhead.

Of course, real Bitplanes do not necessarily halve the candidate set after every split. Some bitplanes may be highly correlated, one branch may contain almost every document, and alternative branches also add work. So \(j^*\) is a starting point for the experiment, not a predicted final answer. 

---

## 3.2.2 Does B require smaller clusters?

At first, it may seem like Bitplanes should always use smaller clusters because smaller clusters produce narrower masks.

However, the simplified equation does not actually give that conclusion.

The scan cost per cluster is proportional to

$$
n,
$$

while the bitmap split cost is proportional to

$$
\frac{n}{64}.
$$

Both shrink roughly linearly when \(n\) becomes smaller.

This is why \(n\) largely disappears from the break-even equation above.

So from the simple model alone:

> **Bitplanes do not automatically prefer smaller clusters.**

Smaller clusters may still help in practice because they can change:

* the number of splits \(S\);
* the number of clusters that must be probed \(P\);
* branch and mask overhead;
* cache behaviour;
* routing recall.

Those effects need to be measured.

### Takeaway

For Bitplanes, the key condition is:

$$
\boxed{
\text{time saved from skipped scores}
>
\text{bitmap + branching overhead}.
}
$$

The experiment should therefore record not only latency, but also:

$$
S,\qquad f_B,\qquad
\text{split-word visits},\qquad
\text{rows fully scored}.
$$

---

# 3.3 When can Backward Walk beat the original method?

We can do the same comparison for the prefix-relaxation version of Backward Walk.

Again, assume the same \(C\) and \(P\) initially.

Method A costs approximately

$$
T_A-T_0
\approx
Pna.
$$

From Section 2, Backward Walk costs approximately

$$
T_C-T_0
\approx
2PL\log_2(n+1)t_b
+
F_Ca
+
T_{\text{ranges}},
$$

where

* \(L\) is the number of prefix depths visited;
* \(t_b\) is the time per key comparison;
* \(F_C\) is the number of distinct documents fully scored.

Define

$$
f_C
=
\frac{F_C}{Pn}.
$$

Then

$$
F_C=f_CPn,
$$

so

$$
T_C-T_0
\approx
2PL\log_2(n+1)t_b
+
f_CPna
+
T_{\text{ranges}}.
$$

Backward Walk beats A when

$$
T_C<T_A.
$$

Therefore,

$$
2PL\log_2(n+1)t_b
+
f_CPna
+
T_{\text{ranges}}
<
Pna.
$$

Rearranging,

$$
(1-f_C)Pna
>
2PL\log_2(n+1)t_b
+
T_{\text{ranges}}.
$$

Dividing by \(P\),

$$
\boxed{
(1-f_C)na
>
2L\log_2(n+1)t_b
+
\frac{T_{\text{ranges}}}{P}.
}
$$

Or equivalently,

$$
\boxed{
f_C
<
1-
\frac{
2L\log_2(n+1)t_b+
T_{\text{ranges}}/P
}{
na
}.
}
$$

Again, the meaning is simple.

The left-hand side of the original inequality represents the **document scoring avoided**.

The right-hand side represents the **cost of walking through the prefix hierarchy**.

The prefix-relaxation version already has one useful property: when ranges are nested, documents seen at a deeper prefix do not need to be scored again when moving to the parent. Only the newly exposed sibling ranges need to be processed. 

---

## 3.3.1 Effect of cluster size

There is an interesting difference from Bitplanes.

For Method A, the work inside one selected cluster grows approximately as

$$
O(n).
$$

For the sorted-key Backward Walk lookup, the navigation component grows approximately as

$$
O(\log n).
$$

So as \(n\) becomes larger,

$$
\frac{\log n}{n}
$$

becomes smaller.

This means that, at least from the lookup-cost perspective, Backward Walk can become more attractive when the candidate pool is large.

This does **not** mean larger clusters are automatically better overall. A larger cluster also contains more documents behind each prefix, and routing recall still changes with \(C\) and \(P\).

But unlike the simplified Bitplane model, C has a clearer asymmetry:

> **the scan it is trying to avoid grows linearly, while the prefix lookup itself grows much more slowly.**

Whether this becomes an actual speedup depends mainly on whether we can stop while

$$
F_C\ll Pn.
$$

---

# 3.3.2 What prefix depth should we start from?

We can use the simple balanced-bit model to get an initial estimate.

Suppose an \(\ell\)-bit prefix occurs with probability

$$
2^{-\ell}.
$$

Across \(P\) clusters containing approximately \(n\) documents each,

$$
E[F_\ell]
\approx
\frac{Pn}{2^\ell}.
$$

Suppose we would like to expose roughly \(M\) candidates.

Set

$$
\frac{Pn}{2^\ell}
\approx
M.
$$

Then

$$
2^\ell
\approx
\frac{Pn}{M},
$$

giving

$$
\boxed{
\ell
\approx
\log_2
\left(
\frac{Pn}{M}
\right).
}
$$

For example, with one million documents in one pool and a rough target of 100 candidates,

$$
\ell
\approx
\log_2
\left(
\frac{1,000,000}{100}
\right)
\approx
13.3.
$$

So prefix lengths around 13–14 would be reasonable places to investigate first under this toy model.

We can start slightly deeper than this and allow Backward Walk to relax until enough candidates are found.

The important limitation is that

$$
\frac{Pn}{2^\ell}
$$

only predicts **bucket occupancy** under independent and balanced bits.

It does not tell us that those documents contain the true top-\(K\) results.

The detailed analysis shows that prefix candidate count and recall are different quantities, and that dependent bits can make the independent-bit prediction inaccurate. 

So the experiment still needs to measure

$$
R_C(\ell)
$$

directly from the returned document IDs.

### Takeaway

For Backward Walk, the key condition is

$$
\boxed{
\text{time saved from avoiding a large scan}
>
\text{prefix-navigation + range overhead}.
}
$$

The main quantities to record are

$$
L,\qquad F_C,\qquad
\text{final prefix depth},\qquad
\text{recall}.
$$

---

# 3.4 From the math to the experiment

The equations above tell us what needs to happen for B or C to win, but they do not give every parameter directly.

Some important quantities are data-dependent:

$$
S,\qquad f_B,\qquad F_C,\qquad
R_B,\qquad R_C.
$$

We therefore use the math to choose sensible settings, then measure these quantities on the actual dataset.

The resulting parameter search can be summarized as:

| Method | Main settings to vary                         | What we are looking for                                             |
| ------ | --------------------------------------------- | ------------------------------------------------------------------- |
| **A**  | \(C,P\)                                       | Smallest selected pool that still reaches the recall target         |
| **B**  | \(C,P,\) leaf size, node budget               | Enough skipped scores to pay for bitmap traversal                   |
| **C**  | \(C,P,\) starting prefix depth, stopping rule | Small prefix candidate set while retaining enough reference results |

For every method \(m\), let

$$
\theta_m
$$

represent one complete configuration.

The final configuration is

$$
\boxed{
\theta_m^*
=
\arg\min_{\theta_m}
T_m(\theta_m)
}
$$

subject to

$$
\boxed{
R_m(\theta_m)\ge\rho
}
$$

and

$$
\boxed{
M_m(\theta_m)\le32\text{ GB}.
}
$$

The final comparison is therefore

$$
T_A(\theta_A^*),
\qquad
T_B(\theta_B^*),
\qquad
T_C(\theta_C^*).
$$

Method B establishes a speedup only if

$$
T_B(\theta_B^*)<T_A(\theta_A^*)
$$

while satisfying the same recall and memory constraints.

Likewise, Method C establishes a speedup only if

$$
T_C(\theta_C^*)<T_A(\theta_A^*).
$$

This also means the three methods do **not** need to use the same cluster count or local parameters in the final comparison. Each method is allowed to choose the configuration that works best for it, as long as the dataset, queries, reference results, recall target, and RAM limit remain unchanged. That matches the comparison principle already used in the current report. 

---

## 3.5 Summary of the conditions

| Method            | Main break-even idea                                                |
| ----------------- | ------------------------------------------------------------------- |
| **Original**      | Minimize the number of routed documents while keeping enough recall |
| **Bitplanes**     | Saved full-score work must exceed bitmap and branch work            |
| **Backward Walk** | Saved scan work must exceed prefix-navigation and range work        |

For B:

$$
\boxed{
(1-f_B)a
>
\frac{Sb}{64}
+
\frac{T_{\text{other},B}}{Pn}
}
$$

For C:

$$
\boxed{
(1-f_C)na
>
2L\log_2(n+1)t_b
+
\frac{T_{\text{ranges}}}{P}
}
$$

These are the two main conditions we will check in the experiments.

The next section therefore does not just ask *which method has the lowest time*. It also records **why**: how many documents were actually avoided, and how much extra index work was required to avoid them.
