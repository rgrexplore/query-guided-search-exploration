Yes. I think the experiment section should be organized around the **result the reader actually wants to see**, not around the implementation details.

The flow should be:

> **What did we test? → what is the fastest valid configuration for each method? → who won? → why did it win/lose? → does that match the math from Section 3?**

I would write it roughly like this.

---

# 4 Experiments

From Section 3, we now have a set of conditions that should tell us when Method B or Method C can outperform Method A.

Now we can test whether those conditions actually happen on the real data.

For every method, we allow it to choose its own parameters. The documents, queries, scoring function, reference results, recall targets and RAM limit remain unchanged.

The main result we care about is simple:

> **At the same recall requirement, what is the fastest configuration we can find for each method?**

After that, we can look at the internal work counters to understand **why** one method is faster than another.

---

# 4.1 Setup

Use the same setup throughout:

$$
N=1,000,000
$$

documents,

$$
d=256
$$

binary document bits,

$$
K=100
$$

returned results, and the same 200 query vectors.

For every query, we compute the exact top-100 under the complete 256-bit binary score. These document IDs are kept fixed as the reference answer for all three methods.

A method reaches 99% recall if, on average, it returns 99 of those 100 reference IDs.

The retrieval timer includes:

```text
cluster selection → local search → top-K result
```

but does not include document embedding generation or index construction.

Each process must remain within the 32 GB RAM requirement.

### Diagram

I would have **one fairness diagram** here.

```text
                     SAME INPUT

              1,000,000 documents
                   200 queries
                       │
         ┌─────────────┼─────────────┐
         │             │             │
      Method A      Method B      Method C
       Scan         Bitplanes    Backward Walk
         │             │             │
         └─────────────┼─────────────┘
                       │
               Same exact Top-100
                       │
               latency + recall
```

Color:

* A = blue
* B = orange
* C = purple
* shared input/evaluation = gray

Below the figure:

> All methods use the same documents, queries, 256-bit score and reference results. Only the search/index strategy and its parameters change.

The purpose is just to remove any doubt that one method is being evaluated on an easier task. This is already one of the stronger parts of the current benchmark design. 

---

# 4.2 Parameter search

Before comparing the methods, we first search over their own parameter spaces.

For A:

$$
(C,P)
$$

For B:

$$
(C,P,\text{leaf size},\text{node budget})
$$

For C:

$$
(C,P,x,M)
$$

where \(x\) is the starting prefix depth and \(M\) is the candidate target or equivalent stopping parameter.

For every configuration, record:

* recall;
* median query latency;
* peak RAM;
* method-specific work counters.

Then for each recall requirement

$$
\rho\in\{80\%,90\%,95\%,99\%\},
$$

select the **fastest configuration whose measured recall is at least \(\rho\)**.

I would keep the parameter sweep details mostly out of the main text. Put the complete grid in an appendix or reproduction section.

---

# 4.3 Main result: which method is fastest?

This should be the **first major result table**.

## Table 1 — Best configuration at each recall target

I would use rows as recall targets:

| Required recall | A time | B time | C time | Fastest |
| --------------- | -----: | -----: | -----: | ------- |
| 80%             |      … |      … |      … | …       |
| 90%             |      … |      … |      … | …       |
| 95%             |      … |      … |      … | …       |
| 99%             |      … |      … |      … | …       |

Under each timing, you can put the achieved recall in smaller text if LaTeX allows it:

| Required        |           A |       B |       C |
| --------------- | ----------: | ------: | ------: |
| 95%             | **2.97 ms** | 3.07 ms | 3.26 ms |
| achieved recall |      95.00% |  95.00% |  95.00% |

But personally I prefer one compact cell:

```text
2.97 ms
95.00%
```

### Highlighting

* **Bold** the fastest valid method in each row.
* Do not highlight merely because a method has lower latency if it **fails recall**.
* If a configuration does not meet the requirement, show something like:

```text
2.1 ms
93.7% ✗
```

and gray the cell.

This is important because otherwise a very approximate configuration can look like the winner.

---

### Main figure — latency vs recall

Right after the table, add the line plot:

* x-axis: required recall: 80, 90, 95, 99%
* y-axis: median retrieval latency in ms
* A: blue circles
* B: orange squares
* C: purple triangles

This is basically the same type of plot already used in your current report. 

The reason to have **both** the table and graph:

* table gives exact numbers;
* graph immediately shows the trend.

For example, perhaps B is slightly faster at 80–90% but loses at 99%. The graph makes that pattern obvious.

---

# 4.4 What parameters produced those results?

Then show the winning settings.

## Table 2 — Selected configuration

Rows should now be the methods:

| Method | \(C\) | \(P\) | Local setting              | Actual recall | Median ms | Peak RAM |
| ------ | ----: | ----: | -------------------------- | ------------: | --------: | -------: |
| A      |     … |     … | —                          |             … |         … |        … |
| B      |     … |     … | \(L=...\), budget \(=...\) |             … |         … |        … |
| C      |     … |     … | \(x=...\), \(M=...\)       |             … |         … |        … |

Do this for the most important operating point, probably **99% recall**.

You do not need one giant table containing every parameter for all four recall targets. That gets unreadable.

If needed, put the complete settings for 80/90/95/99% in an appendix.

The main text should show the 99% result because that is usually the hardest and most interesting target.

---

# 4.5 Why did it win or lose?

This is probably the most important improvement over the current section.

The latency table tells us **what happened**.

Now we connect it directly to Section 3 and explain **why**.

## Table 3 — Actual work at the selected setting

Rows are the methods:

| Method | Documents scored | Bitmap words | Prefix depths | Range lookups | Extra work |
| ------ | ---------------: | -----------: | ------------: | ------------: | ---------- |
| A      |          \(F_A\) |            — |             — |             — | —          |
| B      |          \(F_B\) |            … |             — |             — | … branches |
| C      |          \(F_C\) |            — |         \(L\) |             … | …          |

For B, you could make the columns more specific:

* rows fully scored;
* split-word visits;
* leaf-word visits;
* branch nodes visited.

For C:

* rows fully scored;
* starting depth;
* final depth;
* depths visited;
* range lookups.

Since the operations are different, we **should not add them together as if one bitmap word = one full score = one range lookup**. The point is to expose the work, not turn everything into a fake common unit. That distinction is already emphasized in the detailed report. 

---

## Example interpretation

Suppose B loses.

You may get something like:

```text
A:
295k documents scored

B:
120k documents scored
but
2.4 million bitmap-word visits
```

Then the explanation becomes:

> B reduced the number of full scores by approximately 59%. However, achieving this reduction required 2.4 million bitmap-word visits and additional branch management. The saved scoring work was therefore smaller than the additional Bitplane work, so the break-even condition from Section 3 was not reached.

That is much stronger than:

> “B took 9.1 ms and A took 8.7 ms.”

---

Suppose C wins.

Maybe:

```text
A:
295k rows scored

C:
18k rows scored
5 prefix depths visited
10 range lookups
```

Then:

> Backward Walk exposed only around 6% of the rows that A would have scanned. The cost of five prefix relaxations and their range lookups was much smaller than the cost of scoring the avoided documents. In this setting, C crossed the break-even point predicted in Section 3.

That's exactly the story your paper should tell.

---

# 4.6 Directly check the break-even equations

I think this deserves its own small table.

Section 3 gave us:

For B,

$$
\text{saved scoring time}
>
\text{Bitplane overhead}.
$$

For C,

$$
\text{saved scoring time}
>
\text{prefix-walk overhead}.
$$

So measure those terms.

## Table 4 — Break-even check

| Method | Estimated scoring saved | Extra search overhead | Net effect | Condition met? |
| ------ | ----------------------: | --------------------: | ---------: | -------------- |
| B      |                    … ms |                  … ms |       … ms | Yes / No       |
| C      |                    … ms |                  … ms |       … ms | Yes / No       |

For example:

```text
B
scores avoided:     3.2 ms
bitmap + queues:    4.0 ms
net:                -0.8 ms
condition:          NO
```

```text
C
scores avoided:     5.8 ms
prefix overhead:    1.1 ms
net:                +4.7 ms
condition:          YES
```

This table directly closes the loop with the math section.

---

# 4.7 Time breakdown

If you can instrument this cleanly, I would add one more figure.

### Stacked bar chart at 99% recall

Three bars:

```text
        A              B              C

   ┌────────┐     ┌────────┐     ┌────────┐
   │scoring │     │scoring │     │scoring │
   │        │     ├────────┤     ├────────┤
   │        │     │bitmap  │     │prefix  │
   ├────────┤     ├────────┤     ├────────┤
   │routing │     │routing │     │routing │
   └────────┘     └────────┘     └────────┘
```

Keep each method's outline color:

* A blue
* B orange
* C purple

But the internal components can use lighter/darker shades of the same method color.

For B:

```text
routing
bit ordering
bitmap traversal
document scoring
top-K / other
```

For C:

```text
routing
prefix lookup
range expansion
document scoring
top-K / other
```

### Why this figure matters

This will immediately show a result like:

> B scores dramatically fewer documents, but its orange “bitmap traversal” region is larger than the scoring it saved.

or:

> C adds only a thin prefix-lookup component while removing most of the scoring bar.

It is much easier to understand visually than the formulas alone.

Only do this if the stage timers can be measured without double-counting. The detailed report already warns against adding costs that are already included inside another measured stage. 

---

# 4.8 Parameter sensitivity

Then I would show **why the selected parameters are sensible**, rather than just saying “the sweep found them.”

## For B

A heatmap:

* x-axis = leaf size
* y-axis = node budget
* cell = latency
* maybe one heatmap per cluster count

Mark configurations that fail recall with an `×` or gray them out.

For example:

```text
                 Leaf size
            32   128   512   scan
Budget 256   ×    ×    4.2   3.7
      1024   ×   3.8   3.5   3.7
      4096  4.9  3.6   3.5   3.7
```

**Purpose:** show the tradeoff from Section 3:

small leaves → fewer scores but more bitmap work.

---

## For C

Similar heatmap:

* x-axis = starting prefix depth \(x\)
* y-axis = candidate target \(M\)
* cell = latency
* gray out settings below recall target.

Maybe:

```text
                 Prefix depth
            12    14    16    20
M = 100     ...   ...    ×     ×
M = 500     ...   ...   ...    ×
M = 1000    ...   ...   ...   ...
```

**Purpose:** show:

* very shallow prefix → too many candidates;
* very deep prefix → many relaxations / poor recall;
* somewhere between them is the useful region.

This directly demonstrates the tradeoff you derived mathematically.

---

# 4.9 Memory

Then a simple table.

## Table 5 — Memory

| Method | Base rows |    Extra index | Query scratch | Peak process RAM | <32 GB? |
| ------ | --------: | -------------: | ------------: | ---------------: | :-----: |
| A      |         … |              — |             … |                … |    ✓    |
| B      |         … |    bitplanes … |       masks … |                … |    ✓    |
| C      |         … | prefix index … |      ranges … |                … |    ✓    |

No big discussion unless memory actually becomes the limiting factor.

If B is much larger because it stores the additional transposed bitplanes, just state it.

---

# 4.10 Final result

Then end the experiment section with a very compact conclusion.

The structure depends on what actually happens.

### If A wins everywhere

Write:

> Across the tested recall targets, the original method remained the fastest qualifying configuration. B and C both reduced full scoring in some settings, but the work required to obtain that reduction was larger than the scoring time saved. Neither new method crossed its break-even condition on this dataset.

That is essentially what the current fixed-data experiment found for the current implementations. 

---

### If B wins in some region

Write something like:

> Bitplanes were faster than A at the 80% and 90% recall requirements, where aggressive pruning reduced the scored document count enough to compensate for bitmap traversal. At 95% and 99%, the required traversal became deeper and the bitmap overhead exceeded the saved scoring time, causing A to become faster again.

Then point at the work table.

---

### If C wins in some region

Something like:

> Backward Walk became faster once its prefix index could isolate a sufficiently small candidate set without requiring many relaxation steps. At 99% recall, the selected configuration scored only \(F_C\) documents compared with \(F_A\) for A, while requiring only \(L\) prefix levels. The resulting score reduction exceeded the prefix-navigation overhead predicted in Section 3.

---

### If each method wins somewhere

That would actually be the most interesting result:

> No single method dominated every operating point. A was strongest when high recall required searching a broad region, B performed best when a few query-dependent splits removed many documents, and C performed best when a sufficiently deep Matryoshka prefix isolated a small candidate set.

Then your conclusion becomes **“different methods correspond to different workload regimes”**, rather than trying to declare one universally best algorithm.

---

## I would organize the results in exactly this order

1. **Table 1:** Who is fastest at each recall target?
2. **Figure 1:** Latency vs recall.
3. **Table 2:** What parameters did each winner use?
4. **Table 3:** What work did each method actually perform?
5. **Table 4:** Did B/C cross their mathematical break-even condition?
6. **Figure 2:** Time breakdown at 99%.
7. **Parameter sensitivity plots:** why that configuration was selected.
8. **Table 5:** Memory.
9. **Short conclusion.**

That way the reader gets the answer **immediately**, and then every following table explains the answer in more depth.

The current report already has pieces of this—especially the latency-vs-recall figure, selected settings table, and work-counter table—but reorganizing them this way would connect the experiment much more directly to the mathematical question from Section 3. 
