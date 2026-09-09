# Why the bitplanes can help

A document stores one sign per coordinate. The stored bit is 0 or 1; its value in the score is
-1 or +1. The query keeps its floating-point values.

```text
score = sum(query[i] * document_sign[i])
      = sum(abs(query)) - 2 * mismatch_penalty

mismatch_penalty = sum(abs(query[i]) for signs that disagree)
```

So the first question is not just “how many signs disagree?” A mismatch on a large query
coordinate costs more than one on a small coordinate.

## A small example

Query: `[0.7, 0.5, -0.4, -0.6, 0.3]`. Its preferred signs are `11001`.

| Document | Mismatch penalty | Score |
|---|---:|---:|
| `01001` | 0.7 | 1.1 |
| `00100` | 1.9 | -1.3 |
| `10100` | 1.2 | 0.1 |

If we insist on matching the first bit, only `10100` remains. But `01001` has the better total
score. We need a way to come back to the first bit's other branch.

## Rows and bitplanes

Row storage keeps all bits of one document together. A bitplane turns that around: it stores
one coordinate's sign across the documents in a bucket.

```text
             doc A  doc B  doc C
coordinate 0    0      0      1
coordinate 1    1      0      0
coordinate 2    0      1      1
```

A mask says which documents are still in a node. AND it with a coordinate's bitplane to keep
positive signs. AND it with the complemented bitplane to keep negative signs. The mask already
excludes unused positions at the end of a word.

Within a packed document code, checking selected coordinates is `(document XOR query_signs) AND
required_coordinates`. A zero result means those coordinates match. That coordinate mask is
different from the document-membership masks used by the tree.

## Walking the branches

1. Sort coordinates by decreasing query magnitude. The example order is 1, 4, 2, 3, 5.
2. Start one node per selected bucket and put them in a shared queue.
3. Take a node and split it on the next coordinate. The matching child adds no penalty; the
   other child adds the coordinate's absolute query value. Drop empty children.
4. Usually take the queued node with the smallest accumulated penalty. Optional random
   exploration can choose another queued node instead.
5. Once a node is small, score its rows and keep the best candidates. Also score a node when
   every coordinate has been visited: identical codes can leave a large group at that depth.
6. Stop at the node budget, or when no pending branch can improve the current candidates.

For a node with penalty `P`, its best possible score is `sum(abs(query)) - 2*P`: that assumes
every remaining coordinate matches. If even that bound cannot beat the worst kept candidate,
the branch can be skipped. Equal-score ties are handled by row ID, so equal bounds need care.

Random scheduling does not change this bound. We check the best bound in the queue, not just
the bound of whichever random node happened to be chosen. The code leaves a small conservative
rounding margin between bound arithmetic and lookup-table scoring.

This gives exact binary candidates inside the selected buckets when the budget is unlimited.
It says nothing about documents in unsearched buckets, or about whether binary ranking agrees
with full-vector ranking. Those are measured separately.

## What it costs

A bucket with `m` documents uses roughly `ceil(m/64)` words per bitplane. Branching also costs
queue work, mask allocation/copying, counting set bits and scoring leaf rows. A tight contiguous
scan can still win, especially on small buckets or when many branches must be explored.

With 17 routing bits there are 131,072 possible addresses. One billion documents would average
about 7,629 per address only if occupancy were uniform. Learned embedding signs need not behave
that way. One billion 256-bit codes occupy 32 GB before other index structures and float vectors.
