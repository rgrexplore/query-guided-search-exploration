# Follow one query

Build the native extension using the root README, then run:

```bash
python examples/small_search.py
```

The script has six five-bit documents and one query. It prints the stored bits,
cluster assignments, full scores and correct top two. Then it runs all three
methods and prints how much work each performed.

A positive query coordinate prefers a stored 1. A negative coordinate prefers a
stored 0. For example, the query's fourth coordinate is -0.6, so matching its
preferred 0 contributes +0.6 to the score. A stored 1 contributes -0.6.

Read in this order:

1. `document_bits` and `query`: the inputs.
2. `assignments` and `selected_clusters`: the shared routing step.
3. `full_scores`: an independent calculation of the answer.
4. The three search calls: exact search inside the same selected clusters.
5. Printed counters: full scores, bitmap-word visits and directory lookups.

Both clusters are opened in this example. It isolates the local search algorithms;
it does not demonstrate approximate routing quality or predict large-corpus latency.

For actual measurements, follow `experiments/README.md`. The paper is in
`reports/search-math/`, with implementation references in the measured-study section.
