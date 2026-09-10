# Cluster-count boundary check

This stage adds 512, 2,048 and 4,096 clusters with denser probe counts, using the same 100,000
documents and 100 tuning queries as the preceding refinement. It measured 127 routing choices,
then 2,002 local-search configurations on 77 qualifying choices. All completed.

Scan remained the fastest measured method at each target in this grid. Larger cluster counts
improved the 90% and 95% scan choices; the earlier 1,024-cluster choices remained faster at 80%
and 99%. This closes the immediate upper-boundary check at 4,096 clusters. It does not prove a
global optimum over every possible architecture or parameter value.

Read `search/comparison/README.md` and the full settings CSV there. The selected settings still
need repeated timings and independent evaluation queries. Neither the full study nor the paper
is complete at this checkpoint.

Raw case folders are preserved locally and packaged in verified `cases.zip` files under
`routing/` and `search/`. Extract those archives in their respective directories to reproduce
analysis from a fresh checkout. The compact tables, configurations and archive manifests remain
available without extracting the raw observations.
