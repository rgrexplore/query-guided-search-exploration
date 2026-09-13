# Reading the figures

1,000,000 documents · 1,000 queries · 64 bits

Each curve chooses the fastest tested configuration that reaches the required recall.
A step means the best qualifying choice changed. It does not interpolate between experiments.
Each method chooses its own clusters, probes and local settings, with the same inputs and RAM limit.
A gap means no completed, qualifying setting was found there. A missing point is not a measured loss.
Query time includes routing and native search. Query embeddings are cached.

The comparison CSV uses configurations chosen by the first sweep, measured again in three process blocks.
The minimum and maximum process medians show repeat variation; they are not confidence intervals.
Logical index bytes count stored fields. Process peak includes runtime and index construction.
