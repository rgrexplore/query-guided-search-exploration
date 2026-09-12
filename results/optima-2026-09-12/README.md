# Parameter study, 12 September 2026

The protocol is `docs/optima-study.md`; reproduction commands are in
`reports/search-math-optima/README.md`.

- `fixed/`: 130 tuning settings, then frozen settings on 64 separate queries.
- `adaptive/`: 299 tuning settings, then frozen settings on 64 separate queries.
- `predictions-v1/`: first time model, retained after its large errors were diagnosed.
- `predictions/`: revised frozen model, model-selected settings, and final prediction check.
- `real-replication/`: frozen historical choices repeated on the same 200 real evaluation queries.

All methods use the same score, reference, one CPU thread, and 32 GB decimal RAM requirement.
Each can choose its own router and local settings. The controlled data has twelve strong query
coordinates and tiny remaining weights; it is not a claim about the distribution of text embeddings.

All 429 tuning workers, 60 controlled evaluation workers, and 9 real-replication workers completed.
Conditional exact-count checks passed for 633 tuning and 762 evaluation observations; observations
outside the formulas' conditions remain in the result. These counts include repeated process
blocks and must not be described as independent corpora. The six selected time forecasts met
the unchanged 20% criterion, with maximum observed error below 10%; other fit failures remain.

Raw cases are stored in each stage's verified `cases.zip`. Extract it in that stage directory
to restore the `cases/` tree. Native/source hashes, configurations, query IDs and raw timing
records are retained. The ignored pools can be prepared again; they are not required to inspect
the saved results. No later evaluation result was used to select the frozen settings.
