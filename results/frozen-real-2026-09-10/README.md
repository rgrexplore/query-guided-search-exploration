# Frozen real-data evaluation shortlist

This shortlist was selected before the independent query set was encoded or searched.
It combines 6,821 unique settings from the completed tuning studies, pooling duplicate timings
and averaging randomized-search quality across seeds. It contains 72 distinct settings.

Two predeclared selections are retained for each corpus size, method and recall target:

- Mean: fastest tested setting whose tuning mean recall reaches the target.
- Conservative: fastest tested setting whose lower training-bootstrap percentile reaches it.

The conservative criterion is a selection heuristic, not a population guarantee. Both selections
will be reported separately on the independent queries. A failed mean-target result will not be
hidden by replacing it after evaluation.

`shortlist.json` contains the exact replay inputs and their source cases. `selected.csv` records
the training measurements; `settings.csv` retains all consolidated settings. `provenance.json`
records the sources and bootstrap setup. The earlier pre-canonicalization shortlist is preserved
separately: omitted zero and explicit0.0 exploration were corrected to the same configuration
before this final shortlist was made.
