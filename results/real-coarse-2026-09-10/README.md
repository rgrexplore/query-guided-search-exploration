# Initial real-data grid

The first grid tested 500 configurations on 100,000 MS MARCO passages and 25 tuning queries.
It used the same binary-score reference and routing options for scan, branching and key lookup.
All configurations completed. See `comparison/` for the raw target selections and full table.

The apparent branch advantage at 95% in this coarse grid was not accepted as a result: scan's
probe choices jumped from 92.76% to 100% recall. The following refinement filled that gap and
added larger cluster counts. Keep this run as the earlier observation, not the final conclusion.

`cases.zip` preserves the raw case inputs, logs, query rows and process results. Every archived
file was checked against its local source. Extract it in this directory before rerunning the
comparison command; the original local cases remain on disk and are ignored by Git.
