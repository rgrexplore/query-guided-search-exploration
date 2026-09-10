# Refined real-data comparison

This stage used 100,000 cached MS MARCO passages, 100 tuning queries, a 256-dimensional
binary reference score, and one CPU thread. It measured 97 routing choices, then tested
1,378 configurations on the 53 choices whose exact-scan recall could reach at least 80%.

The initial coarse grid had made branching look faster at 95% recall. Adding intermediate
probe counts and larger cluster counts removed that apparent advantage: scan was faster at
all four target points in this refined grid. The next boundary study checks larger cluster
counts before choices are frozen. These are tuning observations, not a fresh final test.

See `search/comparison/README.md` for the target table and `search/comparison/settings.csv`
for every measured setting. `routing/` retains the earlier exact-scan screen, including choices
below the recall targets. Local-search recall was checked against scan under identical routing.

Raw cases are stored in verified `cases.zip` files in `routing/` and `search/`. Their archive
manifests contain per-file hashes. Extract each ZIP in its own directory before rerunning
analysis; the original folder paths inside each ZIP are preserved. Local raw files were not deleted.
