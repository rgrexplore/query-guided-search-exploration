# Two measured prefix paths

Both examples use K = 1 and the same fixed 78 clusters. The correct-answer column is calculated afterward; it is not available to the live search.

The first example is the lowest query ID that permits an exact stop at depth 32. The second is the query nearest the median nonzero extra-scoring count. Neither is selected for winning a timing comparison.

## Query 19

Ideal code: `00011110100000011110000110011100`. Q = 4.446551. The exact local best document ID is 356003.

Selected depths are shown below. The normal search stops at the final row shown; the diagnostic continued to the root only to check the answers.

| Depth | New rows at this depth | Total scored | Correct / 1 | Best score so far | Unseen score bound | Decision |
|---:|---:|---:|---:|---:|---:|---|
| 32 | 1 | 1 | 1 | 4.446551 | 4.433512 | Stop |

The correct answer first appeared at depth 32; the normal exact search stopped at depth 32. It scored 0 additional documents between those points.

## Query 55

Ideal code: `00001011101100010101100100110011`. Q = 4.676180. The exact local best document ID is 202321.

Selected depths are shown below. The normal search stops at the final row shown; the diagnostic continued to the root only to check the answers.

| Depth | New rows at this depth | Total scored | Correct / 1 | Best score so far | Unseen score bound | Decision |
|---:|---:|---:|---:|---:|---:|---|
| 32 | 0 | 0 | 0 | No result | 4.673120 | Broaden |
| 24 | 0 | 0 | 0 | No result | 4.673120 | Broaden |
| 21 | 3 | 4 | 1 | 4.597932 | 4.673120 | Broaden |
| 16 | 3 | 37 | 1 | 4.597932 | 4.653987 | Broaden |
| 8 | 2964 | 4543 | 1 | 4.597932 | 4.641909 | Broaden |
| 4 | 2281 | 18793 | 1 | 4.597932 | 4.641909 | Broaden |
| 3 | 3496 | 22289 | 1 | 4.597932 | 4.641909 | Broaden |
| 2 | 16 | 22305 | 1 | 4.597932 | 4.582010 | Stop |

The correct answer first appeared at depth 21; the normal exact search stopped at depth 2. It scored 22,301 additional documents between those points.

