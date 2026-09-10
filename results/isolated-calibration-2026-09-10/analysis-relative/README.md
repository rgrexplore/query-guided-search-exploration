# Isolated timing calibration

117 complete cases; 0 failed or limited cases.
The model uses measured work counts and measured routing time. It is not a forecast of recall or work from N alone.
Check sizes were excluded from fitting. Query vectors are shared across sizes.

| Method | Check settings | Within 20% | Worst absolute p50 error |
|---|---:|---:|---:|
| scan | 4 | 4 | 4.3% |
| branch | 24 | 23 | 29.3% |
| keys | 24 | 24 | 11.9% |

Coefficients combine scoring, memory access, result selection and queue work; they are not hardware instruction timings.
Whole-process lifetime peak includes construction. Query RSS observations can miss brief peaks. Logical payload excludes runtime and allocator overhead.
No tuned winner or billion-document prediction is established by this calibration run.
