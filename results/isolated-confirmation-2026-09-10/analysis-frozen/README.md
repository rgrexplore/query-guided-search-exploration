# Isolated timing calibration

52 complete cases; 0 failed or limited cases.
The model uses measured work counts and measured routing time. It is not a forecast of recall or work from N alone.
Frozen model loaded from results/isolated-calibration-2026-09-10/analysis-relative/models.json

| Method | Check settings | Within 20% | Worst absolute p50 error |
|---|---:|---:|---:|
| scan | 4 | 4 | 3.7% |
| branch | 24 | 21 | 27.2% |
| keys | 24 | 24 | 7.2% |

Coefficients combine scoring, memory access, result selection and queue work; they are not hardware instruction timings.
Whole-process lifetime peak includes construction. Query RSS observations can miss brief peaks. Logical payload excludes runtime and allocator overhead.
No tuned winner or billion-document prediction is established by this calibration run.
