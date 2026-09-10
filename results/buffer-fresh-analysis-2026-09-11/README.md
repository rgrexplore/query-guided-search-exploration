# Buffer-size cost measurements

These are kernel measurements on constructed input buffers, not a retrieval-quality benchmark.
Raw elapsed times include the work named by each operation; input generation is excluded.

## Leaf model checked on a larger buffer

Fit: fixed cost + physical bitmap words × word cost + scored rows × gathered-row cost.
Only 1M/4M buffers fit the coefficients; the 16M buffer checks their transfer.

| Document pool | Remaining rows | Measured ms | Predicted ms | Error |
|---:|---:|---:|---:|---:|
| 16,000,000 | 62,492 | 8.2715 | 7.2111 | -12.8% |
| 16,000,000 | 3,900 | 0.8759 | 0.6338 | -27.6% |
| 16,000,000 | 15,653 | 2.3955 | 1.9531 | -18.5% |
| 16,000,000 | 1,958 | 0.4554 | 0.4158 | -8.7% |

All source tables, including split/root/release times and OS peaks, are retained.
A path profile holds only the measured planes and masks. Its RAM is not the full d-dimensional index RAM.
The 32GB case cannot hold 1B 256-bit codes plus 64-bit document IDs: that payload alone needs 40GB.
Larger-RAM scenarios still require a separate full-query model and explicit extrapolation limits.
