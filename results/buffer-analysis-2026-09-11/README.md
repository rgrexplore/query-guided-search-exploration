# Buffer-size cost measurements

These are kernel measurements on constructed input buffers, not a retrieval-quality benchmark.
Raw elapsed times include the work named by each operation; input generation is excluded.

## Leaf model checked on a larger buffer

Fit: fixed cost + physical bitmap words × word cost + scored rows × gathered-row cost.
Only 1M/4M buffers fit the coefficients; the 16M buffer checks their transfer.

| Document pool | Remaining rows | Measured ms | Predicted ms | Error |
|---:|---:|---:|---:|---:|
| 16,000,000 | 15,778 | 0.6935 | 0.6918 | -0.2% |
| 16,000,000 | 3,902 | 0.2386 | 0.2569 | +7.7% |
| 16,000,000 | 62,692 | 4.3995 | 2.4097 | -45.2% |
| 16,000,000 | 1,957 | 0.1592 | 0.1857 | +16.6% |

All source tables, including split/root/release times and OS peaks, are retained.
A path profile holds only the measured planes and masks. Its RAM is not the full d-dimensional index RAM.
The 32GB case cannot hold 1B 256-bit codes plus 64-bit document IDs: that payload alone needs 40GB.
Larger-RAM scenarios still require a separate full-query model and explicit extrapolation limits.
