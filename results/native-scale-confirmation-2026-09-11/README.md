# Frozen native cost-model check

The coefficients were frozen before this 8M-document run. All complete searches matched their independent reference.
This tests corpus-size transfer on prescribed query workloads, not globally optimal routing or new-query generalization.

| Support | Method | Measured ms | Predicted ms | Error |
|---|---|---:|---:|---:|
| adaptive | branch | 0.9000 | 0.8591 | -4.5% |
| fixed | branch | 0.8538 | 0.8844 | +3.6% |
| adaptive | scan | 196.6639 | 197.8155 | +0.6% |
| fixed | scan | 197.0291 | 203.0161 | +3.0% |
| adaptive | keys | 202.7669 | 207.0857 | +2.1% |
| fixed | keys | 0.0481 | 0.0518 | +7.7% |

All six errors are within 7.8%. This does not establish that error bound at 1B documents.

To recompute after extracting `cases.zip`:
```bash
python -m experiments.check_native_model results/native-scale-confirmation-2026-09-11 --models results/native-scale-check-2026-09-11/frozen-models.json
```
