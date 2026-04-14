# Conformal Coverage Validation

**Coverage target:** 95%

## Per-Bucket Empirical Coverage (validation split)

Each bucket uses the conformal threshold derived from the calibration split.
CI = 95% bootstrap confidence interval.

| Bucket | k injected | Count | Recall@k | 95% CI |
| --- | ---: | ---: | ---: | --- |
| k1 | 1 | 47 | 1.000 | [1.000, 1.000] |
| k3 | 3 | 3 | 1.000 | [1.000, 1.000] |
| k5 | 5 | 45 | 0.978 | [0.933, 1.000] |
| null (abstain) | 0 | 5 | OOD prec: 0.000 | — |

## Policy Comparison (same validation split)

Accuracy = Recall@injected_k; lower avg_k = fewer tokens injected.

| Policy | Accuracy | 95% CI | Avg k | Null-route % |
| --- | ---: | --- | ---: | ---: |
| fixed_k=1 | 0.970 | [0.930, 1.000] | 1.00 | 0.0% |
| fixed_k=3 | 0.990 | [0.970, 1.000] | 3.00 | 0.0% |
| fixed_k=5 | 0.990 | [0.970, 1.000] | 5.00 | 0.0% |
| gap_adaptive | 0.990 | [0.970, 1.000] | 3.38 | 0.0% |
| conformal_adaptive | 0.940 | [0.890, 0.980] | 2.81 | 5.0% |

## Calibrated Thresholds

| nc_threshold_k1 | -0.242075 |
| --- | --- |
| nc_threshold_k3 | -0.088911 |
| nc_threshold_k5 | 0.151823 |
| nc_threshold_null | 0.211089 |
