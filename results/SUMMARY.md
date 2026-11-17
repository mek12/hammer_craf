# Experimental results - reproducible summary

Current source: original server reports in `results/server_records/`.
Recorded normalized JSON hashes of historical chat transcriptions are verified. No new GPU measurements were performed.

Sample count: 50,705. Units: the first four metrics are percentages; cross-entropy is natural-log loss.

| Model | Accuracy % | Balanced accuracy % | Macro-F1 % | AUC % | Cross-entropy |
|---|---:|---:|---:|---:|---:|
| HAMMER baseline | 85.3427 | 83.8513 | 83.6121 | 92.6660 | 0.526091 |
| CRAF v1, lr=1e-4 | 85.5142 | 83.8879 | 83.7545 | 92.8615 | 0.563057 |
| CRAF v1, lr=1e-5 | 85.5340 | 84.0556 | 83.8226 | 92.8592 | 0.564652 |
| CRAF v2, combined | 85.4156 | 83.7724 | 83.6425 | 92.7468 | 0.530419 |

## Differences from baseline

| Model | Delta AUC (percentage points) | Delta accuracy (percentage points) | Delta macro-F1 (percentage points) | Net additional correct predictions | Delta CE |
|---|---:|---:|---:|---:|---:|
| CRAF v1, lr=1e-4 | +0.1955 | +0.1716 | +0.1424 | +87 | +0.036967 |
| CRAF v1, lr=1e-5 | +0.1932 | +0.1913 | +0.2104 | +97 | +0.038561 |
| CRAF v2, combined | +0.0808 | +0.0730 | +0.0304 | +37 | +0.004328 |

## Error types

| Model | False alarms (real → fake) | Missed manipulations (fake → real) | Selected epoch (one-based) |
|---|---:|---:|---:|
| CRAF v1, lr=1e-4 | 3540 | 3805 | 2 |
| CRAF v1, lr=1e-5 | 3437 | 3898 | 9 |
| CRAF v2, combined | 3568 | 3827 | 13 |

## Limitations

- These are binary detection results only; CRAF grounding performance has not been measured.
- AUC and cross-entropy cannot be recomputed from the aggregated confusion matrix; the JSON values are retained.
- Three example-level test prediction files are now available; additional analysis is in `SERVER_SUMMARY.md`. The single-seed limitation remains.
- V1 patience changed along with the learning rate; differences cannot be attributed solely to the learning rate.
- Validation results for all four V2 conditions are now available; a separate validation table is in `SERVER_SUMMARY.md`.
- No V2 control/penalty/decision test reports are available, so these variants receive no scores in the test table.
- Test results influenced subsequent variant designs: the comparison is exploratory, not an independent final holdout evaluation.
