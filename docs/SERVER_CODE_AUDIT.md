> This document records the initial CRAF archive review. The subsequently supplied
> HAMMER config/run_args, 11-epoch log, and test record provide additional evidence;
> see `../results/HAMMER_TRAINING_SUMMARY.md` and `../qa/HAMMER_TRAINING_REVIEW.md`.

# Server code and experiment provenance audit

November 13, 2026. Raw directories inspected: `hammer-craf/`, `hammer-craf-v2/`; comparison target: `gpu/`. Only file bytes, AST, and JSON/JSONL were read. Server code was not executed, no training/research was performed, and no checkpoint/pickle was loaded. Raw directories and other documents were unchanged during that audit. Detailed hashes and checks: `qa/SERVER_SOURCE_AUDIT.json`. After repository deduplication, original names in this historical audit are resolved through `results/SERVER_EVIDENCE_INDEX.json`.

**Finding:** All 14 raw `.py/.sh` files were byte-identical to their packaged counterparts. Recorded trainer/head/helper SHA values across six runs match the actual files and `gpu/`. This verifies source consistency of the supplied records; it is not independent attestation of the historical server execution.

## Sources and version

| Raw directory | File | SHA-256 | Match with gpu |
|---|---|---|---|
| hammer-craf | `craf_residual.py` | `8fba844c8e377b8ae221b3a771a7754e9f6a6d89e4788194be5b1e22c1d811e6` | identical |
| hammer-craf | `hammer_craf_server.py` | `bb7fd19ebb758a042b665cc5c5b91214e01e2d326f0a87eb9de0139b0af635d1` | identical |
| hammer-craf | `run_craf_server.sh` | `2e547f76615e239957cc57e9759cd4d178e73631963609555880c281fea76906` | identical |
| hammer-craf | `test_hammer_craf_server.py` | `1776776358ac73855adc45c202c63830ce9750587933d49c8fe18efcddfc2d6c` | identical |
| hammer-craf | `train_hammer_4gpu.py` | `4da2a691407ea248bc10d4a7683aa8d5ebdd7658424fc0d535c2162db9f4319b` | identical |
| hammer-craf-v2 | `craf_residual.py` | `8fba844c8e377b8ae221b3a771a7754e9f6a6d89e4788194be5b1e22c1d811e6` | identical |
| hammer-craf-v2 | `craf_residual_v2.py` | `8c88690940a28b4b90a7f67d997e7701392de705be9cf699ed51660370aa4faa` | identical |
| hammer-craf-v2 | `hammer_craf_server.py` | `bb7fd19ebb758a042b665cc5c5b91214e01e2d326f0a87eb9de0139b0af635d1` | identical |
| hammer-craf-v2 | `hammer_craf_v2_server.py` | `0069e26e8fdd46c8206b676437ad7f6be6c2e3fe4e4b547bf88952c56a844167` | identical |
| hammer-craf-v2 | `run_craf_v2_experiments.sh` | `4f9f934c40fe194176a33d7425423ef0de1459477a804333eb1be27f3248cc46` | identical |
| hammer-craf-v2 | `test_craf_residual_v2.py` | `2c2cdb7138e3680f80129d058c1f9d43157f964a356a097b1e49275f23a77ffd` | identical |
| hammer-craf-v2 | `test_hammer_craf_server.py` | `1776776358ac73855adc45c202c63830ce9750587933d49c8fe18efcddfc2d6c` | identical |
| hammer-craf-v2 | `test_hammer_craf_v2_server.py` | `c80a67c7d0e47d183beb2f2e6222e8310ed7567cf4b9a738937153535cc22d31` | identical |
| hammer-craf-v2 | `train_hammer_4gpu.py` | `4da2a691407ea248bc10d4a7683aa8d5ebdd7658424fc0d535c2162db9f4319b` | identical |

V1 `trainer_source_sha256` → `hammer_craf_server.py`, `head_source_sha256` → `craf_residual.py`. V2 trainer → `hammer_craf_v2_server.py`, head → `craf_residual_v2.py`, helper → `hammer_craf_server.py`. The V1 config schema has no separate helper field. Hashes are recorded through `__file__`/the adjacent head file and V2 `shared.__file__`; all records match exactly. Hashes for additional download/test/wrapper code in `gpu/` were also written to JSON; no historical execution is claimed for extra files without server counterparts.

The `torch.load(..., weights_only=True, map_location="cpu")` version in `hammer_craf_server.py`, with `mmap` removed for compatibility with older PyTorch, is identical in the server and package copies. The cache extractor SHA identifies this version; no additional mmap fix is needed. AST-extracted parser defaults are in JSON. V1 train defaults are lr=1e-4/patience=5; V2 defaults are lr=1e-5/patience=8; both use seed=42, head batch=512, and target=30. The V2 parser starts with decision inputs disabled and λ=0; wrapper/config settings explicitly enable variant options.

## Actual run records

These values were read from each run's `run.json` config and `history.json`, not inferred from code defaults. All runs use seed=42, CRAF head batch=512, weight_decay=0.01, AUC selection, and target=30. This batch value does not describe HAMMER DDP/global batch size.

| Run | lr | patience | decision/λ | Completed epochs | History range (zero-based) | Selected epoch (zero-/one-based) |
|---|---:|---:|---|---:|---|---|
| `hammer-craf/craf_lr_1e5` | 1e-05 | 8 | V1/— | 17 | 0–16 | 8 / 9 |
| `hammer-craf/craf_results` | 0.0001 | 5 | V1/— | 7 | 0–6 | 1 / 2 |
| `hammer-craf-v2/craf_v2_combined_seed42` | 1e-05 | 8 | True/0.01 | 21 | 0–20 | 12 / 13 |
| `hammer-craf-v2/craf_v2_control_seed42` | 1e-05 | 8 | False/0.0 | 17 | 0–16 | 8 / 9 |
| `hammer-craf-v2/craf_v2_decision_seed42` | 1e-05 | 8 | True/0.0 | 11 | 0–10 | 2 / 3 |
| `hammer-craf-v2/craf_v2_penalty_seed42` | 1e-05 | 8 | False/0.01 | 24 | 0–23 | 15 / 16 |

All histories are contiguous; selected epochs and validation metrics match the highest AUC in each history. Final patience counters equal their thresholds: records are consistent with early stopping. A target of 30 does not mean 30 epochs were completed. Sums of history `seconds` in JSON cover only recorded epoch durations, not cache extraction, startup, final evaluation, or total end-to-end GPU cost.

## Cache and shared identity

| Manifest | Actual file SHA-256 | Examples | Listed/present shards |
|---|---|---:|---|
| `train.manifest.json` | `dc0a5b227cab25ba197edf35b3325450f3e5b00f317c8fba72e24de2a48e65f1` | 208184 | 4 / 0 |
| `val.manifest.json` | `ee977a873977256152d37933d41d6c8e1d44891b20e7f7927b62cbb3b956d1e7` | 22126 | 4 / 0 |
| `test.manifest.json` | `138949eddca1c1a291a86622ad0bf21f9635113bd83dd09137179b30fed0779f` | 50705 | 4 / 0 |

Train/val manifest SHAs in all six configs match actual manifest file hashes. Backbone/preprocessing/source identity, excluding split and metadata SHA, is identical across the three manifests; the six backbone config fields and three test report cache identities also match. Every extractor SHA matches the actual `hammer_craf_server.py` SHA. Shared backbone, BERT config, and tokenizer hashes do not mean their file bytes are present or verified.

Train–val, train–test, and val–test source-news ID intersections in the manifests are **0**. This was verified for the supplied manifests; it does not verify all image/text duplicates, the upstream data version, or original metadata bytes. The manifests total 281,015 records; this source audit did not resolve the difference from the literature's 230,000 total. Six third-party files in `source_sha256` (`models/HAMMER.py`, `models/xbert.py`, `models/vit.py`, `dataset/dataset.py`, `dataset/utils.py`, `train.py`) are recorded only as hashes; their actual files are absent from the inspected server/package sources. Upstream file-hash/commit identity was not independently verified.

## Predictions and targeted metric checks

Available files, listed by original imported path:

- `hammer-craf/craf_lr_1e5/test_comparison_predictions.jsonl` — 50705 rows, SHA-256 `91adb33981ce03a3cd4c87c55a294c79ea6071a1250df036231a4e7fa06ca3b3`.
- `hammer-craf/craf_results/test_comparison_predictions.jsonl` — 50705 rows, SHA-256 `c05144d0e9e3bd0a3fac3510d8542cc92e4a1b4c7e1605e62a1ab790c1e403a6`.
- `hammer-craf-v2/craf_v2_combined_seed42/test_comparison_predictions.jsonl` — 50705 rows, SHA-256 `9c11983f79ddb1eb0fa1809ad5ed44db3b77c0d919524ccc47cc29e28805e47d`.

Across all three files, indices run from 0–50,704, labels are binary, and probabilities are finite and in [0,1]; index/label/baseline-probability sequences are exactly identical. Reported counts match the test manifest count. Confusion matrices, accuracy, balanced accuracy, macro-F1, and tie-aware rank AUC were recomputed: the largest absolute difference is below 1.12×10⁻¹⁶.

CE was approximately recomputed from the single recorded p_fake column using `1-p_fake`; differences were 1.71×10⁻⁷ for baseline, 1.28×10⁻⁶ for V1 lr1e-4, 1.14×10⁻⁶ for V1 lr1e-5, and 1.60×10⁻⁷ for V2 combined. This is not exact numerical parity; without the original two float32 softmax columns and sklearn environment, the precise cause cannot be established. Probability rounding/normalization is a possible explanation. The direction of CE deterioration is unchanged. The claim that raw predictions are absent is obsolete; paired error analysis is possible. This audit did not produce CIs or p-values.

V2 control/penalty/decision have `run.json`, `history.json`, and `validation_comparison.json`, but no test report or prediction file. Validation AUC: control **0.9249354319**, penalty **0.9238513408**, decision **0.9249796503**, combined **0.9238293595**. Combined is not best by validation AUC; after test-informed design, presenting the protocol as an independent final holdout would be incorrect. It can be reported as a limited validation ablation with one seed and other settings held fixed.

## Document correction guidance for the primary agent

- Report §4/§6 and `docs/REPORT_EVIDENCE_REVIEW.md`: distinguish CRAF from HAMMER when discussing “missing run/config, unknown seed/global batch/total epochs.” CRAF settings and completed epochs are verified above; HAMMER training records arrived after this initial review, as noted at the top.
- Report §5/§8, `results/PROVENANCE.md`, and the evidence review: replace “no control/penalty/decision results” with “validation records available, test results absent.” Do not claim combined superiority or a causal penalty effect.
- Report §7 and provenance/evidence review: “no raw predictions; AUC/CE only user-reported; paired analysis impossible” describes the old scope. Original reports and three paired JSONLs exist; AUC was verified, and approximate CE-check limits are explained above. Prediction rows contain no images/text, so visual explanations of error cases remain unsubstantiated.
- It is correct that selected epochs 2/9/13 are not total epoch counts; totals can now be given as 7/17/21. Since V1 LR and patience changed together, attributing causality solely to LR remains inappropriate.
- Retain test-informed design, one-seed, statistical-significance uncertainty, and absence of grounding/calibration/robustness/SOTA evidence. Recovered records do not remove these scientific limitations.

## Actual missing items and blockers

- At the time of the initial review, full HAMMER baseline logs/config/run arguments and checkpoint-selection traces were absent; only a user-reported epoch-10 excerpt existed. These records were subsequently supplied, as noted above.
- HAMMER/CRAF weight, optimizer, and RNG checkpoints; no `.pt`/`.pth` files. The 12 cache `.pt` shards listed by manifests are absent; shard hashes/features/logits cannot be checked.
- Actual Python/PyTorch/CUDA/driver/hardware environment outputs; requirements files do not prove what was installed.
- Upstream sources identified by hashes, original tokenizer/BERT config/metadata/backbone bytes; identity consistency is not integrity verification of available files.
- Additional seeds, an untouched final test protocol, and test outputs for the other three V2 variants.

Nothing blocks local source/JSON inspection. Independent retraining or checkpoint inference requires the missing files above. `run.json`, `history.json`, original comparison reports, manifests, and three test JSONLs should no longer be listed as missing.
