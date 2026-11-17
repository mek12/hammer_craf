# Report evidence review — November 13, 2026

Target: `reports/Kocak_YZM5110_Research.md`. After the initial CRAF review, the package was updated with HAMMER records. No new web research or GPU training was performed. The eight numbered sections and R1–R6 literature sequence were retained. Structural checks covered the updated main text including references, tables with at most six columns, and 50,705 unique indices/targets/baseline probabilities in the same order across three JSONL files. PDF generation and visual inspection were assigned to the supervising reviewer.

## Mapping claims to sources

The original server files from `hammer-craf/` and `hammer-craf-v2/` were read. Report links point to the permanent copies under `results/server_records/`. The old `results/user_reported/` transcriptions are no longer the only evidence. This document records the November 13 review; current paths after deduplication are resolved through `results/SERVER_EVIDENCE_INDEX.json`.

| Claim group | Local evidence | Verification and limits |
|---|---|---|
| Six completed head-training runs | Two v1 and four v2 `run.json` files | Seed 42, batch 512, max 30, hidden 256, dropout 0.2, weight decay 0.01; file-level evidence, not physical GPU telemetry |
| V1 setting difference | `hammer-craf/craf_results/run.json`; `craf_lr_1e5/run.json` | 1e-4/patience 5 versus 1e-5/patience 8; the LR effect cannot be isolated |
| V2 controlled validation ablation | Four `craf_v2_*_seed42/{run,validation_comparison}.json` records | Shared 1e-5/patience 8; control, penalty, decision, combined available; test only for combined |
| Selected validation values | Six `validation_comparison.json` files | AUC-selected checkpoint; CE was not selected by minimum loss |
| Three test comparisons | Two v1 and combined `test_comparison.json` / `test_comparison_predictions.jsonl` files | Each JSONL contains exactly 50,705 rows, not 50,507 |
| Paired transitions | JSONL `index`, `target`, baseline/craf fake probability | With p_fake≥0.5, errors fixed/introduced: 707/620, 730/633, 368/331; net +87/+97/+37 |
| Histories | Six `history.json` files | Lengths 7/17/17/24/11/21; selected one-based epochs 2/9/9/16/3/13 |
| Timing scope | Training loops in archived v1 `hammer_craf_server.py` and v2 `hammer_craf_v2_server.py` | Both compute `seconds` before writing checkpoint/history; timing covers epoch training and validation |
| Source separation and counts | `hammer-craf/craf_features/{train,val,test}.manifest.json` | Rows 208184/22126/50705; news_ids 56894/7975/18360; all three pairwise source intersections are zero |
| Shared backbone/cache identity | Identity/hash fields in runs and manifests | Shared recorded context; missing `.pt`/`.pth` contents cannot be physically verified |
| Architecture and source size | Residual source files; `results/model_resources.json` | Head parameters 922117/922885; FP32 3688468/3691540 bytes; not the full model or peak VRAM |
| HAMMER retraining/test | `results/server_records/hammer_training/{config.yaml,run_args.json,log.jsonl,test_metrics.json}` | Target 25, recorded 11; selected epoch 10 (zero-based); four task test results; matching CRAF backbone hash |
| Literature | `LITERATURE_MATRIX.md`; `PAPER_READING_GUIDE.md`; `references.bib` | R1–R5 published, R6 preprint; no claim of new web research or personal reading |
| Official code pinning | `NOTICE.md` | Upstream target commit, not the final submission commit |

## Numerical interpretation and retained limitations

- The first four test-table metrics are percentages; validation AUC is on a 0–1 scale; CE is natural-log loss. About 0.2 percentage points is not 20%.
- V2 validation AUC: control 0.9249354319; decision 0.9249796503; penalty 0.9238513408; combined 0.9238293595. Decision−control is only +0.0000442184; the penalty reduces AUC in both matched comparisons.
- Control CE 0.584701 → penalty 0.547143 / combined 0.547165; HAMMER CE 0.541329 remains better. Decision CE is 0.570732. This is not evidence of superior calibration.
- Equality between control validation and low-LR v1 does not mean a control test result exists. No control/penalty/decision test scores were added to the report.
- All three CRAF test conditions have worse CE than the baseline; combined also has lower balanced accuracy. No grounding, robustness, or SOTA superiority is claimed.
- The sums of logged epoch seconds are respectively 45.745 / 109.325 / 116.408 / 157.439 / 74.972 / 140.049 seconds. Both source versions take the timestamp before checkpoint writing. Cache extraction, startup, checkpoint writing, final evaluation, and upstream training are excluded; these are not full wall time, GPU-hours, or VRAM measurements.
- Hardware of 4 × RTX A6000 48 GB is user-reported. HAMMER run_args records world_size 4 and per-GPU batch 64, giving nominal global batch 256; log entries 0–10 show 11 recorded epochs.
- Manifest source separation was verified; repeated examples from the same source and content similarity still require investigation. The difference between 281015 rows and the literature's 230000 pairs remains unresolved.
- Availability of paired predictions removes the old “no raw predictions” limitation; it does not establish independent observations. No CI/p-value/McNemar/bootstrap result was fabricated.
- The influence of test results on later variant design is retained as a protocol violation relative to the instructor's rule against test-based tuning. `test_used=false` describes only the corresponding training/selection record; it does not make these comparisons an independent final holdout.

## Outstanding submission evidence

The student number, public repository URL, and submission commit/tag had not been supplied at the time of review. Cache `.pt` shards, `.pth` weights, and an actual environment inventory are absent. HAMMER's 11-epoch log/config/run_args/test records are now available; continuation toward the target of 25 epochs and the stopping reason remain unknown. V2 control/penalty/decision tests and multiple seeds are absent; additional experiments were not presented as completed. Source-group-aware uncertainty analysis, content-based error examples, an explanation of the data-count discrepancy, and the student's personal reading remain open.

The statistics contributor's `scripts/analyze_server_results.py`, `results/server_analysis.json`, and `results/SERVER_SUMMARY.md` outputs were a separate responsibility; that report-editing task changed only its two assigned report documents. Once available, those outputs could be read to check consistency with transition counts and timing sums. This document does not claim a PDF page count or visual approval.

The final HAMMER addition passed 32 checks with `scripts/analyze_hammer_training.py`.
Epoch durations total 14907.237 seconds; the HAMMER launcher includes checkpoint
writing, whereas CRAF excludes it. These timing scopes were kept separate.
All task-head test results belong to the HAMMER baseline and were not presented
as CRAF grounding gains.
