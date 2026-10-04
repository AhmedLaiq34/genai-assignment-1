# Task 3 training and results report (soft mixture of experts)

Written 2026-10-05. Implementation record: `docs/PHASE_T3_REPORT.md`; design: `docs/TASK3_PLAN.md`; decisions D80 to D101. All numbers are validation numbers (2,944 rows) unless the section says test. The test set was opened once (section 9).

## 1. What was run
Kaggle T4, notebook `t3-pipeline`, one run of `tools/t3_pipeline.py` (`t3-v1`), 2026-10-04 evening UTC, about 9.5 minutes of runner time: prepare 42 s, study 360 s, final training 100 s, evaluation 30 s, export + promotion 8 s. The student capped training at 20 to 25 minutes, so the budgets of the plan were reduced (D100): study 10 trials requested with a 9-minute cap, 1 warm-up + 2 joint epochs per trial, trial validation on 736 rows, MedianPruner (3 start-up trials, warm-up 2 steps), final training 1 warm-up + 8 joint epochs with validation every second epoch. No Kaggle dry run was done; the local dry run (G5) with the real sources had passed. T4 timings (batch 64, AMP): warm-up 0.051 s/step, joint 0.101 s/step, full validation 7.2 s, peak 1,181 MB. The Task 2 sources were verified by sha256 at the start and the end (unchanged).

Model: gate = copy of the Task 2 classifier (98,196 parameters), three experts = the Task 2 specialists (1,322,507 each), identity branch; 4,065,717 parameters in total.

## 2. Study (`studies/t3_moe/`)
- 12 trials in the database: 9 COMPLETE, 1 PRUNED (median), 2 FAIL. The two FAIL trials are the first two attempts: the Kaggle notebook had no W&B secret and each child process stopped on the missing API key; the supervisor retried and then switched to offline logging (`WANDB_MODE=offline`). Trial 0 (the PDF start values) was one of the failed trials and the runner does not re-enqueue it, so the study has no trial-0 result. Timeout not hit.
- Best trial 7, fixed J on the 736-row subset 0.0969 (per epoch 0.0982, 0.0974, 0.0969): joint_lr 7.3e-5, tau 3.64, lambda_c 0.0198, lambda_b 0.0134, recon share r 0.403 (lambda_1 0.403, lambda_s 0.597). No parameter within 10 % of a range edge (positions on the range: joint_lr 0.67, tau 0.86, lambda_c 0.15, lambda_b 0.56, r 0.16; the tau value is nearer the upper end on a log scale, at 0.86 of the range).
- PDF start comparison: because trial 0 failed, the PDF start values (tau 1, joint_lr 5e-5, lambda 0.8 / 0.2 / 0.1 / 0.01) were trained again locally with the same budget as a trial (1 warm-up + 2 joint epochs, 736 validation rows, RTX 3050, seed 42; `artifacts/pdfstart_t3/`): val J 0.1036, 0.1039, 0.1038 against 0.0969 for the tuned trial 7. This is a separate local run, not a study trial.
- Search ranges as in plan B18 (provisional, not widened). Plots: `report/figures/task3/t3_optuna_*.png`; table `report/tables/task3/study_summary.*`.

## 3. Final training (`artifacts/runs/task3/20261004-2005_kaggle_t3moe_final_t3/`)
Validation J per epoch (full validation, every second epoch): epoch 0 (the Task 2 copy in soft mode at tau 3.64) 0.1025; epoch 2 0.0955; epoch 4 0.0949; epoch 6 **0.0932** (best, SSIM 0.846); epoch 8 0.0935; epoch 9 (last) 0.0933. No collapse flag at any validation. The warm-up (gate only) epoch is epoch 1. The gate accuracy fell from 0.991 (epoch 0) to 0.946 to 0.957 during joint training. Curve figure: `report/figures/task3/training_curves.png`.

## 4. Comparison with Tasks 1 and 2 (validation, identical tensors; `comparison_cond_severity.csv`)
| System | J | SSIM |
|---|---|---|
| input (do nothing) | 0.1558 | 0.729 |
| Task 1 universal autoencoder | 0.1237 | 0.791 |
| Task 2 oracle routing | 0.1028 | 0.828 |
| Task 2 predicted routing | 0.1032 | 0.828 |
| **Task 3 soft mixture** | **0.0932** | **0.846** |

Task 3: MAE 0.0329, PSNR 28.89 dB. By condition (J, input / Task 2 predicted / Task 3): clean 0 / 0.0027 / 0.0056; salt-and-pepper 0.3417 / 0.1146 / 0.1125; blur 0.0908 / 0.1145 / 0.0970; occlusion 0.1909 / 0.1809 / 0.1578. By severity, blur low 0.0505 / 0.1143 / 0.0815 and blur medium 0.0970 / 0.1121 / 0.0994: the soft system is better than the Task 2 predicted routing on blur at every severity, and still worse than the input on low and medium blur and on clean images. Task 3 beats the input on salt-and-pepper (all severities) and on occlusion (all severities).

## 5. Routing analysis (PDF page 7; `weights_by_class_severity.csv`, `routing_heatmap.png`)
Mean weights (identity / salt / blur / occlusion):
- clean: 0.836 / 0.027 / 0.045 / 0.092.
- salt-and-pepper low / medium / high: salt 0.932 / 0.985 / 0.989.
- blur low / medium / high: blur 0.551 / 0.676 / 0.680, identity 0.387 / 0.277 / 0.273.
- occlusion low / medium / high: occlusion 0.614 / 0.771 / 0.850, identity 0.330 / 0.188 / 0.111.

The gate keeps weight on the identity branch for mild blur and mild occlusion, where the specialists alone are worse than the input (Task 2 blur low: expert J 0.1173 vs input 0.0505). Lower identity weight with higher severity follows the same pattern. Gate accuracy (argmax = true class) 0.946 against 0.991 for the Task 2 classifier on the same rows: clean 0.999, salt-and-pepper 0.997, blur 0.927, occlusion 0.860. A low accuracy on blur and occlusion is the identity weight winning the argmax on mild rows; it has to be read together with J (section 4). The classification weight of the best trial is small (0.02), so the gate is less tied to the label than in Task 2.

Examples (`examples_dominant.png`, `examples_distributed.png`, `examples.csv`): a row is dominant when its largest weight is at least 0.95 (739 of 2,944 validation rows) and distributed when it is at most 0.60 (327 rows); 8 rows of each kind are drawn, 2 per class where possible.

Expert activity (`expert_activity.csv`, thresholds in plan B19): no branch is inactive (mean weights 0.344 identity, 0.258 salt, 0.177 blur, 0.221 occlusion; share of rows where the branch is the argmax 0.30 / 0.25 / 0.23 / 0.22) and no branch dominates rows of another class (largest mean weight of a branch on another class: identity on blur rows, 0.308).

Expert drift (`expert_drift.csv`, each expert's own output alone on its own class rows, 736 rows each): Task 3 vs the Task 2 specialist, J salt 0.1142 vs 0.1146, blur 0.1190 vs 0.1153, occlusion 0.1866 vs 0.1812. The blur and occlusion experts are slightly worse on their own class after joint training (+0.0036 and +0.0054); the system as a whole improves because of the weights.

## 6. Settings that matter
Experts stayed in eval mode (BatchNorm statistics kept) in both stages (D83); CE on the raw logits, mixture with softmax(G/tau) (D81); the balance loss is the PDF's quadratic form (D88); warm-up trains the gate only at 2e-4, the joint stage one AdamW group at 7.3e-5 with cosine decay (D85); collapse thresholds 0.02 / 0.9 and analysis thresholds 0.95 / 0.60 / 0.02 / 0.5 are recommendations (D95, D98). Validation severities are tertiles of the sampled parameter.

## 7. Export and application
`models/onnx/t3_soft_moe.onnx` (outputs `output` and `weights`, tau 3.64 inside the graph): parity on Kaggle 4.2e-7 (`output`), local re-check 8.9e-7 (`output`) and 2.3e-7 (`weights`), weight-sum error 1.2e-7, dominant branch agrees on 16 of 16 inputs; rows in `report/tables/onnx_parity.csv`. `/api/soft` answers 200 on 10 calls, inference about 15 to 30 ms per image on the laptop CPU (Docker backend: 46 ms for one call including the first model load); the Soft workspace check of `app/frontend_v2/verification/real_backend.mjs` passed and the screenshots are in `report/figures/app/app_soft-*.png`.

## 8. Tracking
The 11 offline W&B runs of the Kaggle session (10 trial runs and the final run) were synced locally to entity `ahmedlaiq34`, project `genai-a1` after the run (the Kaggle session had no API key). The local PDF-start run was made with the tracker off.

## 9. Test set (run once, with the student's approval, D101)
36,690 test cases (`artifacts/eval/task3/20261005-012220_test/`): J input 0.1898, Task 1 0.1330, Task 2 oracle 0.1277, Task 2 predicted 0.1276, **Task 3 0.1142** (SSIM 0.811 vs 0.787 for Task 2 predicted); gate accuracy 0.904. By condition (Task 3 J): clean 0.0065, salt-and-pepper 0.1149, blur 0.1078, occlusion 0.1557. Same ordering as on validation.

## 10. Limits to state in the report
- Reduced budgets (D100): 10 trials requested, 12 recorded (2 failed), 1 + 2 epochs per trial, final 1 + 8 epochs; a longer search could change the best values.
- The study has no trial-0 result; the PDF-start comparison comes from a separate local run with the same epochs.
- One run, one seed; no confidence intervals.
- The gate accuracy is lower than in Task 2 (section 5); clean images and mild blur are not restored better than the input.
- The notebook ran without the W&B secret; the runs were uploaded afterwards.

## 11. Files
Checkpoint `models/checkpoints/t3_soft_moe.pt` (sha256 `767c08ba45511746cfe02ae61a28349dc249c4b173c42bdf68426075c1702a29`), ONNX `models/onnx/t3_soft_moe.onnx`, final config `configs/task3_moe_final.yaml`, study `studies/t3_moe/` and `artifacts/optuna/t3_moe.db`, evaluation `artifacts/eval/task3/20261005-011630_val/` and `..._test/`, Kaggle logs `artifacts/logs/t3_kaggle/`, tables `report/tables/task3/` (+ `test/`), figures `report/figures/task3/` (+ `test/`).
