# Task 2 training report (Kaggle T4, 2026-10-04)

Written by the Task 2 lead session. Source: the Kaggle run `t2` (code version `t2-v3`, `dry_run=False`, Tesla T4, 15:37 to 16:19 UTC, about 42 minutes), downloaded as `t2_results.zip` (kept in `dist/kaggle_results/`). Every number below was read from the files of that run, and the key ones were re-computed locally (section 6). All evaluation is on the **validation manifest**; the locked test set was not used for Task 2.

## 1. What was run (budgets D47)

| Stage | Setting | Result |
|---|---|---|
| Classifier study `t2_classifier` | 12 trials x 8 epochs, median pruner (5 startup trials, warm-up 5) | **12 trials: 11 complete, 1 pruned, 0 failed** (8.8 min) |
| Classifier final training | 30 epochs, best checkpoint by val macro-F1 | best epoch 26, **val macro-F1 0.9908** (3.0 min) |
| Specialist shared study `t2_specialist_shared` | 10 trials x 8 epochs per corruption (a trial = 3 trainings), median pruner (5 startup trials, warm-up 2) | **10 trials: 6 complete, 4 pruned, 0 failed** (15.5 min) |
| Specialist finals | 50 epochs each, same architecture, three separate runs | salt best epoch 49, blur 49, occlusion 47 (3.5 / 5.0 / 3.5 min) |
| Evaluation, ONNX export + parity, promotion | validation manifest only | all passed |

Completed-trial counts are the numbers in the table above (12 trials in the classifier study, 10 in the specialist study, 8 epochs per trial in both).

## 2. Study results

**Classifier** (maximise val macro-F1): best trial 10, value 0.9813 at 8 epochs: lr 0.00218, batch 32, channels `16-32-64-128`, dropout 0.400, weight decay 8.9e-05. The three best trials (8, 10, 11) all use batch 32, channels `16-32-64-128` and dropout 0.40 to 0.44 (0.9789, 0.9813, 0.9782); the lowest-scoring complete trials (0.76 to 0.92) used batch 128 or the smallest learning rates. The one pruned trial had macro-F1 0.690 (batch 64, lr 0.0002, channels `16-32-64`). **Range edges:** batch 32 is the smallest choice, dropout 0.40 to 0.44 is near the top of its range (0 to 0.5), lr 0.0022 to 0.0028 is near the top of its range (1e-4 to 3e-3). The optimum may lie outside the searched ranges.

**Specialists** (minimise the mean J of salt, blur, occlusion): best trial 5, mean J 0.1674 at 8 epochs (J_salt 0.1635, J_blur 0.1322, J_occlusion 0.2065): lr 0.000375, bottleneck 2048 latent values, channels (base) 64, batch 64, L1/SSIM weight alpha 0.589. Only 6 trials completed; the 4 pruned trials were three with base channels 16 (two of them with batch 128, one with batch 32) and one with channels 32, batch 32 and alpha 0.84. **Range edges:** base channels 64 is the largest choice, alpha 0.59 is near the lower end of its range (0.5 to 0.95). The optimisation history, parameter importance and parallel-coordinate plots are in `studies/t2_specialist_shared/` (the classifier's in `studies/t2_classifier/`).

## 3. Final results (validation manifest, 2944 rows = 736 per condition)

J = 0.5 L1 + 0.5 (1 - SSIM), lower is better; "input" is the do-nothing baseline (the corrupted input scored against the clean target).

| Condition | Input J | Oracle-routed J | Predicted-routed J | Oracle SSIM | Input SSIM | Beats its input? |
|---|---|---|---|---|---|---|
| clean | 0.0000 | 0.0000 (identity) | 0.0027 | 1.000 | 1.000 | n/a (bypass) |
| salt and pepper | 0.3417 | **0.1146** | 0.1146 | 0.809 | 0.359 | yes |
| gaussian blur | 0.0908 | **0.1153** | 0.1145 | 0.808 | 0.844 | no (J 0.1153 vs 0.0908) |
| occlusion | 0.1909 | **0.1812** | 0.1809 | 0.697 | 0.712 | yes, narrowly (-0.010 J) |
| **overall** | 0.1558 | **0.1028** | **0.1032** | 0.828 | 0.729 | yes |

By severity (oracle J vs input J):
- salt: low 0.1096 vs 0.2508, medium 0.1152 vs 0.3595, high 0.1189 vs 0.4157 (large gains everywhere);
- occlusion: low 0.1562 vs 0.1304 (**worse**), medium 0.1847 vs 0.1978, high 0.2101 vs 0.2629;
- blur: low 0.1173 vs 0.0505 (**much worse**), medium 0.1121 vs 0.0970 (**worse**), high 0.1167 vs 0.1173 (about equal).

**Blur (D43, accepted by the student):** the blur specialist's J is higher than its blurred input's at low and medium severity and about equal at high severity (overall 0.1153 vs 0.0908). **Occlusion** lowers J at medium and high severity and is higher than the input for the smallest occlusions (low severity). The PDF design (identity bypass for clean, one specialist per corruption) is used as specified.

**Classifier** (val, 2944 rows): accuracy 0.9908, macro precision 0.9909, macro recall 0.9908, macro-F1 0.9908. Per class (precision / recall / F1): clean 0.990 / 0.973 / 0.982, salt 1.000 / 1.000 / 1.000, blur 0.979 / 0.993 / 0.986, occlusion 0.995 / 0.997 / 0.996. Row-normalised confusion matrix: clean row [0.973, 0, 0.022, 0.005], salt row [0, 1, 0, 0], blur row [0.007, 0, 0.993, 0], occlusion row [0.003, 0, 0, 0.997]. Files: `classifier_report.json`, `confusion_normalised.csv/.png` in the evaluation folder.

**Routing failures** (classifier errors that change the route): 27 of 2944 rows (0.92 percent).
- 20 clean images were sent to an expert (16 to the blur expert, 4 to the occlusion expert);
- 5 blurred and 2 occluded images were sent to the identity bypass (predicted "clean");
- the mean J change on these rows is +0.041 (a higher J), with a range from -0.197 to +0.191. Some misroutes lower J: the low-severity blurred images that the classifier calls "clean" receive the identity output, which has a lower J than the blur specialist's output for mild blur. This is why predicted-routing J for blur (0.1145) is slightly lower than oracle-routing J (0.1153).
- The 20 clean images sent to experts cost J 0.0027 on average over all clean rows.

Figures: `routing_failures_worst.png` (target | input | oracle output | predicted output | error) in `artifacts/eval/task2/20261004-213747_val/`.

## 4. Notes on the training and the model

- **Training curves.** The specialist validation J was still decreasing slowly at the end of the 50 epochs (salt 0.1181 at epoch 30, 0.1146 at epoch 50; blur 0.1177 to 0.1154; occlusion 0.1848 to 0.1814), with best epochs 49, 49 and 47. The classifier's best epoch was 26 of 30 (macro-F1 0.9898 at epoch 30).
- **Reference diagnostic.** An earlier 30-epoch diagnostic run with the provisional starting values (4096 latent values, base 64, lr 1.7e-4, batch 32, alpha 0.86) gave salt 0.1183, blur 0.1009, occlusion 0.1727 on the specialists' own corruptions; the final models (Optuna best trial: 2048 latent values, base 64, lr 3.75e-4, batch 64, alpha 0.589, 50 epochs) gave 0.1146, 0.1153, 0.1812. The two runs differ in epochs, batch size, learning rate and L1/SSIM weight, so they are not a controlled comparison, and the provisional values are not an Optuna result. The final models use the study's best trial, as the PDF specifies.
- **Latent grid, not a vector (D40, D42).** The specialists use the convolutional latent grid (2048 values, no skip connections), because the dense vector latent only reproduced an 8x8 thumbnail in Task 1. This must be stated in the report.
- **Val severities are tertiles** of the sampled parameter (D-level contract), not the fixed test severities; the test set was not used.
- The routing-failure analysis is on the validation set only.

## 5. Files and where they are

| What | Where |
|---|---|
| Final runs (ckpt_best, ckpt_last, metrics.jsonl, sample grids, confusion files) | `artifacts/runs/task2_classifier/20261004-1547_kaggle_t2cls_final_t2/`, `artifacts/runs/task2_specialist_{salt,blur,occlusion}/<run>/` |
| Studies (trials.csv, plots, DB) | `studies/t2_classifier/`, `studies/t2_specialist_shared/`; working DBs in `artifacts/optuna/` |
| Final configs | `configs/task2_classifier_final.yaml`, `configs/task2_specialist_final.yaml` |
| Evaluation (Kaggle) and local re-evaluation | `artifacts/eval/task2/20261004-161804_val/` and `20261004-213747_val/` (identical numbers) |
| Promoted checkpoints | `models/checkpoints/t2_classifier.pt`, `t2_ae_salt.pt`, `t2_ae_blur.pt`, `t2_ae_occlusion.pt` (not in git) |
| ONNX | `models/onnx/t2_classifier.onnx`, `t2_ae_salt.onnx`, `t2_ae_blur.onnx`, `t2_ae_occlusion.onnx` + `.meta.json` sidecars (`smoke: false`) |
| Hashes | `models/MANIFEST.json` (merged with Task 1's entry) |
| Parity table | `report/tables/onnx_parity.csv` (4 Task 2 rows, tag `final`, merged below Task 1's row) |
| Run logs and summary | `artifacts/logs/t2/` (`t2_status.log`, `t2_summary.json`); the original Kaggle zip is in `dist/kaggle_results/t2_results.zip` |

sha256 of the promoted checkpoints: classifier `7d4a9a1f8a073e64e8cc21deec4db98d51ad48dc04eb9b5b4215f49102ece028`, salt `4f8b2edb835c3b967fc8ef4ea1d43a4092841e3bfde4bd583dfb2c3a086eafee`, blur `0d2aede1d0636d4f65115b2c0d064319cc7a0fd22b55a24fdf9d1af9a750d74c`, occlusion `d8e82568179939796c1461265bed6af9412f5a727f6093bcc2b9370c2b64e1b4`. Task 3 must assert these at startup.

## 6. Verification done locally (this machine, after downloading the zip)

| Check | Result |
|---|---|
| sha256 and size of the four promoted checkpoints against `models/MANIFEST.json` | all four match; none is a smoke checkpoint (`run.smoke` false) |
| ONNX sidecars (`onnx_sha256`, `checkpoint_sha256`) against the files and the manifest | match, `smoke: false` |
| Fresh PyTorch vs ONNX Runtime parity on 16 local val inputs (4 per condition) | classifier max-abs diff 7.6e-06, salt 6.6e-07, blur 7.5e-07, occlusion 1.2e-06; tolerance 1e-4, all passed (Kaggle run: 6.7e-06, 4.5e-07, 5.7e-07, 5.4e-07) |
| Routing parity (ONNX classifier picks the same class as PyTorch) | 16 of 16, and the 16 inputs contain all four classes |
| Independent re-evaluation of the promoted checkpoints on the full val manifest | oracle J 0.1028, predicted J 0.1032, accuracy 0.991, 27 misroutes: identical to the Kaggle evaluation |
| Test set | not used by any Task 2 command. `artifacts/test_access.log` contains one line (21:07 local, 16:07 UTC) written about 30 minutes before the local re-evaluation; it did not come from Task 2 (all Task 2 evaluations use `final_test=False`, and my run added no line), most likely the Task 1 session |

## 7. Not done

- Test-set evaluation (needs the student's approval; the Task 2 test numbers for the report will come from one `--final-test` run later, together with Tasks 1 and 3).
- Docker Compose run with the real Task 2 ONNX files and the real `/api/hard`, and a visual check of the Hard-Routed page in a browser (the backend code and tests are done; the real models are now in `models/onnx/`).
- Task 2 figures and tables for the LaTeX report are not generated yet (the evaluation folder holds the inputs).

## 8. App check with the real ONNX files

Backend started locally (`uvicorn app.main:app`, CPU) with `models/onnx/`; script and raw results: `artifacts/eval/task2/api_check.json`, example images `artifacts/eval/task2/api_*_{input,output}.png` and `api_check_montage.png`.

- `GET /api/health`: `t2_classifier`, `t2_salt`, `t2_blur`, `t2_occlusion` all `loaded: true`, `smoke: false`, sha256 equal to the ONNX sidecars.
- `POST /api/hard` with the sample `Abyssinian_201`, seed 42, 10 calls (no corruption; salt-and-pepper, blur and occlusion at low, medium and high): all returned HTTP 200 with four probabilities summing to 1, a predicted class, the matching expert, `identity_bypass` and the four timing fields. In all 10 calls the predicted class was the applied corruption (clean for no corruption, with p(clean) 0.997); the smallest winning probability was 0.9877 (occlusion, low). Clean input used `expert: identity`, `identity_bypass: true`, `timing_ms.expert: 0`.
- One upload through `file=` returned 200 (blur/high predicted as `gaussian_blur`, expert `blur`); a text file returned 415; `POST /api/universal` still returned 200.
- Docker Compose (run later, after Docker Desktop was started): `docker compose build` and `docker compose up -d` succeeded, both containers started with `models/onnx` mounted read-only. Through nginx on `http://localhost:8080` the same script gave identical results: health shows the four Task 2 models loaded with `smoke: false` and matching sha256, all 10 `/api/hard` calls returned 200 with the correct expert (predicted class equals the applied corruption in all 10), the upload returned 200, the text file 415, `/api/universal` 200, and the index page loaded (HTTP 200). Results: `artifacts/eval/task2/api_check_docker.json`.
- Browser check (Chrome driven by Playwright against the Docker stack, with the second frontend `app/frontend_v2` now used by Compose): `app/frontend_v2/verification/real_backend.mjs`, results and screenshots in `app/frontend_v2/verification/real/` (copies for the report in `report/figures/app/`). The Hard-Routed page showed, for none / salt / blur / occlusion at medium severity, four labelled probability bars (top class 99.7 / 100 / 99.98 / 99.96 percent), the predicted corruption, the selected expert or Identity bypass, the input and restored images, the timing breakdown and a download link; no uncaught page errors. This was an automated run, not a manual look by the student.

## 9. Report assets (script `tools/t2_report_assets.py`, about 6 seconds, reads existing files only)

Tables in `report/tables/task2/` (CSV and LaTeX): `results_oracle`, `results_predicted` (condition x severity with MAE, SSIM, PSNR, J, `SSIM_input`, `J_input`, count), `classifier_report`, `confusion_normalised`, `misroute_confusion`, `study_summary` (counts and best trial of both studies), `model_facts` (parameters: classifier 98,196; each specialist 1,322,507; constructor arguments). Figures in `report/figures/task2/`: `training_curves.png`, `j_by_condition.png` (input vs oracle vs predicted), `restoration_examples_{salt,blur,occlusion}.png` (rows of target | input | output | error x4, medium-severity validation rows), `confusion_normalised.png`, `routing_failures_worst.png`, and the Optuna plots of both studies (`cls_*`, `spec_*`: optimisation history, parameter importance, parallel coordinates). The numbers in the tables are identical to those in sections 2 and 3 of this report.

## 10. Test results (final evaluation, run once with the student's approval)

`scripts/evaluate.py --task t2cls --final-test` with the four promoted checkpoints; evaluation folder `artifacts/eval/task2/20261004-234854_test/`; 3,669 test images x 10 conditions = 36,690 cases (identical manifest tensors for both routing modes); the access is the second line of `artifacts/test_access.log` (2026-10-04 18:48 UTC). Tables and figures: `report/tables/task2/test/`, `report/figures/task2/test/` (built by `tools/t2_report_assets.py --eval-dir <test folder>`).

| Condition (images) | Input J | Task 1 universal J | Task 2 oracle J | Task 2 predicted J |
|---|---|---|---|---|
| clean (3,669) | 0.0000 | 0.1046 | 0.0000 (identity) | 0.0037 |
| salt and pepper (11,007) | 0.3307 | 0.1107 | 0.1186 | 0.1185 |
| gaussian blur (11,007) | 0.1076 | 0.1237 | 0.1273 | 0.1273 |
| occlusion (11,007) | 0.1944 | 0.1742 | 0.1799 | 0.1783 |
| **overall (36,690)** | 0.1898 | 0.1330 | **0.1277** | **0.1276** |

SSIM overall: input 0.671, Task 1 0.775, Task 2 oracle 0.787, Task 2 predicted 0.787. The same Task 1 test evaluation is in `report/tables/t1_test_results.csv`.

- **How to read the overall row:** the four conditions are not equally weighted (clean 10 percent, each corruption 30 percent of the cases). Task 2's lower overall J comes from clean images, which the identity bypass leaves unchanged (J 0.0000, against 0.1046 for the universal autoencoder). On the three corrupted conditions Task 1's J is lower than Task 2's for each one; the mean over the three corrupted conditions is 0.1362 for Task 1, 0.1419 for Task 2 oracle and 0.1414 for Task 2 predicted.
- **By severity (oracle J vs input J):** salt lower than the input at every severity (low 0.1136 vs 0.2062, medium 0.1171 vs 0.3504, high 0.1251 vs 0.4354). Blur: higher than the input at low (0.1155 vs 0.0361) and medium (0.1150 vs 0.1122), lower at high (0.1514 vs 0.1744). Occlusion: higher at low (0.1451 vs 0.0913), lower at medium (0.1742 vs 0.1817) and high (0.2205 vs 0.3103).
- **Classifier on the test manifest:** accuracy 0.9909, macro precision 0.9828, macro recall 0.9878, macro-F1 0.9852. Per class F1: clean 0.957, salt 1.000, blur 0.996, occlusion 0.988. Clean has the lowest precision (0.941) because 208 occluded images were predicted as clean.
- **Routing failures:** 335 of 36,690 cases (0.91 percent). By true class (counts of predicted class): clean to blur 64 and to occlusion 37; salt to clean 4; blur to clean 10 and to occlusion 5; occlusion to clean 208 and to blur 7. The mean J change on misroutes is -0.0141 (J is on average lower after a misroute): the occluded images predicted as clean receive the identity output, whose J (the input's) is lower than the specialist's for the smaller occlusions (occlusion low: predicted-routing J 0.1403 vs oracle 0.1451). The largest J increases are clean photos routed to an expert; in `routing_failures_worst.png` the worst cases are mostly clean photos with large dark backgrounds sent to the occlusion expert, whose output then differs strongly from the clean photo.
- The validation numbers of sections 3 to 4 are unchanged; the test numbers are of the same kind (same model, never used for any choice).
