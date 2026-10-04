# Task 1: what we did and the problems we solved (for the Task 2 agent's review)

Audience: the agent that wrote Task 2's code and will now train it on Kaggle. Purpose: use this as a **review checklist against your own work**.
Every item is a mistake or obstacle that cost Task 1 real time. For each one: the symptom, the root cause, the fix, and **how to check whether Task 2 has the same problem**.
Written 2026-10-04 by the Task 1 lead, before the final Task 1 training run on Kaggle finished (its result is not in this file).
Companion file: `docs/LESSONS_FROM_TASK1.md` (the measurements behind the modelling decisions). This file is the chronology and the checklist.

## 0. Where Task 2 stands (from a quick audit of the repository on 2026-10-04 evening)

Already done correctly in Task 2 (verified by reading the code, not by running it): sampler seed offset, stale `RUNNING` trials marked `FAIL`, budget counts only `COMPLETE`+`PRUNED`
(`tasks/task2/runs.py`), conv latent in `configs/task2_specialist.yaml`, pruner warm-ups raised (classifier `n_startup 8 / n_warmup 5`, specialist `8 / 2`), `data_root` passed to the
parity checks in `tools/t2_pipeline.py`, failure-tail lines in the runner, `dirs_exist_ok=True` and `mkdir` in the notebook cells. `tools/t2_pipeline.py` parses.
Still to review: sections 2.9 to 2.13 below (Kaggle specifics) and the checklist in section 4.

## 1. Timeline of Task 1 (2026-10-03 and 10-04)

1. **Setup.** Python 3.13 venv, torch cu128 on the RTX 3050. A pip install collided with a second one started while the first was still running (WinError 32). Scaffold scripts were named `scripts/*.py.py`; renamed.
2. **Build.** Sub-agents wrote the corruptions/split/manifests (23 tests), shared utilities (18 tests), the dataset layer and samplers (14 tests), the Task 1 model/train/tune/evaluate/export, a backend and a frontend. A session restart interrupted two agents; they were resumed.
3. **Smoke run** (50 steps, kill and resume, ONNX parity): worked, but two resumes hung or failed from memory exhaustion (see 2.1).
4. **First Optuna study attempt (12 trials, stopped).** W&B entity wrong, a trial hung, a trial repeated, the pruner pruned too early (2.2 to 2.5).
5. **Unattended runner** `tools/t1_overnight.py` written and dry-run tested; the dense-model study (40 trials), final training (100 epochs), evaluation, ONNX export and promotion all ran overnight. One bug at the last stage (2.6).
6. **Audit against the PDF.** The numbers looked fine, the images did not: blurry blobs. The result was weak (2.7).
7. **Diagnosis in about an hour** with throwaway scripts (calibration, overfit test, bottleneck sweep, conv-latent prototypes). Decision D40: conv latent.
8. **Move to Kaggle.** New notebook, code and data zips, runner options. The dry run on Kaggle found three problems the local dry run could not (2.9 to 2.12).
9. **Final Task 1 run committed on Kaggle** (in progress when this was written).

## 2. Problems and fixes

### 2.1 Memory exhaustion on the Windows laptop (15.6 GB RAM)
- Symptom: a resumed smoke run hung for more than 10 minutes; `WinError 1455` (paging file too small); later, mid-study, a DataLoader worker died with `WinError 1114` loading `torch\lib\shm.dll` and the trial hung forever.
- Root cause: several Python processes at once (training, pytest runs from another session, stale processes from interrupted sessions). Every Windows DataLoader worker re-imports torch, about 0.5 GB each. C: (where the pagefile lives) was 98 to 100 percent full.
- Fix: `num_workers=0` for training data and `val_workers=0`; run one heavy process at a time; temp files on D: (`TMP`/`TEMP`, pytest `--basetemp`).
- Check Task 2: does any Windows-side run use workers? Are there leftover processes before a run (`tasklist`)? On Kaggle (Linux) this does not apply, the profile's `num_workers: 2` is fine.

### 2.2 Wrong W&B entity
- Symptom: `CommError: entity ahmedlaiq not found during upsertBucket` at the first trial, which also left a FAILED trial in the study DB.
- Root cause: username `ahmedlaiq` is not the entity; the account's entity is **`ahmedlaiq34`**.
- Fix: `WANDB_ENTITY=ahmedlaiq34`; delete the polluted DB (only after the student agreed).
- Check Task 2: the runner env sets `WANDB_ENTITY`/`WANDB_PROJECT`; on Kaggle the key comes from the secret `WANDB_API_KEY`. The Task 1 dry run printed "wandb logged in as entity ahmedlaiq34", so a missing secret shows up in the first lines.

### 2.3 A hung trial
- See 2.1. A trial killed from outside stayed `RUNNING` in the SQLite DB forever.
- Fix: at study start every `RUNNING` trial becomes `FAIL`; the supervisor kills a child whose log is silent for 20 minutes and restarts it.
- Check Task 2: `runs.py` marks them, and `tools/t2_pipeline.py` has the stall detection (`STALL_MINUTES = 20`), both verified by reading the code.

### 2.4 A repeated trial after a restart
- Symptom: after resuming, trial 2 had exactly the parameters of trial 0.
- Root cause: `TPESampler(seed=42)` restarts its random stream when the process restarts.
- Fix: `seed = 42 + number of existing trials` (the Task 2 code does this).

### 2.5 The pruner killed trials before they had a chance
- Symptom: 5 of 11 trials pruned at epoch 1 to 3; four of the five used batch 128.
- Root cause: the validation metric sits on a plateau (J about 0.46) for the first 4 to 6 epochs, then falls. The median pruner (warm-up of 1 epoch) compared trials before they left the plateau, and a large batch means fewer steps per epoch, so it leaves the plateau later: the pruner selected for speed, not quality. It also fires the "fewer than half of trials completed" stop condition.
- Fix: `n_startup_trials 8`, `n_warmup_steps 8` for 15-epoch trials (no pruning before epoch 9).
- Check Task 2: **look at one real learning curve for each study before trusting the warm-up number**, especially the classifier (macro-F1 may sit near 0.25, chance, for several epochs). The specialist study reports after each *corruption* (steps 1, 2, 3), not epochs, so check what warm-up 2 actually skips (the first two corruptions, i.e. pruning is only possible after the third).

### 2.6 `promote()` crashed on the real `models/MANIFEST.json`
- Symptom: at the last stage of the overnight run, `AttributeError: 'str' object has no attribute 'get'`.
- Root cause: the repository's manifest is `{"version": 1, "models": []}`; the code assumed a bare list. The dry run used a temporary `models/` folder, so it never read the real file.
- Fix: accept both layouts and keep the wrapper (D39, test added). The supervisor's automatic retry then completed the stage.
- Check Task 2: promotion of 4 checkpoints calls the same function (fixed). **Dry runs that write to temp folders do not test the real files; open the real `MANIFEST.json` once.**

### 2.7 The dense model was weak (the biggest finding)
- Symptom: validation J 0.3134, SSIM 0.454; outputs were blobs; clean inputs reconstructed no better than corrupted ones.
- Diagnosis (details in `LESSONS_FROM_TASK1.md` section 3): the model reproduced an 8x8 thumbnail (the thumbnail's J is 0.3131); a 4x bigger bottleneck changed nothing; the same model overfits 64 images to J 0.09, so the code is fine; a flatten + dense latent generalises badly from 2,944 images.
- Fix: convolutional latent grid (D40). A 16x16x16 latent gives J 0.139 / SSIM 0.761 in 30 epochs.
- My first hypothesis ("the bottleneck range is capped at 512") was **wrong** and was only discovered by measuring before re-running a 2-hour study. Measure first.
- Check Task 2: `configs/task2_specialist.yaml` now uses the conv latent. For the blur specialist, compare with the "do nothing" baseline (blur input J 0.0908; the all-conditions conv model scored 0.116 on blur, worse than the input).

### 2.8 Tests that silently depend on the real config
- Symptom: after `configs/task1_universal.yaml` received real budgets, a test that expected a `TBD` error ran the full 40-trial study; after switching to the conv latent, 8 pipeline tests failed because their tiny config inherited `latent: conv, bottleneck_dim: 4096`.
- Fix: tests pin their own tiny model (`latent="dense", depth=4, base_channels=8, bottleneck_dim=32`) and build the TBD case explicitly.
- Check Task 2: run the whole suite after any config change; look for tests that call `load_config(None, "local")` and then only override some keys. Run the Task 2 tests with `CUDA_VISIBLE_DEVICES=-1` (without it 2 routing tests fail with a device mismatch).

### 2.9 Kaggle: a hard-coded local data path (the failure of the first Kaggle dry run)
- Symptom: stages study/final_config/train/evaluate passed on the T4, **export failed 4 times**: `FileNotFoundError: /kaggle/working/repo/data/splits/pets_split.json`.
- Root cause: `verify_parity(..., data_root="local")` defaults to the repository's `data/` folder; on Kaggle the data is under `/kaggle/input/...`. Evaluation worked because it takes the config's data root.
- Fix: pass `data_root=<the run's config>` (done in `t1_overnight.py`, and in `t2_pipeline.py`).
- Check Task 2: **`task2_export.py` and `onnx_verify.py` still have `data_root="local"` as the default** (`verify_task2_parity`, `verify_routing_parity`, `val_inputs`). The runner passes `data_root` explicitly (verified at `t2_pipeline.py` lines 278 to 294), so it is safe there, but any other caller (scripts, notebooks, manual runs on Kaggle) would break the same way. Consider making the argument required, or defaulting to the profile in use. Also grep for `resolve_data_paths("local")` in anything that can run on Kaggle (`tools/diag_quick_checks.py` does, it is local-only by design).
- A local dry run **cannot** find this class of bug. The only dry run that counts for paths is the one on Kaggle itself. Reproduce locally by temporarily hiding `data/` (I did, then restored it) or by unzipping `pets_data.zip` into a fake `input/pets-data` folder and passing that as `data_root`.

### 2.10 Kaggle: the notebook kept running OLD code after a new dataset version
- Symptom: after uploading a corrected `genai_code.zip` as a new version of the code dataset, the notebook still failed the same way and `"data_root=final_cfg" in tools/t1_overnight.py` printed `False`.
- Root cause: a notebook keeps its input pinned to the version it was attached with; Cell 1 originally copied the code only `if not os.path.isdir("/kaggle/working/repo")`, so the old copy also survived inside the session.
- Fix: (a) Cell 1 now always runs `shutil.copytree(CODE_DIR, "/kaggle/working/repo", dirs_exist_ok=True)` (keeps `artifacts/` for resume, overwrites code); (b) the reliable way to get new code in is a **new dataset** (`genai-code-2`), removing the old input from the notebook and restarting the session.
- Check Task 2: `docs/KAGGLE_T2_STEPS.md` says to upload "a new version of your private code dataset". That is exactly what failed for Task 1. Use a **new dataset name** instead (the notebook finds `tools/t2_pipeline.py` by glob, so the name does not matter), remove the old code dataset from the notebook, and verify with a one-line check (`"<some new string>" in Path(".../tools/t2_pipeline.py").read_text()`) before a long run.

### 2.11 Kaggle: Cell 4 failed because nothing existed to copy
- Symptom: `FileNotFoundError: /kaggle/working/t1_v2_results` in `shutil.make_archive`.
- Root cause: in dry-run mode the outputs live in `artifacts/dryrun_overnight/...`, not in the paths Cell 4 collected, so the staging folder was never created.
- Fix: create the staging folder first, include the dry-run folders and choose the status-log path by `DRY_RUN`.
- Check Task 2: its Cell 4 has `stage.mkdir(parents=True)` and the dry-run paths (verified in the notebook source). Re-check that the real-run paths in Cell 4 match what the Task 2 runner really writes (run ids, `eval` folders).

### 2.12 A debugging cell must not stay in a committed notebook
- The extra "diagnostic" cell (reads a dry-run log that does not exist in a fresh session) would have raised an error at the end of the 2-hour commit and marked the whole run failed. It was deleted before committing.
- Check Task 2: open the notebook and make sure it has exactly the four intended cells before **Save & Run All (Commit)**.

### 2.13 Kaggle setup facts that were not obvious (all confirmed in the screenshots/logs)
- Internet must be **On**; the GPU option is `T4 x2` but the code uses only one GPU; the environment should be "Always use latest" (the pinned one is an older snapshot); Persistence "Files only" keeps `/kaggle/working` if a session restarts.
- A dataset mounts at `/kaggle/input/datasets/<user>/<name>/` (nested); the data lookup (`pets/dataset.py::_find_dir`) now searches up to 8 levels (test added). The code dataset is found by globbing for `tools/<runner>.py`.
- The W&B key goes in an Add-ons > Secrets entry named `WANDB_API_KEY` and must be ticked for the notebook; the runner prints "wandb logged in as entity ahmedlaiq34" at the start when it works.
- The Kaggle container runs Python 3.13 (`/usr/lib/python3.13`); the code runs there unchanged.
- Use **Save Version > Save & Run All (Commit)** for the unattended run; **Run All** in the editor is an interactive session that can stop with the browser. A factory reset clears `/kaggle/working`.
- The dry run takes a few minutes and is mandatory; it found two of the three Kaggle problems above.

### 2.14 Small mistakes worth avoiding (caught by tests or by the dry run)
- A scripted edit wrote `"\n"` as a real line break inside an f-string in `tools/t1_overnight.py` (`SyntaxError`). It was caught only because the runner was dry-run again after editing. **After editing a runner or a notebook, parse it and dry-run it.**
- A scripted multi-edit aborted midway on a failed string match and wrote nothing; re-apply edits one at a time and check that each target string exists.
- A shell filter (`grep`) buffered the progress output of a background job; read the run's own metrics file instead.
- Background commands cannot be waited on for more than 10 minutes in the foreground; run long jobs in the background with a log file.
- Watchers (monitors) expire after about 30 minutes; the real "finished" signal is the background job's own exit.

## 3. Decisions taken (all recorded in `docs/DECISIONS.md`)

D16 and D17 local training with the approved budget (40 trials x 15 epochs, 100 final epochs), D18 W&B entity, D19 study robustness fixes and runner, D37 test that depended on the real config,
D39 `promote()` layouts, D40 conv latent, D41 Kaggle transport. Task 2's own D42 onwards mirror them.

## 4. Review checklist for Task 2 before the Kaggle run

1. `grep -rn 'data_root="local"\|resolve_data_paths("local")'` over `src/`, `tools/`, `scripts/`: every call reachable on Kaggle gets the run's config (section 2.9).
2. Open the real `models/MANIFEST.json` and run `promote()` on a copy of the real layout once (section 2.6).
3. Every study: take one real learning curve (the quick-check tool or one short training) and set the pruner warm-up after the plateau; confirm the specialist steps (1, 2, 3 corruptions) mean what the config comments say (section 2.5).
4. Confirm the sampler seed offset, RUNNING-to-FAIL and answered-trial budget by a two-trial dry run that you interrupt (the Task 1 dry run killed the study child mid-trial and the pipeline recovered).
5. Baseline: every restoration table has `J_input`/`SSIM_input` columns; the blur specialist is compared with its input (section 2.7). Report honestly if it does not beat the input.
6. The classifier: chance-level macro-F1 is about 0.25; check the first epochs and that the balanced sampler gives exact class counts (tested).
7. Code transport: a new Kaggle dataset (not a new version), old input removed, a one-line "new code present" check (section 2.10).
8. Notebook: exactly the intended cells, `DRY_RUN = True` first, then `False` with a Commit run (sections 2.11 to 2.13).
9. The W&B secret is attached and the first lines of Cell 3 print `wandb logged in as entity ahmedlaiq34`.
10. The runner prints the tail of a failing stage's log into the notebook output (verified in `t2_pipeline.py`), so a failure needs no second round trip; keep that.
11. No test data: the Task 2 pipeline's evaluation stage must be on the validation manifest (`final_test=False`); the locked test set needs the student's approval.
12. Time estimate: do not trust a number from the local laptop for a T4; read the first trial's time from the Kaggle log and re-estimate.

## 5. Unknowns (not verified)

- The final Task 1 result on Kaggle (study best trial, final J, ONNX file) was not available when this was written; see `docs/PHASE_T1_TRAINING_REPORT.md` when it exists.
- Whether Kaggle allows two GPU notebooks to run at the same time on the student's account was not checked; `docs/KAGGLE_T2_STEPS.md` wisely says to wait for the Task 1 run to finish.
- The blur-specialist hypothesis (a larger latent might help) is untested.
- No T4 timing for the conv model exists yet.
