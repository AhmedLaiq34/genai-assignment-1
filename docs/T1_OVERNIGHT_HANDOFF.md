# Task 1 overnight run: handoff for the session that drives it

Written 2026-10-04 by the Opus session that diagnosed the first study attempt and prepared this run.
The student is asleep. Your job: launch the run, watch it, fix problems yourself, and when it is done
write `docs/PHASE_T1_TRAINING_REPORT.md`. Only stop early if the run cannot be repaired.

## 1. What happened before (do not repeat it)

The first study attempt (12 trials, 2026-10-04 02:09 to 02:28) was stopped. Its DB is kept at
`artifacts/optuna/old/t1_universal_attempt1.db` and its logs at `artifacts/logs/t1_study_part1.log` and
`artifacts/logs/t1_study.log`. Best trial there: trial 0, val J 0.3555 (lr 3.6e-4, batch 32, bottleneck 512,
64 channels, dropout 0.25, alpha 0.60). Root causes and fixes (decision D19 in `docs/DECISIONS.md`):

| Problem | Cause | Fix now in place |
|---|---|---|
| Trial hung; DataLoader worker died with WinError 1114 | Memory/commit pressure: C: was nearly full (pagefile), Task 2 window ran pytest at the same time, every Windows worker re-imports torch (~0.5 GB) | Training data loaded with `num_workers=0` (no worker processes); temp files on D: |
| 5 of 11 trials pruned at epochs 1 to 3 | Val J sits on a ~0.46 plateau for the first 4 to 6 epochs; pruner warm-up was 1 epoch | MedianPruner `n_startup_trials 8`, `n_warmup_steps 8` |
| Trial 2 repeated trial 0 after the restart | TPE seed 42 re-drew the same first sample | Sampler seed = 42 + number of existing trials (`task1/tune.py`) |
| Killed trial stayed RUNNING | Process killed mid-trial | RUNNING trials are marked FAIL at start-up |
| A failed trial used up the budget | Budget counted every finished trial | Only COMPLETE + PRUNED count toward `n_trials=40`; failed trials are retried |
| Wrong W&B entity | Username `ahmedlaiq` is not the entity | `WANDB_ENTITY=ahmedlaiq34` |

The study was restarted from scratch with the fixes (the student approved losing the first attempt).
Objective (J = 0.5*L1 + 0.5*(1-SSIM)), search ranges, 40 trials, 15 epochs per trial and 100 final epochs are
unchanged.

## 2. The runner

`tools/t1_overnight.py` runs every stage as a child process and supervises it:

`study` (40 trials) -> `final_config` (writes `configs/task1_final.yaml` from the best trial)
-> `train` (100 epochs, fixed run id, resumes from its own `ckpt_last.pt`) -> `evaluate` (VAL manifest only)
-> `export` (`models/onnx/t1_universal_ae.onnx` + sidecar with smoke=false, parity on 16 val inputs,
row in `report/tables/onnx_parity.csv`) -> `promote` (`models/checkpoints/t1_universal_ae.pt`, `models/MANIFEST.json`).

- A crashed child is restarted after 60 s; a child whose log is silent for 20 min is killed and restarted.
- Stage state: `artifacts/logs/t1_overnight_state.json`. Human-readable progress: `artifacts/logs/t1_overnight_status.log`.
  Per-stage logs: `artifacts/logs/t1_<stage>.log`. Final facts: `artifacts/logs/t1_overnight_summary.json`.
- It stops itself only if one stage fails 4 times in a row without progress (status line `STOPPED: ...`).
- If W&B keeps failing it switches to `WANDB_MODE=offline` and runs `wandb sync` at the end.
- It asks Windows not to sleep while it runs.
- Rerunning the same command continues where it stopped (finished stages are skipped).

Verified on 2026-10-04 with `--dry-run` (tiny data, tmp folders, no W&B): all six stages passed, and a
second dry run where the study child was killed mid-trial recovered on its own (trial marked FAIL, retried,
pipeline completed). `tests/test_task1_pipeline.py`: 9 passed.

Expected time: study about 2 to 2.5 h, final training about 30 to 45 min, the rest a few minutes.

## 3. Launch (once)

First check it is not already running and not already finished:
```
tasklist | grep -i python
cat artifacts/logs/t1_overnight_status.log    # may not exist yet
```
Launch from the repo root with the Bash tool, `run_in_background: true` (you get a notification when it exits):
```
.venv/Scripts/python.exe tools/t1_overnight.py > artifacts/logs/t1_overnight_console.log 2>&1
```
Do not set TRACKER/WANDB variables yourself; the runner sets them for its children.

## 4. Watching

- Arm a Monitor on the status log (it gets one line per stage event and every failure):
  `tail -n 0 -F artifacts/logs/t1_overnight_status.log`
- For trial-level progress: `grep -E "Trial [0-9]+ (finished|pruned|failed)" artifacts/logs/t1_study.log | tail`
- During `train`: `tail -3 artifacts/logs/t1_train.log` (one line per epoch).
- First progress report: confirm the first trial's W&B run appears (the URL is printed in `t1_study.log`,
  project https://wandb.ai/ahmedlaiq34/genai-a1).
- Report progress each time you check. Do not poll more often than every few minutes.

## 5. If something goes wrong (fix it, then rerun the launch command; the runner resumes)

1. Read the end of `artifacts/logs/t1_<stage>.log` and `t1_overnight_status.log`.
2. Typical cases:
   - **CUDA out of memory in one trial:** the trial is marked FAIL and the study continues by itself. Only act if
     it repeats: check `nvidia-smi` for another process on the GPU (the Task 2 window is CPU-only; `CUDA_VISIBLE_DEVICES=-1`).
   - **WinError 1114 / 1455 / "paging file" / "No space left on device":** memory or disk. Check C: free space
     (`df -h /c`). Free space by deleting only *your own* temp data (`artifacts/tmp/*`, `artifacts/pytest_tmp/*`,
     `artifacts/dryrun_overnight/`). Never delete data, splits, manifests, studies or other sessions' files.
   - **W&B errors:** the runner goes offline after 2 failures and syncs at the end. Nothing to do unless sync fails;
     then run `.venv/Scripts/python.exe -m wandb sync --include-offline wandb/offline-run-*` later.
   - **A Python exception in the code:** fix the bug in the Task 1 file it points to
     (`src/genai/tasks/task1/*`, `src/genai/models/autoencoder.py`, `src/genai/export/onnx_*.py`, `tools/t1_overnight.py`),
     run `pytest tests/test_task1_pipeline.py -q --basetemp=artifacts/pytest_tmp/t1`, record the fix in
     `docs/DECISIONS.md` (append-only; next free number after the last row), then relaunch.
   - **STOPPED after 4 failures:** same as above: diagnose, fix, relaunch. The stage's attempt counter keeps
     counting; the consecutive-failure counter restarts with the new supervisor.
3. Do **not**: change the objective, the search ranges, the budgets or the pruner mid-study; delete the study DB;
   run `--final-test` or read test data; edit Task 2 files (`src/genai/tasks/task2/**`, `models/classifier.py`,
   `export/task2_export.py`, `configs/task2_*`, `app/**`, `notebooks/**`, `scripts/*.py`); commit to git.
4. If the whole PC rebooted (e.g. Windows Update): just relaunch; everything resumes.

## 6. When the status log says `ALL STAGES DONE`

Write `docs/PHASE_T1_TRAINING_REPORT.md` from `artifacts/logs/t1_overnight_summary.json`, the eval folder
named in it (`table_cond_severity.csv`, `summary.json`, `representative_12.png`, `worst_4.png`), the parity
result and the logs. Contents:

- Benchmark confirmation (2026-10-04, batch 128: 0.107 s/step AMP off, 0.082 s/step AMP on; report said 0.107 / 0.0825).
- First attempt and why it was stopped (section 1 of this file), then the restarted study.
- Study: trials completed / pruned / failed, best trial number, best J, best parameters, full search space
  (from the summary), pruner settings, 15 epochs per trial, 40-trial budget (failed trials do not count).
- Final training: final train loss and val J, best epoch and its val J, first-epoch values, wall-clock.
- Validation table per condition (clean, salt_pepper, gaussian_blur, occlusion; MAE, SSIM, PSNR, J) and the
  severity rows. Note: val severities are tertile labels of the sampled training-range parameter, not the fixed
  test severities.
- ONNX parity (max-abs, mean-abs, tolerance 1e-4, passed), ONNX sha256, smoke=false.
- Promoted checkpoint path and sha256 from `models/MANIFEST.json`.
- W&B run URLs (in the summary) and the W&B mode used (online/offline).
- Wall-clock per stage, every failure/retry and what fixed it.
- Not done: test-set evaluation (locked), Tasks 2 to 4, Stitch design.
- Numbers only, no "good"/"bad". Flag oddities: one condition far worse than the others, flat curves, best
  epoch at the very end (still improving) or very early (overfitting), a pruning-heavy study, a best trial at the
  edge of a search range.

Then tell the student in one line that the GPU is free so the Task 2 window may use it.
