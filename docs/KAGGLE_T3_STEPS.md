# Task 3 on Kaggle: steps for the student

Everything for Task 3 runs in ONE notebook (`notebooks/kaggle_t3_pipeline.ipynb`), unattended, after Task 2 is finished.
Task 3 starts from the four Task 2 checkpoints (gate = classifier, experts = the three specialists) and the Task 1 autoencoder
(comparison column only), so they travel to Kaggle in a small extra dataset:
`prepare` (finds the five checkpoints, checks their sha256, measures seconds per step on the T4, prints the projected minutes of the
real run and stops if the projection is above the limit) -> Optuna study `t3_moe` -> final training (warm-up with frozen experts,
then joint fine-tuning) -> validation evaluation (Task 1, Task 2 and Task 3 on the same tensors, routing analysis) -> ONNX file with
`output` and `weights` + parity check -> promotion of `t3_soft_moe` -> second sha256 check of the five source checkpoints.
Nothing in it reads the test set. Design and budgets: `docs/TASK3_PLAN.md` (sections C and F); the budgets need your approval first.

**Expected time (estimate, NOT measured on the T4):** the runner takes about 19 to 30 minutes (plan section C): study 10 to 16 minutes
(12 trials of 1 warm-up + 3 joint epochs, capped at 16 minutes), final training 5 to 8 minutes (2 + 12 epochs), evaluation 1 to 2 minutes,
export + promotion about 1 minute, stage polling and process starts about 1 minute. The notebook setup (copy + `pip install`) comes on top
and is not included. The dry run (step 2 below) replaces these estimates by a measurement: the `prepare` stage prints the seconds per step
and the **projected minutes of the real run**; the real run is blocked (exit code 3) if the projection is above 35 minutes
(`budget.max_minutes`).

## 0. Files (a data dataset you already have, plus TWO NEW datasets)

| File | Becomes |
|---|---|
| `dist/cloud_data/pets_data.zip` | your existing private data dataset (no change needed) |
| `dist/cloud_models/t3_sources.zip` (`.venv\Scripts\python.exe tools\package_t3_sources.py`) | a **new private dataset** `t3-sources` (about 50 MB: the four Task 2 checkpoints, the Task 1 checkpoint, `SOURCES.json` with their sha256) |
| `dist/cloud_code/genai_code.zip` (`.venv\Scripts\python.exe tools\package_cloud_code.py`, rebuilt for Task 3) | a **new dataset** (not a new version), `genai-code-t3` |

The packaging script refuses (and writes nothing) if a file is missing or its sha256 differs from the Task 2 report / `models/MANIFEST.json`.
Package after the assistant has reported that its local checks passed; it prints the size and the hashes (keep them for the report).

Why a new dataset for the code: in Task 1 a new *version* of the code dataset was not picked up (a notebook stays pinned to the version it
was attached with) and the run kept failing with the old code. If you change code later: run `tools\package_cloud_code.py` again, upload it as
another NEW dataset (`genai-code-t3-2`), remove the old code input from the notebook (Add Input / the x next to it), and restart the session.
Never upload `models/`, `artifacts/`, `data/raw/` or the test data: only `t3_sources.zip` carries checkpoints.

## 1. Kaggle setup

1. Datasets > New Dataset > upload `t3_sources.zip` > Private > name `t3-sources`.
2. Datasets > New Dataset > upload `genai_code.zip` > Private > name `genai-code-t3`.
3. Code > New Notebook > File > Import Notebook > `notebooks/kaggle_t3_pipeline.ipynb`. It must have exactly 4 code cells; delete any other cell.
4. Add Input > your pets data dataset, `genai-code-t3` and `t3-sources` (remove older code inputs, for example `genai-code-t2`).
   Settings: **Accelerator = GPU T4 x2** (one GPU is used), **Internet = On**, Environment = "Always use latest", Persistence = "Files only".
5. Add-ons > Secrets > `WANDB_API_KEY` (the same secret as before), ticked for this notebook. W&B: entity `ahmedlaiq34`, project `genai-a1`, group `task3`.

## 2. Run

1. **Dry run.** Cell 2 starts with `DRY_RUN = True` and `BUDGET_OVERRIDES = []`: Run All once (a few minutes, tiny budgets: 2 trials of
   1 + 1 epochs, final 1 + 1 epochs). It must end with `ALL STAGES DONE`. Check:
   - Cell 1 prints `code version: t3-v1` (anything else = old code is still attached) and the path of `t2_classifier.pt`
     (if it asserts, the `t3-sources` dataset is not an input);
   - the status log says `preflight: wandb logged in as entity ahmedlaiq34` (otherwise the secret is missing or not ticked);
   - `prepare` prints `sources verified (sha256)` with four hashes (they must equal the four hashes printed by `package_t3_sources.py`),
     the `benchmark` line with seconds per step, and `projected minutes of the REAL run ... TOTAL <n> min`;
   - `evaluate` and `export_promote` pass (they read the data from `/kaggle/input`), and `export_promote` prints
     `sources verified again at the end: unchanged`.

   Send the status log (the last 3000 characters are printed by Cell 4, the whole file is in the results zip) to the assistant.
2. **Budget.** If the projected total is above 35 minutes, the real run stops at `prepare` (nothing is trained). Then put entries into
   `BUDGET_OVERRIDES` in Cell 2, in this order, one more each time until the projection fits (plan section C):
   `"timeout_minutes=10"`, `"train.joint_epochs=8"`, `"epochs_per_trial.joint=2"`, `"train.val_every_epochs=2"`.
   The overrides are applied to the study and to the final configuration and are echoed into the status log
   (`budget overrides (--set): ...`). Do not raise a budget above the approved values without asking.
3. **Real run.** Set `DRY_RUN = False`, then **Save Version > Save & Run All (Commit)** (runs in the background, the browser may be closed).
   "Run All" in the editor is an interactive session that can stop when the browser closes.
4. If the session stops: open the notebook and re-run with the **same `TAG`**. Finished stages are skipped, an interrupted study continues,
   an interrupted training resumes from its last checkpoint (this needs the files in `/kaggle/working`, so restart from the saved version's output).

## 3. When it says `ALL STAGES DONE`

Cell 4 builds `/kaggle/working/t3_results.zip`. Download it and give it to the assistant (do not unzip it over the repository yourself). It contains:

- `studies/t3_moe/` (trials.csv, plots, study DB) and `artifacts/optuna/t3_moe.db`
- `configs/task3_moe_final.yaml` (study config + the best trial)
- `runs/task3/<run_id>/` (final training: `ckpt_best.pt`, `ckpt_last.pt`, `metrics.jsonl`, samples) and `eval/task3/...` (move both under `artifacts/`)
- `models/checkpoints/t3_soft_moe.pt`, `models/MANIFEST.json`, `models/onnx/t3_soft_moe.onnx` (+ sidecar), `report/tables/onnx_parity.csv`
- `artifacts/logs/t3/` (status log, per-stage logs, `t3_summary.json`)
- after a dry run only: `artifacts/dryrun_t3/` without checkpoints and ONNX files

Careful with `models/MANIFEST.json` and `report/tables/onnx_parity.csv`: the zip's copies only know Task 3 (the code dataset has no `models/` folder).
They are **merged** with the existing files, never overwritten. The assistant places the runs under `artifacts/runs/task3/`, the evaluation under
`artifacts/eval/task3/`, the study under `studies/t3_moe/`, re-checks the hashes and the ONNX parity locally, re-runs the validation evaluation
(the numbers must match Kaggle), produces the report figures and tables and writes the Task 3 training report.

## 4. Things that can go wrong

- `code dataset not found (or an old one ...)`: the code dataset is missing, or it is an old one without `tools/t3_pipeline.py`.
- `sources dataset not found`: `t3-sources` is not an input of the notebook.
- `pip install` fails: Internet is off.
- Data not found: the pets data dataset is not an input.
- `ValueError ... sha256`: a source checkpoint differs from the Task 2 file. Nothing was trained. Upload `t3_sources.zip` again from the packaging script, never edit the files.
- `STOPPED: prepare: projected time above budget.max_minutes`: see step 2 of section 2 (`BUDGET_OVERRIDES`); this is not a crash.
- `--set KEY: this key is not in configs/task3_moe.yaml`: a typo in `BUDGET_OVERRIDES`.
- W&B errors: the secret is missing or not ticked; after two failures the runner logs offline and syncs at the end.
- The runner stops after one stage failed 4 times in a row without progress; the reason is the last line of
  `artifacts/logs/t3/t3_status.log` and the stage log next to it (`t3_<stage>.log`).
