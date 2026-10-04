# Task 2 on Kaggle: steps for the student

Everything for Task 2 runs in ONE notebook (`notebooks/kaggle_t2_pipeline.ipynb`), unattended, after the Task 1 run has finished:
classifier study (12 trials x 8 epochs) -> classifier final training (30 epochs) -> specialist shared study (10 trials, each trial
trains salt, blur and occlusion for 8 epochs) -> three specialist final trainings (50 epochs each) -> validation evaluation
(oracle and predicted routing, with the "do nothing" baseline) -> the four ONNX files + parity checks -> promotion of the four checkpoints.
Nothing in it reads the test set. Budgets: decision D47 in `docs/DECISIONS.md` (replaces the 40-trial plan of D44;
the report states the number of trials that actually completed).

**Expected time:** about 1.1 hours (65 to 70 minutes). It is a guess from the local RTX 3050 (classifier 3.3 s/epoch, specialist about
7.5 s/epoch, mixed precision on), NOT a T4 measurement:
classifier study about 5 min, classifier final 2 min, specialist study about 30 min (less with pruning), three specialist finals about
19 min, evaluation + export + promotion + the runner's 30-second stage polls about 8 to 10 min.
Re-estimate it from the first Kaggle status lines: note when `cls_study` finishes and how long the first specialist trial takes. If a
specialist trial takes more than about 3.5 minutes, lower the specialist `n_trials` in `configs/task2_specialist.yaml` (and rebuild the
code zip). If time is left, the specialist study could go up to 15 trials at most (about 75 to 80 minutes in total), not more.

## 0. Files (the same data dataset as Task 1, a NEW code dataset)

| File | Becomes |
|---|---|
| `dist/cloud_data/pets_data.zip` | your existing private data dataset (no change needed) |
| `dist/cloud_code/genai_code.zip` (rebuilt for Task 2) | a **new dataset** (not a new version), e.g. `genai-code-t2` |

Why a new dataset: in Task 1 a new *version* of the code dataset was not picked up (a notebook stays pinned to the version it was attached
with) and the run kept failing with the old code. (Task 1's run does not need to be finished first: Task 2 uses its own notebook and its
own `/kaggle/working`.) If you change code later:
`.venv\Scripts\python.exe tools\package_cloud_code.py`, upload it as another NEW dataset (`genai-code-t2-2`), remove the old code input
from the notebook (Add Input / the x next to it), and restart the session.

## 1. Kaggle setup

1. Datasets > New Dataset > upload `dist/cloud_code/genai_code.zip` > Private > name it `genai-code-t2`.
2. Code > New Notebook > File > Import Notebook > `notebooks/kaggle_t2_pipeline.ipynb`. It must have exactly 4 code cells; delete any other cell.
3. Add Input > your data dataset and `genai-code-t2` (remove the old Task 1 code dataset input). Settings: **Accelerator = GPU T4 x2**
   (one GPU is used), **Internet = On**, Environment = "Always use latest", Persistence = "Files only".
4. Add-ons > Secrets > `WANDB_API_KEY` (the same secret as Task 1), ticked for this notebook. W&B group: `task2`.

## 2. Run

1. Cell 2 starts with `DRY_RUN = True`: run all cells once (about 10 minutes, tiny data). It must end with `ALL STAGES DONE`.
   Check three things in its output:
   - Cell 1 prints `code version: t2-v3` (a different value means old code is still attached);
   - the first status lines say `preflight: wandb logged in as entity ahmedlaiq34` (otherwise the secret is missing or not ticked);
   - the stages `evaluate` and `export` pass (they are the ones that read the data from `/kaggle/input`; Task 1's export stage failed there once).
2. Set `DRY_RUN = False`, then **Save Version > Save & Run All (Commit)** (runs in the background, browser may be closed).
   "Run All" in the editor is an interactive session that can stop when the browser closes.
3. If the session stops: open the notebook and re-run with the **same `TAG`**. Finished stages are skipped, an interrupted study continues,
   an interrupted training resumes from its last checkpoint (this needs the files in `/kaggle/working`, so restart from the saved version's output).

## 3. When it says `ALL STAGES DONE`

Cell 4 builds `/kaggle/working/t2_results.zip`. Download it and unzip it into the repository root. It contains:

- `studies/t2_classifier/`, `studies/t2_specialist_shared/` (trials.csv, plots, study DB) and `artifacts/optuna/`
- `configs/task2_classifier_final.yaml`, `configs/task2_specialist_final.yaml`
- `runs/task2_classifier/...`, `runs/task2_specialist_{salt,blur,occlusion}/...` and `eval/task2/...` (move both under `artifacts/`)
- `models/checkpoints/t2_*.pt`, `models/MANIFEST.json` (sha256 of the four checkpoints), `models/onnx/t2_*.onnx` (+ sidecars),
  `report/tables/onnx_parity.csv`
- `artifacts/logs/t2/` (status log, per-stage logs, `t2_summary.json`)

Careful with `models/MANIFEST.json` and `report/tables/onnx_parity.csv`: the zip's copies only know Task 2. Tell the assistant instead of
overwriting them; it merges them with the Task 1 entries, checks the sha256 values and the ONNX parity locally, and writes the Task 2 training report.

## 4. Things that can go wrong

- `code dataset not found (or an old version ...)`: the code dataset is missing or still the old version without `tools/t2_pipeline.py`.
- `pip install` fails: Internet is off.
- Data not found: the data dataset is not an input.
- W&B errors: the secret is missing or not ticked; after two failures the runner logs offline and syncs at the end.
- The runner stops only after one stage failed 4 times in a row without progress; the reason is the last line of
  `artifacts/logs/t2/t2_status.log` and the stage log next to it.
