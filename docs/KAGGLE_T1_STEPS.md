# Task 1 on Kaggle: steps for the student

The study (40 trials x 15 epochs), the final 100-epoch training, the validation evaluation, the ONNX export and the
parity check all run in ONE notebook (`notebooks/kaggle_t1_pipeline.ipynb`). Expected time on a T4: about 1.5 to 2.5 hours
(not measured on a T4; the same model takes about 8 s per epoch on the local RTX 3050). Nothing in it reads the test set.

## 0. Files to upload (both already built; neither goes to GitHub)

| File | Size | Becomes |
|---|---|---|
| `dist/cloud_data/pets_data.zip` | 151.6 MB | private Kaggle Dataset, e.g. `pets-data` (cache, splits, manifests; no test cache) |
| `dist/cloud_code/genai_code.zip` | 112 KB | private Kaggle Dataset, e.g. `genai-code` (src, configs, scripts, tools) |

Rebuild the code zip after any code change: `.venv\Scripts\python.exe tools\package_cloud_code.py`
(then upload it as a new version of the code dataset).

## 1. Kaggle setup (one time)

1. kaggle.com > Datasets > New Dataset > upload `pets_data.zip` > Visibility **Private** > Create. Repeat for `genai_code.zip`.
2. kaggle.com > Code > New Notebook > File > Import Notebook > `notebooks/kaggle_t1_pipeline.ipynb`.
3. In the notebook: **Add Input** > your two datasets. Settings: **Accelerator = GPU T4 x2** (one GPU is used), **Internet = On**
   (needed for `pip install` and Weights & Biases; Kaggle may ask you to verify your phone number first).
4. Add-ons > Secrets > add `WANDB_API_KEY` with your key from https://wandb.ai/authorize and tick it for this notebook.
   (Without it the runner logs offline and you would have to run `wandb sync` yourself.)

## 2. Run

1. Cell 2: leave `DRY_RUN = True` for the first run (about 5 minutes, tiny data) to prove the setup works, then set `DRY_RUN = False`.
2. Run all cells, or use **Save Version > Save & Run All** to let it run in the background with the browser closed.
   While it runs, the notebook prints the runner's status lines; the W&B project is https://wandb.ai/ahmedlaiq34/genai-a1 (group `task1`).
3. If the session stops, open the notebook again and re-run with the **same `TAG`**: it resumes (finished stages are skipped,
   an interrupted study continues, an interrupted training resumes from its last checkpoint).
   This only works if `/kaggle/working` still holds the files: run interactively, or restart from the saved version's output.

## 3. When it says `ALL STAGES DONE`

Cell 4 builds `/kaggle/working/t1_v2_results.zip`. Download it (right-side Output panel) and unzip it into the repository root:

- `studies/t1_universal_v2/` (trials.csv, plots, study DB), `configs/task1_final_v2.yaml`
- `runs/task1/<run_id>/` (ckpt_best.pt, metrics.jsonl, samples) and `eval/task1/<run_id>_val/` (tables, grids)
- `models/onnx/t1_universal_ae.onnx` (+ sidecar), `models/checkpoints/t1_universal_ae.pt`, `report/tables/onnx_parity.csv`
- `artifacts/logs/v2/` (status log, per-stage logs, summary json)

Move `runs/` and `eval/` under `artifacts/` (`artifacts/runs/task1/...`, `artifacts/eval/task1/...`), then tell the assistant:
it will promote the checkpoint into `models/MANIFEST.json`, verify the ONNX parity locally, and write `docs/PHASE_T1_TRAINING_REPORT.md`.

## 4. Things that can go wrong

- `code dataset not found`: the code dataset is not added as an input to the notebook (Add Input).
- `pip install` fails: Internet is off.
- Data not found: the data dataset is not added as an input; the loader looks under `/kaggle/input` (up to 8 folders deep) for `cache/pets128`.
- W&B errors: the secret is missing or not ticked for this notebook; the runner switches to offline logging after two failures.
- The runner stops itself only after one stage fails 4 times in a row without progress; the reason is the last line of the status log
  (`artifacts/logs/v2/t1_overnight_status.log`) and the stage log next to it.
