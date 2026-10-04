# Task 4 on Kaggle: steps for the student

Everything for Task 4 runs in ONE notebook (`notebooks/kaggle_t4_pipeline.ipynb`), unattended:
data check -> GPU benchmark -> final cGAN training (resumes after a reset) -> split into `t4_generator.pt` / `t4_discriminator.pt`
-> `t4_generator.onnx` + parity check (16 val photos x 3 style ids) -> validation evaluation.
Nothing in it reads the test set. The confirmatory Optuna study is NOT run (decision: skipped, the final training uses the logged
best trial #25). Design: `docs/TASK4_PLAN.md`.

**Before the real run you must set the number of epochs.** `train.epochs` in `configs/task4_final.yaml` is `TBD_AFTER_BENCHMARK`
and the pipeline refuses to start the training while it is TBD. The value is now 120 (cut from your intended 200 to fit a 30 to 40 minute budget: about 20 to 25 minutes of training at the laptop's 8 to 13 s/epoch, plus about 6 minutes of setup, export and evaluation; the T4 speed is not measured yet). Rule (plan C17): measure seconds per epoch
(`python scripts/benchmark.py --model t4 --batch 8 --amp both`, on the target GPU; on Kaggle the pipeline's `benchmark` stage also
prints it), then use 200 only if `200 x s/epoch x 1.1` fits in one session; otherwise let the run continue over several sessions
(section 4). The log of the old Colab study suggests about 20 s/epoch, which is only an estimate to replace with the measurement.

## 0. Files

| File | Becomes |
|---|---|
| `dist/cloud_data/fs2k_data.zip` (`python scripts/prepare_fs2k.py --package`) | a **new private dataset**, e.g. `fs2k-data` (no test images inside) |
| `dist/cloud_code/genai_code.zip` (`python tools/package_cloud_code.py`, rebuilt after the epochs are set) | a **new dataset** (not a new version), e.g. `genai-code-t4` |

**Never upload:** `data/raw/`, the FS2K test images, `artifacts/`, `models/`, `app/`, `.venv/`, `node_modules/`.
A new code version of an existing dataset is not picked up by a notebook that already has it attached (Task 1 lesson): always upload
the changed code as another NEW dataset (`genai-code-t4-2`), remove the old code input from the notebook, restart the session.

## 1. Kaggle setup

1. Datasets > New Dataset > upload `fs2k_data.zip` > Private. Same for `genai_code.zip` (name `genai-code-t4`).
2. Code > New Notebook > File > Import Notebook > `notebooks/kaggle_t4_pipeline.ipynb`. It must have exactly 4 code cells.
3. Add Input > the FS2K data dataset and `genai-code-t4`. Settings: **Accelerator = GPU T4 x2** (one GPU is used),
   **Internet = On**, Persistence = "Files only".
4. Add-ons > Secrets > `WANDB_API_KEY`, ticked for this notebook. W&B project `genai-a1`, group `task4`.

## 2. Run

1. Cell 2 starts with `DRY_RUN = True`: run all cells once (a few minutes, tiny data). It must end with `ALL STAGES DONE`.
   Check:
   - Cell 1 prints `code version: t4-v1` (anything else = old code attached);
   - the status lines say `preflight: wandb logged in as entity ahmedlaiq34`;
   - `data_check` prints 899 train / 159 val pairs and the style counts (it reads the data from `/kaggle/input`, nested folders are found);
   - `export` passed (ONNX parity).
2. Set `DRY_RUN = False`, then **Save Version > Save & Run All (Commit)** (background run, the browser may be closed).

## 3. When it says `ALL STAGES DONE`

Cell 4 builds `/kaggle/working/t4_results.zip`. Download it and unzip it into the repository root. It contains the run folder
(`runs/task4/<run_id>/`: `ckpt_best.pt`, `ckpt_last.pt`, `metrics.jsonl`, `samples/`, snapshots), `eval/task4/...`, `models/onnx/t4_generator.onnx`
(+ sidecar), `artifacts/promote_t4/t4_generator.pt` and `t4_discriminator.pt`, `report/tables/onnx_parity.csv`, `docs/BENCHMARKS.md`
and `artifacts/logs/t4/`. Move `runs` and `eval` under `artifacts/`.
Careful with `report/tables/onnx_parity.csv` and `docs/BENCHMARKS.md`: the zip's copies only know Task 4. **Merge** them with the
existing files, do not overwrite. Do not copy a Kaggle-made `.onnx` into `models/onnx/` without checking its sha256 locally first
(plan F steps 7 to 8: promote locally with `common.checkpoint.promote`, then export and verify locally).

## 4. After a reset (the session stops before the training is finished)

1. Open the notebook, Save Version again with the same `TAG`. Finished stages are skipped when `/kaggle/working` is intact.
2. If it is a NEW version (empty `/kaggle/working`): Add Input > the previous version's **Output** as an input. The pipeline
   (`resume_roots = /kaggle/input`) finds `runs/task4/<run_id>/ckpt_last.pt` in it, copies that run folder and continues
   (same optimiser states, same fixed sample photos); the benchmark stage is skipped on a continuation.
3. `ckpt_last.pt` is written every 10 minutes and at each epoch end, so a reset loses at most about 10 minutes.

## 5. Things that can go wrong

- `code dataset not found (or an old version ...)`: the code dataset is missing or still an old one.
- `train.epochs ... is still TBD_AFTER_BENCHMARK`: set it in `configs/task4_final.yaml`, rebuild and upload the code zip as a new dataset.
- Data not found / `unexpected split sizes`: the FS2K data dataset is not an input, or it is not the packaged zip.
- W&B errors: the secret is missing or not ticked; after two failures the runner logs offline and syncs at the end.
- GAN training can collapse (plan H1): watch the W&B curves and the sample grids in the first hour; a resume does not repair a collapse.
- The runner stops only after one stage failed 4 times in a row without progress; the reason is the last line of
  `artifacts/logs/t4/t4_status.log` and the stage log next to it.
