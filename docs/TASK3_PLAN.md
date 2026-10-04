# Task 3 plan: jointly trained soft mixture-of-experts restoration

Planner output of 2026-10-04 (planning only; nothing here was run). Authoritative design for the Task 3 implementation team (`docs/WORKER_PROMPT_T3.md`).
Labels as in `docs/IMPLEMENTATION_PLAN.md`: **[PDF]** obligation of the assignment (PDF pages 6 and 7), **[CON]** existing contract (`docs/CONTRACTS.md`), **[REC]** recommendation of this plan (record as a DECISIONS row; the student may change it), **[OPT]** optional, only if time is left.
Every runtime figure below is either measured (quoted from the Task 2 Kaggle run, `docs/PHASE_T2_TRAINING_REPORT.md`) or marked **estimate**.

---

## A. Specification checklist

| # | Obligation | Type | Planned file / function | Proof (test or artifact) |
|---|---|---|---|---|
| A1 | Gate initialised from the Task 2 classifier | PDF p.6 | `models/moe.py` `SoftMoE.load_from_task2` (gate = `CorruptionClassifier` rebuilt from `t2_classifier.pt`) | `test_moe.py::test_gate_equals_classifier` (gate logits == classifier logits at load); checkpoint `source_checkpoints` |
| A2 | Experts initialised from the three specialists | PDF p.6 | same function, `experts["salt"/"blur"/"occlusion"]` from `t2_ae_*.pt` via `task2.routing.load_component` (keeps its cond_id checks) | `test_moe.py::test_experts_equal_specialists` |
| A3 | Identity branch for clean images, receives the corrupted input | PDF p.6, CON 3.2 | `SoftMoE.forward`: branch 0 = `x` | `test_moe.py::test_formula_matches_manual` |
| A4 | `w = softmax(G(x)/tau)`, `x_hat = w0 x + w1 A_salt(x) + w2 A_blur(x) + w3 A_occ(x)` | PDF p.6 | `SoftMoE.forward` | `test_formula_matches_manual`, `test_weights_sum_to_one` |
| A5 | Pretrained source checkpoints read-only, sha256 asserted, unchanged after training | CON 3.8 | `tasks/task3/sources.py` `find_sources`, `verify_sources`; re-check at the end of `export_promote` | `test_moe.py::test_wrong_sha_refused`; `test_task3_train.py::test_sources_byte_identical` |
| A6 | Warm-up with frozen experts (gate only), then joint fine-tuning of everything at a smaller lr | PDF p.6 | `tasks/task3/train.py` `_train` (stage `warmup` then `joint`) | `test_experts_unchanged_in_warmup` (parameter + buffer hash), `test_joint_gradients_nonzero` |
| A7 | Loss `l1 L1 + ls (1-SSIM) + lc CE + lb L_balance` | PDF p.6 | `train.py` `moe_loss` | `test_task3_train.py::test_loss_terms` (hand-computed values) |
| A8 | Balance loss over a balanced batch | PDF p.6, CON 3.5 | `moe_loss` (quadratic form), `balanced_batch` sampler | `test_balance_zero_for_perfect_routing`, sampler already tested |
| A9 | Optuna over joint lr, tau, lambda_c, lambda_b, reconstruction weighting | PDF p.6, CON 3.13 | `tasks/task3/tune.py` `run_study`, `configs/task3_moe.yaml` `tuned_params` | `test_task3_tune.py` dry run (2 trials) ; `studies/t3_moe/trials.csv` |
| A10 | Pruning on poor configurations and routing collapse | PDF p.6 (may), CON 3.6 | `tune.py` (MedianPruner + `collapse_flags`) | `test_collapse_prunes` (forced collapse) |
| A11 | Average expert weights per true corruption type and severity; heatmap | PDF p.7 | `tasks/task3/evaluate.py` `evaluate_task3` -> `weights_by_class_severity.csv`; `tools/t3_report_assets.py` heatmap | `report/figures/task3/routing_heatmap.png` |
| A12 | Examples: one expert dominates / weights distributed | PDF p.7 | `evaluate_task3` -> `examples_dominant.png`, `examples_distributed.png` | files in the eval folder and `report/figures/task3/` |
| A13 | Inactive expert / expert dominating unrelated inputs | PDF p.7 | `evaluate_task3` -> `expert_activity.csv` | `report/tables/task3/expert_activity.*` |
| A14 | Comparison with Task 1 and Task 2 on identical tensors (+ "do nothing") | CON (P5), lessons | `evaluate_task3` computes input, T1, T2 oracle, T2 predicted, T3 in one loop | `comparison_cond_severity.csv` |
| A15 | Workspace "Soft Mixture-of-Experts Restoration": four weights, reconstruction, inference time, strongest contributors | PDF p.7 | `app/backend/app/soft.py` + `/api/soft` in `main.py`; UI already exists in `app/frontend_v2` (weights bars, dominant) | `tests/test_backend_soft.py`; `real_backend.mjs` Soft step; screenshots `report/figures/app/app_soft-*.png` |
| A16 | Complete soft pipeline exported to ONNX (outputs `output`, `weights`; softmax and tau inside) | PDF p.7, CON 3.9 | `export/task3_export.py` `SoftMoEExport`, `export_t3`, `verify_t3_parity` | `test_onnx_t3.py`; row(s) in `report/tables/onnx_parity.csv` |
| A17 | Tracking (W&B group `task3`), checkpoint `t3_soft_moe`, promotion | PDF p.2, CON 3.8, 3.12 | `train.py`, pipeline stage `export_promote` | W&B runs; `models/MANIFEST.json` entry |

---

## B. Design decisions (record as D80 onwards, see section I)

| # | Decision | Reason |
|---|---|---|
| B1 [REC] | **Gate = an exact copy of the Task 2 `CorruptionClassifier`** (same architecture, its internal `[0,1] -> [-1,1]` normalisation, its weights). `G(x)` = its raw logits. | PDF asks for it; the copy reproduces Task 2 routing at step 0, so any later change is caused by training. |
| B2 [REC] | **CE is applied to the raw logits** `CE(G(x), y)`, the mixture uses `softmax(G(x)/tau)`. | CE then means exactly what it meant in Task 2 (independent of tau); tau only shapes how sharp the mixing is. With CE on tempered logits, a large tau would push the logits to grow by the factor tau and undo the softening. |
| B3 [REC] | **tau is a fixed hyperparameter per run** (not learned), the same in warm-up, joint stage, evaluation, ONNX and backend; stored in the checkpoint config `model.tau` and baked into the ONNX graph as a constant. | Simple and explainable; a learned tau is a second mechanism to analyse. |
| B4 [REC] | **Experts are always in `eval()` mode** (BatchNorm uses and keeps its running statistics, dropout 0.03 off), also in the joint stage where their weights do get gradients. `SoftMoE.train(mode)` is overridden to set only the gate's mode. During warm-up the experts additionally have `requires_grad=False` and run under `torch.no_grad()`. | Each specialist's BN statistics were learned on its own corruption only; in the mixture every expert sees all four classes, so train-mode BN would overwrite those statistics and change the expert at inference. Train-mode BN would also change buffers during the frozen warm-up (breaks "experts unchanged"). `no_grad` makes warm-up steps cheaper. |
| B5 [REC] | **Gate in `train()` mode in both stages** (BN batch statistics, dropout 0.40), exactly as when the classifier was trained. | Balanced batches have the same class mix as classifier training, so the gate's BN statistics stay valid. |
| B6 [REC] | **Optimiser AdamW, weight decay 1e-4** (specialists' value). Warm-up: gate parameters only, constant `train.warmup_lr = 2e-4` (about 1/10 of the classifier's final lr 2.18e-3). Joint: **one** parameter group (gate + experts) at the tuned `joint_lr`, range below the warm-up lr, cosine decay over the joint steps. No grad clipping (as Task 2). | One joint lr is what the PDF asks to tune; separate gate/expert lrs would add a parameter without a PDF reason. |
| B7 [REC] | **AMP**: the four networks run under fp16 autocast; logits and branch outputs are cast to float32; softmax, mixture and all loss terms in float32; GradScaler. | Same practice as Task 2 (loss in float32). |
| B8 [REC] | **Reconstruction weighting** is searched as one share `r` = `recon_l1_share`: `lambda_1 = r`, `lambda_s = 1 - r`, range [0.3, 0.95]; the reconstruction part is then `task1.train.restoration_loss(x_hat, clean, alpha=r)` (reused). | Keeps the scale of the reconstruction term fixed, so lambda_c and lambda_b keep their meaning across trials. The study score is the fixed J (CON 3.6), so trials stay comparable whatever r is. The PDF start (0.8 / 0.2) is r = 0.8. |
| B9 [REC] | **Balance loss = the PDF's quadratic form** `sum_k (mean_batch(w_k) - 1/4)^2`, computed on the balanced batch. Not replaced. | It is 0 for perfect routing on a balanced batch, so it does not fight correct routing (numbers in H3). Alternatives considered and not used (cite if discussed): the importance/load loss of Shazeer et al. 2017 ("Outrageously large neural networks", ICLR) and the load-balancing loss of Fedus et al. 2021/2022 ("Switch Transformers", JMLR); both target many experts and token routing, not a 4-branch convex mixture. |
| B10 [REC] | **Sampler `balanced_batch`, batch 64 (16 per class)** fixed, not tuned. 2,944 training images -> 46 steps per epoch, as the specialists. | PDF requires balanced batches for L_balance; batch is not in the PDF's list for Task 3; 64 is the specialists' final batch. Peak memory is measured by the benchmark. |
| B11 [REC] | **Checkpoint** `build_checkpoint(...)` plus: `config.component = "soft_moe"`, `config.model = {"gate": <classifier model cfg>, "experts": {"salt": cfg, "blur": cfg, "occlusion": cfg}, "tau": float, "branches": ["identity","salt","blur","occlusion"]}` (enough for `SoftMoE.from_config`), `config.source_checkpoints = {name: {"file": basename, "sha256": ...}}`, `config.run_id`, `stage` ("warmup"/"joint"), `step_in_epoch`, RNG state, epoch loss sums. `run.smoke` is true if the run or any source checkpoint is a smoke checkpoint. | Rebuild without the Task 2 files; provenance; a fixture-built model can never be promoted or exported into `models/onnx`. |
| B12 [REC] | **Provenance check**: `verify_sources(paths)` hashes the four Task 2 files (and T1 when present) against `TASK2_SHA256` before anything is loaded; called at the start of every training run, every trial (each trial reloads the originals) and again at the end of the pipeline. Mismatch or missing file -> `ValueError` naming the file and both hashes. Never falls back to random weights. | CON 3.8; the Kaggle transport is new (section F). |
| B13 [REC] | **Epoch budget**: trial = 1 warm-up + 3 joint epochs; final = 2 warm-up + 12 joint epochs. Optuna steps and `metrics.jsonl` use one epoch counter across both stages (warm-up epochs first). | From the time budget (section C); to be confirmed by the gate G2 learning curve. |
| B14 [REC] | **Best checkpoint** = lowest fixed J on the validation set among **joint-stage epochs that are not collapsed** (flags of B16). An "epoch 0" validation (before any training) is run and logged in the final training only, for the report; it is not eligible. If every eligible epoch is collapsed the run ends with a clear error. | A promoted T3 model must be a trained one; epoch 0 shows what the Task 2 copy gives in soft mode. |
| B15 [REC] | **Validation sets**: trials use `trial_val_subset = 736` rows (the first 736 val-manifest rows; whole images, 184 per class, asserted); final training and evaluation use all 2,944 rows. Training uses all 2,944 training images (no train subset). | A full validation pass through three autoencoders costs about as much as a training epoch (section C). |
| B16 [CON/REC] | **Collapse flags** on every validation: (a) any branch with mean weight < `collapse.min_mean_weight` = 0.02 over the validation rows; (b) for a true class c, a branch k != c whose mean weight over the rows of class c is > `collapse.max_foreign_weight` = 0.9 (branch 0 counts as class 0). Study: a flagged trial is pruned (`user_attr pruned_reason = "collapse"`), from the first validation on. Thresholds are a recommendation and are stated in the report. | CON 3.6. |
| B17 [REC] | **Study**: TPE via `task2.runs.open_study` (seed offset, RUNNING -> FAIL), budget counted with `n_answered`; `n_trials = 12` plus `timeout_minutes = 16` (whichever comes first; the report states the actual counts); MedianPruner `n_startup_trials 4`, `n_warmup_steps 2` (pruning possible from step 2 = end of the first joint epoch; provisional until G2); the PDF start values are **enqueued as trial 0** (`lambda_1 0.8, lambda_s 0.2, lambda_c 0.1, lambda_b 0.01`, `tau 1.0`, `joint_lr 5e-5`; tau and joint_lr are not given by the PDF: 1.0 is the plain softmax, 5e-5 the geometric middle of the lr range). | Robustness fixes of Tasks 1 and 2; trial 0 lets the report compare the tuned result with the PDF's starting point. |
| B18 [REC] | **Search ranges (PROVISIONAL)**: `joint_lr` log [1e-5, 2e-4]; `tau` log [0.5, 5.0]; `lambda_c` log [0.01, 1.0]; `lambda_b` log [0.001, 0.1]; `recon_l1_share` uniform [0.3, 0.95]. After G2, if the median top-1 minus top-2 logit margin on validation is m, check that tau = 5 gives a top weight clearly below 1 (sigmoid-like estimate `1/(1+3 e^{-m/tau})`); widen the tau range if not. | PDF values sit inside every range; tau above 1 is needed for gradients (H1). |
| B19 [REC] | **Analysis definitions**: dominant row = max weight >= 0.95; distributed row = max weight <= 0.60 (if fewer than 8 such rows exist, take the 8 rows with the lowest max weight and print the threshold reached); branch "inactive" = mean weight < 0.02 or never the argmax on any validation row; branch "dominates unrelated inputs" = mean weight > 0.5 on the rows of another true class. | Makes the PDF p.7 questions answerable with one table; thresholds are stated in the report. |

---

## C. Budget (recommendation; derived from measured Task 2 speeds; student approval required)

**Measured (Kaggle T4, AMP on, 2026-10-04):** one specialist, 50 epochs in 3.5 to 5.0 min including up to 30 s of supervisor polling, 46 steps per epoch, val pass on 736 rows about 1 s. So one specialist epoch is about 3.6 to 6.0 s, of which about 2.6 to 5.0 s training, i.e. **about 0.06 to 0.11 s per specialist step**. Classifier 30 epochs in 3.0 min (92 steps/epoch at batch 32); its val pass 1 to 3 s.

**Estimates (to be replaced by the benchmark stage on the T4):**

| Quantity | Estimate | How |
|---|---|---|
| joint step (gate + 3 experts, fwd + bwd, B 64) | 0.17 to 0.44 s -> **8 to 20 s per epoch** | 3 to 4 x specialist step (given) |
| warm-up step (experts fwd only, `no_grad`; gate fwd + bwd) | 0.06 to 0.22 s -> **3 to 10 s per epoch** | 1 to 2 x specialist step (estimate) |
| full validation pass (2,944 rows, 3 experts + gate) | **13 to 15 s** | 3 experts x 4 x ~1 s + classifier 1 to 3 s |
| subset validation pass (736 rows) | **about 4 s** | a quarter of the above |
| one trial (1 warm-up + 3 joint epochs + 4 subset validations + loading) | **about 50 to 95 s** | 3..10 + 24..60 + 16 + ~5 s |

**Run plan (pipeline `tools/t3_pipeline.py`, 5 stages):**

| Stage | Content | Estimated time |
|---|---|---|
| `prepare` | find + sha256-check sources; benchmark: 5 warm-up + 20 timed steps of each step type at B 64 AMP on, one full and one subset val pass, peak memory; print projected minutes per stage; **gate**: stop if the projection exceeds `budget.max_minutes` (35) | about 1 to 2 min |
| `study` | 12 trials x (1 + 3) epochs, timeout 16 min, pruning | 10 to 16 min (cap) |
| `final_train` | write `configs/task3_moe_final.yaml` (best trial), then 2 warm-up + 12 joint epochs, full validation every epoch + epoch 0 | 5 to 8 min |
| `evaluate` | full val manifest: input, T1, T2 oracle, T2 predicted, T3; tables and figures | 1 to 2 min |
| `export_promote` | ONNX + parity (both outputs), promotion of `t3_soft_moe`, re-hash of the sources | about 1 min |
| supervisor overhead | **exit check every 2 s** instead of 30 s (stall check unchanged), 5 child starts (torch import each, estimate 10 s) | about 1 min |
| **Total (runner)** | | **about 19 to 30 min**; notebook setup (copy + pip install) comes on top and is not measured |

Polling cost in the Task 2 runner: `time.sleep(30)` before every exit check, i.e. up to 30 s lost per stage (11 stages). The Task 3 runner sleeps 2 s between exit checks and has 5 stages.

**If the first status lines are slower than estimated** (the `prepare` stage prints the measured seconds per step and the projection), cut in this order, through `BUDGET_OVERRIDES` in notebook Cell 2 (no new code upload needed):
1. `timeout_minutes=10` (fewer trials; counts are reported);
2. `train.joint_epochs=8` for the final run;
3. `epochs_per_trial.joint=2`;
4. `train.val_every_epochs=2` for the final run.
If it is faster (one trial < 50 s), `n_trials` may go up to 16 with the same timeout. Never more without a new approval.

---

## D. Module specification

Shared names: `src/genai/tasks/task3/__init__.py` (lead): `BRANCH_NAMES = ("identity", "salt", "blur", "occlusion")`; `EXPERT_NAMES = ("salt", "blur", "occlusion")`; `TASK2_SHA256 = {"classifier": "7d4a9a1f8a073e64e8cc21deec4db98d51ad48dc04eb9b5b4215f49102ece028", "salt": "4f8b2edb835c3b967fc8ef4ea1d43a4092841e3bfde4bd583dfb2c3a086eafee", "blur": "0d2aede1d0636d4f65115b2c0d064319cc7a0fd22b55a24fdf9d1af9a750d74c", "occlusion": "d8e82568179939796c1461265bed6af9412f5a727f6093bcc2b9370c2b64e1b4"}` (Task 2 report §5; also in `models/MANIFEST.json`); `SOURCE_FILES = {"classifier": "t2_classifier.pt", "salt": "t2_ae_salt.pt", "blur": "t2_ae_blur.pt", "occlusion": "t2_ae_occlusion.pt", "t1": "t1_universal_ae.pt"}`; `PROMOTED_NAME = "t3_soft_moe"`; `ONNX_KEY = "t3_soft_moe"`; `TRACKER_GROUP = "task3"`; `RUN_GROUP = "task3"`. T1's sha256 is read from `models/MANIFEST.json` by the packaging script and stored in the package's `SOURCES.json`; T1 is optional everywhere (comparison column only).

### D1 `src/genai/models/moe.py` (Agent M)
```python
class SoftMoE(nn.Module):
    def __init__(self, gate_cfg: dict, expert_cfgs: dict, tau: float = 1.0)   # expert_cfgs = {"salt": cfg, "blur": cfg, "occlusion": cfg}
    # attributes: gate (CorruptionClassifier), experts (nn.ModuleDict, keys EXPERT_NAMES), tau (python float), hparams
    def forward(self, x, return_branches: bool = False)
        # x N x 3 x 128 x 128 in [0,1] -> (x_hat N x 3 x 128 x 128, w N x 4 float32, logits N x 4 float32[, branches N x 4 x 3 x 128 x 128])
        # branches[:,0] = x ; experts under torch.no_grad() when experts_frozen ; w = softmax(logits.float()/tau)
        # x_hat = (w[:, :, None, None, None] * branches.float()).sum(1)   (a convex combination: stays in [0,1], no clamp)
    def freeze_experts(self) -> None      # requires_grad False on every expert parameter
    def unfreeze_experts(self) -> None
    @property experts_frozen -> bool
    def train(self, mode: bool = True)   # gate.train(mode); experts always eval() (B4); returns self
    def model_config(self) -> dict       # the B11 "model" section
    @classmethod from_config(cls, model_cfg: dict) -> "SoftMoE"
    @classmethod load_from_task2(cls, paths: dict, tau: float, expected_sha256: dict | None = TASK2_SHA256) -> "SoftMoE"
        # verify_sources first (hash), then task2.routing.load_component for each of the 4, copy state_dicts into new modules
def parameter_hash(module: nn.Module) -> str   # sha256 over state_dict (parameters AND buffers) in key order
def describe(model: SoftMoE) -> str             # parameter counts per part
```
Expected parameter count with the real sources: 98,196 + 3 x 1,322,507 = 4,065,717 (assert in the real-model check, not in unit tests).

### D2 `src/genai/tasks/task3/sources.py` (Agent M)
```python
def find_sources(cfg: dict, need_t1: bool = False) -> dict      # {"classifier": Path, ..., "t1": Path | None}
    # search order: cfg["sources"]["dir"] (explicit) -> recursive glob for SOURCE_FILES under resolve(cfg["data_root"]) (Kaggle input; nested folders) -> repo models/checkpoints
    # no hidden "local" default: data_root comes from the run's config; error lists every place searched
def verify_sources(paths: dict, expected: dict = TASK2_SHA256, t1_sha: str | None = None) -> dict   # {name: sha256}; ValueError on mismatch / missing
def source_record(paths: dict) -> dict          # {name: {"file": basename, "sha256": ...}} for checkpoints and summary.json
```

### D3 `src/genai/tasks/task3/train.py` (Agent T)
```python
def run_training(cfg: dict, resume: str | None = None, on_checkpoint=None) -> Path        # plan 10.1
def _train(cfg, resume=None, on_checkpoint=None, report_fn=None, sources: dict | None = None) -> tuple[Path, float]
    # report_fn(step, val_J, flags) after every validation of a finished epoch; may raise optuna.TrialPruned
def moe_loss(x_hat, clean, logits, w, labels, lam: dict) -> tuple[Tensor, dict]
    # lam = {lambda_1, lambda_s, lambda_c, lambda_b}; returns total and {"recon", "ce", "balance"} (floats) ; float32
def balance_loss(w: Tensor) -> Tensor            # sum_k (w[:, k].mean() - 0.25) ** 2
def validate(model, loader, device) -> dict       # evaluate_restoration numbers + mean_w (4), mean_w_by_class (4x4), gate_accuracy, flags
def collapse_flags(mean_w, mean_w_by_class, cfg) -> dict   # {"collapsed": bool, "reasons": [...]}
def measure_step_times(cfg, batch: int, amp: bool, steps: int, warmup: int) -> dict
    # {warmup_s_per_step, joint_s_per_step, val_full_s, val_subset_s, peak_mb, params}; used by bench_t3 and the pipeline
```
Behaviour: modelled on `task2.classifier._train` (resume incl. RNG and `_SkipFirst` mid-epoch skip, atomic `ckpt_last` / `ckpt_best`, `metrics.jsonl`, `tracking.init_run(..., group="task3")`, `on_checkpoint`). Data: `PetsTrainDataset("train", "balanced_batch", ...)` + `make_batch_sampler("balanced_batch", ...)`; val loader = `task1.train.build_val_loader` with `train.val_subset`. Stage switch after `warmup_epochs`: new AdamW over all parameters + cosine scheduler over the joint steps; the checkpoint stores `stage` so resume rebuilds the right optimizer. Run dir `<output_root>/runs/task3/<run_id>/`, run_id desc `t3moe`. `metrics.jsonl` row per epoch: `epoch, stage, global_step, train_loss, train_recon, train_ce, train_balance, train_mean_w[4], lr, val_J, val_l1, val_ssim, val_psnr, val_mean_w[4], val_mean_w_by_class[4][4], val_gate_accuracy, collapsed, collapse_reasons, epoch_seconds` (+ an `epoch: 0` row in final runs). Sample grid every `sample_every_epochs`: target | input | x_hat | |error| x4 for 8 fixed rows (2 per class).

### D4 `src/genai/tasks/task3/tune.py` (Agent T)
```python
def run_study(cfg: dict, on_checkpoint=None) -> Path       # plan 10.1; returns studies/t3_moe
PARAM_TARGETS = {"joint_lr": "train.joint_lr", "tau": "model.tau", "lambda_c": "train.lambda_c",
                 "lambda_b": "train.lambda_b", "recon_l1_share": ("train.lambda_1", "train.lambda_s")}   # second: r and 1 - r
PDF_START = {"joint_lr": 5e-5, "tau": 1.0, "lambda_c": 0.1, "lambda_b": 0.01, "recon_l1_share": 0.8}
def make_trial_config(cfg, values, trial_number) -> dict   # epochs_per_trial, trial_val_subset; output_root/trials (D27, D29)
def write_final_config(cfg: dict, study, out_path) -> Path  # base config + best params + final epochs + `final_config_source` block
def dry_run_overrides(cfg: dict, tmp_root: Path) -> dict    # study name + "_dryrun", tmp storage/run/export folders, run.smoke
```
Uses `task2.runs.open_study`, `n_answered`, `task1.tune.suggest_params`, `export_study`; `study.enqueue_trial(PDF_START)` only when the study has no trials yet; `study.optimize(..., n_trials=remaining, timeout=timeout_minutes*60)`. Trial user attrs: `run_id`, `val_J_per_epoch`, `mean_w_by_class` (last validation), `pruned_reason` ("median" or "collapse"). After each trial: DB copy to `persist_root` + `on_checkpoint`. TBD guard: `n_trials`, `epochs_per_trial`, `timeout_minutes` must not be `TBD_AFTER_BENCHMARK`.

### D5 `configs/task3_moe.yaml` (Agent T) and `configs/task3_moe_final.yaml` (written by `write_final_config`)
Keys: `task: t3`, `component: soft_moe`, `seed: 42`, `data_config`, `run {desc: t3moe, smoke: false, project: genai-a1}`, `sources {dir: null}`, `model {tau: 1.0}`, `train {batch_size: 64, warmup_epochs: 2, joint_epochs: 12, warmup_lr: 2.0e-4, joint_lr: 5.0e-5, weight_decay: 1.0e-4, lambda_1: 0.8, lambda_s: 0.2, lambda_c: 0.1, lambda_b: 0.01, amp: true, scheduler: cosine, grad_clip: null, train_subset: null, val_subset: null, val_every_epochs: 1, val_batch_size: 64, checkpoint_every_minutes: 10, log_every_steps: 20, sample_every_epochs: 2, max_steps: null, pause_after_steps: null}`, `collapse {min_mean_weight: 0.02, max_foreign_weight: 0.9}`, `study: t3_moe`, `n_trials: 12`, `timeout_minutes: 16`, `epochs_per_trial {warmup: 1, joint: 3}`, `trial_val_subset: 736`, `trial_train_subset: null`, `sampler {name: TPE, seed: 42}`, `pruner {name: Median, n_startup_trials: 4, n_warmup_steps: 2}`, `enqueue_pdf_start: true`, `storage: artifacts/optuna/t3_moe.db`, `study_dir: studies/t3_moe`, `budget {max_minutes: 35}`, `eval {max_rows: null, batch_size: 64, n_examples: 8}`, `tuned_params` (B18, Task 1 format). Comment on every budget value: "RECOMMENDATION, approved by the student on <date> (D8x)".

### D6 `src/genai/tasks/task3/evaluate.py` (Agent E)
```python
def run_evaluation(cfg: dict, checkpoint, final_test: bool = False) -> Path     # plan 10.1
def evaluate_task3(cfg: dict, checkpoint, final_test: bool = False, sources: dict | None = None) -> Path
def load_soft_moe(ckpt_path) -> tuple[SoftMoE, dict]
```
One pass over the manifest (val; test only with `final_test=True`, logged by `PetsManifestDataset`); for each batch the SAME tensors go through: input baseline, T1 (if its checkpoint was found), Task 2 `HardRoutedSystem` oracle and predicted (reuse `task2.routing`), T3 with `return_branches=True`. Output folder `<output_root>/eval/task3/<YYYYmmdd-HHMMSS>_<val|test>/`:
- `per_image.csv`: `row, image_id, cond, severity, true_class, w_identity, w_salt, w_blur, w_occlusion, dominant, max_w, entropy, MAE, SSIM, PSNR, J, SSIM_input, J_input, J_t1, J_t2_oracle, J_t2_predicted, t2_predicted_class, J_branch_identity, J_branch_salt, J_branch_blur, J_branch_occlusion` (the last four: each branch output alone, for expert-drift analysis);
- `comparison_cond_severity.csv` (rows cond x severity + per cond + overall; columns J and SSIM of input, T1, T2 oracle, T2 predicted, T3; count);
- `weights_by_class_severity.csv` (mean of the four weights per true class x severity; clean has one row), `routing_heatmap.png`;
- `expert_activity.csv` (per branch: mean weight overall, share of rows where it is the argmax, mean weight on own-class rows, max mean weight on another class's rows, flags `inactive` / `dominates_unrelated`, B19);
- `expert_drift.csv` (per expert: J of the branch output alone on its own class rows, T3 vs the Task 2 specialist on the same rows);
- `examples_dominant.png`, `examples_distributed.png` (rows: target | input | the four branch outputs titled with their weights | x_hat | |error| x4; `n_examples` rows each, 2 per class where possible), `examples.csv` (which rows were drawn and the threshold used);
- `summary.json` (overall J of all systems, input baseline, gate accuracy vs the Task 2 classifier accuracy on the same rows, collapse flags, expert activity flags, source and T3 checkpoint sha256, manifest / split sha256, tau, lambdas).

### D7 `src/genai/export/task3_export.py` (Agent E)
```python
class SoftMoEExport(nn.Module):   # wraps SoftMoE in eval(); forward(x) -> (output, weights); tau a Python constant in the graph
def export_t3(name: str, ckpt_path, out_path) -> None       # opset 17 (ONNX_OPSET), dynamic batch on input/output/weights, dynamo=False, sidecar .meta.json like Task 2 (smoke flag, onnx_sha256, checkpoint_sha256, tau)
def verify_t3_parity(name, ckpt_path, onnx_path, n: int = 16, inputs=None, tag: str = "", csv_path=None, data_root=None) -> dict
    # data_root REQUIRED when inputs is None (no "local" default, D46); 16 val rows, 4 per class; max/mean abs diff for output AND weights,
    # |sum(weights)-1| <= 1e-5 per row, dominant-branch agreement; appends rows "t3_soft_moe" and "t3_soft_moe:weights" with onnx_verify.FIELDS
```
Refuses to write a smoke model under `models/onnx/` (Task 2 rule). Import `onnx_verify.val_inputs`, `FIELDS`, `PARITY_CSV`; do not edit `onnx_verify.py`.

### D8 Benchmark (lead): `bench_t3` in `scripts/benchmark.py`, registered in `BENCH_FUNCTIONS` under `t3`; calls `task3.train.measure_step_times` with sources from `find_sources(cfg)`; rows to `docs/BENCHMARKS.md`.

### D9 App (Agent A writes `soft.py` and tests; the lead pastes the `main.py` hookup)
`app/backend/app/soft.py`:
```python
def run_soft(img01: np.ndarray, get_session) -> tuple[np.ndarray, dict, dict]
    # session "t3_soft_moe": run(["output", "weights"], {"input": img01[None]})
    # returns output [3,128,128], {"weights": [4 floats, 6 decimals], "dominant": BRANCH_NAMES[argmax], "dominant_id": int,
    #   "ranking": branch names sorted by weight}, {"inference": ms}
```
`main.py`: replace the 501 stub by `POST /api/soft` with the same form fields as `/api/hard`, `check_fields` -> `get_session("t3_soft_moe")` (503 if missing) -> `prepare_input` -> `run_soft` -> `{**image_fields(...), **routing, "timing_ms": {"preprocess", "inference", "total"}}`. `dominant` values: `identity | salt | blur | occlusion` (the frontend shows them via `readable()`; weights order matches its `EXPERT_LABELS`). `/api/health` already lists `t3_soft_moe` (it is in `ONNX_FILES`). The frontend `app/frontend_v2` is not edited.

### D10 Kaggle runner and packaging (Agent K)
- `tools/package_t3_sources.py`: reads `models/MANIFEST.json`, checks the four Task 2 files against `TASK2_SHA256` and T1 against its manifest entry, writes `dist/cloud_models/t3_sources.zip` with the five `.pt` files + `SOURCES.json` (`{file: sha256}`); refuses on any mismatch. Prints size and hashes.
- `tools/t3_pipeline.py`: copy of the Task 2 supervisor structure (`Paths`, state file, child stages, stall detection, W&B offline fallback, `--data-root`, `--tag`, `--dry-run`, summary) with `PIPELINE_VERSION = "t3-v1"`, `POLL_SECONDS = 2` for the exit check (stall check every 30 s as before), stages `prepare, study, final_train, evaluate, export_promote` (section C), `--set KEY=VALUE` repeatable (budget overrides, applied to the study and final configs and echoed into the status log), `t3_summary.json` (study counts COMPLETE / PRUNED / FAIL, best trial, range-edge flags, timeout hit or not, stage seconds, benchmark row, final curve, evaluation summary, parity, source hashes before/after). Dry run: `artifacts/dryrun_t3/`, 2 trials x (1 + 1) epochs, train subset 256, val subset 64 (multiple of 4), final 1 + 1 epochs, real benchmark measurement but no gate stop. Evaluation and export get the run's config as data root (no "local" default).
- `notebooks/kaggle_t3_pipeline.ipynb`: exactly 4 cells like the Task 2 notebook: (1) find `tools/t3_pipeline.py` and `t2_classifier.pt` under `/kaggle/input` by glob (assert both, print `code version`), copy the code, pip install, `nvidia-smi`; (2) `DEVICE_PROFILE`, `TAG = "t3"`, `DRY_RUN = True`, `WANDB_SECRET`, `BUDGET_OVERRIDES = []`; (3) run the pipeline; (4) bundle `t3_results.zip` (`studies/t3_moe`, `configs/task3_moe_final.yaml`, `models/` (only t3 files exist there), `report/tables/onnx_parity.csv`, `artifacts/logs/<TAG>`, `artifacts/optuna/t3_moe.db`, `/kaggle/working/runs/task3`, `/kaggle/working/eval/task3`, the dry-run folders; staging folder created first).
- `docs/KAGGLE_T3_STEPS.md`: section F steps 1 to 6 in the format of `docs/KAGGLE_T2_STEPS.md`, with the expected `code version: t3-v1`.

### D11 `tools/t3_report_assets.py` (Agent E)
Reads an evaluation folder, the final run folder and `studies/t3_moe/`; writes CSV + LaTeX tables to `report/tables/task3/` and figures to `report/figures/task3/` (list in section G). Asserts a `_val` folder unless `--allow-test` is passed (for the approved test run). Reads files only.

---

## E. Verification plan (all on CPU, `CUDA_VISIBLE_DEVICES=-1`, `--basetemp=artifacts/pytest_tmp/<name>`, TMP/TEMP on D:)

Fixtures: `tests/t3_fixtures.py` (Agent M): `make_t3_sources(tmp)` = `t2_fixtures.make_fixture_checkpoints` + a tiny T1 checkpoint, returns `(paths, sha_dict)`; tests pass `expected_sha256=sha_dict`. Data: `tiny_pets_root` (conftest). Tests pin tiny models and budgets **inside the test** and never read `configs/task3_moe.yaml` budgets.

| Test file | Tests |
|---|---|
| `tests/test_moe.py` (M) | weights sum to 1 (atol 1e-6) for random inputs and tau in {0.5, 1, 5}; `x_hat` equals a manual loop over the four branches; gate logits equal the source classifier's; experts equal the source specialists; output in [0,1]; `train()` keeps experts in eval; `freeze_experts` -> no grad; wrong sha256 -> ValueError; missing file -> ValueError; `find_sources` finds files in a nested fake `input/datasets/u/t3-sources/` folder; `from_config(model_config())` + state_dict round-trip |
| `tests/test_task3_train.py` (T) | loss terms against hand-computed values; balance 0 for one-hot balanced weights and > 0 for collapsed ones; warm-up: `parameter_hash` of experts (params + buffers) identical before/after, gate hash changed; joint: non-zero gradient norm for gate and every expert after one step; source files byte-identical (sha256) after a smoke run; resume from `pause_after_steps` continues `global_step` and the stage; checkpoint config rebuilds the model; smoke flag propagates from fixture sources |
| `tests/test_task3_tune.py` (T) | dry run 2 trials x (1 + 1) epochs, tiny subset: DB, `trials.csv`, user attrs; trial 0 has the PDF start values; each trial calls `load_from_task2` again (counter) and starts from the source weights (hash); forced collapse (monkeypatched `validate` returning a 0.95 foreign weight) -> PRUNED with `pruned_reason == "collapse"`; TBD guard raises; `write_final_config` output loads |
| `tests/test_task3_eval.py` (E) | fixture evaluation on 32 rows: all files exist, `per_image.csv` columns, weights columns sum to 1, comparison table has the five systems, T1 column absent (not crashing) without T1; `final_test=True` writes to a tmp test-access log only |
| `tests/test_onnx_t3.py` (E) | export of a fixture model to tmp; parity of `output` and `weights` < 1e-4 on 16 inputs covering 4 classes; weights sum to 1; dynamic batch (1 and 5); refusal to write a smoke model under `models/onnx`; `verify_t3_parity` without inputs and without `data_root` raises |
| `tests/test_backend_soft.py` (A) | tiny random ONNX built from a fixture `SoftMoEExport` in a tmp model dir: response fields (`weights` 4 values summing to 1 within 1e-4, `dominant` in BRANCH_NAMES, `timing_ms.inference`, images, `corruption_applied`, `params`, `seed`); 503 when the file is missing; 4xx on a bad upload; `/api/universal` and `/api/hard` unchanged |
| `tests/test_backend.py` (lead) | remove `"soft"` from the 501 stub test (the parametrisation becomes empty: delete that test and say so in the report) |
| `tests/test_t3_pipeline.py` (K) | `package_t3_sources` on fixture files with a fake manifest (refuses a wrong hash); pipeline `--dry-run` against `tiny_pets_root` + fixture sources unpacked into a nested fake input folder (`--data-root`), ends with `ALL STAGES DONE`, writes `t3_summary.json`; exit poll uses `POLL_SECONDS` |

**Gates before the real run, in order** (the worker prompt makes the lead check them in this order):
- **G1** full `pytest tests -q` passes (existing count + new tests reported).
- **G2 measure first (local, real checkpoints, a few minutes):** `tools/t3_quick_check.py` (lead; reads only): on the full val manifest (GPU if free, else the first 736 rows on CPU) print (a) J of the T3 copy at init for tau in {1, 2, 4} vs Task 2 predicted 0.1032 / oracle 0.1028 (same rows); (b) mean max weight and median top-1 minus top-2 logit margin; (c) mean weight per true class at init; (d) GPU only: 1 warm-up + 3 joint epochs at the PDF start, val J per epoch, mean weights per class per epoch, seconds per step. Result decides B18 (tau range) and B17 (pruner warm-up); write the numbers into the phase report.
- **G3** `scripts/benchmark.py --model t3 --batch 64 --amp on` on the laptop GPU if free (laptop numbers, labelled; the T4 numbers come from the Kaggle dry run).
- **G4** `scripts/tune.py --task t3 --dry-run` (2 trials).
- **G5** full local `tools/t3_pipeline.py --dry-run --data-root <fake nested input folder with pets_data + t3_sources unzipped>` ends with `ALL STAGES DONE`; Task 2 file hashes unchanged.
- **G6** real-model checks locally: export the init model (not promoted, written to `artifacts/fixtures/task3/onnx/`) and run parity on real val inputs; `/api/soft` against it with `uvicorn` (10 calls; results to `artifacts/eval/task3/api_check_init.json`). Proves the real-size graph exports.
- **G7** Kaggle dry run (section F step 4).
- **G8** student approves the budgets (section C, with the T4 projection printed by G7) -> real run.
After the real run: Docker Compose check, `real_backend.mjs` Soft step (update: run Soft for none / salt / blur / occlusion at medium, check four weights summing to 1, the dominant branch shown, screenshot `soft-<cond>.png`; the Sketch part of the file is not changed).

---

## F. Execution runbook for the student

1. **Approve** the budget of section C (or change it) and answer the worker prompt's first question.
2. **Package** (after the implementation team reports G1 to G6 passed): `.venv\Scripts\python.exe tools\package_t3_sources.py` -> `dist/cloud_models/t3_sources.zip` (about 50 MB for the Task 2 files plus the T1 file); `.venv\Scripts\python.exe tools\package_cloud_code.py` -> `dist/cloud_code/genai_code.zip`.
3. **Kaggle setup**: Datasets > New Dataset > `t3_sources.zip` > Private > name `t3-sources`. Datasets > New Dataset > `genai_code.zip` > Private > a **new** name `genai-code-t3` (never a new version). Code > New Notebook > Import `notebooks/kaggle_t3_pipeline.ipynb`; it must have exactly 4 code cells. Add Input: your pets data dataset, `genai-code-t3`, `t3-sources` (remove older code inputs). Accelerator GPU T4, Internet On, Persistence "Files only". Secret `WANDB_API_KEY` ticked (entity `ahmedlaiq34`, project `genai-a1`, group `task3`).
4. **Dry run**: `DRY_RUN = True`, Run All. Check: Cell 1 prints `code version: t3-v1` and the path of `t2_classifier.pt`; the status log shows `sources verified` with the four hashes, `preflight: wandb logged in as entity ahmedlaiq34`, the `prepare` benchmark line with seconds per step and the **projected real-run minutes**; `evaluate` and `export_promote` pass; it ends with `ALL STAGES DONE`. Send the status log to the assistant; budgets are adjusted with `BUDGET_OVERRIDES` if the projection is above 35 minutes.
5. **Real run**: `DRY_RUN = False`, Save Version > Save & Run All (Commit). On interruption re-run with the same `TAG`.
6. **Download** `/kaggle/working/t3_results.zip` and give it to the assistant. Do not unzip `models/MANIFEST.json` or `report/tables/onnx_parity.csv` over the local files: they are **merged** (the Kaggle copies only contain Task 3). The assistant places runs under `artifacts/runs/task3/`, the evaluation under `artifacts/eval/task3/`, the study under `studies/t3_moe/` and `artifacts/optuna/`, `configs/task3_moe_final.yaml`, `models/checkpoints/t3_soft_moe.pt`, `models/onnx/t3_soft_moe.onnx` + sidecar, re-checks hashes and parity locally, re-runs `evaluate_task3` on the val manifest (numbers must match Kaggle), runs `tools/t3_report_assets.py`, and writes `docs/PHASE_T3_TRAINING_REPORT.md`.
7. **App check**: `docker compose up -d --build`, open http://localhost:8080, Soft Mixture-of-Experts workspace with none / salt / blur / occlusion; run `app/frontend_v2/verification/real_backend.mjs`.
8. **Test set (once, only with your explicit approval, logged in `artifacts/test_access.log`)**: one `--final-test` session for T1, T2 and T3 together: `python scripts/evaluate.py --task t3 --ckpt models/checkpoints/t3_soft_moe.pt --final-test` (computes T3 and, on the same tensors, input, T1, T2 oracle and predicted), then `tools/t3_report_assets.py --eval-dir <..._test> --allow-test`.

---

## G. Report evidence checklist for Task 3

Figures `report/figures/task3/`: `t3_architecture.png` (gate + identity + three experts + weighted sum; drawn by the script with matplotlib boxes, or by the student); `routing_heatmap.png` **[PDF]** (rows: true class x severity, columns: four branches, mean weight written in each cell); `weights_distribution.png` (per true class, box plots of the four weights); `examples_dominant.png`, `examples_distributed.png` **[PDF]**; `training_curves.png` (val J per epoch with the warm-up / joint boundary, train recon / CE / balance, mean weight per branch per epoch, epoch 0 marked); `j_comparison.png` (input / T1 / T2 oracle / T2 predicted / T3 per condition); Optuna `t3_optuna_history.png`, `t3_optuna_importance.png`, `t3_optuna_parallel.png`; app screenshots `report/figures/app/app_soft-*.png`.
Tables `report/tables/task3/` (CSV + LaTeX): `comparison_cond_severity`, `weights_by_class_severity`, `expert_activity`, `expert_drift`, `study_summary` (counts COMPLETE / PRUNED / FAIL, pruned reasons, timeout hit, best trial, best params, PDF-start trial 0 result, range-edge flags: best value within 10 % of a range end on the log or linear scale), `final_config`, `gate_vs_classifier` (gate accuracy vs Task 2 classifier on the same rows), `model_facts` (4,065,717 parameters, tau, lambdas). Plus the T3 row in `report/tables/onnx_parity.csv`.
Facts the text must state: budgets as run (trials, epochs per trial, timeout, final 2 + 12 epochs or the overrides used); the actual counts; which parameters sit at range edges; tau and the four lambdas of the final model and of trial 0; the collapse thresholds 0.02 / 0.9 and the B19 thresholds as recommendations; that experts kept BN in eval mode and why; the epoch-0 J (soft copy of Task 2) next to the trained J; what the gate does on mild blur (the identity branch on low-severity blur: weight measured, compared with the blur expert's J 0.1173 vs input 0.0505 at low severity in Task 2); validation severities are tertiles of the sampled parameter; all numbers are validation numbers unless the approved test run happened; references actually used (PDF-suggested balance loss; Shazeer et al. 2017 and Fedus et al. 2022 only if discussed; Jacobs et al. 1991 "Adaptive mixtures of local experts" for the mixture-of-experts idea).

---

## H. Risks and open questions

- **H1 Softmax saturation.** The Task 2 classifier reaches 0.99 accuracy with very confident outputs, so at tau = 1 the weights are almost one-hot and the gradients through `softmax` are close to zero: warm-up may change nothing. G2 measures the logit margin and the init max weight; the tau range goes above 1 (B18). If even tau = 5 leaves the top weight above 0.99 on most rows, widen the range before the study (lead decision, DECISIONS row).
- **H2 Expert drift.** In the joint stage each expert receives gradient from every row, weighted by its weight. With near one-hot weights the gradient comes mostly from its own class; `expert_drift.csv` measures it (branch-alone J on own rows, T3 vs Task 2). A drift that worsens an expert is reported, not hidden.
- **H3 Balance term vs correct routing.** On a balanced batch perfect routing gives 0. Example: if all low-severity blur rows (about 1/12 of a batch) moved fully to the identity branch, the mean weights would be 1/4 + 1/12 and 1/4 - 1/12, balance = 2 x (1/12)^2 = 0.0139, times lambda_b 0.01 = 1.4e-4, which is small next to the reconstruction gain on those rows. So the PDF form should not block that behaviour; lambda_b near the top of its range could. The heatmap answers it.
- **H4 CE vs the identity branch on mild blur.** CE pushes blur rows to branch 2, reconstruction may prefer branch 0 on mild blur. A lower gate "accuracy" on blur is then expected and must be read together with J. A study whose best trial has very small lambda_c should be flagged in the report (the gate is then less tied to the runtime label, which the PDF says CE is for).
- **H5 Time.** Three experts per step. Mitigations are built in (no_grad experts in warm-up, val subset in trials, study timeout, cut list in C). The first real numbers come from the Kaggle dry run's `prepare` stage.
- **H6 ONNX with four branches.** All four branches always run (about 3 x the hard-routed cost per image on CPU; measured in G6). Exporting the real-size graph early (G6) removes the risk before Kaggle.
- **H7 Collapse rule vs useful behaviour.** Rule (b) would prune a trial that sends > 90 % of the weight on blur rows to identity; that is unlikely on all blur rows (only mild blur benefits) but, if it happens in many trials, report it and discuss whether the threshold is right; do not change it during the study.

**Questions for the student (only these change the plan):**
- Q1 Is anything else using the laptop GPU or Docker right now? (decides local GPU work for G2 / G3 / G6)
- Q2 Do you approve the budget of section C (12 trials x (1 + 3) epochs with a 16-minute cap, final 2 + 12 epochs, runner about 19 to 30 minutes, estimate) or do you want a different total?
Everything else is decided above; the student may override any [REC] row.

---

## I. Agents, ownership and order

**Do not edit:** Task 1 and Task 2 files (`src/genai/tasks/task1/**`, `task2/**`, `models/autoencoder.py`, `models/classifier.py`, `configs/task1_*`, `configs/task2_*`, `tools/t1_*`, `tools/t2_*`, `notebooks/kaggle_t1*`, `kaggle_t2*`), `src/genai/common/**`, `src/genai/pets/**`, `app/frontend/**`, `app/frontend_v2/**` except `app/frontend_v2/verification/real_backend.mjs`, `src/genai/export/{onnx_export,onnx_verify,task2_export}.py`, Task 4 files, `models/checkpoints/*` and `models/onnx/*` (read-only), `docs/CONTRACTS.md`, `docs/IMPLEMENTATION_PLAN.md`, this plan.
**Shared files, lead only, additive, re-read immediately before each edit** (another window may be editing them): `src/genai/tasks/cli.py` (no change expected: `t3` exists), `scripts/{benchmark,tune,export_onnx,evaluate}.py`, `app/backend/app/main.py`, `tests/test_backend.py`, `tools/package_cloud_code.py` (no change expected), `docs/DECISIONS.md`, `docs/AI_USE_LOG.md`, `docs/BENCHMARKS.md`, `docs/TRACEABILITY.md`.

| Agent | Owns | Returns to the lead |
|---|---|---|
| Lead | `src/genai/tasks/task3/__init__.py`, `tools/t3_quick_check.py`, shared files above, `docs/PHASE_T3_REPORT.md` | gates G1 to G6, the report |
| M (model) | `src/genai/models/moe.py`, `src/genai/tasks/task3/sources.py`, `tests/t3_fixtures.py`, `tests/test_moe.py` | exact signatures, `model_config()` layout, fixture helper |
| T (training + study) | `src/genai/tasks/task3/{train,tune}.py`, `configs/task3_moe.yaml`, `tests/test_task3_train.py`, `tests/test_task3_tune.py` | `_train` signature, checkpoint layout, `measure_step_times`, dry-run output |
| E (evaluation + ONNX + report assets) | `src/genai/tasks/task3/evaluate.py`, `src/genai/export/task3_export.py`, `tools/t3_report_assets.py`, `tests/test_task3_eval.py`, `tests/test_onnx_t3.py` | eval folder listing, parity rows (fixture), ONNX op problems at once |
| A (app) | `app/backend/app/soft.py`, `tests/test_backend_soft.py`, `app/frontend_v2/verification/real_backend.mjs` (Soft step only) | the exact `main.py` snippet for the lead, the API contract |
| K (Kaggle) | `tools/t3_pipeline.py`, `tools/package_t3_sources.py`, `notebooks/kaggle_t3_pipeline.ipynb`, `docs/KAGGLE_T3_STEPS.md`, `tests/test_t3_pipeline.py` | dry-run status log, notebook cell count |

**Order:** Lead: `task3/__init__.py` first (minutes). **Wave 1** (parallel): M; A (can build its test ONNX from a 20-line stand-in module with the same outputs until M's wrapper exists); K (supervisor skeleton and packaging). **Wave 2** (as soon as M returns, about the first hour): T and E in parallel. **Wave 3**: K wires the stages to T and E; lead adds scripts, the `main.py` hookup, the `test_backend.py` change, then runs the gates G1 to G6. DECISIONS rows: B1 to B19 become **D80 to D98** (`B<n>` -> `D(79+n)`); D53 to D59 may be used by other windows and D60 to D79 are reserved for Task 4. Re-read `docs/DECISIONS.md` right before appending; if D80 or later is already taken, continue after the highest existing number and note the mapping in the phase report.
