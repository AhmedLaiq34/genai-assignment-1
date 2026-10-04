# Submission existence audit

Date: 2026-10-05 (Asia/Karachi).

Scope: the student's latest instruction was to check existence only and skip tests or time-consuming work. This audit uses directory listings, Git metadata, and selected existing documents/source files. **OK means the stated files or records exist; it does not certify runtime correctness.** No tests, builds, training, evaluation, Docker commands, downloads, uploads, clone checks, or binary hashing were run. No official test images were opened.

Requirements reference: `ASSIGNMENT_TRANSCRIPTION.md`, PDF pages 1–9. No original assignment PDF was found among the project files listed; this audit cannot resolve discrepancies against it. The incorrect printed March 16, 2024 deadline is not used.

## Verified OK

### Repository and implementation inventory

| Item | Status | Evidence | Remaining action |
|---|---|---|---|
| Four task implementations | OK | `src/genai/tasks/task1/`, `task2/`, `task3/`, `task4/`; models in `src/genai/models/{autoencoder,classifier,moe,cgan}.py` | Explain methodology and results in the report; execution not checked. |
| Configurations | OK | `configs/task1_final_v2.yaml`, `task2_classifier_final.yaml`, `task2_specialist_final.yaml`, `task3_moe_final.yaml`, `task4_final.yaml`; study and dataset configs also present | Use the final configurations in the report and README. |
| Dependencies and package metadata | OK | `requirements.txt`, `requirements_kaggle.txt`, `pyproject.toml`, `app/backend/requirements.txt`, frontend package manifests and lockfiles | Installation completeness not checked. |
| Data-preparation scripts | OK | `scripts/prepare_pets.py`, `scripts/prepare_fs2k.py`; `src/genai/pets/`, `src/genai/fs2k/` | Document dataset sources, directory layout, and commands. |
| Splits and deterministic corruption manifests | OK | Tracked `data/splits/pets_split.json`, `fs2k_split.json`; `data/manifests/pets_{val,test}_manifest.jsonl` and both `.sha256` sidecars | Actual hashes not recomputed. |
| Split-count evidence | OK | `report/tables/dataset_split.csv`: Oxford 2,944 train / 736 validation / 3,669 test; `report/tables/task4/fs2k_split_counts.csv`: 899 / 159 / 1,046 | These are existing table values, not independently regenerated counts. |
| Complete corruption configuration assets | OK | `report/tables/corruption_config.{csv,tex}`, `report/figures/corruption_examples.png`, `configs/data_pets.yaml` | Interpret the configuration and deterministic severities in the report. |
| Training, tuning, evaluation entry points | OK | `scripts/train.py`, `tune.py`, `evaluate.py`; corresponding task modules | Commands were not executed. |
| Optuna records for all tasks | OK | DBs, trial CSVs and plots in `studies/t1_universal_v2/`, `t2_classifier/`, `t2_specialist_shared/`, `t3_moe/`, `task4_cgan/` | Task 4 also has `best_params.json` and `trials_from_console_log.csv`; its existing presentation decision is retained. Database consistency not tested. |
| Final run/evaluation folders | OK | Task folders exist under `artifacts/runs/` and `artifacts/eval/`; phase reports describe the final runs | Local ignored evidence is not part of a clone. |
| Eight local checkpoints | OK | `models/checkpoints/`: Task 1; Task 2 classifier and three specialists; Task 3; Task 4 generator and discriminator | Hashes and Task 2 immutability were not reverified. |
| Seven inference ONNX files and sidecars | OK | `models/onnx/`: Task 1; Task 2 classifier and three specialists; Task 3 full pipeline; Task 4 generator; each has `.meta.json` | Publish model download links; binaries are ignored by Git. |
| Existing ONNX parity evidence | OK | `report/tables/onnx_parity.csv` contains passing rows for all seven models and an additional Task 3 weights row | Existing evidence only. Task 4 row records 48 cases; `docs/PHASE_T4_REPORT.md` states these cover 16 photos × three styles. No parity rerun or freshness check. |
| React/Tailwind frontend and FastAPI backend | OK | `app/frontend_v2/`, `app/backend/`; Compose selects `frontend_v2` | Runtime not checked. Old `app/frontend/` also remains. |
| Required workspace names | OK | Exact four names occur in `app/frontend_v2/src/App.jsx` | Display/layout not checked live. |
| Required backend operations | OK | Routes for `/api/health`, `/api/universal`, `/api/hard`, `/api/soft`, `/api/sketch` in `app/backend/app/main.py` | No HTTP calls or response validation. |
| Webcam and download implementation | OK | `SketchWorkspace.jsx` contains `getUserMedia`; shared `UI.jsx` has PNG download links | Webcam permission and downloads not exercised. |
| Task 4 style conditioning in both networks | OK | `src/genai/models/cgan.py`: distinct `nn.Embedding` tables in G and D; embeddings concatenated into their inputs; G also uses its embedding at the bottleneck | Static source evidence only. |
| Docker packaging | OK | `app/backend/Dockerfile`, `app/frontend_v2/Dockerfile`, `docker-compose.yml`, frontend nginx configuration | Compose documents port 8080 and mounts ONNX files read-only; backend port 8000 is internal. Stack not started. |
| Tests and historical verification records | OK | `tests/`, phase reports, `app/frontend_v2/verification/` scripts/results/screenshots | No current pass/fail counts are claimed. |
| AI-use log | OK | `docs/AI_USE_LOG.md` | Convert relevant entries into the report appendix and verify your understanding. |
| Git repository and populated development branch | OK | Local `git branch -vv`: `dev` at `f4f064c`, tracking `origin/dev`; remote URL is `https://github.com/AhmedLaiq34/genai-assignment-1.git` | Remote availability, default branch, and grader access were not checked over the network. |

### Report evidence inventory

| Required evidence | Status | Existing evidence / precise gap |
|---|---|---|
| Task 1 architecture | OK | `report/figures/t1_architecture.png` |
| Task 2 architecture/pipeline | OK | `report/figures/t2_pipeline.png`; autoencoder architecture shared with Task 1; model facts in `report/tables/task2/model_facts.*` |
| Task 3 architecture | OK | `report/figures/task3/t3_architecture.png` |
| Task 4 architecture diagram | MISSING | No diagram found in `report/figures/task4/`; draw G, D, their style embeddings, U-Net skips and PatchGAN output using `src/genai/models/cgan.py`. Existing `tools/t4_report_assets.py` does not produce this diagram. |
| Loss definitions and training explanations | PROBLEM | Definitions exist in assignment/source/phase documents, but no report manuscript exists to present them. |
| Shared dataset/corruption tables | OK | `report/tables/dataset_split.*`, `corruption_config.*`, Task 4 split-count CSV |
| Optuna search spaces/final configurations | PROBLEM | Shared `search_spaces.*` and `final_configs.*` cover Tasks 1–2; Task 3 has `report/tables/task3/final_config.*` and study summary; Task 3/4 search configs and Task 4 final config exist. Consolidated report-ready coverage of all four tasks is unfinished. |
| Trial outcomes and best-trial evidence | OK | Task 1 trials CSV; Task 2/3 study-summary tables; Task 4 study/trials files. No database cross-check performed. |
| Training/validation curves | OK | Task 1 `t1_training_curves.png`; Task 2/3 `training_curves.png`; Task 4 `losses.png`, `val_metrics.png`, `d_probabilities.png`, `sample_timeline.png` |
| Tasks 1–3 quantitative tables | OK | `t1_{val,test}_results.csv`; Task 2 oracle/predicted tables and `test/`; Task 3 condition/severity comparisons and `test/` |
| Classifier metrics/confusion matrices | OK | `report/tables/task2/classifier_report.*`, `confusion_normalised.*`, corresponding test assets and figures |
| Task 3 routing analysis | OK | `weights_by_class_severity.*`, `expert_activity.*`, `expert_drift.*`; heatmaps, weight distributions, dominant/distributed example grids, including test assets |
| Task 4 per-style results | OK | `report/tables/task4/{val,test}_by_style.csv`; results/style-variation grids in `report/figures/task4/` |
| Task 1 twelve examples/four failures | OK | `t1_test_representative_12.png`, `t1_test_worst_4.png`; `report/text/failure_analysis_t1_t2.md` and failure-analysis CSVs. Image counts/content not visually rechecked. |
| Task 2 routing failures | OK | `report/figures/task2/routing_failures_worst.png` and test version; failure-analysis text and misroute tables |
| Task 3 dominating/distributed/inactive-expert evidence | OK | Dominant/distributed grids and expert-activity tables; discussion must still be written in the report. |
| Task 4 failures | OK | `report/figures/task4/failures.png`, `failures_test.png` |
| Restoration grids/error-map evidence | OK | Task 1 and Task 2 restoration grids exist; phase reports describe their target/input/output/error panels. Visual correctness not rechecked. |
| Application screenshots: Tasks 1–3 and system info | OK | `report/figures/app/app_universal.png`, `app_hard-*.png`, `app_soft-*.png`, `app_system-panel.png` |
| Application screenshots: Task 4 | PROBLEM | Existing tracked screenshots at `app/frontend_v2/verification/sketch_real/{sample-style1,sample-style2,sample-style3,upload-style1}.png`; none in `report/figures/app/`. Reference the existing files or place selected copies among report assets. Webcam capture evidence not found there. |
| W&B screenshots, all tasks | PROBLEM | Eight local images in `report/figures/wandb/`, but ignored by `.gitignore:24` (`wandb/`). They are absent from tracked report files. |
| W&B run links | PROBLEM | Task 1 explicit run URL is recorded in `docs/PHASE_T1_TRAINING_REPORT.md`; Task 4 report records run ID `ugt4ngh7`; Task 2/3 reports describe tracking but no explicit run URLs were found in those reports. Collect all four final run links. Live dashboards not checked. |
| Original Stitch workspace evidence | OK | Four workspace images in `report/figures/stitch/` |
| Stitch prompts/canvas overview | MISSING | No prompts file or canvas overview found in that folder. Useful supporting evidence; the PDF requires original design evidence, not these specific additional filenames. |
| IEEE LaTeX manuscript and bibliography | MISSING | `report/main.tex` and `report/references.bib` absent; existing `.tex` files are generated tables, not a manuscript. |
| Compiled report PDF | MISSING | No PDF found under `report/` or `docs/`. |
| Demo video / YouTube link | MISSING | No video artifact found under `report/` or `docs/`, and no actual YouTube link found in searched README/docs/report text. A video may exist elsewhere; its submission link is not recorded here. |

## Problems found

Ordered by submission impact. No preparation fixes were applied because the latest request limits work to checking existence.

1. **MISSING — IEEE LaTeX report and compiled PDF.** Assets exist, but `report/README.md` still says the source goes here. Create the manuscript, bibliography, interpreted results/failures, and AI-use appendix, then compile. Fix applied: no.
2. **MISSING — recorded YouTube demonstration link.** Record the required five-to-seven-minute demonstration and put its YouTube URL in the report. No measured duration is claimed. Fix applied: no.
3. **PROBLEM — model distribution is incomplete.** All seven ONNX files are local and Git-ignored. `scripts/fetch_models.py` raises `NotImplementedError`; `models/MANIFEST.json` contains checkpoint metadata but no model download URLs or ONNX distribution entries. Prepare downloads with ONNX hashes and a working downloader. Fix applied: no.
4. **PROBLEM — README is obsolete.** It says “Scaffold only,” “no models implemented or trained,” and contains TODOs for implemented scripts, the application, and downloads. Replace it with the actual setup/data/train/tune/evaluate/export/Compose guide. Fix applied: no.
5. **PROBLEM / STUDENT — branch and access readiness.** Local `main` and `origin/main` point to the initial empty commit; `git ls-tree --name-only main` returns no files. `dev` is populated. If the remote default branch remains `main`, a normal clone will be empty. Select `dev` as default or merge into `main`, and ensure grader access. Remote default/private status not verified. Fix applied: no; no commits, pushes, merges or settings changes.
6. **PROBLEM — W&B evidence exists only locally.** `git check-ignore -v report/figures/wandb/task1_wandb.png` identifies the broad `wandb/` ignore rule. Preserve the eight report screenshots in the submission/source package; explicitly add reviewed images or approve scoping that ignore rule to the root run directory. Fix applied: no.
7. **MISSING / PROBLEM — remaining report presentation.** Task 4 architecture diagram, complete all-task search-space presentation, explicit final run URLs for Tasks 2–4, report integration of Task 4 app screenshots, and written interpretation remain. Fix applied: no.
8. **PROBLEM — stale traceability and historical status text.** `docs/TRACEABILITY.md` still says “All status: not started”; many G/D/T1/A rows are stale, G9 references the old frontend, T2g says Docker/browser checks were not done, T3e says report figures are not generated, and T4g says the frontend was not checked. Later phase reports and existing screenshots provide historical evidence of those activities. Update statuses with evidence and retain any unverified runtime qualifications. Fix applied: no.
9. **PROBLEM — Compose has no healthchecks.** Compose and both Dockerfiles exist; `docker-compose.yml` uses a simple dependency list and contains no healthchecks. The backend source has missing-model handling and `/api/health`, but readiness was not exercised. Add healthchecks in a later preparation pass. Fix applied: no.

## Missing items the student must make

1. Write `report/main.tex` using IEEEtran conference format and `report/references.bib`; include distinct Task 1–4 sections, shared dataset/application sections, limitations, conclusion and AI-use appendix. Use the existing tables/figures and verify citations rather than inventing them. Explain the four Task 1 failures and Task 2 misrouting analysis from `report/text/failure_analysis_t1_t2.md`.
2. Draw the Task 4 architecture diagram from the actual generator/discriminator source. Include separate style embeddings in both networks, generator skips, and local PatchGAN discrimination.
3. Collect exact W&B final-run links for all tasks. Include the existing screenshots and ensure they are packaged despite Git ignoring their folder.
4. Include the existing Task 4 app screenshots in the report; capture webcam operation if you need evidence for that path. Existing Stitch screenshots can be included now; preserve original prompts/canvas evidence if available.
5. Publish the chosen model distribution and insert a valid download link in README/report. A source-only clone currently lacks all inference binaries.
6. Record a five-to-seven-minute demo: Compose startup → upload → runtime corruption → universal restoration → hard-routing probabilities/expert → soft-routing weights → face-to-sketch/style selection → PNG download → W&B records for all four tasks. Upload to YouTube and put only its link in the report.
7. Compile and inspect the PDF, then assemble any recommended LaTeX source zip with its figures/tables and bibliography.

Existing asset generators, if later regeneration is needed: `tools/make_t1_report_assets.py`, `t2_report_assets.py`, `t3_report_assets.py`, `t4_report_assets.py`, `report_common_assets.py`, and `failure_analysis_t1_t2.py`. They were not run; review their inputs and test-data access before using them.

## Decisions needed from the student

| Decision | Recommendation |
|---|---|
| Report drafting assistance | Start the report next. A generated skeleton requires your explicit approval under the original audit prompt; none was created. |
| Model distribution | Use GitHub Release assets for all seven ONNX files plus a hash-checking downloader. Actual local sizes: Task 1 and each Task 2 specialist 5.29 MB, classifier 0.39 MB, Task 3 16.28 MB, Task 4 76.80 MB (decimal MB). Alternatives: track the smaller files and release Task 4, or Git LFS. Uploads and ignore changes need approval. |
| Default branch | Make the populated `dev` branch default, or merge the finished project into `main`; ensure the submission links target populated content. |
| Grader access | Verify repository visibility; if private, make public or grant the grader access. The audit prompt describes it as private, but no remote lookup was performed. |
| Ignored W&B report screenshots | Preserve reviewed screenshots with an explicit add or approved ignore-rule correction. Keep generated root W&B run folders ignored. |

These are outstanding submission decisions, not requests to restart skipped tests or the clone check.

## Submission checklist

- [x] Source, configs, dependencies, preparation/training/tuning/evaluation/export scripts exist.
- [x] Study folders for all tasks exist.
- [x] Local checkpoint and inference ONNX files exist.
- [x] Existing parity CSV has passing records covering all seven models.
- [x] Frontend/backend, Dockerfiles and Compose exist.
- [x] Most quantitative/visual report assets and original Stitch workspace images exist.
- [ ] Finish and compile the IEEE LaTeX report, including interpreted results, failures and AI-use appendix.
- [ ] Include the missing Task 4 diagram and complete all-task search/configuration presentation.
- [ ] Include W&B screenshots and final run links; ensure screenshots are present in the submitted source package.
- [ ] Include Task 4 application screenshots and discuss webcam support with appropriate evidence.
- [ ] Replace the scaffold README and downloader stub; publish model download links.
- [ ] Resolve populated default branch and grader repository access.
- [ ] Upload the demo to YouTube; confirm its real duration is five to seven minutes.
- [ ] Put GitHub, YouTube and model-download links inside the report.
- [ ] Confirm the real deadline in Google Classroom; disregard the incorrect printed PDF deadline.
- [ ] Submit the report PDF to Google Classroom; a LaTeX source zip is recommended.
- [ ] Do not upload the demo video directly to Classroom; submit its YouTube link within the report.
- [ ] Review the source package for secrets; exclude datasets, credentials, local environments and run folders.
- [ ] Be able to explain the complete submitted system and run it on previously unseen images.

## What you could not check and why

- **Runtime correctness:** tests, builds, smoke steps, Docker startup, endpoints, browser UI, webcam and downloads were skipped at your request. Existing historical reports are not new verification.
- **Hash/parity freshness and model immutability:** binary hashing, checkpoint comparison, ONNX inspection and parity reruns were skipped. The manifest and parity records merely exist.
- **Test-set integrity:** `artifacts/test_access.log` exists and was read before the scope change. It records Oxford accesses at 2026-10-04 16:07:12, 18:48:54 and 20:22:19 UTC, and FS2K at 20:10:16 UTC. Phase reports attribute them respectively to final Tasks 1, 2, 3 and 4 evaluations. Log lines do not themselves record purpose, and this existence audit does not certify absence of unlogged access or test-based checkpoint selection.
- **Fresh-clone execution and remote state:** no network clone, remote settings check, GitHub visibility check, model-link check or W&B dashboard visit. The previous permission questions became unnecessary after the scope was reduced.
- **Security audit:** no full working-tree/history secret scan. The active `artifacts/kaggle_config/kaggle.json` was not opened or printed; `git check-ignore` confirms it is ignored, which does not establish historical absence of secrets. `git ls-files -ci --exclude-standard` returned no tracked-but-ignored files.
- **Original PDF, report compilation and video:** no original PDF found in the listed project files; no manuscript/PDF/video submission link exists in the inspected submission locations. TeX installation was not checked.
- **Visual and statistical quality:** images were listed, not visually judged; metrics were read as existing evidence, not recomputed.

Work performed: added this audit and appended one entry to `docs/AI_USE_LOG.md`. No implementation, README, requirements, manifest, models, run folders, Git ignore rules or branch state were changed.

## Preparation update after the existence audit

The student subsequently authorized direct PDF generation and model publication. `report/main.pdf` now contains GitHub, the supplied YouTube URL (`https://youtu.be/4-InqI11ZQA`), and the published `models-v1` release with individual links to all seven ONNX files. The downloader is implemented; the manifest contains ONNX hashes, sizes and release URLs; README model/report sections are updated. All seven GitHub asset SHA-256 digests and sizes match the manifest. Repository visibility remains private and the default branch remains `main`; grader access/default branch still need attention. No commit or push was performed. The PDF uses IEEE-style layout and was generated directly, as explicitly requested by the student, rather than via LaTeX.

Visibility update: the student explicitly requested public access. GitHub CLI now confirms the repository is PUBLIC; models-v1 release assets inherit public access. The default branch remains unchanged.
