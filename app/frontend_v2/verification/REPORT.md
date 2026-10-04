# Verification and design comparison

Verified on 2026-10-04 in Windows PowerShell, Node 24.19.0, npm 11.17.0, using installed Google Chrome through Playwright. Browser rendering and the mock used CPU; Chrome launched with `--disable-gpu`. No Docker or model inference was started.

## Commands executed

All npm/Node project commands ran inside `D:\Generative AI\Assignment_01\app\frontend_v2`.

| Command | Outcome |
| --- | --- |
| `node --version` | v24.19.0 |
| `npm --version` | PowerShell blocked the npm.ps1 wrapper; used npm.cmd thereafter. |
| `npm.cmd install` | First sandboxed attempt failed with registry EACCES. |
| `npm.cmd install --fetch-retries=0` | Approved network retry succeeded; cache and installation stayed in this folder. |
| `npm.cmd install --fetch-retries=0` after dependency review | Attempted braces override failed because version 3.0.4 is not published. Removed that override. |
| `npm.cmd install --fetch-retries=0` after removing override | Succeeded; upgraded sharp to 0.35.4. Reported 5 high advisories in the development dependency tree. |
| `npm.cmd run build` | Production build passed. Repeated after final source changes. |
| `node mock/server.mjs` | Standalone mock on 127.0.0.1:8000; restarted after sharp update. |
| `npm.cmd run dev -- --port 5173 --strictPort` | Vite at 127.0.0.1:5173; restarted after fixing watcher exclusions. |
| `npm.cmd run verify` | Browser checks passed; final run writes results.json, screenshots, and downloaded PNGs. |

Supporting read-only commands inspected the request, local instructions, screenshot files, specified frontend/backend sources, API contract, assignment transcription, installed browser paths, and install/build logs. `Copy-Item` copied four display sample PNGs into `mock/fixtures/`. No git state-changing commands were used.

During debugging, a small `node -e` probe posted mock configuration and requested the Vite proxy; it confirmed connection refusal after the Windows watcher crashed. The first browser run exposed that crash on a locked download file. Excluding verification files fixed it. A subsequent run exposed a test-only limitation: Chrome omits file multipart payloads from Playwright's `request.postData()`. Verification now observes FormData entries before submission and checks the resulting server output. Neither issue remains in the passing run.

## Browser checks

- All four exact workspace names, initial empty state, disabled Run before source selection.
- Universal, hard, and soft: source selection, synthetic corruption, loading state, successful image responses, downloadable PNGs, routing bars where applicable, 501/503 errors, and successful retry.
- Hard clean prediction shows Identity bypass.
- Uploads default to no further corruption; the multipart form contains a file and no sample_id. Custom Gaussian parameters and a valid severity value are sent correctly.
- Wrong MIME type and files over 10 MB fail client validation. Unreadable PNG bytes produce a visible server error.
- Sketch: default 501; success for Style 1, Style 2, Style 3 after enabling the mock; original/result comparison and download.
- Camera permission denial is explicit; a synthetic webcam captures a PNG; tracks end after capture. No actual camera was used.
- System: online/offline/recovery, model loaded/missing states, refresh, and last inference time.
- Screenshots at 1600 px desktop and 390 px phone width; all four phone layouts have no horizontal overflow.
- No uncaught browser page errors in the passing run.

## Stitch comparison: known differences

All four original PNGs were inspected before implementation and compared visually against rendered browser screenshots. The shared lavender canvas, indigo accents, white rounded cards, two-column proportions, navigation, source thumbnails, shaded upload panel, image pairs, routing bars and action buttons follow the references. This is not a pixel-identical reconstruction; all known differences follow.

**Shared differences**

- The screenshots contain portrait/architecture/nature/scientific imagery. The app displays actual API images; mock restoration examples use four supplied pet dataset samples at the backend's native 128 × 128 resolution. Mock sketch verification also uses a sample fixture. No screenshot portrait is represented as a model result.
- Arial/Helvetica system fonts and small inline SVG icons approximate the design. Original font/icon assets were not supplied. Font sizes, line wrapping, italic usage, badge widths, and text density differ, especially in the header and metadata. Body/helper text is smaller than the Stitch renders. Exact shadow and border shades are approximations from the images.
- Header version, GPU, CUDA, FP16, TensorRT, warm-buffer, code/settings/profile decorations are replaced by the app title, research label, exact workspace names, and a live System disclosure. No unsupported hardware claims appear.
- Upload controls expose a native file chooser for accessibility and accept PNG/JPEG/WebP up to 10 MB. RAW, EXR, TIFF, 25/64 MB, and direct GPU ingestion advertised in the mockups are not supported by the requested implementation.
- Imagery uses square, contained image panels without fake PSNR/SSIM/LPIPS, loss, color-profile, tensor-shape, super-resolution, channel-count, or quality overlays. The API does not supply these measurements.
- Unsupported comparison sliders, contour differences, fullscreen/loupe tools, export telemetry, model hyperparameters, help/settings buttons, and simulation/debug switches are absent. Actual empty/loading/error/success feedback takes their place.
- Metadata uses readable returned JSON and real timing fields, rather than the decorative horizontal scientific metadata strips. Extra metadata/status cards alter page height. Footer wording is simplified.
- Phone navigation wraps, controls wrap, columns stack, and images stack. Only desktop references were provided, so the phone arrangement is an adaptation.

**Universal**

- Samples and upload are combined into one source card instead of two separate cards.
- The top statistics card includes inference time and pipeline only; PSNR and SSIM are unavailable.
- Custom numeric parameter fields replace the always-visible dual sliders. Only fields for the chosen corruption are shown; seed is editable. The result download sits in a timing card.

**Hard-routed**

- The heading uses the exact workspace name with a short pipeline description rather than the breadcrumb row. Corruption controls reuse the shared horizontal choices instead of a 2 × 2 grid.
- Low/medium/high are the valid severities; the screenshot's Extreme option is absent. There is no confidence threshold: the backend uses argmax and identity bypass for clean.
- Bars place labels above tracks, rather than keeping all low-probability rows on one line. Real classifier/expert labels replace invented model names, SLA, and dispatcher status.
- Returned corruption settings get their own card; bottom timing/download and true status feedback replace telemetry and simulated router state controls.

**Soft mixture-of-experts**

- The required corruption/severity/custom parameter controls replace the static mixed-degradation profile and temperature slider, since the API has no temperature/mixed-corruption field.
- Weight bars are sorted by contribution for display, while their labels preserve the API index mapping (identity, salt, blur, occlusion). Dominant/secondary contributors are emphasized. A segmented blend bar and exact weight sum use the response values.
- The static latent-space equations, inspect-weight control, made-up model/version labels, top debug-state tabs, and quality metric chips are replaced by plain explanation, returned settings, and actual status.

**Face-to-sketch**

- The source starts empty. A source preview appears after upload/capture, with a Remove action; no face crosshair/ROI overlay is drawn.
- Style 1/2/3 are named exactly as specified. Fine Graphite/Charcoal Drama/Minimalist Ink, pencil grades, hatching density, and shading values are not promised by the contract.
- Upload and camera controls are native accessible controls rather than the exact two-segment tab. The output uses the shared comparison cards and does not reserve the reference's large blank lavender area below portraits.
- Only PNG download is supported. SVG/vector export, resolution/DPI claims, transfer to restoration, and debug showcase cards are absent.

## Limits and remaining dependency finding

- Real FastAPI requests, ONNX model output/quality, actual webcam hardware, physical phone devices, Safari/Firefox, and container build/nginx networking were not checked. The required proxy configuration is supplied, but Docker was deliberately not started.
- Mock success does not establish restoration quality or prove the real backend supports soft/sketch. The inspected backend implements hard routing and currently returns 501 for soft/sketch.
- Final install succeeds but npm reports **5 high-severity advisories in development dependencies**, stemming from the braces dependency under Tailwind's build tooling. The attempted patched braces version was unavailable in the registry. No speculative override is retained. The sharp advisory was resolved by updating sharp. The production nginx stage contains built static assets, not these Node development dependencies.
- Chrome is required to rerun the supplied script as written; it uses an existing installation, not a newly downloaded browser. Browser fake-media flags validate UI plumbing only.
