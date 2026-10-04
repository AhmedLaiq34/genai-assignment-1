# Restoration Studio

React + Vite + Tailwind frontend for the four assignment workspaces. Everything in this implementation, including the mock, dependencies, npm cache, and verification artifacts, lives in `app/frontend_v2/`. The existing frontend/backend are not imported or modified.

## Install and run

Use Node.js 22 or newer. In PowerShell, `npm.cmd` avoids systems that block the `npm.ps1` wrapper.

```powershell
cd 'D:\Generative AI\Assignment_01\app\frontend_v2'
npm.cmd install
```

For the mock, open two terminals in this folder:

```powershell
# Terminal 1: standalone mock API, port 8000
npm.cmd run mock
```

```powershell
# Terminal 2: frontend
npm.cmd run dev -- --port 5173 --strictPort
```

Open http://127.0.0.1:5173. Universal, hard, and soft restoration return illustrative results. Sketch intentionally returns **HTTP 501** by default. Open System to see model availability.

To exercise successful sketch generation, enable all mock endpoints while the mock is running:

```powershell
Invoke-RestMethod -Uri http://127.0.0.1:8000/__mock/config -Method Post -ContentType application/json -Body '{"failures":{}}'
```

To restore the default or simulate errors/loading:

```powershell
Invoke-RestMethod -Uri http://127.0.0.1:8000/__mock/config -Method Post -ContentType application/json -Body '{"failures":{"sketch":501,"hard":503},"delay":1500}'
```

`/__mock/config` belongs only to the local mock. Restarting the mock restores its defaults. It is bound to loopback and is not included in the production image. The UI explicitly labels mock results. The mock applies CPU image transformations, returns the original source as the simulated restoration, and uses grayscale/contrast/threshold transformations for the three simulated sketch styles. It does **not** run a trained model or demonstrate model quality. The four bundled sample fixtures are copies of existing backend display samples.

## Real backend

Stop the mock with Ctrl+C. Have the existing FastAPI backend running at `http://localhost:8000`, then run the same frontend dev command. `vite.config.js` proxies `/api` to that address. Backend setup/model training is outside this folder's scope.

- Restoration sends exactly one of `file` or `sample_id`, plus `corruption`, `severity`, `seed`, and optional JSON `params`.
- Custom parameters use `p`, `kernel`/`sigma`, or `n_rects`/`coverage`. The severity field remains `low`, `medium`, or `high` even with custom parameters.
- Uploads can be corrupted like samples. Tick **Already corrupted** to lock the corruption controls and send `corruption=none` (restore a damaged image as it is).
- Sketch sends `file` and `style` (1, 2, or 3). These are categorical conditions; artistic names are not assumed.
- HTTP 501 and 503 show “Not available yet” without crashing. Other errors can be retried.
- PNG, JPEG, and WebP uploads are limited to 10 MB. The server still validates image contents. Previews revoke object URLs when replaced or unmounted.
- System polls health every 30 seconds, supports manual refresh, lists all reported models, and remembers the most recent successful inference time. Hard inference time is classifier + expert time; total/preprocessing times are also shown.
- Webcam capture requires localhost or HTTPS. Permission denial and missing devices have explicit messages. Tracks stop on capture, close, upload, workspace change, or unmount.

## Face-to-Sketch with the real backend (Task 4)

`POST /api/sketch` is implemented by the backend (`app/backend/app/sketch.py`) and uses `models/onnx/t4_generator.onnx`. This frontend sends `file` or `sample_id` plus `style` (1, 2, 3) and reads `photo_png_b64`, `sketch_png_b64`, `style_id` (0 to 2), `style_label`, `timing_ms`. The Face-to-Sketch workspace lists six display-only sample photos from `GET /api/sketch/samples` (backend folder `app/backend/samples_sketch/`); upload and webcam work as before. 503 means the generator file is missing. The mock server serves the same sample list and a `style_label`.

## Build and container files

```powershell
npm.cmd run build
npm.cmd run preview
```

The build is in `dist/`. Vite preview serves static assets; use the dev server for the configured API proxy, or nginx for the production proxy.

`Dockerfile` builds with Node and serves with nginx. `nginx.conf` proxies `/api/` to `http://backend:8000`, retains the `/api` path, accepts 11 MB request bodies for 10 MB file uploads plus multipart overhead, and supports SPA routing. The existing backend must be reachable as `backend` on the container network. No Docker commands were run and no repository Compose file was changed.

## Code map

- `src/App.jsx`: navigation and shared shell; no overview screen.
- `src/api.js`: all application fetch calls, multipart assembly, errors, response validation.
- `src/hooks.js`: object URL lifetime, file validation, asynchronous inference state.
- `src/components/UI.jsx`: cards, buttons, chips, badges, option tabs, bars, image panels, uploads, downloads, and icons.
- `src/components/InputColumn.jsx`: shared sample/upload/corruption controls.
- `src/components/RestorationWorkspace.jsx`: shared restoration workspace orchestration.
- `src/components/Results.jsx`: result images, routing probabilities/weights, timing, metadata, download.
- `src/components/SketchWorkspace.jsx`: sketch styles and webcam lifecycle.
- `src/components/SystemPanel.jsx`: live health/model status.
- `tailwind.config.js`: palette, fonts, radii, spacing, shadows; `src/styles.css` supplies shared component classes.
- `mock/server.mjs`: independent mock server, backed by local fixture PNGs.

Controls use native buttons, inputs, radio buttons and labels. Current choices expose pressed/current states, errors use alerts, progress bars have accessible names and values, focus indicators are visible, and animation respects reduced-motion preferences. Below desktop width the columns stack; image comparisons stack on narrow phones.

## Verification

With the mock and dev server running, and Google Chrome installed:

```powershell
npm.cmd run verify
```

The Playwright script uses the existing Chrome installation, disables GPU acceleration, and uses a fake camera. No browser binary download is necessary. To use an installed Edge instead, change the `channel` in `verification/browser.mjs` to `msedge`.

See [verification/REPORT.md](verification/REPORT.md) for exact commands, known differences from Stitch, dependency findings, and untested integrations. `verification/results.json` records successful checks. Desktop/mobile screenshots and downloaded PNGs are saved alongside it. The dev watcher ignores verification files to avoid Windows download locks.

## Connected to the real backend (Docker Compose)

`docker-compose.yml` at the repository root builds this folder as the `frontend` service (`context: app/frontend_v2`); its nginx proxies `/api/` to the `backend` service, which mounts `models/onnx` read-only. Run `docker compose up --build` and open http://localhost:8080. The first, plain frontend is still in `app/frontend`.

`node verification/real_backend.mjs` (BASE defaults to `http://localhost:8080`; use `BASE=http://127.0.0.1:5173` with the dev server and a backend on :8000) drives the real UI against the real models and writes screenshots and `results.json` to `verification/real/`. Checked on 2026-10-04 with the Task 1 and Task 2 models: samples, System panel (Task 1 and four Task 2 models Loaded; Soft MoE and Sketch Missing), Universal restoration, Hard-routed restoration (clean to Identity bypass, salt, blur and occlusion to the matching expert with four probability bars), upload with `corruption=none`, the server error for an unreadable image, and HTTP 501 for Soft MoE and Sketch shown as "Not available yet". No backend code change was needed: the request and response formats of this frontend already match the backend.
