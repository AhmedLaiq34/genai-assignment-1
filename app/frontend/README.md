# Frontend (React + Vite + Tailwind)

Single page with four workspaces in a top nav. Only **Universal Restoration** is functional;
Hard-Routed, Soft MoE and Face-to-Sketch are "not implemented yet" stubs with their final layout
sketched (probability bars, 4-weight bars, Style 1/2/3 + webcam button).

> **TODO (Phase 6):** this layout is a plain function-over-style placeholder. It must be rebuilt
> from the Google Stitch design.

## Run
```
npm install
npm run dev          # http://localhost:5173, proxies /api -> http://localhost:8000
npm run mock         # optional: fake backend on :8000 (mock/server.mjs), no real models needed
npm run build        # production build into dist/
```
Docker: `Dockerfile` builds with Node, serves with nginx; `nginx.conf` proxies `/api` to `http://backend:8000`
(`client_max_body_size 20m`).

## Files
- `src/api.js` - the only place that calls `fetch` (base path `/api`).
- `src/App.jsx` - nav + tab switching.
- `src/components/UniversalWorkspace.jsx` - the working page.
- `src/components/Stubs.jsx` - the three placeholder workspaces.
- `mock/server.mjs` - mock API.

## API assumptions (match these in the backend, or edit `src/api.js`)
- `GET /api/samples` returns a JSON list of items `{id, url?}`. A bare list of ids, or `{samples: [...]}`, also works.
  If an item has no `url`, the image is fetched from `GET /api/samples/{id}`.
- `POST /api/universal` is multipart with either `file` (upload) or `sample_id`, plus `corruption`
  (`none|salt_pepper|gaussian_blur|occlusion`) and `seed`.
  When corruption is not `none`, it also sends either `severity` (`low|medium|high`) or, for custom values,
  `params` as a JSON string (and no `severity`):
  - salt_pepper: `{"p": 0.08}`
  - gaussian_blur: `{"kernel": 5, "sigma": 1.5}`
  - occlusion: `{"n_rects": 2, "coverage": 0.2}`
- Response: `{input_png_b64, output_png_b64, corruption_applied, params, timing_ms:{preprocess,inference,total}}`
  (base64 without a data-URL prefix).
- Errors: non-2xx with JSON `{detail: ...}`; `detail` is shown to the user.
