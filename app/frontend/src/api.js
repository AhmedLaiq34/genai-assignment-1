// All backend calls live here. Base path is /api (Vite proxy in dev, nginx in Docker).
const BASE = "/api";

// Turn a failed response into an Error carrying the server's `detail` message.
async function check(res) {
  if (res.ok) return res;
  let msg = `HTTP ${res.status}`;
  try {
    const body = await res.json();
    if (body.detail) {
      msg = typeof body.detail === "string" ? body.detail : JSON.stringify(body.detail);
    }
  } catch {
    /* body was not JSON; keep the status text */
  }
  throw new Error(msg);
}

// GET /api/samples -> list of clean samples. Assumed items: {id, url?} (see README).
export async function listSamples() {
  const res = await check(await fetch(`${BASE}/samples`));
  const json = await res.json();
  // Accept either a bare list or {samples: [...]}; accept items that are plain ids too.
  const list = Array.isArray(json) ? json : json.samples || [];
  return list.map((item) => {
    const id = typeof item === "object" ? item.id : item;
    const url = (typeof item === "object" && item.url) || `${BASE}/samples/${encodeURIComponent(id)}`;
    return { id: String(id), url };
  });
}

// The multipart form sent to /api/universal and /api/hard (same fields for both).
// Input = {file | sampleId, corruption, severity, params, seed}
function buildForm({ file, sampleId, corruption, severity, params, seed }) {
  const fd = new FormData();
  if (file) fd.append("file", file);
  else fd.append("sample_id", sampleId);
  fd.append("corruption", corruption);
  if (corruption !== "none") {
    if (params) fd.append("params", JSON.stringify(params));
    else fd.append("severity", severity);
  }
  fd.append("seed", String(seed));
  return fd;
}

// POST /api/universal -> {input_png_b64, output_png_b64, corruption_applied, params, timing_ms:{preprocess,inference,total}}
export async function runUniversal(input) {
  const res = await check(await fetch(`${BASE}/universal`, { method: "POST", body: buildForm(input) }));
  return res.json();
}

// POST /api/hard -> the universal fields (timing_ms has preprocess, classifier, expert, total) plus
// {probs[4], predicted, predicted_id, expert, identity_bypass}
export async function runHard(input) {
  const res = await check(await fetch(`${BASE}/hard`, { method: "POST", body: buildForm(input) }));
  return res.json();
}
