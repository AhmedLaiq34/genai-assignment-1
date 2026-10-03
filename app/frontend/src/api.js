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

// POST /api/universal. Input = {file | sampleId, corruption, severity, params, seed}
export async function runUniversal({ file, sampleId, corruption, severity, params, seed }) {
  const fd = new FormData();
  if (file) fd.append("file", file);
  else fd.append("sample_id", sampleId);
  fd.append("corruption", corruption);
  if (corruption !== "none") {
    if (params) fd.append("params", JSON.stringify(params));
    else fd.append("severity", severity);
  }
  fd.append("seed", String(seed));
  const res = await check(await fetch(`${BASE}/universal`, { method: "POST", body: fd }));
  return res.json();
}
