// All network requests live here. Vite and nginx both forward /api to FastAPI.
async function request(path, options = {}) {
  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), 120000);
  try {
    const response = await fetch(`/api/${path}`, { ...options, signal: options.signal || controller.signal });
    const body = await response.json().catch(() => null);
    if (!response.ok) {
      const detail = typeof body?.detail === 'string' ? body.detail : JSON.stringify(body?.detail || 'Request failed.');
      throw new Error([501, 503].includes(response.status) ? `Not available yet. ${detail}` : detail);
    }
    if (!body) throw new Error('The server returned an unreadable response.');
    return body;
  } catch (error) {
    if (error.name === 'AbortError') throw new Error('The request timed out. Please try again.');
    if (error instanceof TypeError) throw new Error('Cannot reach the server. Check the connection and try again.');
    throw error;
  } finally { clearTimeout(timeout); }
}

export const getHealth = () => request('health');
export async function listSamples() {
  const body = await request('samples');
  return (Array.isArray(body) ? body : body.samples || []).map(item => {
    const id = String(typeof item === 'object' ? item.id : item);
    return { id, url: item.url || `/api/samples/${encodeURIComponent(id)}` };
  });
}

// Display-only face photos for the Face-to-Sketch workspace (GET /api/sketch/samples).
export async function listSketchSamples() {
  const body = await request('sketch/samples');
  return (Array.isArray(body) ? body : []).map(item => ({ id: String(item.id), url: item.url || `/api/sketch/samples/${encodeURIComponent(item.id)}` }));
}

export async function runModel(workspace, input) {
  const form = new FormData();
  if (input.file) form.append('file', input.file);
  else if (input.sampleId) form.append('sample_id', input.sampleId);
  if (workspace === 'sketch') form.append('style', input.style);
  else {
    form.append('corruption', input.corruption);
    // Custom parameters replace a preset, but severity must still be a valid backend enum.
    form.append('severity', input.severity === 'custom' ? 'medium' : input.severity);
    form.append('seed', String(input.seed));
    if (input.params && input.corruption !== 'none') form.append('params', JSON.stringify(input.params));
  }
  const result = await request(workspace, { method: 'POST', body: form });
  const images = workspace === 'sketch' ? ['photo_png_b64', 'sketch_png_b64'] : ['input_png_b64', 'output_png_b64'];
  if (images.some(key => typeof result[key] !== 'string' || !result[key])) throw new Error('The server did not return both images. Please try again.');
  const vector = workspace === 'hard' ? result.probs : workspace === 'soft' ? result.weights : null;
  if (['hard', 'soft'].includes(workspace) && (!Array.isArray(vector) || vector.length !== 4 || vector.some(x => !Number.isFinite(x) || x < 0 || x > 1) || Math.abs(vector.reduce((a, b) => a + b, 0) - 1) > 0.02)) {
    throw new Error('The server returned invalid routing values. Please try again.');
  }
  return result;
}
export const pngSource = value => value ? `data:image/png;base64,${value}` : null;
