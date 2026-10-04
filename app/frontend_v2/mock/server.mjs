// Standalone CPU-only demo server. No imports from the existing frontend/backend.
import http from 'node:http';
import { readFile } from 'node:fs/promises';
import sharp from 'sharp';

const ids = ['Abyssinian_201', 'Bengal_33', 'shiba_inu_68', 'great_pyrenees_91'];
const fixtures = Object.fromEntries(await Promise.all(ids.map(async id => [id, await readFile(new URL(`./fixtures/${id}.png`, import.meta.url))])));
const failures = { sketch: 501 }; // Required unavailable endpoint; tests can enable its happy path.
let delay = 350;
const send = (res, status, body) => { res.writeHead(status, { 'Content-Type': 'application/json', 'Cache-Control': 'no-store' }); res.end(JSON.stringify(body)); };
const fail = (status, message) => Object.assign(new Error(message), { status });

async function readBody(req) {
  const chunks = []; let size = 0;
  for await (const chunk of req) {
    size += chunk.length;
    if (size > 11 * 1024 * 1024) throw fail(413, 'Maximum image size is 10 MB.');
    chunks.push(chunk);
  }
  return Buffer.concat(chunks);
}
function parameters(corruption, severity, custom) {
  if (corruption === 'none') return { type: 'clean' };
  const level = ['low', 'medium', 'high'].indexOf(severity);
  const defaults = {
    salt_pepper: { p: [0.05, 0.15, 0.3][level] },
    gaussian_blur: { kernel: [3, 5, 9][level], sigma: [0.7, 1.5, 3][level] },
    occlusion: { n_rects: [1, 2, 3][level], coverage: [0.08, 0.2, 0.4][level] },
  };
  const p = custom || defaults[corruption];
  if (!p || typeof p !== 'object' || Array.isArray(p)) throw fail(422, 'Invalid custom parameters.');
  if (corruption === 'salt_pepper' && !(Number.isFinite(p.p) && p.p >= 0 && p.p <= 1)) throw fail(422, 'Noise density must be 0–1.');
  if (corruption === 'gaussian_blur' && !(Number.isInteger(p.kernel) && p.kernel >= 3 && p.kernel <= 15 && p.kernel % 2 === 1 && Number.isFinite(p.sigma) && p.sigma > 0)) throw fail(422, 'Use an odd kernel from 3 to 15 and a positive sigma.');
  if (corruption === 'occlusion' && !(Number.isInteger(p.n_rects) && p.n_rects >= 1 && p.n_rects <= 5 && p.coverage >= 0.02 && p.coverage <= 0.6)) throw fail(422, 'Use 1–5 rectangles and coverage 0.02–0.6.');
  return { type: corruption, ...p };
}
async function corrupt(png, params, seed) {
  if (params.type === 'clean') return png;
  if (params.type === 'gaussian_blur') return sharp(png).blur(Math.max(0.3, params.sigma)).png().toBuffer();
  const { data, info } = await sharp(png).removeAlpha().raw().toBuffer({ resolveWithObject: true });
  let state = seed >>> 0;
  const random = () => { state = (Math.imul(state, 1664525) + 1013904223) >>> 0; return state / 4294967296; };
  if (params.type === 'salt_pepper') {
    for (let i = 0; i < data.length; i += 3) if (random() < params.p) { const color = random() < 0.5 ? 0 : 255; data.fill(color, i, i + 3); }
  } else {
    const side = Math.floor(Math.sqrt(info.width * info.height * params.coverage / params.n_rects));
    for (let n = 0; n < params.n_rects; n++) {
      const x = Math.floor(random() * (info.width - side)); const y = Math.floor(random() * (info.height - side));
      for (let row = y; row < y + side; row++) data.fill(20, (row * info.width + x) * 3, (row * info.width + x + side) * 3);
    }
  }
  return sharp(data, { raw: info }).png().toBuffer();
}

const server = http.createServer(async (req, res) => {
  try {
    const pathname = new URL(req.url, 'http://localhost').pathname;
    // Local verification control. It is not part of the production API and is never bundled.
    if (req.method === 'POST' && pathname === '/__mock/config') {
      const config = JSON.parse((await readBody(req)).toString());
      if (config.failures) { for (const key of Object.keys(failures)) delete failures[key]; Object.assign(failures, config.failures); }
      if (config.delay != null) delay = Math.max(0, Math.min(5000, config.delay));
      return send(res, 200, { failures, delay });
    }
    const route = pathname.replace('/api/', '');
    if (failures[route]) return send(res, failures[route], { detail: `The ${route} model is not available yet (mock response).` });
    if (req.method === 'GET' && route === 'health') {
      const models = Object.fromEntries(['t1_universal', 't2_classifier', 't2_salt', 't2_blur', 't2_occlusion', 't3_moe', 't4_generator'].map(key => [key, { loaded: key !== 't4_generator' || !failures.sketch }]));
      return send(res, 200, { status: 'ok', mock: true, models });
    }
    if (req.method === 'GET' && route === 'samples') return send(res, 200, ids.map(id => ({ id, url: `/api/samples/${id}` })));
    if (req.method === 'GET' && route === 'sketch/samples') return send(res, 200, ids.map(id => ({ id, url: `/api/sketch/samples/${id}` })));
    if (req.method === 'GET' && route.startsWith('sketch/samples/')) {
      const fixture = fixtures[decodeURIComponent(route.slice(15))];
      if (!fixture) throw fail(404, 'Unknown sample.');
      res.writeHead(200, { 'Content-Type': 'image/png' }); return res.end(fixture);
    }
    if (req.method === 'GET' && route.startsWith('samples/')) {
      const fixture = fixtures[decodeURIComponent(route.slice(8))];
      if (!fixture) throw fail(404, 'Unknown sample.');
      res.writeHead(200, { 'Content-Type': 'image/png' }); return res.end(fixture);
    }
    if (req.method !== 'POST' || !['universal', 'hard', 'soft', 'sketch'].includes(route)) throw fail(404, 'Not found.');
    const body = await readBody(req);
    let form;
    try { form = await new Request('http://localhost', { method: 'POST', headers: { 'Content-Type': req.headers['content-type'] || '' }, body }).formData(); }
    catch { throw fail(422, 'Expected a multipart form.'); }
    const file = form.get('file'); const sampleId = form.get('sample_id');
    if ((!file && !sampleId) || (file && sampleId)) throw fail(400, 'Choose exactly one file or sample_id.');
    if (file && file.size > 10 * 1024 * 1024) throw fail(413, 'Maximum image size is 10 MB.');
    const bytes = file ? Buffer.from(await file.arrayBuffer()) : fixtures[sampleId];
    if (!bytes) throw fail(404, 'Unknown sample.');
    let original;
    try { original = await sharp(bytes).rotate().resize(128, 128, { fit: 'cover' }).removeAlpha().png().toBuffer(); }
    catch { throw fail(415, 'The uploaded file is not a readable image.'); }
    await new Promise(resolve => setTimeout(resolve, delay));
    const b64 = buffer => buffer.toString('base64');
    if (route === 'sketch') {
      const style = Number(form.get('style'));
      if (![1, 2, 3].includes(style)) throw fail(422, 'Style must be 1, 2, or 3.');
      let sketch = sharp(original).grayscale().normalise();
      if (style === 2) sketch = sketch.linear(1.6, -45);
      if (style === 3) sketch = sketch.threshold(125);
      return send(res, 200, { mock: true, photo_png_b64: b64(original), sketch_png_b64: b64(await sketch.png().toBuffer()), style_id: style - 1, style_label: `Style ${style}`, timing_ms: { inference: 51.2, total: 54.1 } });
    }
    const corruption = form.get('corruption') || 'none'; const severity = form.get('severity') || 'medium';
    const index = ['none', 'salt_pepper', 'gaussian_blur', 'occlusion'].indexOf(corruption);
    if (index < 0 || !['low', 'medium', 'high'].includes(severity)) throw fail(422, 'Invalid corruption or severity.');
    const seed = Number(form.get('seed') || 42);
    if (!Number.isInteger(seed) || seed < 0) throw fail(422, 'Seed must be a non-negative integer.');
    let custom = null;
    try { if (form.get('params')) custom = JSON.parse(form.get('params')); } catch { throw fail(422, 'Parameters must be valid JSON.'); }
    const params = parameters(corruption, severity, custom);
    const input = await corrupt(original, params, seed);
    const result = { mock: true, input_png_b64: b64(input), output_png_b64: b64(original), corruption_applied: corruption, params, seed, timing_ms: { preprocess: 1.2, inference: 38.4, total: 40.1 } };
    if (route === 'hard') Object.assign(result, { probs: [0, 1, 2, 3].map(i => i === index ? 0.91 : 0.03), predicted: ['clean', 'salt_pepper', 'gaussian_blur', 'occlusion'][index], expert: ['identity', 'salt', 'blur', 'occlusion'][index], identity_bypass: index === 0, timing_ms: { preprocess: 1.2, classifier: 4.2, expert: index ? 37.9 : 0, total: index ? 43.3 : 5.4 } });
    if (route === 'soft') Object.assign(result, { weights: [0.082, 0.124, 0.548, 0.246], dominant: 'blur', timing_ms: { preprocess: 1.2, inference: 47.8, total: 49 } });
    return send(res, 200, result);
  } catch (error) { send(res, error.status || 500, { detail: error.status ? error.message : 'Mock server could not process the request.' }); }
});
server.listen(Number(process.env.PORT || 8000), '127.0.0.1', () => console.log('Mock API: http://127.0.0.1:8000 (sketch returns 501 by default)'));
