// End-to-end check of the UI against the REAL backend (real ONNX models), e.g. the Docker stack on :8080.
//   node verification/real_backend.mjs                       (BASE defaults to http://localhost:8080)
//   BASE=http://127.0.0.1:5173 node verification/real_backend.mjs   (Vite dev server + backend on :8000)
// Unlike browser.mjs it needs no mock. The Soft step needs the real t3_soft_moe.onnx; the Face-to-Sketch step needs the real t4_generator.onnx.
import { chromium } from 'playwright';
import assert from 'node:assert/strict';
import { mkdir, writeFile } from 'node:fs/promises';
import { fileURLToPath } from 'node:url';

const BASE = process.env.BASE || 'http://localhost:8080';
const directory = fileURLToPath(new URL('./real/', import.meta.url));
await mkdir(directory, { recursive: true });
const sample = fileURLToPath(new URL('../../backend/samples/Abyssinian_201.png', import.meta.url));

const browser = await chromium.launch({ channel: 'chrome', headless: true, args: ['--disable-gpu'] });
const context = await browser.newContext({ viewport: { width: 1600, height: 1100 }, acceptDownloads: true });
const page = await context.newPage();
page.setDefaultTimeout(30000);
await page.addInitScript(() => {
  const append = FormData.prototype.append;
  FormData.prototype.append = function (...args) {
    const value = append.apply(this, args);
    window.lastFormFields = Object.fromEntries([...this.entries()].map(([key, item]) => [key, item instanceof File ? item.name : item]));
    return value;
  };
});
const errors = [], checks = [], details = {};
page.on('pageerror', error => errors.push(error.message));
const names = { universal: 'Universal Restoration', hard: 'Hard-Routed Restoration', soft: 'Soft Mixture-of-Experts Restoration', sketch: 'Face-to-Sketch Generator' };
const navigate = async id => { await page.getByRole('navigation').getByRole('button', { name: names[id], exact: true }).click(); await page.getByRole('heading', { name: names[id], exact: true }).waitFor(); };
const run = () => page.getByRole('button', { name: /^(Run Universal|Classify & Route|Run Soft MoE|Generate Sketch)/ });
const corruption = label => page.getByRole('group', { name: 'Corruption type' }).getByRole('button', { name: label, exact: true });
const success = () => page.getByRole('link', { name: /Download.*PNG/ }).waitFor();
const shot = name => page.screenshot({ path: `${directory}${name}.png`, fullPage: true });
const bars = () => page.getByRole('progressbar').evaluateAll(els => els.map(e => [e.getAttribute('aria-label'), Number(e.getAttribute('aria-valuenow'))]));

try {
  await page.goto(BASE);
  await page.getByRole('button', { name: 'Select sample Abyssinian_201' }).waitFor();
  assert(await run().isDisabled());
  checks.push('Page loads, the sample thumbnails come from the real /api/samples, Run disabled before a source is chosen.');

  // System panel: live health from the backend
  await page.getByText('Online', { exact: false }).first().waitFor();
  await page.locator('summary').filter({ hasText: 'System' }).click();
  const panel = page.getByLabel('System panel');
  for (const model of ['t1_universal', 't2_classifier', 't2_salt', 't2_blur', 't2_occlusion']) {
    const row = panel.locator('li').filter({ hasText: model });
    assert((await row.first().innerText()).includes('Loaded'), `${model} not shown as loaded`);
  }
  details.system = await panel.locator('li').allInnerTexts();
  await shot('system-panel');
  await page.locator('summary').filter({ hasText: 'System' }).click();
  checks.push('System panel: Online; t1_universal and the four Task 2 models shown as Loaded.');

  // Universal
  await navigate('universal');
  await page.getByRole('button', { name: 'Select sample Abyssinian_201' }).click();
  await corruption('Salt & pepper').click();
  await run().click();
  await success();
  assert(await page.getByRole('img', { name: 'Restored output', exact: true }).evaluate(i => i.complete && i.naturalWidth === 128));
  await shot('universal');
  checks.push('Universal: sample + salt & pepper -> real restoration, 128 px output image, timing and download shown.');

  // Hard-routed: each corruption and the identity bypass
  await navigate('hard');
  details.hard = {};
  for (const [label, expectedClass, expectedExpert] of [['None', 'clean', 'Identity bypass'], ['Salt & pepper', 'salt pepper', 'salt'],
                                                        ['Gaussian blur', 'gaussian blur', 'blur'], ['Occlusion', 'occlusion', 'occlusion']]) {
    await page.getByRole('button', { name: 'Select sample Abyssinian_201' }).click();
    await corruption(label).click();
    if (label !== 'None') await page.getByRole('group', { name: 'Severity' }).getByRole('button', { name: 'Medium', exact: true }).click();
    await run().click();
    await success();
    const text = await page.locator('main').innerText();
    assert(text.includes(`Predicted corruption: ${expectedClass}`), `${label}: predicted class text missing`);
    assert(text.includes(`Selected expert: ${expectedExpert}`), `${label}: selected expert text missing`);
    const values = await bars();
    assert.equal(values.length, 4);
    assert(Math.abs(values.reduce((a, [, v]) => a + v, 0) - 100) < 3 || Math.abs(values.reduce((a, [, v]) => a + v, 0) - 1) < 0.03);
    details.hard[label] = values;
    await shot(`hard-${label.toLowerCase().replace(/[^a-z]+/g, '-')}`);
  }
  checks.push('Hard-routed: none -> Identity bypass; salt, blur, occlusion -> the matching expert; four probability bars each time.');

  // Upload path
  await navigate('universal');
  const upload = page.getByLabel('Upload image', { exact: true });
  await upload.setInputFiles(sample);
  await run().click();
  const payload = await page.evaluate(() => window.lastFormFields);
  assert.equal(payload.corruption, 'none'); assert(!('sample_id' in payload)); assert(payload.file);
  await success();
  checks.push('Upload: sends file + corruption=none (already corrupted), real backend answers with images.');
  await upload.setInputFiles({ name: 'broken.png', mimeType: 'image/png', buffer: Buffer.from('invalid image bytes') });
  await run().click();
  await page.getByRole('alert').first().waitFor();
  details.brokenUpload = await page.getByRole('alert').first().innerText();
  checks.push(`Unreadable image: the server's error is shown (${details.brokenUpload}).`);

  // Soft mixture of experts (Task 3, needs the real t3_soft_moe.onnx): each corruption at medium severity
  await navigate('soft');
  details.soft = {};
  const branchNames = ['identity', 'salt', 'blur', 'occlusion'];
  for (const [label, file] of [['None', 'none'], ['Salt & pepper', 'salt'], ['Gaussian blur', 'blur'], ['Occlusion', 'occlusion']]) {
    await page.getByRole('button', { name: 'Select sample Abyssinian_201' }).click();
    await corruption(label).click();
    if (label !== 'None') await page.getByRole('group', { name: 'Severity' }).getByRole('button', { name: 'Medium', exact: true }).click();
    await run().click();
    await success();
    assert(await page.getByRole('img', { name: 'Restored output', exact: true }).evaluate(i => i.complete && i.naturalWidth === 128), `${label}: restored image missing`);
    const values = await bars();
    assert.equal(values.length, 4, `${label}: expected four weight bars`);
    const sum = values.reduce((a, [, v]) => a + v, 0);   // the bars show percent or fractions: accept both
    assert(Math.abs(sum - 100) < 3 || Math.abs(sum - 1) < 0.03, `${label}: weights sum to ${sum}`);
    const text = await page.locator('main').innerText();
    const dominant = text.match(/Dominant branch:\s*(\w+)/)?.[1];
    assert(branchNames.includes(dominant), `${label}: dominant branch missing or unknown (${dominant})`);
    assert(/Routing weight sum\s*1\.00\d?/.test(text) || /Routing weight sum\s*0\.99\d/.test(text), `${label}: routing weight sum not shown as 1`);
    details.soft[label] = { weights: values, dominant };
    await shot(`soft-${file}`);
  }
  checks.push('Soft MoE: none, salt, blur, occlusion at medium -> real restoration, four weight bars summing to 1, the dominant branch shown (screenshots soft-<cond>.png).');

  // Face-to-Sketch: the real t4_generator.onnx (see also verification/sketch_real.mjs for the full check)
  await navigate('sketch');
  await page.getByLabel('Upload image', { exact: true }).setInputFiles(sample);
  await run().click();
  await success();
  await shot('sketch');
  checks.push('Face-to-Sketch: a real sketch is returned and shown (Download link present).');

  assert.deepEqual(errors, [], `uncaught page errors: ${errors.join('; ')}`);
  checks.push('No uncaught browser page errors.');
} finally {
  await writeFile(`${directory}results.json`, JSON.stringify({ base: BASE, checks, details, pageErrors: errors }, null, 2));
  await browser.close();
}
console.log(checks.map(c => `OK  ${c}`).join('\n'));
