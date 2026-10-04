// End-to-end checks use a real browser and the standalone mock, not the production backend.
import { chromium, request } from 'playwright';
import assert from 'node:assert/strict';
import { mkdir, writeFile, readFile } from 'node:fs/promises';
import { fileURLToPath } from 'node:url';

const directory = fileURLToPath(new URL('./', import.meta.url));
await mkdir(directory, { recursive: true });
const api = await request.newContext({ baseURL: 'http://127.0.0.1:8000' });
const browser = await chromium.launch({ channel: 'chrome', headless: true, args: ['--disable-gpu', '--use-fake-device-for-media-stream', '--use-fake-ui-for-media-stream'] });
const context = await browser.newContext({ viewport: { width: 1600, height: 1100 }, permissions: ['camera'], acceptDownloads: true });
const page = await context.newPage();
page.setDefaultTimeout(15000);
// Chrome omits multipart bodies containing files from request.postData().
// Record FormData entries before transmission, without reading any image bytes.
await page.addInitScript(() => {
  const append = FormData.prototype.append;
  FormData.prototype.append = function (...args) {
    const value = append.apply(this, args);
    window.lastFormFields = Object.fromEntries([...this.entries()].map(([key, item]) => [key, item instanceof File ? item.name : item]));
    return value;
  };
});
const errors = [];
const checks = [];
page.on('pageerror', error => errors.push(error.message));
const names = { universal: 'Universal Restoration', hard: 'Hard-Routed Restoration', soft: 'Soft Mixture-of-Experts Restoration', sketch: 'Face-to-Sketch Generator' };
const fixture = fileURLToPath(new URL('../mock/fixtures/Abyssinian_201.png', import.meta.url));
const config = async failures => { const response = await api.post('/__mock/config', { data: { failures, delay: 500 } }); assert(response.ok()); };
const navigate = async id => { await page.getByRole('navigation').getByRole('button', { name: names[id], exact: true }).click(); await page.getByRole('heading', { name: names[id], exact: true }).waitFor(); };
const run = () => page.getByRole('button', { name: /^(Run Universal|Classify & Route|Run Soft MoE|Generate Sketch)/ });
async function success() { await page.getByRole('link', { name: /Download.*PNG/ }).waitFor(); }
async function error() { await page.getByRole('alert').filter({ hasText: 'Not available yet' }).waitFor(); }
async function screenshot(name) { await page.screenshot({ path: `${directory}${name}.png`, fullPage: true }); }

try {
  await config({ sketch: 501 });
  await page.goto('http://127.0.0.1:5173');
  await page.getByRole('button', { name: 'Select sample Abyssinian_201' }).waitFor();
  assert(await run().isDisabled());
  await screenshot('universal-empty-desktop');
  checks.push('Initial empty state; Run disabled until a source is selected.');

  for (const id of ['universal', 'hard', 'soft']) {
    await navigate(id);
    await page.getByRole('button', { name: 'Select sample Abyssinian_201' }).click();
    await page.getByRole('group', { name: 'Corruption type' }).getByRole('button', { name: id === 'universal' ? 'Salt & pepper' : 'Gaussian blur', exact: true }).click();
    await run().click();
    await page.getByText('Processing…', { exact: true }).waitFor();
    await success();
    const output = page.getByRole('img', { name: 'Restored output', exact: true });
    assert(await output.evaluate(image => image.complete && image.naturalWidth === 128));
    if (id !== 'universal') assert.equal(await page.getByRole('progressbar').count(), 4);
    await screenshot(`${id}-desktop`);
    await page.setViewportSize({ width: 390, height: 844 });
    await screenshot(`${id}-mobile`);
    assert(await page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth), `${id}: mobile overflow`);
    await page.setViewportSize({ width: 1600, height: 1100 });
    const downloaded = page.waitForEvent('download');
    await page.getByRole('link', { name: /Download.*PNG/ }).click();
    const download = await downloaded;
    await download.saveAs(`${directory}${id}-download.png`);
    assert((await readFile(`${directory}${id}-download.png`)).subarray(1, 4).equals(Buffer.from('PNG')));
    await config({ [id]: id === 'hard' ? 503 : 501, sketch: 501 });
    await run().click(); await error();
    assert.equal(await page.getByRole('link', { name: /Download.*PNG/ }).count(), 0);
    await config({ sketch: 501 });
    await run().click(); await success();
    checks.push(`${names[id]}: sample, loading, successful image/routing response, PNG download, HTTP ${id === 'hard' ? 503 : 501}, retry, desktop and mobile.`);
  }

  await navigate('hard');
  await page.getByRole('button', { name: 'Select sample Abyssinian_201' }).click();
  await run().click(); await success();
  await page.getByText('Identity bypass', { exact: true }).waitFor();
  checks.push('Hard routing clean input selects Identity bypass.');

  await navigate('universal');
  const upload = page.getByLabel('Upload image', { exact: true });
  await upload.setInputFiles({ name: 'notes.txt', mimeType: 'text/plain', buffer: Buffer.from('test') });
  await page.getByRole('alert').filter({ hasText: 'Choose a PNG' }).waitFor();
  await upload.setInputFiles({ name: 'large.png', mimeType: 'image/png', buffer: Buffer.alloc(10 * 1024 * 1024 + 1) });
  await page.getByRole('alert').filter({ hasText: 'too large' }).waitFor();
  await upload.setInputFiles(fixture);
  assert(!(await page.getByRole('checkbox', { name: /Already corrupted/ }).isChecked()));   // uploads can be corrupted by default
  assert(await page.getByRole('group', { name: 'Corruption type' }).getByRole('button', { name: 'Gaussian blur' }).isEnabled());
  await page.getByRole('checkbox', { name: /Already corrupted/ }).check();
  assert(await page.getByRole('group', { name: 'Corruption type' }).getByRole('button', { name: 'Gaussian blur' }).isDisabled());
  await run().click();
  const payload = await page.evaluate(() => window.lastFormFields);
  assert.equal(payload.corruption, 'none');
  assert(!('sample_id' in payload));
  await success();
  await page.getByRole('checkbox', { name: /Already corrupted/ }).uncheck();
  await page.getByRole('group', { name: 'Corruption type' }).getByRole('button', { name: 'Gaussian blur' }).click();
  await page.getByRole('checkbox', { name: 'Use custom parameters' }).check();
  await page.getByRole('spinbutton', { name: 'Kernel (odd, 3–15)' }).fill('7');
  await page.getByRole('spinbutton', { name: 'Sigma', exact: true }).fill('2.5');
  await run().click();
  const custom = await page.evaluate(() => window.lastFormFields);
  assert.equal(custom.params, '{"kernel":7,"sigma":2.5}');
  assert.equal(custom.severity, 'medium');
  await success();
  await upload.setInputFiles({ name: 'broken.png', mimeType: 'image/png', buffer: Buffer.from('invalid image bytes') });
  await run().click();
  await page.getByRole('alert').filter({ hasText: 'not a readable image' }).waitFor();
  checks.push('Client rejects incorrect MIME/oversized files; corrupt PNG server error; upload sends none; custom kernel/sigma and valid severity verified in multipart.');

  await navigate('sketch');
  await page.getByLabel('Upload image', { exact: true }).setInputFiles(fixture);
  await run().click(); await error();
  await screenshot('sketch-unavailable-desktop');
  await config({});
  for (const style of [1, 2, 3]) {
    await page.getByRole('radio', { name: new RegExp(`Style ${style}`) }).check();
    await run().click(); await success();
    await page.getByText(`Style ${style}`, { exact: true }).last().waitFor();
  }
  await screenshot('sketch-desktop');
  await page.setViewportSize({ width: 390, height: 844 });
  await screenshot('sketch-mobile');
  assert(await page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth));
  await page.setViewportSize({ width: 1600, height: 1100 });
  const sketchDownload = page.waitForEvent('download');
  await page.getByRole('link', { name: /Download sketch/ }).click();
  await (await sketchDownload).saveAs(`${directory}sketch-download.png`);
  await page.evaluate(() => {
    window.originalCamera = navigator.mediaDevices.getUserMedia.bind(navigator.mediaDevices);
    navigator.mediaDevices.getUserMedia = async () => { throw new DOMException('Denied', 'NotAllowedError'); };
  });
  await page.getByRole('button', { name: 'Use Webcam' }).click();
  await page.getByRole('alert').filter({ hasText: 'Camera permission denied' }).waitFor();
  await page.evaluate(() => { navigator.mediaDevices.getUserMedia = window.originalCamera; });
  await page.getByRole('button', { name: 'Use Webcam' }).click();
  await page.waitForFunction(() => document.querySelector('video')?.videoWidth > 0);
  await page.evaluate(() => { window.capturedStream = document.querySelector('video').srcObject; });
  await page.getByRole('button', { name: 'Capture photo' }).click();
  await page.getByText('webcam-photo.png', { exact: true }).first().waitFor();
  assert(await page.evaluate(() => window.capturedStream.getTracks().every(track => track.readyState === 'ended')));
  await run().click(); await success();
  checks.push('Sketch: default 501, all three style success paths, download, permission denial, synthetic webcam capture and track cleanup, desktop/mobile. Real camera not tested.');

  await page.locator('summary').click();
  await page.getByRole('heading', { name: 'System', exact: true }).waitFor();
  await page.getByRole('button', { name: 'Refresh status' }).click();
  await page.getByText('t1_universal', { exact: true }).waitFor();
  await config({ health: 503 });
  await page.getByRole('button', { name: 'Refresh status' }).click();
  await page.getByRole('alert').filter({ hasText: 'health model' }).waitFor();
  await config({ sketch: 501 });
  await page.getByRole('button', { name: 'Refresh status' }).click();
  await page.getByText('Missing', { exact: true }).waitFor();
  checks.push('System health: online, offline, recovery, per-model loaded/missing, last inference.');
  assert.deepEqual(errors, []);
  await writeFile(`${directory}results.json`, JSON.stringify({ passed: true, checks, pageErrors: errors }, null, 2));
  console.log(checks.join('\n'));
} catch (error) {
  await screenshot('failure');
  console.error((await page.locator('body').innerText()).slice(-5000));
  throw error;
} finally {
  await config({ sketch: 501 }).catch(() => {});
  await browser.close(); await api.dispose();
}
