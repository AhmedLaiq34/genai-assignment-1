// End-to-end check of the Face-to-Sketch workspace against the REAL backend and the real t4_generator.onnx.
//   BASE=http://127.0.0.1:5173 node verification/sketch_real.mjs   (Vite dev server + backend on :8000)
//   node verification/sketch_real.mjs                              (Docker stack on http://localhost:8080)
// Writes screenshots and results.json to verification/sketch_real/.
import { chromium } from 'playwright';
import assert from 'node:assert/strict';
import { mkdir, writeFile } from 'node:fs/promises';
import { fileURLToPath } from 'node:url';

const BASE = process.env.BASE || 'http://localhost:8080';
const directory = fileURLToPath(new URL('./sketch_real/', import.meta.url));
await mkdir(directory, { recursive: true });
const upload = fileURLToPath(new URL('../../backend/samples/Abyssinian_201.png', import.meta.url));

const browser = await chromium.launch({ channel: 'chrome', headless: true, args: ['--disable-gpu'] });
const context = await browser.newContext({ viewport: { width: 1600, height: 1100 }, acceptDownloads: true });
const page = await context.newPage();
page.setDefaultTimeout(30000);
const errors = [], checks = [], details = {};
page.on('pageerror', error => errors.push(error.message));
const shot = name => page.screenshot({ path: `${directory}${name}.png`, fullPage: true });
const generate = () => page.getByRole('button', { name: /^Generate Sketch/ });
const outputSrc = () => page.getByRole('img', { name: 'Generated sketch' }).first().getAttribute('src');

try {
  await page.goto(BASE);
  await page.getByRole('navigation').getByRole('button', { name: 'Face-to-Sketch Generator', exact: true }).click();
  await page.getByRole('heading', { name: 'Face-to-Sketch Generator', exact: true }).waitFor();

  // 1. sample photo from GET /api/sketch/samples, style 1
  const samples = page.getByRole('button', { name: /^Select sample / });
  await samples.first().waitFor();
  details.sampleCount = await samples.count();
  assert.equal(details.sampleCount, 6, 'six display-only sample photos expected');
  assert(await generate().isDisabled(), 'Generate must be disabled before a photo is chosen');
  await samples.first().click();
  await generate().click();
  await page.getByRole('link', { name: /Download sketch/ }).waitFor();
  const first = await outputSrc();
  assert(first && first.startsWith('data:image/png;base64,'), 'generated sketch image missing');
  await page.getByText('Style 1', { exact: true }).first().waitFor();
  await shot('sample-style1');
  checks.push('Sample photo + Style 1: real sketch returned, caption "Style 1", download link shown.');

  // 2. another style gives a different sketch of the same photo
  await page.getByRole('radio', { name: /Style 2/ }).check();
  await generate().click();
  await page.getByRole('link', { name: /Download sketch/ }).waitFor();
  await page.getByText('Style 2', { exact: true }).first().waitFor();
  const second = await outputSrc();
  assert.notEqual(second, first, 'Style 2 must produce a different image than Style 1');
  await shot('sample-style2');
  await page.getByRole('radio', { name: /Style 3/ }).check();
  await generate().click();
  await page.getByText('Style 3', { exact: true }).first().waitFor();
  const third = await outputSrc();
  assert(third !== first && third !== second, 'Style 3 must differ from Styles 1 and 2');
  await shot('sample-style3');
  checks.push('Styles 1, 2 and 3 of the same photo give three different sketches; the captions read Style 1/2/3 (backend style_label).');

  // 3. own upload
  await page.getByLabel('Upload image', { exact: true }).setInputFiles(upload);
  await page.getByRole('radio', { name: /Style 1/ }).check();
  await generate().click();
  await page.getByRole('link', { name: /Download sketch/ }).waitFor();
  await shot('upload-style1');
  checks.push('Uploaded photo: sketch generated.');

  // 4. download works
  const [download] = await Promise.all([page.waitForEvent('download'), page.getByRole('link', { name: /Download sketch/ }).click()]);
  details.downloadName = download.suggestedFilename();
  assert(details.downloadName.endsWith('.png'));
  checks.push(`Download gives a PNG (${details.downloadName}).`);

  assert.deepEqual(errors, [], `uncaught page errors: ${errors.join('; ')}`);
  checks.push('No uncaught browser page errors.');
} finally {
  await writeFile(`${directory}results.json`, JSON.stringify({ base: BASE, checks, details, pageErrors: errors }, null, 2));
  await browser.close();
}
console.log(checks.map(c => `OK  ${c}`).join('\n'));
