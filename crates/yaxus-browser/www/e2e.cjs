// End-to-end test of the browser page in desktop Chrome: loads the 0.8B GGUF, checks the
// translations against the CLI's, and checks that a GPU error is reported and recovered
// from. Not run by CI (it needs a GPU, Chrome and the model).
//
//   ./crates/yaxus-browser/www/build.sh
//   npm install --prefix /tmp/pw playwright-core
//   NODE_PATH=/tmp/pw/node_modules node crates/yaxus-browser/www/e2e.cjs <plamo-3-translate-0.8b-Q4_K_M.gguf>
//
// CHROME overrides the browser path; HEADFUL=1 shows the window.
const http = require('node:http');
const fs = require('node:fs');
const path = require('node:path');
const { chromium } = require('playwright-core');

const gguf = process.argv[2];
if (!gguf) {
  console.error('usage: node crates/yaxus-browser/www/e2e.cjs <gguf>');
  process.exit(2);
}
const CHROME = process.env.CHROME || {
  darwin: '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome',
  linux: '/usr/bin/google-chrome',
  win32: 'C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe',
}[process.platform];

// `yaxus translate --model 0.8b:Q4_K_M --draft none --to <lang>` (f32 and int16 agree).
const MID_EN =
  'Machine translation has improved quickly in recent years. Small models can now run on a laptop, ' +
  'and some of them even run inside a web browser. This makes it possible to translate private ' +
  'documents without sending them to a server.';
const LONG_EN = Array(12)
  .fill(
    'The committee met on Tuesday to review the budget for the coming year. ' +
      'Several members argued that the proposed cuts to the library would hurt students, ' +
      'while others pointed out that the city had already postponed repairs to two bridges.',
  )
  .join('\n\n');
const CASES = [
  { name: 'en->ja', to: 'ja', text: 'The cat is sleeping on the sofa.', want: '猫がソファで寝ている。' },
  { name: 'ja->en', to: 'en', text: '吾輩は猫である。名前はまだ無い。', want: 'I am a cat. I have no name yet.' },
  {
    name: 'en->ja paragraph',
    to: 'ja',
    text: MID_EN,
    want:
      '機械翻訳は近年急速に進化しています。小型のモデルはノートパソコンで動作するようになり、' +
      '中にはウェブブラウザ内で動作するものさえあります。' +
      'これにより、プライベートな文書をサーバーに送信することなく翻訳することが可能になりました。',
  },
  { name: 'en->ja polite', to: 'ja', form: 'polite', text: 'The cat is sleeping on the sofa.', want: '猫がソファで寝ています。' },
  // 12 paragraphs: 12 segments, joined by blank lines.
  // Each paragraph's prompt begins with the earlier ones, so most of its KV is reused.
  { name: 'en->ja document', to: 'ja', text: LONG_EN, check: (out) => out.split('\n\n').length === 12, reused: true },
];

// Test-only: keeps the worker's GPUDevice where the test can reach it, to break it.
// `self.__rejectFirst` makes the first device request fail (an adapter that advertises
// more than it grants); `self.__fakeBinding` makes the device report a smaller storage
// binding than it has (WebGPU never grants less than 128 MB, which the 0.8B fits);
// `self.__failBigBuffers` makes every buffer over 64 MB invalid (an allocation that fails);
// `self.__failNextSubmit` makes an invalid buffer at the next submission (an error in the
// middle of a translation), and `self.__delayErrors` holds uncaptured-error events back
// that many milliseconds (WebGPU does not say when they arrive).
const SHIM =
  'const __onerror = Object.getOwnPropertyDescriptor(GPUDevice.prototype, "onuncapturederror");\n' +
  'Object.defineProperty(GPUDevice.prototype, "onuncapturederror", {\n' +
  '  get() { return __onerror.get.call(this); },\n' +
  '  set(h) { __onerror.set.call(this, h && ((e) => (self.__delayErrors ? setTimeout(() => h(e), self.__delayErrors) : h(e)))); },\n' +
  '});\n' +
  'const __submit = GPUQueue.prototype.submit;\n' +
  'GPUQueue.prototype.submit = function (b) {\n' +
  '  if (self.__failNextSubmit) {\n' +
  '    self.__failNextSubmit = false;\n' +
  '    self.__device.createBuffer({ size: 4, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.STORAGE });\n' +
  '  }\n' +
  '  return __submit.call(this, b);\n' +
  '};\n' +
  'const __createBuffer = GPUDevice.prototype.createBuffer;\n' +
  'GPUDevice.prototype.createBuffer = function (d) {\n' +
  '  if (self.__failBigBuffers && d.size > (64 << 20)) d = { ...d, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.STORAGE };\n' +
  '  return __createBuffer.call(this, d);\n' +
  '};\n' +
  'const __requestDevice = GPUAdapter.prototype.requestDevice;\n' +
  'GPUAdapter.prototype.requestDevice = async function (d) {\n' +
  '  if (self.__rejectFirst && !self.__rejected) { self.__rejected = true; throw new DOMException("test: rejected", "OperationError"); }\n' +
  '  const device = await __requestDevice.call(this, d);\n' +
  '  if (self.__fakeBinding) {\n' +
  '    const limits = {};\n' +
  '    for (const k in device.limits) limits[k] = device.limits[k];\n' +
  '    limits.maxStorageBufferBindingSize = self.__fakeBinding;\n' +
  '    Object.defineProperty(device, "limits", { value: limits });\n' +
  '  }\n' +
  '  self.__device = device;\n' +
  '  return device;\n' +
  '};\n';
const FAULTS = {
  'device lost': () => self.__device.destroy(),
  'validation error': () =>
    self.__device.createBuffer({ size: 4, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.STORAGE }),
};

function serve(root) {
  const types = { '.html': 'text/html', '.js': 'text/javascript', '.wasm': 'application/wasm' };
  const server = http.createServer((req, res) => {
    const file = path.join(root, decodeURIComponent(new URL(req.url, 'http://x').pathname));
    const target = file.endsWith('/') ? path.join(file, 'index.html') : file;
    if (!target.startsWith(root) || !fs.existsSync(target)) {
      res.writeHead(404).end();
      return;
    }
    res.writeHead(200, { 'content-type': types[path.extname(target)] || 'application/octet-stream' });
    fs.createReadStream(target).pipe(res);
  });
  return new Promise((resolve) => server.listen(0, '127.0.0.1', () => resolve(server)));
}

const settled = (page) =>
  page.waitForFunction(
    () => {
      const s = document.querySelector('#status');
      return /準備できました|^完了$|^中止しました$/.test(s.textContent) || s.classList.contains('err');
    },
    null,
    { timeout: 600_000 },
  );
const state = (page) =>
  page.evaluate(() => ({
    status: document.querySelector('#status').textContent,
    err: document.querySelector('#status').classList.contains('err'),
    out: document.querySelector('#output').textContent,
    stats: document.querySelector('#stats').textContent,
    canTranslate: !document.querySelector('#translate').disabled,
  }));

async function load(page) {
  await page.locator('#load').click();
  await settled(page);
  const s = await state(page);
  if (s.err) throw new Error(`load failed: ${s.status}`);
}

async function translate(page, c) {
  await page.locator('#target').selectOption(c.to);
  if (c.to === 'ja') await page.locator('#form').selectOption(c.form || '');
  await page.locator('#source').fill(c.text);
  await page.locator('#translate').click();
  await settled(page);
  return state(page);
}

(async () => {
  const root = path.resolve(__dirname);
  if (!fs.existsSync(path.join(root, 'pkg', 'yaxus_browser.js'))) throw new Error('run ./crates/yaxus-browser/www/build.sh first');
  const server = await serve(root);
  const browser = await chromium.launch({
    executablePath: CHROME,
    headless: !process.env.HEADFUL,
    args: ['--enable-unsafe-webgpu', ...(process.platform === 'darwin' ? ['--use-angle=metal'] : [])],
  });
  let failures = 0;
  const check = (name, ok, detail) => {
    console.log(`${ok ? 'ok  ' : 'FAIL'} ${name}${detail ? ` — ${detail}` : ''}`);
    if (!ok) failures++;
  };
  try {
    const page = await browser.newPage();
    let flags = '';
    await page.route('**/worker.js', async (route) => {
      const res = await route.fetch();
      const body = await res.text();
      const at = body.indexOf('\n', body.indexOf('import init')) + 1;
      const shim = SHIM + flags;
      await route.fulfill({ response: res, body: body.slice(0, at) + shim + body.slice(at) });
    });
    await page.goto(`http://127.0.0.1:${server.address().port}/`);

    // A file that is not a GGUF is an ordinary error.
    await page.locator('#file').setInputFiles(path.join(root, 'index.html'));
    await settled(page);
    let s = await state(page);
    check('a non-GGUF file is refused', s.err && !s.canTranslate, s.status);

    // Counts and sizes beyond wasm32's usize are errors, not truncated (2^32 tensors
    // would read as none).
    const u32le = (v) => { const b = Buffer.alloc(4); b.writeUInt32LE(v); return b; };
    const u64le = (v) => { const b = Buffer.alloc(8); b.writeBigUInt64LE(BigInt(v)); return b; };
    const str = (t) => Buffer.concat([u64le(t.length), Buffer.from(t)]);
    const header = (tensors, kvs) => Buffer.concat([Buffer.from('GGUF'), u32le(3), u64le(tensors), u64le(kvs)]);
    const bad = {
      '2^32 tensors': header(2n ** 32n, 0),
      'an alignment of 2^63': Buffer.concat([
        header(1, 1), str('general.alignment'), u32le(10), u64le(2n ** 63n),
        str('t'), u32le(1), u64le(4), u32le(0), u64le(0), Buffer.alloc(64),
      ]),
    };
    for (const [name, buffer] of Object.entries(bad)) {
      await page.locator('#file').setInputFiles({ name: 'bad.gguf', mimeType: 'application/octet-stream', buffer });
      await settled(page);
      s = await state(page);
      check(`a GGUF with ${name} is refused on wasm32`, s.err && /does not fit/.test(s.status), s.status);
    }

    const t0 = Date.now();
    await page.locator('#file').setInputFiles(gguf);
    await settled(page);
    s = await state(page);
    if (s.err) throw new Error(`load failed: ${s.status}`);
    console.log(`loaded in ${((Date.now() - t0) / 1000).toFixed(1)} s`);
    console.log((await page.locator('#info').textContent()).replace(/^/gm, '  '));

    // As in the server's page, the segments of a document are translated side by side by default.
    check('the default is the automatic number of chains', (await page.locator('#chains').inputValue()) === '0', '');
    for (const c of CASES) {
      s = await translate(page, c);
      const ok = !s.err && (c.want ? s.out === c.want : c.check(s.out)) && (!c.reused || /再利用/.test(s.stats));
      check(c.name, ok, ok ? s.stats : `${s.status} | got ${JSON.stringify(s.out.slice(0, 200))}`);
    }

    // The document again with segments translated side by side: all of them, in order, and the
    // choice is the page's own (`#chains`).
    for (const [value, label] of [['2', '2 chains'], ['0', 'automatic chains (3 for 12 segments)']]) {
      await page.locator('#chains').selectOption(value);
      s = await translate(page, CASES[4]);
      const parts = s.out.split('\n\n');
      // Side by side, a chain's first segment has no history: the prompts are not those of one chain (2888 tokens).
      const side = !/prompt 2888 tokens/.test(s.stats);
      check(`${label}: the document is translated in full`, !s.err && parts.length === 12 && parts.every((p) => p.trim()) && side, s.stats);
    }
    await page.locator('#chains').selectOption('1');

    // A stop: the translation ends with what it has, soon after, and the model is still there.
    await page.locator('#chains').selectOption('0');
    await page.locator('#target').selectOption('ja');
    // (a long one: 36 paragraphs, about 3,500 characters when done)
    await page.locator('#source').fill([CASES[4].text, CASES[4].text, CASES[4].text].join('\n\n'));
    await page.locator('#translate').click();
    await page.waitForFunction(() => document.querySelector('#output').textContent.length > 0, null, { timeout: 120_000 });
    const stopAt = Date.now();
    await page.locator('#stop').click();
    await settled(page);
    s = await state(page);
    check('a stop ends the translation with what it had', !s.err && s.status === '中止しました' && /中止/.test(s.stats) && s.out.length > 0 && s.out.length < 3000, `${s.status} (${Date.now() - stopAt} ms) ${s.out.length} chars`);
    s = await translate(page, CASES[0]);
    check('after a stop the model translates again', !s.err && s.out === CASES[0].want, s.out);

    // Segments may be translated side by side: the text of each is kept apart and laid out in order.
    const laid = await page.evaluate(async () => {
      const { Layout } = await import('./layout.js');
      const el = document.createElement('div');
      const layout = new Layout(el);
      layout.plan(['\n\n', '\n\n', ' ', '\n\n']);
      layout.add(2, 'C');
      layout.add(0, 'A');
      layout.add(3, 'D');
      layout.add(1, 'B');
      layout.flush();
      return el.textContent;
    });
    check('segments that arrive out of order are laid out in order', laid === 'A\n\nB C\n\nD', JSON.stringify(laid));

    for (const [name, fault] of Object.entries(FAULTS)) {
      await page.workers().at(-1).evaluate(fault);
      s = await translate(page, CASES[0]);
      check(`${name} is reported`, s.err && !s.canTranslate, s.status.split('\n')[0]);
      await load(page);
      s = await translate(page, CASES[0]);
      check(`${name}: works after loading again`, !s.err && s.out === CASES[0].want, s.out);
    }

    // An error in the middle of a translation, reported late: that translation fails,
    // not the next one.
    await page.workers().at(-1).evaluate(() => {
      self.__delayErrors = 2000;
    });
    await page.locator('#target').selectOption('ja');
    await page.locator('#source').fill(CASES[2].text);
    await page.locator('#translate').click();
    await page.waitForFunction(() => /翻訳しています/.test(document.querySelector('#status').textContent));
    await page.workers().at(-1).evaluate(() => {
      self.__failNextSubmit = true;
    });
    await settled(page);
    s = await state(page);
    check('a late GPU error fails the translation it happened in', s.err && !s.canTranslate, s.status.split('\n')[0]);
    await load(page);

    // The same with segments translated side by side (the automatic number of chains): one
    // error fails the whole translation, once, and the model loads again.
    await page.locator('#chains').selectOption('0');
    await page.locator('#target').selectOption('ja');
    await page.locator('#source').fill([CASES[4].text, CASES[4].text].join('\n\n'));
    await page.locator('#translate').click();
    await page.waitForFunction(() => document.querySelector('#output').textContent.length > 0, null, { timeout: 120_000 });
    await page.workers().at(-1).evaluate(() => {
      self.__failNextSubmit = true;
    });
    await settled(page);
    s = await state(page);
    check('a GPU error while segments run side by side fails the translation', s.err && !s.canTranslate, s.status.split('\n')[0]);
    await load(page);
    s = await translate(page, CASES[0]);
    check('side by side: works after loading again', !s.err && s.out === CASES[0].want, s.out);

    // A buffer that fails while the weights are uploaded is a load error.
    flags = 'self.__failBigBuffers = true;\n';
    await page.locator('#load').click();
    await settled(page);
    s = await state(page);
    check('a failed allocation while loading is a load error', s.err && !s.canTranslate, s.status.split('\n')[0]);

    // The first device request is rejected: the retry with the base limits (a 128 MB
    // binding) works, and the 0.8B fits it whole (its largest buffer is 124 MB).
    flags = 'self.__rejectFirst = true;\n';
    await load(page);
    let warn = await page.locator('#warn').textContent();
    s = await translate(page, CASES[0]);
    check('a rejected device request is retried', !s.err && s.out === CASES[0].want && !warn, warn || s.out);

    // A 110 MiB binding: the Q6_K output matrix (a 123.9 MB buffer) is loaded as Q4
    // (82.6 MB), the page says so, and translation still works (the output may differ
    // from the CLI's).
    flags = `self.__fakeBinding = ${110 << 20};\n`;
    await load(page);
    warn = await page.locator('#warn').textContent();
    check('a matrix over the binding limit loads as Q4', /Q4/.test(warn), warn);
    s = await translate(page, CASES[0]);
    check('a matrix over the binding limit: still translates', !s.err && s.out.length > 0, s.out);
  } finally {
    await browser.close();
    server.close();
  }
  console.log(failures ? `${failures} failed` : 'all passed');
  process.exitCode = failures ? 1 : 0;
})().catch((e) => {
  console.error('FAILED:', e.message);
  process.exitCode = 1;
});
