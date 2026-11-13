// End-to-end test of the browser page in desktop Chrome: loads the 0.8B GGUF, checks the
// translations against the CLI's, and checks that a GPU error is reported and recovered
// from. Not run by CI (it needs a GPU, Chrome and the model).
//
//   ./web/local/build.sh
//   npm install --prefix /tmp/pw playwright-core
//   NODE_PATH=/tmp/pw/node_modules node web/local/e2e.cjs <plamo-3-translate-0.8b-Q4_K_M.gguf>
//
// CHROME overrides the browser path; HEADFUL=1 shows the window.
const http = require('node:http');
const fs = require('node:fs');
const path = require('node:path');
const { chromium } = require('playwright-core');

const gguf = process.argv[2];
if (!gguf) {
  console.error('usage: node web/local/e2e.cjs <gguf>');
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
  { name: 'en->ja document', to: 'ja', text: LONG_EN, check: (out) => out.split('\n\n').length === 12 },
];

// Test-only: keeps the worker's GPUDevice where the test can reach it, to break it.
const SHIM =
  'const __requestDevice = GPUAdapter.prototype.requestDevice;\n' +
  'GPUAdapter.prototype.requestDevice = async function (d) {\n' +
  '  const device = await __requestDevice.call(this, d);\n' +
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
      return /準備できました|^完了$/.test(s.textContent) || s.classList.contains('err');
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
  if (!fs.existsSync(path.join(root, 'pkg', 'yaxus_web.js'))) throw new Error('run ./web/local/build.sh first');
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
    await page.route('**/worker.js', async (route) => {
      const res = await route.fetch();
      const body = await res.text();
      const at = body.indexOf('\n', body.indexOf('import init')) + 1;
      await route.fulfill({ response: res, body: body.slice(0, at) + SHIM + body.slice(at) });
    });
    await page.goto(`http://127.0.0.1:${server.address().port}/`);

    // A file that is not a GGUF is an ordinary error.
    await page.locator('#file').setInputFiles(path.join(root, 'index.html'));
    await settled(page);
    let s = await state(page);
    check('a non-GGUF file is refused', s.err && !s.canTranslate, s.status);

    const t0 = Date.now();
    await page.locator('#file').setInputFiles(gguf);
    await settled(page);
    s = await state(page);
    if (s.err) throw new Error(`load failed: ${s.status}`);
    console.log(`loaded in ${((Date.now() - t0) / 1000).toFixed(1)} s`);
    console.log((await page.locator('#info').textContent()).replace(/^/gm, '  '));

    for (const c of CASES) {
      s = await translate(page, c);
      const ok = !s.err && (c.want ? s.out === c.want : c.check(s.out));
      check(c.name, ok, ok ? s.stats : `${s.status} | got ${JSON.stringify(s.out.slice(0, 200))}`);
    }

    for (const [name, fault] of Object.entries(FAULTS)) {
      await page.workers().at(-1).evaluate(fault);
      s = await translate(page, CASES[0]);
      check(`${name} is reported`, s.err && !s.canTranslate, s.status.split('\n')[0]);
      await load(page);
      s = await translate(page, CASES[0]);
      check(`${name}: works after loading again`, !s.err && s.out === CASES[0].want, s.out);
    }
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
