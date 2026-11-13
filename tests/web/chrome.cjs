// Run against a built app and a locally served web/local directory. See README.md.
const assert = require('node:assert/strict');
const { chromium } = require('playwright-core');

async function testGpuFailure(browser, label, duringTranslation) {
  const page = await browser.newPage();
  await page.route('**/worker.js', async route => {
    const response = await route.fetch();
    const source = await response.text();
    const injection = `const originalCreateBuffer = GPUDevice.prototype.createBuffer;
      GPUDevice.prototype.createBuffer = function(desc) {
        if (desc.label === ${JSON.stringify(label)}) desc = {...desc, size: this.limits.maxBufferSize + 4};
        return originalCreateBuffer.call(this, desc);
      };\n`;
    await route.fulfill({ response, body: injection + source });
  });
  await page.goto(process.env.YAXUS_WEB_URL || 'http://localhost:8765/');
  await page.locator('#model').setInputFiles(process.env.YAXUS_GGUF);
  await page.locator('#load').click();
  await page.waitForFunction(() => /読み込みました|エラー/.test(document.querySelector('#status').textContent), null, { timeout: 180_000 });
  if (duringTranslation) {
    assert.match(await page.locator('#status').textContent(), /読み込みました/);
    await page.locator('#input').fill('吾輩は猫である。');
    await page.locator('#translate').click();
    await page.waitForFunction(() => /完了|エラー/.test(document.querySelector('#status').textContent), null, { timeout: 180_000 });
  }
  assert.match(await page.locator('#status').textContent(), /エラー: WebGPU:/);
  assert(await page.locator('#translate').isDisabled());
  await page.close();
}

(async () => {
  assert(process.env.YAXUS_GGUF, 'Set YAXUS_GGUF to the 0.8B Q4_K_M GGUF');
  assert(process.env.CHROME, 'Set CHROME to the Chrome executable');
  const browser = await chromium.launch({ executablePath: process.env.CHROME, headless: false });
  try {
    const page = await browser.newPage();
    page.on('pageerror', error => console.error(error));
    await page.goto(process.env.YAXUS_WEB_URL || 'http://localhost:8765/');
    assert(await page.evaluate(async () => !!await navigator.gpu?.requestAdapter()), 'WebGPU is unavailable');
    await page.locator('#model').setInputFiles(process.env.YAXUS_GGUF);
    await page.locator('#load').click();
    await page.waitForFunction(() => /読み込みました|エラー/.test(document.querySelector('#status').textContent), null, { timeout: 180_000 });
    assert.match(await page.locator('#status').textContent(), /読み込みました/);
    await page.evaluate(() => {
      window.outputUpdates = 0;
      new MutationObserver(() => window.outputUpdates++).observe(document.querySelector('#output'), { childList: true, subtree: true, characterData: true });
    });
    for (const [target, input, expected] of [['en', '吾輩は猫である。名前はまだ無い。', /cat/i], ['ja', 'The cat is sleeping on the sofa.', /猫/]]) {
      await page.locator('#target').selectOption(target);
      await page.locator('#input').fill(input);
      await page.evaluate(() => { window.outputUpdates = 0; });
      await page.locator('#translate').click();
      assert(await page.locator('#load').isDisabled(), 'model loading must be disabled while translating');
      await page.waitForFunction(() => /完了|エラー/.test(document.querySelector('#status').textContent), null, { timeout: 180_000 });
      assert.match(await page.locator('#status').textContent(), /完了/);
      assert(await page.locator('#load').isEnabled(), 'model loading must be re-enabled after translating');
      const output = await page.locator('#output').textContent();
      assert.match(output, expected, 'translation should mention the cat');
      assert(await page.evaluate(() => window.outputUpdates) > 1, 'translation must stream');
      console.log(`${target}: ${output}`);
    }
    await page.locator('#target').selectOption('en');
    await page.locator('#input').fill('猫がソファで寝ています。'.repeat(45));
    await page.locator('#translate').click();
    await page.waitForFunction(() => /完了|終了|エラー/.test(document.querySelector('#status').textContent), null, { timeout: 180_000 });
    assert.match(await page.locator('#status').textContent(), /上限に達した/);
    await page.locator('#model').setInputFiles({ name: 'invalid.gguf', mimeType: 'application/octet-stream', buffer: Buffer.from('invalid') });
    await page.locator('#load').click();
    await page.waitForFunction(() => document.querySelector('#status').textContent.includes('エラー'), null, { timeout: 10_000 });
    assert(await page.locator('#translate').isDisabled(), 'an invalid model must not be usable');

    // GGUF counts are 64-bit even on wasm32 and must not silently wrap to zero.
    const oversizedCount = Buffer.alloc(24);
    oversizedCount.write('GGUF', 0);
    oversizedCount.writeUInt32LE(3, 4);
    oversizedCount.writeBigUInt64LE(2n ** 32n, 8);
    await page.locator('#model').setInputFiles({ name: 'overflow.gguf', mimeType: 'application/octet-stream', buffer: oversizedCount });
    await page.locator('#load').click();
    await page.waitForFunction(() => /too many GGUF tensors/.test(document.querySelector('#status').textContent), null, { timeout: 10_000 });
    assert(await page.locator('#translate').isDisabled());

    // Validation failures during initialization and inference both reach the UI.
    await testGpuFailure(browser, 'argmax', false);
    await testGpuFailure(browser, 'web readback', true);
  } finally {
    await browser.close();
  }
})().catch(error => { console.error(error); process.exitCode = 1; });
