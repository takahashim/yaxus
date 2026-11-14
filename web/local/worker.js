// The model runs here, off the page's thread: loading and translating never block the
// page, and the page can throw the whole worker away (wasm memory, WebGPU device and
// all) after an error it cannot trust, or before loading another file.
//
// Messages in:  {type: 'load', file}  {type: 'translate', text, target, form, chains}
// Messages out: {type: 'status', message}  {type: 'read', done, total}
//               {type: 'ready', info}  {type: 'progress', done, total}
//               {type: 'plan', breaks}  {type: 'text', segment, text}
//               {type: 'done', stats}
//               {type: 'error', message, fatal}
import init, { ModelBuffer, load_model } from './pkg/yaxus_web.js';

let translator = null;
// One request at a time, in order (the page disables its buttons meanwhile, but a
// message may still arrive while the previous one is being handled).
let queue = Promise.resolve();

const post = (msg) => postMessage(msg);
const status = (message) => post({ type: 'status', message });

onmessage = ({ data }) => {
  queue = queue.then(() => handle(data));
};

async function handle(data) {
  try {
    if (data.type === 'load') await load(data.file);
    else if (data.type === 'translate') await translate(data);
  } catch (e) {
    post({ type: 'error', message: e?.message || String(e), fatal: isFatal(e) });
  }
}

// A panic aborts the wasm instance (it shows as a RuntimeError), and an error of the GPU
// leaves the device unusable (`broken`); either way only a new worker helps.
function isFatal(e) {
  if (e instanceof WebAssembly.RuntimeError) return true;
  try {
    return translator?.broken === true;
  } catch {
    return true;
  }
}

async function load(file) {
  status('wasm を読み込んでいます…');
  await init();
  status('ファイルを読んでいます…');
  const buffer = new ModelBuffer(file.size);
  const reader = file.stream().getReader();
  let done = 0;
  let shown = -1;
  for (;;) {
    const chunk = await reader.read();
    if (chunk.done) break;
    buffer.push(chunk.value);
    done += chunk.value.length;
    const pct = Math.floor((100 * done) / file.size);
    if (pct !== shown) {
      shown = pct;
      post({ type: 'read', done, total: file.size });
    }
  }
  translator = await load_model(buffer, status);
  post({
    type: 'ready',
    info: {
      device: translator.device || (await adapterName()),
      maxBinding: translator.max_binding,
      loadMs: translator.load_ms,
      reducedToQ4: translator.reduced_to_q4,
    },
  });
}

// Browsers leave the adapter's description empty; vendor and architecture are given.
async function adapterName() {
  try {
    const info = (await navigator.gpu.requestAdapter())?.info;
    return [info?.vendor, info?.architecture].filter(Boolean).join(' ') || 'WebGPU';
  } catch {
    return 'WebGPU';
  }
}

async function translate({ text, target, form, chains, neighbor }) {
  if (!translator) throw new Error('モデルが読み込まれていません。');
  // 0 is the automatic number of chains.
  translator.set_chains(chains ?? 1);
  translator.set_neighbor_context(!!neighbor);
  const stats = await translator.translate(
    text,
    target,
    form || null,
    (breaks) => post({ type: 'plan', breaks: Array.from(breaks) }),
    (segment, text) => post({ type: 'text', segment, text }),
    (done, total) => post({ type: 'progress', done, total }),
  );
  const plain = {
    promptTokens: stats.prompt_tokens,
    outputTokens: stats.output_tokens,
    maxNew: stats.max_new,
    segments: stats.segments,
    stop: stats.stop,
    reusedTokens: stats.reused_tokens,
    elapsedMs: stats.elapsed_ms,
    prefillMs: stats.prefill_ms,
    decodeMs: stats.decode_ms,
    tokensPerSecond: stats.tokens_per_second,
  };
  stats.free();
  post({ type: 'done', stats: plain });
}
