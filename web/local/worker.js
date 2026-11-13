import init, { load_model } from './pkg/yaxus_web.js';
let translator;
let pending = Promise.resolve();
const status = message => postMessage({type:'status', message});
onmessage = ({data}) => { pending = pending.then(() => handle(data)); };
async function handle(data) {
  try {
    if (data.type === 'load') {
      translator?.free(); translator = undefined;
      status('WASM を読み込んでいます…');
      await init();
      status('モデルを読み込んでいます…');
      translator = await load_model(data.file);
      postMessage({type:'ready'});
    } else if (data.type === 'translate') {
      if (!translator) throw new Error('モデルを読み込んでください。');
      const truncated = await translator.translate(data.text, data.target, text => postMessage({type:'text', text}));
      postMessage({type:'done', truncated});
    }
  } catch (error) {
    const message = String(error);
    const fatal = error instanceof WebAssembly.RuntimeError || message.startsWith('WebGPU:') || message.includes('GPU readback');
    postMessage({type:'error', message, fatal});
  }
}
