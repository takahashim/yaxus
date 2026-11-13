const $ = id => document.getElementById(id);
let worker;
let ready = false;
function startWorker() {
  worker = new Worker('./worker.js', { type: 'module' });
  worker.onmessage = onWorkerMessage;
  worker.onerror = e => {
    $('status').textContent = `エラー: ${e.message}`;
    resetWorker();
  };
}
function resetWorker() {
  ready = false;
  $('translate').disabled = true;
  $('load').disabled = false;
  worker.terminate();
  worker = null;
}
function onWorkerMessage({data}) {
  if (data.type === 'status') $('status').textContent = data.message;
  if (data.type === 'ready') { ready = true; $('status').textContent = 'モデルを読み込みました。'; $('translate').disabled = false; $('load').disabled = false; }
  if (data.type === 'text') $('output').textContent += data.text;
  if (data.type === 'done') { $('translate').disabled = false; $('load').disabled = false; $('status').textContent = data.truncated ? '出力の上限に達したため、翻訳を途中で終了しました。' : '翻訳が完了しました。'; }
  if (data.type === 'error') {
    $('status').textContent = `エラー: ${data.message}`;
    if (data.fatal) resetWorker();
    else { $('load').disabled = false; $('translate').disabled = !ready; }
  }
}
startWorker();
$('load').onclick = () => {
  const file = $('model').files[0];
  if (!file) { $('status').textContent = 'GGUF ファイルを選択してください。'; return; }
  if (!worker) startWorker();
  ready = false; $('translate').disabled = true; $('load').disabled = true;
  worker.postMessage({type:'load', file});
};
$('translate').onclick = () => {
  $('output').textContent = ''; $('translate').disabled = true; $('load').disabled = true;
  $('status').textContent = '翻訳中…';
  worker.postMessage({type:'translate', text:$('input').value, target:$('target').value});
};
