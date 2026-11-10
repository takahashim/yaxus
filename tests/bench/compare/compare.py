"""Compare yakusel and Ollama on the same prompts.

    python compare.py BACKEND MODE [options]

BACKEND: ollama:<model> (raw PFML3 prompts to /api/generate, built with the model's
chat template) or yakusel (POST /v1/translate on YAKUSEL_URL with YAKUSEL_TOKEN).
MODE: set FILE (sentences, `lang<TAB>text`), doc FILE (paragraphs, client-side chains
with the same history window as yakusel), par FILE N (N concurrent sentences).
Writes outputs to --out (one per line, newlines escaped).

Environment: MODEL_DIR (the HF model directory, for the chat template), OLLAMA_URL
(default http://127.0.0.1:11435), YAKUSEL_URL, YAKUSEL_TOKEN. Needs transformers for
the Ollama backend. Start yakusel with --no-response-cache when repeating runs.
"""

import argparse
import json
import os
import ssl
import sys
import threading
import time
import urllib.request

OLLAMA = os.environ.get("OLLAMA_URL", "http://127.0.0.1:11435")
YAKUSEL = os.environ.get("YAKUSEL_URL", "https://127.0.0.1:18443")
TOKEN = os.environ.get("YAKUSEL_TOKEN", "")
STOP = ["<|plamo:tag|>", "<|plamo:bos|>"]
CTX = ssl._create_unverified_context()
_tok = None


def tokenizer():
    global _tok
    if _tok is None:
        from transformers import AutoTokenizer

        _tok = AutoTokenizer.from_pretrained(os.environ["MODEL_DIR"], trust_remote_code=True)
    return _tok


def post(url, body, headers=None):
    h = {"content-type": "application/json", **(headers or {})}
    req = urllib.request.Request(url, json.dumps(body).encode(), h)
    with urllib.request.urlopen(req, context=CTX if url.startswith("https") else None, timeout=3600) as r:
        return json.load(r)


def translate(backend, text, lang, history=()):
    """Returns (output, prompt_tokens, generated, prefill_s, total_s)."""
    t0 = time.time()
    if backend.startswith("ollama:"):
        msgs = []
        for s, t in history:
            msgs += [{"role": "user", "content": s}, {"role": "assistant", "content": t}]
        msgs.append({"role": "user", "content": text})
        prompt = tokenizer().apply_chat_template(msgs, add_generation_prompt=True, tokenize=False, target_lang=lang)
        r = post(
            f"{OLLAMA}/api/generate",
            {
                "model": backend[7:],
                "prompt": prompt,
                "raw": True,
                "stream": False,
                "keep_alive": "30m",
                "options": {"temperature": 0, "top_k": 1, "num_predict": 256, "num_ctx": 8192, "stop": STOP},
            },
        )
        return (
            r["response"],
            r.get("prompt_eval_count", 0),
            r.get("eval_count", 0),
            r.get("prompt_eval_duration", 0) / 1e9,
            time.time() - t0,
        )
    body = {"text": text, "target_lang": lang, "history": [{"source": s, "target": t} for s, t in history]}
    r = post(f"{YAKUSEL}/v1/translate", body, {"authorization": f"Bearer {TOKEN}"})
    u = r["usage"]
    return r["text"], u["prompt_tokens"], u["completion_tokens"], u["prefill_ms"] / 1e3, time.time() - t0


class Window:
    """yakusel_api::HistoryWindow."""

    def __init__(self, budget):
        self.start, self.budget = 0, budget

    def select(self, pairs):
        if self.budget == 0:
            return []
        chars = lambda p: len(p[0]) + len(p[1])
        self.start = min(self.start, len(pairs))
        if sum(map(chars, pairs[self.start :])) > self.budget:
            start = len(pairs) - 1
            kept = chars(pairs[start])
            while start > self.start and kept + chars(pairs[start - 1]) <= self.budget // 4:
                start -= 1
                kept += chars(pairs[start])
            self.start = start
        return pairs[self.start :]


def paragraphs(path, limit):
    out, cur = [], []
    for line in open(path, encoding="utf-8").read().split("\n"):
        if line.strip():
            cur.append(line)
        elif cur:
            out.append("\n".join(cur))
            cur = []
    if cur:
        out.append("\n".join(cur))
    return out[:limit]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("backend")
    ap.add_argument("mode", choices=["set", "doc", "par"])
    ap.add_argument("file")
    ap.add_argument("n", nargs="?", type=int, default=8)
    ap.add_argument("--to", default="en")
    ap.add_argument("--chains", type=int, default=1)
    ap.add_argument("--max-paragraphs", type=int, default=40)
    ap.add_argument("--history-chars", type=int, default=4000)
    ap.add_argument("--out")
    a = ap.parse_args()

    if a.mode in ("set", "par"):
        items = [l.split("\t", 1) for l in open(a.file, encoding="utf-8").read().splitlines() if l.strip()]
    else:
        paras = paragraphs(a.file, a.max_paragraphs)
    translate(a.backend, "こんにちは。", "en")  # warm-up / load the model
    results = {}
    t0 = time.time()
    if a.mode == "set":
        for i, (lang, text) in enumerate(items):
            results[i] = translate(a.backend, text, lang)
    elif a.mode == "par":
        picked = [items[i % len(items)] for i in range(a.n)]

        def run(i):
            results[i] = translate(a.backend, picked[i][1], picked[i][0])

        th = [threading.Thread(target=run, args=(i,)) for i in range(a.n)]
        [t.start() for t in th]
        [t.join() for t in th]
    else:
        n, k = len(paras), max(1, min(a.chains, len(paras)))
        chains, at = [], 0
        for c in range(k):
            ln = n // k + (1 if c < n % k else 0)
            chains.append(list(range(at, at + ln)))
            at += ln

        def run(segs):
            w, done = Window(a.history_chars), []
            for i in segs:
                hist = w.select([p for p in done if p[0].strip() and p[1].strip()])
                results[i] = translate(a.backend, paras[i], a.to, hist)
                done.append((paras[i], results[i][0]))

        th = [threading.Thread(target=run, args=(c,)) for c in chains]
        [t.start() for t in th]
        [t.join() for t in th]
    wall = time.time() - t0
    rs = [results[i] for i in sorted(results)]
    gen = sum(r[2] for r in rs)
    prompt = sum(r[1] for r in rs)
    decode = sum(r[4] - r[3] for r in rs)
    print(
        f"{a.backend} {a.mode}: {len(rs)} requests, wall {wall:.2f} s, {gen} tokens -> {gen / wall:.1f} tok/s end to end; "
        f"decode {gen / max(decode, 1e-9):.1f} tok/s (per request, summed), mean prefill {1e3 * sum(r[3] for r in rs) / len(rs):.0f} ms, "
        f"prompt tokens {prompt}"
    )
    if a.out:
        with open(a.out, "w", encoding="utf-8") as f:
            for r in rs:
                f.write(r[0].replace("\\", "\\\\").replace("\n", "\\n") + "\n")


if __name__ == "__main__":
    main()
