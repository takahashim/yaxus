"""Generate tokenizer golden data from the reference Python tokenizer.

Usage:
    python gen_tokenizer.py MODEL_DIR > tokenizer.jsonl

Inputs: hand-written edge cases, random strings, and the public-domain excerpts in
corpus/ (see corpus/SOURCES.md). Requires transformers, numba, numpy and torch (the
remote tokenizer is only loaded when torch is installed). Each output line is one of
    {"kind": "encode", "text": str, "ids": [int], "decoded": str}
    {"kind": "prompt", "request": {...}, "prompt": str, "ids": [int]}
    {"kind": "prompt_error", "request": {...}}
"""

import json
import random
import sys
from pathlib import Path

from transformers import AutoTokenizer

model_dir = Path(sys.argv[1])
tok = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)

HANDCRAFTED = [
    "",
    " ",
    "  ",
    "   ",
    "a",
    "Hello, world!",
    " Hello  world   again    and     more",
    "会議は来週の月曜日に変更されました。",
    "吾輩は猫である。名前はまだ無い。",
    "ｶﾀｶﾅ ﾃｽﾄ　全角スペース　",
    "Ünïcödé façade naïve coöperate",
    "Привет, мир! Как дела?",
    "你好，世界。我们明天见。",
    "안녕하세요 세계",
    "مرحبا بالعالم",
    "नमस्ते दुनिया",
    "สวัสดีชาวโลก",
    "😀😃😄 👨‍👩‍👧‍👦 🇯🇵 👍🏽",
    "aaaa",
    "aaaaaaaaaaaaaaaa",
    "ーーーー",
    "!!!???....",
    "\n\n\n\n\n",
    "line1\nline2\r\nline3\ttab",
    "\x00\x01\x1c\x1d\x7f",
    "<|plamo:msg|>",
    "<|plamo:tag|>input<|plamo:msg|>hi<|plamo:tag|>output",
    "<|plamo:unknown_token|>",
    "<|plamo:has space|>",
    "<|plamo:" + "x" * 64 + "|>",
    "<|plamo:" + "x" * 65 + "|>",
    "<|plamo:<|plamo:msg|>",
    "<|plamo:bos|><|plamo:bos|>",
    "<|plamo:reserved:0x106|>",
    "<|plamo:begin_foo:plamo|>",
    "boundary",
    "\U000F0000\U0010FFFF￿",
    "def f(x):\n    return x  # comment\n",
    "https://example.com/path?q=1&r=2#frag",
    "1234567890 3.14159 1,000,000",
]

POOLS = [
    "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ",
    "      ",
    "\n\t\r",
    "あいうえおかきくけこさしすせそ日本語翻訳会議月曜日ー。、",
    "漢字中文简体繁體",
    "😀👍🏽🇯🇵‍️",
    "<|>plamo:msgtag",
    "　\x1c\x85 ́",
    "0123456789.,;:!?()[]{}'\"",
]


def random_text(rng: random.Random) -> str:
    n = rng.randint(1, 60)
    pools = rng.sample(POOLS, rng.randint(1, 3))
    out = []
    while len(out) < n:
        c = rng.choice(rng.choice(pools))
        out.append(c * (rng.randint(1, 6) if rng.random() < 0.15 else 1))
    return "".join(out)


def corpus() -> list[str]:
    """Public-domain excerpts in corpus/ (see corpus/SOURCES.md), one string per file."""
    root = Path(__file__).resolve().parent / "corpus"
    return [p.read_text(encoding="utf-8") for p in sorted(root.glob("*.txt"))]


def emit(obj):
    print(json.dumps(obj, ensure_ascii=False))


def encode_case(text: str):
    ids = tok.encode(text, add_special_tokens=False)
    emit({"kind": "encode", "text": text, "ids": ids, "decoded": tok.decode(ids)})


rng = random.Random(1234)
for t in HANDCRAFTED:
    encode_case(t)
for text in corpus():
    # Every paragraph, every line, and the whole excerpt as one long input.
    for para in text.split("\n\n"):
        encode_case(para)
        for line in para.split("\n"):
            if line.strip() and line != para:
                encode_case(line)
    encode_case(text)
for _ in range(1000):
    encode_case(random_text(rng))

PROMPTS = [
    {"text": "会議は来週の月曜日に変更されました。", "target_lang": "en"},
    {"text": "The meeting was moved to next Monday.", "target_lang": "ja", "source_lang": "en", "form": "polite"},
    {"text": "The meeting was moved.", "target_lang": "ja", "form": "plain", "context": "社内連絡メール"},
    {"text": "Hello", "target_lang": "ja(easy)"},
    {"text": "Hello", "target_lang": "zh-hans", "source_lang": "ja(easy)"},
    {
        "text": "The owner approved it.",
        "target_lang": "ja",
        "glossary": [{"source": "owner", "target": "オーナー"}, {"source": "approve", "target": "承認"}],
        "history": [{"source": "First line.", "target": "最初の行。"}, {"source": "Second.", "target": "二番目。"}],
        "source_lang": "en",
    },
    {"text": "x", "target_lang": "en", "context": ""},
    # Errors
    {"text": "   ", "target_lang": "en"},
    {"text": "<|plamo:msg|>", "target_lang": "en"},
    {"text": "x", "target_lang": "EN"},
    {"text": "x", "target_lang": "en-us"},
    {"text": "x", "target_lang": "en", "form": "polite"},
    {"text": "x", "target_lang": "ja", "form": "casual"},
    {"text": "x", "target_lang": "ja", "glossary": [{"source": "", "target": "y"}]},
    # Added later (appended so that the earlier lines stay as they were).
    {"text": "x", "target_lang": "en", "context": "   "},
    {"text": "x", "target_lang": "ja-easy", "form": "polite"},
    {"text": "x", "target_lang": "en", "context": "<|plamo:msg|>"},
    {"text": "x", "target_lang": "en", "history": [{"source": "<|plamo:tag|>", "target": "y"}]},
    {"text": "x", "target_lang": "en", "glossary": [{"source": "a", "target": "<|plamo:val|>"}]},
]

for req in PROMPTS:
    messages = []
    for h in req.get("history", []):
        messages.append({"role": "user", "content": h["source"]})
        messages.append({"role": "assistant", "content": h["target"]})
    messages.append({"role": "user", "content": req["text"]})
    kwargs = {k: v for k, v in req.items() if k not in ("text", "history")}
    try:
        prompt = tok.apply_chat_template(messages, add_generation_prompt=True, tokenize=False, **kwargs)
    except Exception:
        emit({"kind": "prompt_error", "request": req})
        continue
    ids = tok.apply_chat_template(messages, add_generation_prompt=True, tokenize=True, **kwargs)
    if isinstance(ids, dict) or hasattr(ids, "keys"):
        ids = ids["input_ids"]
    emit({"kind": "prompt", "request": req, "prompt": prompt, "ids": list(ids)})
