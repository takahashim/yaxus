"""Generate model golden data (last-token logits and greedy outputs) with HF transformers.

Usage:
    python gen_logits.py MODEL_DIR > logits_2b.jsonl

Runs the model in float32 on CPU (needs ~4x the bf16 size in RAM). Each line:
    {"name", "ids", "top_ids", "top_logits", "logits_sum", "logits_sq_sum", "greedy"}
"""

import json
import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_dir = sys.argv[1]
max_new = int(sys.argv[2]) if len(sys.argv) > 2 else 48
torch.manual_seed(0)
tok = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_dir, trust_remote_code=True, dtype=torch.float32)
model.eval()

CASES = [
    ("ja2en", [{"role": "user", "content": "会議は来週の月曜日に変更されました。"}], {"target_lang": "en"}),
    (
        "en2ja_polite",
        [{"role": "user", "content": "The meeting has been moved to next Monday. Please update your calendar."}],
        {"target_lang": "ja", "source_lang": "en", "form": "polite"},
    ),
    (
        "en2zh_glossary_history",
        [
            {"role": "user", "content": "The owner approved the plan."},
            {"role": "assistant", "content": "所有者批准了该计划。"},
            {"role": "user", "content": "The owner will review it again tomorrow."},
        ],
        {"target_lang": "zh-hans", "glossary": [{"source": "owner", "target": "负责人"}]},
    ),
]

with torch.no_grad():
    for name, messages, kwargs in CASES:
        ids = tok.apply_chat_template(messages, add_generation_prompt=True, tokenize=True, **kwargs)
        if hasattr(ids, "keys"):
            ids = ids["input_ids"]
        ids = list(ids)
        x = torch.tensor([ids])
        logits = model(x).logits[0, -1].float()
        top = torch.topk(logits, 20)
        out = model.generate(x, max_new_tokens=max_new, do_sample=False, eos_token_id=[1, 16])
        greedy = out[0, len(ids):].tolist()
        print(
            json.dumps(
                {
                    "name": name,
                    "ids": ids,
                    "top_ids": top.indices.tolist(),
                    "top_logits": top.values.tolist(),
                    "logits_sum": logits.sum().item(),
                    "logits_sq_sum": (logits * logits).sum().item(),
                    "greedy": greedy,
                    "greedy_text": tok.decode([t for t in greedy if t not in (1, 16)]),
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
