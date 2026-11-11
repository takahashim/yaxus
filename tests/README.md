# Test data

## Outputs of PLaMo models

Some files here were generated with PLaMo 3 Translate models by Preferred Networks,
Inc. They are outputs ("生成物") under the
[PLaMo Community License](https://plamo.preferredai.jp/info/plamo-community-license-en).

- `golden/tokenizer.jsonl`: token ids and decoded texts from the PLaMo tokenizer, made by
  `golden/gen_tokenizer.py`.
- `golden/logits_0_8b.jsonl`, `golden/logits_2b.jsonl`, `golden/logits_8b.jsonl`: top
  logits and greedy outputs of pfnet/plamo-3-translate-0.8b, -2b and -8b on a few prompts,
  made by `golden/gen_logits.py`.
- `bench/ref_default.txt`, `bench/ref_f32.txt`: translations of `bench/sentences.tsv` by
  pfnet/plamo-3-translate-2b through yaxus.

**Notice:** a model trained using these outputs is subject to the PLaMo Community License
(section 5).

## Other text

- **`golden/corpus/`:** excerpts of public-domain works, used as input text. The sources
  are in [golden/corpus/SOURCES.md](golden/corpus/SOURCES.md).
- **`bench/sentences.tsv`:** sentences written for this project. One of them is the
  opening of 『吾輩は猫である』, a public-domain work.
