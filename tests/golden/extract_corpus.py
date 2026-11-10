"""Make the test corpus excerpts in corpus/ from downloaded public-domain texts.

Usage:
    python extract_corpus.py DOWNLOAD_DIR corpus/

DOWNLOAD_DIR must contain (see corpus/SOURCES.md):
    rashomon.html  https://www.aozora.gr.jp/cards/000879/files/127_15260.html
    neko.html      https://www.aozora.gr.jp/cards/000148/files/789_14547.html
    alice.txt      https://www.gutenberg.org/cache/epub/11/pg11.txt
    faust.txt      https://www.gutenberg.org/cache/epub/2229/pg2229.txt
    miserables.txt https://www.gutenberg.org/cache/epub/17489/pg17489.txt
    pg23839.txt    https://www.gutenberg.org/cache/epub/23839/pg23839.txt
    pg14741.txt    https://www.gutenberg.org/cache/epub/14741/pg14741.txt
"""

import html, re, sys
from pathlib import Path

src, out = Path(sys.argv[1]), Path(sys.argv[2])
MAX = 5000

def gutenberg(name, anchor, nth):
    t = (src / name).read_text(encoding="utf-8").replace("\r", "")
    body = t.split("*** START OF", 1)[1].split("\n", 1)[1].split("*** END OF", 1)[0]
    i = -1
    for _ in range(nth):
        i = body.index(anchor, i + 1)
    return body[i:]

def aozora(name):
    t = (src / name).read_bytes().decode("shift_jis", errors="strict")
    m = re.search(r'<div class="main_text">(.*?)<div class="bibliographical_information">', t, re.S)
    body = m.group(1)
    body = re.sub(r"<rp>.*?</rp>|<rt>.*?</rt>", "", body, flags=re.S)
    # Aozora gaiji (JIS X 0213 characters shown as images): the Unicode character for
    # each one found, identified from its description in the alt text.
    gaiji = {"第4水準2-12-93": "扭", "第3水準1-88-81": "眶"}  # てへん+丑, 目+匡
    def gaiji_char(m):
        alt = m.group(1)
        u = re.search(r"U\+([0-9A-Fa-f]{4,5})", alt)
        if u:
            return chr(int(u.group(1), 16))
        code = re.search(r"第\d水準[\d-]+", alt)
        if code and code.group(0) in gaiji:
            return gaiji[code.group(0)]
        # Unknown ones are kept as a marker and rejected only if they end up in the excerpt.
        return "\x00GAIJI " + alt + "\x00"
    body = re.sub(r'<img[^>]*alt="([^"]*)"[^>]*>', gaiji_char, body)
    body = re.sub(r"<br\s*/?>", "\n", body)
    body = re.sub(r"<[^>]+>", "", body)
    body = html.unescape(body)
    # One paragraph per line in Aozora texts; separate them with blank lines.
    lines = [l.strip() for l in body.split("\n")]
    return "\n\n".join(l for l in lines if l)

def excerpt(text):
    paras = [p.strip("\n") for p in re.split(r"\n\s*\n", text) if p.strip()]
    out, n = [], 0
    for p in paras:
        if n + len(p) > MAX and out:
            break
        out.append(p)
        n += len(p)
    return "\n\n".join(out) + "\n"

works = {
    "ja_rashomon.txt": aozora("rashomon.html"),
    "ja_wagahai.txt": aozora("neko.html"),
    "en_alice.txt": gutenberg("alice.txt", "CHAPTER I.", 2),
    "de_faust.txt": gutenberg("faust.txt", "Zueignung", 2),
    "fr_miserables.txt": gutenberg("miserables.txt", "En 1815, M. Charles-François-Bienvenu Myriel", 1),
    "zh_lunyu.txt": gutenberg("pg23839.txt", "學而第一", 1),
    "ru_derzhavin.txt": gutenberg("pg14741.txt", "МОЛИТВА", 1),
}
for name, text in works.items():
    e = excerpt(text)
    if "\x00GAIJI" in e:
        raise SystemExit(f"{name}: unresolved gaiji in the excerpt: {re.findall(chr(0) + 'GAIJI ([^' + chr(0) + ']*)', e)}")
    (out / name).write_text(e, encoding="utf-8")
    print(f"{name}: {len(e)} chars, {e.count(chr(10)+chr(10))+1} paragraphs | {e[:40]!r}")
