# Test corpus sources

Excerpts (the opening ~5000 characters, whole paragraphs) of public-domain works, used as
real-world text for the tokenizer golden tests (`../gen_tokenizer.py`). They are made by
`../extract_corpus.py` from the files below.

All authors died more than 100 years ago, so the works are in the public domain in Japan,
the United States and the EU. Project Gutenberg's headers, footers and license text are
not included (only the text between its START and END markers is used). Aozora Bunko
texts follow its modern-orthography base editions; ruby annotations are removed and the
two JIS X 0213 characters that Aozora shows as images in `ja_rashomon.txt` are written as
the Unicode characters 扭 and 眶.

| File | Work | Author (died) | Source |
|---|---|---|---|
| `ja_rashomon.txt` | 羅生門 (1915) | 芥川龍之介 (1927) | 青空文庫 https://www.aozora.gr.jp/cards/000879/files/127_15260.html (底本: 「芥川龍之介全集1」ちくま文庫、筑摩書房 1986) |
| `ja_wagahai.txt` | 吾輩は猫である (1905) | 夏目漱石 (1916) | 青空文庫 https://www.aozora.gr.jp/cards/000148/files/789_14547.html (底本: 「夏目漱石全集1」ちくま文庫、筑摩書房 1987) |
| `en_alice.txt` | Alice's Adventures in Wonderland (1865) | Lewis Carroll (1898) | Project Gutenberg eBook #11, https://www.gutenberg.org/ebooks/11 |
| `de_faust.txt` | Faust: Der Tragödie erster Teil (1808) | Johann Wolfgang von Goethe (1832) | Project Gutenberg eBook #2229, https://www.gutenberg.org/ebooks/2229 |
| `fr_miserables.txt` | Les Misérables, Tome I: Fantine (1862) | Victor Hugo (1885) | Project Gutenberg eBook #17489, https://www.gutenberg.org/ebooks/17489 |
| `zh_lunyu.txt` | 論語 | 孔子 (479 BC) and disciples | Project Gutenberg eBook #23839, https://www.gutenberg.org/ebooks/23839 |
| `ru_derzhavin.txt` | Духовные оды | Гавриил Державин (1816) | Project Gutenberg eBook #14741, https://www.gutenberg.org/ebooks/14741 |
