# yaxus

**Built with PLaMo**

yaxus runs [PLaMo 3 Translate](https://huggingface.co/pfnet/plamo-3-translate-2b) as a
translation server and command-line tool, written in Rust. It does not use llama.cpp or
Python. It has its own model code, tokenizer and GPU kernels (WGSL on Vulkan through wgpu)
in one process.

- **Server:** `yaxus serve` serves HTTPS with a bearer token. It provides a JSON/SSE API
  and a web UI, and other machines connect through `yaxus client` or a browser.
- **Local translation:** `yaxus translate` translates on the same machine without a server.
- **Long documents:** documents are split into chains of paragraphs. A chain is translated
  in order with its earlier translations as history, and chains run in parallel.
- **Models:** the 0.8B, 2B and 8B models, as Hugging Face checkpoints (safetensors) or GGUF
  files. Q8_0, Q4_0, Q6_K, Q4_K, Q3_K and Q2_K tensors keep their bit widths on the GPU;
  other types are converted on load.
- **Speculative decoding:** the 0.8B model drafts by default. The output is the same as
  without a draft, only faster.
- **Many requests at once:** the KV cache is paged and shared between requests with a
  common prefix. The engine batches up to 16 requests in one pass and caches responses.
- **CPU backend:** a CPU backend serves as the reference the GPU kernels are tested against.

The model weights and tokenizer are not part of this repository. They are distributed by
Preferred Networks under the PLaMo Community License (see [License](#license)).

This is a personal project. It is developed and measured on one machine: Ubuntu 24.04 with
a Radeon 780M (RDNA3 integrated GPU) and Mesa RADV. Other GPUs should work, but have not
been tested. The design notes and every measurement are in [DESIGN.md](DESIGN.md)
(in Japanese).

## Requirements

- **Rust:** the current stable toolchain (edition 2024).
- **GPU:** a GPU with a Vulkan driver. The fast decode kernels need 32-lane subgroups, and
  integer dot products help the most. On Linux the user must be in the `render` group.
  Without a GPU, `--backend cpu` works, but slowly.
- **Model:** a model in the Hugging Face cache. For example, with the
  [`hf` CLI](https://huggingface.co/docs/huggingface_hub/guides/cli):

  ```sh
  hf download pfnet/plamo-3-translate-2b                                  # default model
  hf download pfnet/plamo-3-translate-0.8b-GGUF --include "*Q2_K.gguf"    # draft model
  hf download pfnet/plamo-3-translate-2b-GGUF --include "*Q4_K_M.gguf"    # for --model 2b:Q4_K_M
  ```

## Install

```sh
cargo install --path crates/yaxus-cli
```

`.cargo/config.toml` builds for the CPU of the machine that runs the build
(`target-cpu=native`). For a binary that runs on other machines, build with
`RUSTFLAGS=""`.

## Usage

Translate locally:

```sh
yaxus translate --to en "吾輩は猫である。名前はまだ無い。"
yaxus translate --to ja --form polite < document.txt      # paragraphs separated by blank lines
yaxus translate --model 2b:Q4_K_M --to en < document.txt   # a GGUF file from the HF cache
```

Run a server:

```sh
yaxus serve                     # listens on 0.0.0.0:8443
yaxus serve --san host.example  # add a name to the server certificate
```

The first start creates an access token and a local CA under `~/.config/yaxus/`. The
server certificate is signed by that CA. Copy `ca.pem` to the client machines, or import
it into a browser to avoid certificate warnings.

On another machine:

```sh
export YAXUS_SERVER=https://host.example:8443 YAXUS_TOKEN=... YAXUS_CA=~/yaxus-ca.pem
yaxus client --to en < document.txt
yaxus client --to ja -g "owner=オーナー" "The owner approved it."
```

In a browser, open `https://host.example:8443/` and enter the token. The API is
`POST /v1/translate`, `POST /v1/translate/document` and `GET /v1/languages`
(see `crates/yaxus-api`).

### Speed and precision

By default (`--prefill-precision auto`, `--decode-precision auto`: the device's choice,
`int16` on the AMD GPUs measured so far), matrix products round their input rows to
15 bits per block of 32 values and use packed integer dot products. The logits differ from an all-f32 computation by about 6e-3. In the tests so far,
the translations were identical.

- **Reference:** `--prefill-precision f32 --decode-precision f32 --draft none
  --no-kv-cache` computes everything in f32.
- **Faster prefill:** `--prefill-precision int8` rounds as llama.cpp does. It is 1.5 times
  faster at prefill, but changes some translations of difficult text.

`yaxus bench` measures pass times, sentence sets and documents.

On the Radeon 780M, against Ollama v0.40.16 with the same Q4_K_M GGUF files (seconds,
lower is better):

| | 2B: yaxus | 2B: Ollama | 8B: yaxus | 8B: Ollama |
|---|---|---|---|---|
| 19 sentences | 6.10 | 9.91 | 12.99 | 28.5 |
| Document, 14 paragraphs (in order) | 34.2 | 39.9 | 76.0 | 113.2 |
| Document, 40 paragraphs (8 chains) | 56.7 | 70.5 | 152.3 | 176.4 |
| 16 concurrent requests | 1.54 | 4.01 | 4.37 | 10.79 |

A single stream without a draft model decodes at about the same speed as Ollama
(2B: 44.3 against 45.3 tokens/s), because both are bound by memory bandwidth.

## Development

- **Default tests:** `cargo test --workspace`. Tests that need model files are ignored, or
  skip themselves when the files are missing.
- **GPU and model tests:** `cargo test --release -p yaxus-backend-wgpu -- --ignored`.
  These need a GPU and the models in the Hugging Face cache.
- **Software Vulkan:** the kernel tests also run on lavapipe, as in CI:
  `YAXUS_SOFTWARE_GPU=1 VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/lvp_icd.json`.
- **CI:** runs `cargo fmt --check` and `cargo clippy` with warnings as errors, and the
  tests ([.github/workflows/ci.yml](.github/workflows/ci.yml)).

## License

The code in this repository is licensed under either of

- Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE))
- MIT license ([LICENSE-MIT](LICENSE-MIT))

at your option.

The PLaMo models are not included, and these licenses do not cover them.

- **Model license:** PLaMo 3 Translate (weights, tokenizer and model code) is provided by
  Preferred Networks, Inc. under the PLaMo Community License
  ([English](https://plamo.preferredai.jp/info/plamo-community-license-en),
  [日本語](https://plamo.preferredai.jp/info/plamo-community-license-ja)). Using a model
  through yaxus means agreeing to that license.
- **Commercial use:** commercial use of the models requires registration with Preferred
  Networks, and in some cases a separate agreement.
- **Affiliation:** yaxus is an independent implementation, not affiliated with or endorsed
  by Preferred Networks.

Some test data are outputs of PLaMo models, as described in
[tests/README.md](tests/README.md). The literary excerpts in `tests/golden/corpus/` are
public-domain works, listed in
[tests/golden/corpus/SOURCES.md](tests/golden/corpus/SOURCES.md).
