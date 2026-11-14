#!/bin/sh
# Builds the browser front end into web/local/pkg.
#
#   ./web/local/build.sh          # release wasm + wasm-bindgen
#
# Needs the wasm32-unknown-unknown target and a wasm-bindgen CLI whose version matches
# the crate (0.2.130):
#   rustup target add wasm32-unknown-unknown
#   cargo install wasm-bindgen-cli --version 0.2.130
set -eu

cd "$(dirname "$0")/../.."

# SIMD128 helps the tokenizer and Q6_K repacking; every current Chrome has it.
# Thin LTO is a few percent on the host-side parsing and quantization.
RUSTFLAGS="${RUSTFLAGS:--C target-feature=+simd128}" \
CARGO_PROFILE_RELEASE_LTO=thin \
  cargo build -p yaxus-web --target wasm32-unknown-unknown --release "$@"

wasm-bindgen --target web --out-dir web/local/pkg \
  target/wasm32-unknown-unknown/release/yaxus_web.wasm

# Optional size/speed pass (Binaryen).
if command -v wasm-opt >/dev/null 2>&1; then
  wasm-opt -O3 -o web/local/pkg/yaxus_web_bg.wasm web/local/pkg/yaxus_web_bg.wasm
fi

echo "built web/local/pkg"
