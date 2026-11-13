#!/bin/sh
set -eu
cd "$(dirname "$0")/../.."
RUSTFLAGS='' cargo build --release --target wasm32-unknown-unknown -p yaxus-web
wasm-bindgen --target web --out-dir web/local/pkg target/wasm32-unknown-unknown/release/yaxus_web.wasm
