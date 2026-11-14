#!/bin/sh
# Serves the browser front end (ES modules and wasm need HTTP, not file://).
#   ./crates/yaxus-browser/www/serve.sh [port]
set -eu
cd "$(dirname "$0")"
exec python3 -m http.server "${1:-8000}"
