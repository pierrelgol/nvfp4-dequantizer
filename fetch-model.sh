#!/usr/bin/env bash
set -euo pipefail

URL="https://huggingface.co/surogate/gemma-3-270m-it-NVFP4/resolve/main/model.safetensors"
DESTINATION="${1:-models/gemma-3-270m-it-NVFP4/model.safetensors}"

command -v curl >/dev/null || {
    echo "error: curl is required" >&2
    exit 1
}

mkdir -p "$(dirname "$DESTINATION")"
curl --fail --location --retry 3 --continue-at - \
    --output "$DESTINATION" \
    "$URL"
