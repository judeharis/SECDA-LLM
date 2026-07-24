#!/bin/sh
# Download and unpack Wikitext-2 into SECDA-LLM/perplexity.

set -eu

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
ZIP="wikitext-2-raw-v1.zip"
DATA_DIR="wikitext-2-raw"
FILE="$DATA_DIR/wiki.test.raw"
URL="https://huggingface.co/datasets/ggml-org/ci/resolve/main/$ZIP"

have_cmd() {
    for cmd in "$@"; do
        command -v "$cmd" >/dev/null 2>&1 || return 1
    done
}

die() {
    printf "%s\n" "$*" >&2
    exit 1
}

dl() {
    if [ -f "$2" ]; then
        return
    fi

    if have_cmd wget; then
        wget "$1" -O "$2"
    elif have_cmd curl; then
        curl -L "$1" -o "$2"
    else
        die "Please install wget or curl"
    fi
}

have_cmd unzip || die "Please install unzip"

cd "$SCRIPT_DIR"

if [ ! -f "$FILE" ]; then
    dl "$URL" "$ZIP"
    unzip -o "$ZIP"
    rm -f -- "$ZIP"
fi

printf "Wikitext-2 is ready at: %s/%s\n" "$SCRIPT_DIR" "$FILE"
printf "Use with llama-perplexity: -f %s/%s\n" "$SCRIPT_DIR" "$FILE"
