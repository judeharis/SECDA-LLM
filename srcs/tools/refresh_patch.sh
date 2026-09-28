#!/usr/bin/env bash
# Regenerate a tool patch from an edited copy of the pinned llama.cpp's file.
#   srcs/tools/refresh_patch.sh <name> <edited file>
#   name: test-backend-ops | llama-bench | cli
# The edited file must start from the CURRENT upstream file: after a llama.cpp
# bump, first apply the old patch to the new upstream (patch --merge, or by
# hand), fix it up, then run this. Never feed it a patched copy from an older
# build: the diff would silently revert upstream's own changes.
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd); root=$(cd "$here/../.." && pwd)
case "${1:-}" in
  test-backend-ops) rel=tests/test-backend-ops.cpp ;;
  llama-bench)      rel=tools/llama-bench/llama-bench.cpp ;;
  cli)              rel=tools/cli/cli.cpp ;;
  *) echo "usage: $0 <test-backend-ops|llama-bench|cli> <edited file>" >&2; exit 1 ;;
esac
edited=${2:?edited file}
[ -f "$edited" ] || { echo "no such file: $edited" >&2; exit 1; }
tmp=$(mktemp); trap 'rm -f "$tmp"' EXIT
rc=0; diff -u --label "a/$rel" --label "b/$rel" "$root/llama.cpp/$rel" "$edited" > "$tmp" || rc=$?
[ $rc -eq 1 ] || { echo "diff failed or no changes (rc=$rc); patch left unchanged" >&2; exit 1; }
mv "$tmp" "$here/patches/$1.patch"; trap - EXIT
echo "wrote $here/patches/$1.patch"
