#!/bin/bash

# # Function to search and replace a string in a file
# replace_string_in_file() {
#   local file="$1"
#   local search="$2"
#   local replace="$3"
#   sed -i "s|${search}|${replace}|g" "$file"
# }



# vivado_2019_path=$(jq -r '.vivado_2019_path' ../config.json)
# vivado_2024_path=$(jq -r '.vivado_2024_path' ../config.json)

# # Remove the last occurrence of '2024.1/bin/' from vivado_2024_path if present
# vivado_2024_path="${vivado_2024_path%/2024.1/bin/}"
# # Remove the last occurrence of '2019.2/bin/' from vivado_2019_path if present
# vivado_2019_path="${vivado_2019_path%/2019.2/bin/}"


# # Example usage: search and replace in devcontainer.json
# replace_string_in_file "../.devcontainer/devcontainer.json" "VIVADO_2019_PATH" $vivado_2019_path
# replace_string_in_file "../.devcontainer/devcontainer.json" "VIVADO_2024_PATH" $vivado_2024_path


# Link the SECDA backend into llama.cpp's ggml/src: the fork's
# ggml_add_backend(SECDA) adds ggml/src/ggml-secda. The link is local to this
# checkout (the fork doesn't track it) and is kept out of the fork's git status,
# along with the run outputs the SECDA tools write. ggml-secda.h needs no link:
# the ggml-secda target exports its directory.
set -euo pipefail
cd "$(dirname "$0")"
top="$(git -C llama.cpp rev-parse --show-toplevel 2>/dev/null || true)"
if [ "$top" != "$(realpath llama.cpp)" ]; then
  echo "llama.cpp is not checked out: run 'git submodule update --init llama.cpp' first" >&2
  exit 1
fi
ln -sfn ../../../srcs/ggml_backend/ggml-secda ./llama.cpp/ggml/src/ggml-secda
rm -f ./llama.cpp/ggml/include/ggml-secda.h   # old header link, no longer used
excl="$(cd llama.cpp && realpath -m "$(git rev-parse --git-path info/exclude)")"
mkdir -p "$(dirname "$excl")"
for p in /ggml/src/ggml-secda /.vscode/ /results/ tbo.csv llama_perf.csv prf.csv /_gstats/ /_plans/; do
  grep -qxF "$p" "$excl" 2>/dev/null || echo "$p" >> "$excl"
done


# pushd ../tensorflow/tensorflow/lite/examples/
# ln -s -f ../../../../src/secda_apps/ ./ 
# popd
# pushd ../tensorflow/tensorflow/lite/delegates/utils/
# ln -s -f ../../../../../src/secda_delegates/ ./
# ln -s -f ../../../../../src/.clang-format ./
# popd