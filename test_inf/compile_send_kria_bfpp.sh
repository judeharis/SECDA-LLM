#!/bin/bash
set -e

# jq -r '.secda_framework_path + "@" + .kria_board_hostname' _secda_automation/config.json
# pushd /mnt/Crucial/WorkspaceB/LLMs/update_lpp_new/llama.cpp/
pushd ../

compile=0
send=0
run=0

usage() {
  cat <<EOF
Usage: $0 -a <board_addr> -p <port>

Options:
  -a, --board-addr   Remote SSH target, e.g. ubuntu@example.com
  -p, --port         Remote SSH port, e.g. 2222
  -h, --help         Show this help
EOF
}

board_addr="ubuntu@jharis.ddns.net"
port=2203

while [[ $# -gt 0 ]]; do
  case "$1" in
    -a|--board-addr)
      if [[ -z "${2:-}" || "${2}" == -* ]]; then
        echo "Error: --board-addr requires a value."
        usage
        exit 1
      fi
      board_addr="$2"
      shift 2
      ;;
    -p|--port)
      if [[ -z "${2:-}" || "${2}" == -* ]]; then
        echo "Error: --port requires a value."
        usage
        exit 1
      fi
      port="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1"
      usage
      exit 1
      ;;
  esac
done

if [[ -z "${board_addr}" || -z "${port}" ]]; then
  echo "Error: both --board-addr and --port are required."
  usage
  exit 1
fi

if ! [[ "${port}" =~ ^[0-9]+$ ]]; then
  echo "Error: --port must be a number."
  exit 1
fi

echo "[compile_send_kria_bfpp] Target board: ${board_addr}, port: ${port}"

#================================================================================================
workspace_path="/mnt/Crucial/WorkspaceB/LLMs/SECDA_LLM/llama.cpp/"

# Helper: configure, build, and send build artifacts to target board
# Args:
#   1) cmake_flags (e.g. "${allq}")
#   2) remote_subdir (e.g. "bin_pre", "bin_q2q3")
#   3) remote_exe_name (name to give the deployed llama-cli on target)
do_build_and_send() {
  local cmake_flags="$1"
  local remote_subdir="$2"
  local remote_exe_name="$3"

  local preload_flag="$4" # optional 4th arg for preload build

  cmake --fresh "-DCMAKE_INSTALL_PREFIX=${workspace_path}out/install/SECDA-aarch64-release" ${cmake_flags} -DCMAKE_SYSTEM_PROCESSOR=aarch64 -DCMAKE_SYSTEM_NAME=Linux -DCMAKE_C_COMPILER=/usr/bin/aarch64-linux-gnu-gcc -DCMAKE_CXX_COMPILER=/usr/bin/aarch64-linux-gnu-g++ -DCMAKE_BUILD_TYPE=Release ${preload_flag} -DBUILD_ARM=ON -DBUILD_KRIA=ON -DGGML_SECDA=ON -S${workspace_path} "-B${workspace_path}out/build/SECDA-aarch64-release"
  cmake --build "${workspace_path}out/build/SECDA-aarch64-release" --parallel 26 --target llama-cli --
  # cp "${workspace_path}out/build/SECDA-aarch64-release/src/libllama.so" "${workspace_path}out/build/SECDA-aarch64-release/bin/libllama.so"
  # cp "${workspace_path}out/build/SECDA-aarch64-release/ggml/src/libggml.so" "${workspace_path}out/build/SECDA-aarch64-release/bin/libggml.so"
  ssh -p "${port}" "${board_addr}" "mkdir -p /home/ubuntu/Workspace/secda_llm/bfpp/${remote_subdir}"
  rsync -r -avz -e "ssh -p ${port}" "${workspace_path}out/build/SECDA-aarch64-release/bin/llama-cli" "${board_addr}:/home/ubuntu/Workspace/secda_llm/bfpp/${remote_subdir}/${remote_exe_name}"
  rsync -r -avz -e "ssh -p ${port}" "${workspace_path}out/build/SECDA-aarch64-release/bin/" "${board_addr}:/home/ubuntu/Workspace/secda_llm/bfpp/${remote_subdir}/"
  # rsync -r -avz -e "ssh -p ${port}" "${workspace_path}out/build/SECDA-aarch64-release/bin/libggml.so" "${board_addr}:/home/ubuntu/Workspace/secda_llm/bfpp/${remote_subdir}/"
  # rsync -r -avz -e "ssh -p ${port}" "${workspace_path}out/build/SECDA-aarch64-release/bin/libllama.so" "${board_addr}:/home/ubuntu/Workspace/secda_llm/bfpp/${remote_subdir}/"
}

# CPU build & send (converted to function)
do_build_and_send_cpu() {
  local cmake_flags="$1"
  local remote_subdir="$2"
  local remote_exe_name="$3"

  cmake --fresh "-DCMAKE_INSTALL_PREFIX=${workspace_path}out/install/SECDA-aarch64-release" ${cmake_flags} -DCMAKE_SYSTEM_PROCESSOR=aarch64 -DCMAKE_SYSTEM_NAME=Linux -DCMAKE_C_COMPILER=/usr/bin/aarch64-linux-gnu-gcc -DCMAKE_CXX_COMPILER=/usr/bin/aarch64-linux-gnu-g++ -DCMAKE_BUILD_TYPE=Release -DNOPERF=ON -DBUILD_ARM=ON -DBUILD_KRIA=ON -S${workspace_path} "-B${workspace_path}out/build/SECDA-aarch64-release"
  cmake --build "${workspace_path}out/build/SECDA-aarch64-release" --parallel 26 --target llama-cli --
  # cp "${workspace_path}out/build/SECDA-aarch64-release/src/libllama.so" "${workspace_path}out/build/SECDA-aarch64-release/bin/libllama.so"
  # cp "${workspace_path}out/build/SECDA-aarch64-release/ggml/src/libggml.so" "${workspace_path}out/build/SECDA-aarch64-release/bin/libggml.so"
  ssh -p "${port}" "${board_addr}" "mkdir -p /home/ubuntu/Workspace/secda_llm/bfpp/${remote_subdir}"
  rsync -r -avz -e "ssh -p ${port}" "${workspace_path}out/build/SECDA-aarch64-release/bin/llama-cli" "${board_addr}:/home/ubuntu/Workspace/secda_llm/bfpp/${remote_subdir}/${remote_exe_name}"
  rsync -r -avz -e "ssh -p ${port}" "${workspace_path}out/build/SECDA-aarch64-release/bin/" "${board_addr}:/home/ubuntu/Workspace/secda_llm/bfpp/${remote_subdir}/"
  # rsync -r -avz -e "ssh -p ${port}" "${workspace_path}out/build/SECDA-aarch64-release/bin/libggml.so" "${board_addr}:/home/ubuntu/Workspace/secda_llm/bfpp/${remote_subdir}/"
  # rsync -r -avz -e "ssh -p ${port}" "${workspace_path}out/build/SECDA-aarch64-release/bin/libllama.so" "${board_addr}:/home/ubuntu/Workspace/secda_llm/bfpp/${remote_subdir}/"
}

source ./_secda_automation/scripts/exp_configs.sh

idx=0
for runtime in "${runtimes_array[@]}"; do
  cmake_flags="${bin_flags_array[$idx]}"
  bin_dir="${bins_array[$idx]}"
  bin_name="${binaries_array[$idx]}"
  echo "Building and sending runtime ${runtime} with cmake flags: ${cmake_flags}, bin dir: ${bin_dir}, bin name: ${bin_name}"
  idx=$((idx + 1))
  if [ "$bin_dir" == "bin_cpu" ]; then
    do_build_and_send_cpu "${cmake_flags}" "${bin_dir}" "${bin_name}"
    continue
  fi
  do_build_and_send "${cmake_flags}" "${bin_dir}" "${bin_name}" "-DACC_PRELOAD=ON"
done

# Invoke CPU build/send
# cpu_flags="-DNOPERF=ON -DGGML_SECDA=OFF"
# do_build_and_send_cpu "${cpu_flags}" "bin_cpu" "llama_cli_v1_cpu"

#================================================================================================

popd
