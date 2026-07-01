#!/bin/bash
set -e

# pushd /mnt/Crucial/WorkspaceB/LLMs/update_lpp_new/llama.cpp/
pushd ../llama.cpp/

compile=0
send=0
run=0

usage() {
  cat <<EOF
Usage: $0 [-a <board_addr>] [-p <port>] [-d <board_path>] [-s <board_sub>]

Options:
  -a, --board-addr   Remote SSH target, e.g. ubuntu@example.com
  -p, --port         Remote SSH port, e.g. 2222
  -d, --board-path   Remote board workspace path, e.g. /home/ubuntu/Workspace/secda_llm/
  -s, --board-sub    Remote board subdirectory, e.g. test_inf
  -h, --help         Show this help
EOF
}


board_addr="ubuntu@jharis.ddns.net"
port=2205
board_path="/home/ubuntu/Workspace/secda_llm/"
board_sub="test_inf"

while [[ $# -gt 0 ]]; do
  case "$1" in
  -a | --board-addr)
    if [[ -z "${2:-}" || "${2}" == -* ]]; then
      echo "Error: --board-addr requires a value."
      usage
      exit 1
    fi
    board_addr="$2"
    shift 2
    ;;
  -p | --port)
    if [[ -z "${2:-}" || "${2}" == -* ]]; then
      echo "Error: --port requires a value."
      usage
      exit 1
    fi
    port="$2"
    shift 2
    ;;
  -d | --board-path)
    if [[ -z "${2:-}" || "${2}" == -* ]]; then
      echo "Error: --board-path requires a value."
      usage
      exit 1
    fi
    board_path="$2"
    shift 2
    ;;
  -s | --board-sub)
    if [[ -z "${2:-}" || "${2}" == -* ]]; then
      echo "Error: --board-sub requires a value."
      usage
      exit 1
    fi
    board_sub="$2"
    shift 2
    ;;
  -h | --help)
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

if [[ -z "${board_addr}" || -z "${port}" || -z "${board_path}" || -z "${board_sub}" ]]; then
  echo "Error: board address, port, board path, and board subdirectory must all be set."
  usage
  exit 1
fi

if ! [[ "${port}" =~ ^[0-9]+$ ]]; then
  echo "Error: --port must be a number."
  exit 1
fi

echo "[compile_send_kria_bfpp] Target board: ${board_addr}, port: ${port}, path: ${board_path}, subdir: ${board_sub}"

#================================================================================================
workspace_path="/mnt/Crucial/WorkspaceB/SECDA/SECDA-LLM/llama.cpp/"
echo "[compile_send_kria_bfpp] Workspace path: ${workspace_path}"
echo "[compile_send_kria_bfpp] Board path: ${board_path}"
echo "[compile_send_kria_bfpp] Board subdir: ${board_sub}"
echo "[compile_send_kria_bfpp] Board address: ${board_addr}"
echo "[compile_send_kria_bfpp] Board port: ${port}"

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

  cmake --fresh "-DCMAKE_INSTALL_PREFIX=${workspace_path}out/install/Kria_SECDA/${remote_subdir}/" ${cmake_flags} -DCMAKE_SYSTEM_PROCESSOR=aarch64 -DCMAKE_SYSTEM_NAME=Linux -DCMAKE_C_COMPILER=/usr/bin/aarch64-linux-gnu-gcc -DCMAKE_CXX_COMPILER=/usr/bin/aarch64-linux-gnu-g++ -DCMAKE_BUILD_TYPE=Release ${preload_flag} -DBUILD_ARM=ON -DBUILD_KRIA=ON -DGGML_SECDA=ON -S${workspace_path} "-B${workspace_path}out/build/Kria_SECDA/${remote_subdir}"
  cmake --build "${workspace_path}out/build/Kria_SECDA/${remote_subdir}" --parallel 26 --target llama-cli --
  ssh -p "${port}" "${board_addr}" "mkdir -p /${board_path}/$board_sub/${remote_subdir}"
  rsync -r -avz -e "ssh -p ${port}" "${workspace_path}out/build/Kria_SECDA/${remote_subdir}/bin/llama-cli" "${board_addr}:/${board_path}/$board_sub/${remote_subdir}/${remote_exe_name}"
  rsync -r -avz -e "ssh -p ${port}" "${workspace_path}out/build/Kria_SECDA/${remote_subdir}/bin/" "${board_addr}:/${board_path}/$board_sub/${remote_subdir}/"
}

# CPU build & send (converted to function)
do_build_and_send_cpu() {
  local cmake_flags="$1"
  local remote_subdir="$2"
  local remote_exe_name="$3"

  cmake --fresh "-DCMAKE_INSTALL_PREFIX=${workspace_path}out/install/Kria_CPU/${remote_subdir}/" ${cmake_flags} -DCMAKE_SYSTEM_PROCESSOR=aarch64 -DCMAKE_SYSTEM_NAME=Linux -DCMAKE_C_COMPILER=/usr/bin/aarch64-linux-gnu-gcc -DCMAKE_CXX_COMPILER=/usr/bin/aarch64-linux-gnu-g++ -DCMAKE_BUILD_TYPE=Release -DBUILD_ARM=ON -DBUILD_KRIA=ON -S${workspace_path} "-B${workspace_path}out/build/Kria_CPU/${remote_subdir}"
  cmake --build "${workspace_path}out/build/Kria_CPU/${remote_subdir}" --parallel 26 --target llama-cli --
  ssh -p "${port}" "${board_addr}" "mkdir -p /${board_path}/$board_sub/${remote_subdir}"
  rsync -r -avz -e "ssh -p ${port}" "${workspace_path}out/build/Kria_CPU/${remote_subdir}/bin/llama-cli" "${board_addr}:/${board_path}/$board_sub/${remote_subdir}/${remote_exe_name}"
  rsync -r -avz -e "ssh -p ${port}" "${workspace_path}out/build/Kria_CPU/${remote_subdir}/bin/" "${board_addr}:/${board_path}/$board_sub/${remote_subdir}/"
}


source ../benchmark/configs/exp_configs.sh

idx=0
for runtime in "${runtimes_array[@]}"; do
  cmake_flags="${bin_flags_array[$idx]}"
  bin_dir="${bins_array[$idx]}"
  bin_name="${binaries_array[$idx]}"
  echo "Building and sending runtime ${runtime} with cmake flags: ${cmake_flags}, bin dir: ${bin_dir}, bin name: ${bin_name}"
  idx=$((idx + 1))
  if [ "$bin_dir" == "bin_cpu" ]; then
    do_build_and_send_cpu "${cmake_flags}" "${bin_dir}" "${bin_name}"
    # do_build_and_send_cpu_profiled "${cmake_flags}" "${bin_dir}" "${bin_name}"
    continue
  fi
  do_build_and_send "${cmake_flags}" "${bin_dir}" "${bin_name}" "-DACC_PRELOAD=OFF"
  # do_build_and_send_profiled "${cmake_flags}" "${bin_dir}" "${bin_name}" "-DACC_PRELOAD=OFF"
done

# Invoke CPU build/send
# cpu_flags="-DNOPERF=ON -DGGML_SECDA=OFF"
# do_build_and_send_cpu "${cpu_flags}" "bin_cpu" "llama_cli_v1_cpu"

#================================================================================================

popd
