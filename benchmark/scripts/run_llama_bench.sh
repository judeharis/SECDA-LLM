#!/bin/bash
set -eo pipefail

enable_power=false

BOARD_PATH="/home/ubuntu/Workspace/secda_llm"
BOARD_SUB="benchmark"
BENCHMARK_ROOT="${BOARD_PATH}/${BOARD_SUB}"
RESULTS_DIR_REL="../results"
MODEL_DIR="${BOARD_PATH}/models"
COMMANDS_FILE="${BENCHMARK_ROOT}/commands_bench.txt"

POWER_START_SCRIPT="${BENCHMARK_ROOT}/scripts/start_power_logging_KRIAv2.sh"
POWER_STOP_SCRIPT="${BENCHMARK_ROOT}/scripts/stop_power_logging_KRIAv2.sh"

LOAD_BITSTREAM_PY="${HOME}/load_bitstream.py"
BOARD_BITSTREAMS_DIR="${BOARD_PATH}/bitstreams"
HOST_BITSTREAMS_DIR="/home/ubuntu/bitstreams"
CLEAR_BITSTREAM_FILE="CPU_KRIA_1_0.bit"
DEFAULT_BITSTREAM_FILE="CPU_1_0.bit"

UDMABUF_GLOB="/dev/udmabuf*"
UDMABUF_PREFIX="/dev/udmabuf"
UDMABUF_MGR="/dev/u-dma-buf-mgr"
MEMINFO_PATH="/proc/meminfo"
DROP_CACHES_PATH="/proc/sys/vm/drop_caches"
TRACE_FILE="sds_trace_data.dat"

BENCH_N="${BENCH_N:-0}"
BENCH_P="${BENCH_P:-0}"
BENCH_B="${BENCH_B:-8}"
BENCH_PG="${BENCH_PG:-32,8}"
FLAGS="--no-warmup -r 1"

active_power_pid=""
active_power_pid_file=""

parse_args() {
  while [[ $# -gt 0 ]]; do
    case "$1" in
      --power)
        enable_power=true
        shift
        ;;
      *)
        echo "Unknown option: $1"
        exit 1
        ;;
    esac
  done
}

load_configs() {
  source ./exp_configs.sh

  models=("${models_array[@]}")
  model_names=("${model_names_array[@]}")
  bitstreams=("${bitstreams_array[@]}")
  binaries=("${binaries_array[@]}")
  acc_tag=("${acc_tags_array[@]}")
  bins=("${bins_array[@]}")
  accelerated=("${accelerated_array[@]}")
  threads=(1)
}

init_commands_log() {
  rm -f "${COMMANDS_FILE}"
  touch "${COMMANDS_FILE}"
}

stop_active_power_logger() {
  if [[ -n "${active_power_pid}" && -x "${POWER_STOP_SCRIPT}" ]]; then
    "${POWER_STOP_SCRIPT}" "${active_power_pid}" "${active_power_pid_file}" >/dev/null 2>&1 || true
  fi
  active_power_pid=""
  active_power_pid_file=""
}

clear_bitstream() {
  echo "-----------------------------------------------------------"
  echo "Clearing Bitstream"
  echo "-----------------------------------------------------------"
  python3 "${LOAD_BITSTREAM_PY}" -q "${BOARD_BITSTREAMS_DIR}/${CLEAR_BITSTREAM_FILE}"
}

cleanup_temp_files() {
  rm -f "${TRACE_FILE}"
}

error_exit() {
  local line_no="$1"
  local source_file="$2"
  stop_active_power_logger
  echo "error at line ${line_no} in ${source_file}"
  clear_bitstream
  cleanup_temp_files
  exit 1
}

clear_udmabuf_if_exists() {
  local idx="$1"
  if [[ -e "${UDMABUF_PREFIX}${idx}" ]]; then
    echo "delete udmabuf${idx}" >"${UDMABUF_MGR}"
    echo "Cleared ${UDMABUF_PREFIX}${idx}"
  fi
}

clear_all_udmabuf() {
  local dev idx
  for dev in ${UDMABUF_GLOB}; do
    [[ -e "$dev" ]] || continue
    idx="${dev#${UDMABUF_PREFIX}}"
    [[ -n "$idx" ]] || continue
    clear_udmabuf_if_exists "$idx"
  done
}

clear_udma() {
  echo "-----------------------------------------------------------"
  echo "Clear UDMA"
  echo "-----------------------------------------------------------"
  cat "${MEMINFO_PATH}" | grep -i cma
  clear_all_udmabuf
  cat "${MEMINFO_PATH}" | grep -i cma
}

drop_caches() {
  local sleep_secs="$1"
  sudo sh -c "/bin/echo 3 > ${DROP_CACHES_PATH}"
  sleep "${sleep_secs}"
}

start_power_logger() {
  local power_log_file="$1"
  local power_pid_file="$2"

  active_power_pid=""
  active_power_pid_file=""
  if [[ "${enable_power}" == true ]]; then
    if [[ -x "${POWER_START_SCRIPT}" && -x "${POWER_STOP_SCRIPT}" ]]; then
      "${POWER_START_SCRIPT}" 0.05 "${power_log_file}" "${power_pid_file}"
      active_power_pid=$(cat "${power_pid_file}" 2>/dev/null || true)
      active_power_pid_file="${power_pid_file}"
    else
      echo "Power logger scripts not found/executable. Skipping power logging."
    fi
  fi
}

check_cmd_status() {
  local status="$1"
  if [[ "${status}" -ne 0 ]]; then
    echo "llama-bench command failed with status ${status}"
    exit "${status}"
  fi
}

move_if_exists() {
  local src="$1"
  local dst="$2"
  [[ -f "${src}" ]] && mv -f "${src}" "${dst}"
}

run_single_binary() {
  local model="$1"
  local mn="$2"
  local thread="$3"
  local binary="$4"
  local acc="$5"
  local bin_folder="$6"
  local tag="$7"
  local bitstream="$8"

  local bench_binary="${binary}_bench"
  local result_base="bench_${mn}_${thread}_${tag}"
  local power_log_file="${RESULTS_DIR_REL}/${result_base}_power.txt"
  local power_pid_file="${RESULTS_DIR_REL}/${result_base}_power.pid"
  local result_txt="${RESULTS_DIR_REL}/${result_base}.txt"

  echo "========================================"
  echo "Running llama-bench for ${model}_${thread}_${tag}"

  echo "python3 ${LOAD_BITSTREAM_PY} ${BOARD_BITSTREAMS_DIR}/${bitstream}.bit" >>"${COMMANDS_FILE}"
  python3 "${LOAD_BITSTREAM_PY}" "${BOARD_BITSTREAMS_DIR}/${bitstream}.bit"
  drop_caches 3

  cd "${BENCHMARK_ROOT}"
  mkdir -p results
  cd "${BENCHMARK_ROOT}/${bin_folder}"

  if [[ ! -x "./${bench_binary}" ]]; then
    echo "Missing executable: ${BENCHMARK_ROOT}/${bin_folder}/${bench_binary}"
    exit 1
  fi

  echo "sudo env LD_LIBRARY_PATH=\${PWD}/bin:\${LD_LIBRARY_PATH:-} ./${bench_binary} -m ${MODEL_DIR}/${model} -b ${BENCH_B} -n ${BENCH_N} -t ${thread} -p ${BENCH_P} -pg ${BENCH_PG} ${FLAGS}" >>"${COMMANDS_FILE}"

  start_power_logger "${power_log_file}" "${power_pid_file}"

  local cmd_status=0
  LD_LIBRARY_PATH="${PWD}/bin:${LD_LIBRARY_PATH:-}" "./${bench_binary}" -m "${MODEL_DIR}/${model}" -t "${thread}" -b "${BENCH_B}" -n "${BENCH_N}" -p "${BENCH_P}" -pg "${BENCH_PG}" ${FLAGS} \
    2>&1 | tee "${result_txt}" || cmd_status=$?

  stop_active_power_logger
  check_cmd_status "${cmd_status}"

  move_if_exists "prf.csv" "${RESULTS_DIR_REL}/${result_base}_prf.csv"
  move_if_exists "llama_perf.csv" "${RESULTS_DIR_REL}/${result_base}_llama_perf.csv"
  move_if_exists "llama-bench.csv" "${RESULTS_DIR_REL}/${result_base}_llama-bench.csv"

  drop_caches 1
  echo "========================================"
}

run_all_benchmarks() {
  local model mn thread
  for i in "${!models[@]}"; do
    model="${models[$i]}"
    mn="${model_names[$i]}"

    for thread in "${threads[@]}"; do
      for j in "${!binaries[@]}"; do
        run_single_binary \
          "${model}" \
          "${mn}" \
          "${thread}" \
          "${binaries[$j]}" \
          "${accelerated[$j]}" \
          "${bins[$j]}" \
          "${acc_tag[$j]}" \
          "${bitstreams[$j]}"
      done
    done
  done
}

main() {
  parse_args "$@"
  load_configs
  init_commands_log

  trap 'error_exit ${LINENO} ${BASH_SOURCE[0]}' ERR
  trap 'stop_active_power_logger' EXIT

  echo "Running llama-bench on FPGA"
  clear_udma
  python3 "${LOAD_BITSTREAM_PY}" "${HOST_BITSTREAMS_DIR}/${DEFAULT_BITSTREAM_FILE}"

  run_all_benchmarks

  clear_bitstream
  cleanup_temp_files
}

main "$@"
