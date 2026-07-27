#!/bin/bash
set -eo pipefail

enable_power=false
threads=(1)

BOARD_PATH="/home/ubuntu/Workspace/secda_llm"
BOARD_SUB="benchmark"
POWER_START_SCRIPT="${BOARD_PATH}/${BOARD_SUB}/scripts/start_power_logging_KRIAv2.sh"
POWER_STOP_SCRIPT="${BOARD_PATH}/${BOARD_SUB}/scripts/stop_power_logging_KRIAv2.sh"

LOAD_BITSTREAM_PY="${HOME}/load_bitstream.py"
BOARD_BITSTREAMS_DIR="${BOARD_PATH}/bitstreams"
HOST_BITSTREAMS_DIR="/home/ubuntu/bitstreams"
CLEAR_BITSTREAM_FILE="CPU_KRIA_1_0.bit"

UDMABUF_GLOB="/dev/udmabuf*"
UDMABUF_PREFIX="/dev/udmabuf"
UDMABUF_MGR="/dev/u-dma-buf-mgr"
MEMINFO_PATH="/proc/meminfo"
DROP_CACHES_PATH="/proc/sys/vm/drop_caches"
TRACE_FILE="sds_trace_data.dat"

active_power_pid=""
active_power_pid_file=""

parse_args() {
  while [[ $# -gt 0 ]]; do
    case "$1" in
      --power)
        enable_power=true
        shift
        ;;
      -t | --threads)
        IFS=',' read -r -a threads <<< "$2"
        shift 2
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

  synth_test_names=("${synth_test_names_array[@]}")
  synth_test_lines=("${synth_test_lines_array[@]}")
  bitstreams=("${bitstreams_array[@]}")
  binaries=("${binaries_array[@]}")
  acc_tag=("${acc_tags_array[@]}")
  bins=("${bins_array[@]}")
  accelerated=("${accelerated_array[@]}")
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

check_cmd_status() {
  local status="$1"
  local test_name="$2"
  local command_name="$3"

  if [[ "${status}" -ne 0 ]]; then
    echo "${command_name} command failed for ${test_name} with status ${status}"
    exit "${status}"
  fi
}

move_if_exists() {
  [[ -f "$1" ]] && mv -f "$1" "$2"
}

print_test_cases() {
  echo "-----------------------------------------------------------"
  echo "Test cases (${#synth_test_names[@]}):"
  for tn in "${synth_test_names[@]}"; do
    echo "  ${tn}"
  done
  echo "-----------------------------------------------------------"
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

run_single_test() {
  local test_binary="$1"
  local acc="$2"
  local tag="$3"
  local test_name="$4"
  local test_line="$5"
  local thread="$6"

  local result_file="../results/synth_${test_name}_${thread}_${tag}_results.txt"
  local valid_file="../results/synth_${test_name}_${thread}_${tag}_validation.txt"
  local power_log_file="../results/synth_${test_name}_${thread}_${tag}_power.txt"
  local power_pid_file="../results/synth_${test_name}_${thread}_${tag}_power.pid"
  local backend="SECDA"
  local cmd_status=0

  if [[ "${acc}" != true ]]; then
    backend="CPU"
  fi


  echo "" | tee -a "${result_file}"
  echo "========== Test: ${test_name} (threads=${thread}) ==========" | tee -a "${result_file}"

  LD_LIBRARY_PATH="${PWD}/bin:${LD_LIBRARY_PATH:-}" "./${test_binary}" test -b "${backend}" -t "${thread}" --output csv --test-params "${test_line}" \
    >> "${valid_file}" 2>&1 || cmd_status=$?
  check_cmd_status "${cmd_status}" "${test_name}" "tbo_validation"

  move_if_exists "tbo.csv" "../results/synth_${test_name}_${thread}_${tag}_vtbo.csv"

  cmd_status=0
  start_power_logger "${power_log_file}" "${power_pid_file}"
  LD_LIBRARY_PATH="${PWD}/bin:${LD_LIBRARY_PATH:-}" "./${test_binary}" perf -b "${backend}" -t "${thread}" --output csv --test-params "${test_line}" \
    2>&1 | tee -a "${result_file}" || cmd_status=$?

  stop_active_power_logger
  check_cmd_status "${cmd_status}" "${test_name}" "${test_binary}"

  [[ "${acc}" == true ]] && move_if_exists "prf.csv" "../results/synth_${test_name}_${thread}_${tag}_prf.csv"
  move_if_exists "tbo.csv" "../results/synth_${test_name}_${thread}_${tag}_tbo.csv"
  drop_caches 1
}

run_binary_suite() {
  local binary="$1"
  local acc="$2"
  local bin_folder="$3"
  local tag="$4"
  local bitstream="$5"

  local test_binary="${binary}_tbo"

  echo "========================================"
  echo "Running synth bench for ${tag}"

  python3 "${LOAD_BITSTREAM_PY}" "${BOARD_BITSTREAMS_DIR}/${bitstream}.bit"
  drop_caches 3

  cd "${BOARD_PATH}/${BOARD_SUB}/"
  mkdir -p results
  cd "${BOARD_PATH}/${BOARD_SUB}/${bin_folder}"

  if [[ ! -x "./${test_binary}" ]]; then
    echo "Missing executable: ${BOARD_PATH}/${BOARD_SUB}/${bin_folder}/${test_binary}"
    exit 1
  fi

  for thread in "${threads[@]}"; do
    for i in "${!synth_test_names[@]}"; do
      run_single_test "${test_binary}" "${acc}" "${tag}" "${synth_test_names[$i]}" "${synth_test_lines[$i]}" "${thread}"
    done
  done

  echo "========================================"
}

main() {
  parse_args "$@"
  load_configs

  trap 'error_exit ${LINENO} ${BASH_SOURCE[0]}' ERR
  trap 'stop_active_power_logger' EXIT

  print_test_cases
  clear_udma
  clear_bitstream

  for i in "${!binaries[@]}"; do
    run_binary_suite "${binaries[$i]}" "${accelerated[$i]}" "${bins[$i]}" "${acc_tag[$i]}" "${bitstreams[$i]}"
  done

  clear_bitstream
  cleanup_temp_files
}

main "$@"
