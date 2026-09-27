#!/bin/bash
set -eo pipefail

threads=(1)

BOARD_PATH="/home/ubuntu/Workspace/secda_llm"
BOARD_SUB="benchmark"
BENCHMARK_ROOT="${BOARD_PATH}/${BOARD_SUB}"
RESULTS_DIR_REL="../results"
MODEL_DIR="${BOARD_PATH}/models"
DATASET_DIR="${BOARD_PATH}/datasets"
DATASET_FILE="wiki.test.raw"
COMMANDS_FILE="${BENCHMARK_ROOT}/commands_perplexity.txt"

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

PPL_CHUNKS="${PPL_CHUNKS:-64}"
PPL_BATCH="${PPL_BATCH:-16}"
# 512 matches llama-perplexity's own built-in default (set before arg
# parsing in perplexity.cpp), the traditional wikitext-2 PPL@ctx=512
# convention - not the model's trained context like other llama.cpp tools.
PPL_CTX_SIZE="${PPL_CTX_SIZE:-512}"
# SECDA's DimCheck (acc_driver.h) rejects any MUL_MAT where
# (ubatch_size * ceil(K/256)) > 512 (SUP_KNB), K being a layer's reduction
# dim - each llama_decode() call is split into ubatch_size-token chunks, so
# this is the actual per-call N the accelerator sees, not -b. For
# tiny-llama-1.1B (ffn=5632, the largest K) that caps ubatch at ~23; 16
# keeps headroom for larger models before this needs recomputing.
PPL_UBATCH="${PPL_UBATCH:-16}"
FLAGS="--no-warmup"

parse_args() {
  while [[ $# -gt 0 ]]; do
    case "$1" in
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

  models=("${models_array[@]}")
  model_names=("${model_names_array[@]}")
  bitstreams=("${bitstreams_array[@]}")
  binaries=("${binaries_array[@]}")
  acc_tag=("${acc_tags_array[@]}")
  bins=("${bins_array[@]}")
  accelerated=("${accelerated_array[@]}")
}

init_commands_log() {
  rm -f "${COMMANDS_FILE}"
  touch "${COMMANDS_FILE}"
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

check_cmd_status() {
  local status="$1"
  if [[ "${status}" -ne 0 ]]; then
    echo "llama-perplexity command failed with status ${status}"
    exit "${status}"
  fi
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

  local ppl_binary="${binary}_perplexity"
  local result_base="perplexity_${mn}_${thread}_${tag}"
  local result_txt="${RESULTS_DIR_REL}/${result_base}.txt"
  local dataset_path="${DATASET_DIR}/${DATASET_FILE}"

  echo "========================================"
  echo "Running llama-perplexity for ${model}_${thread}_${tag}"

  if [[ ! -f "${dataset_path}" ]]; then
    echo "Missing perplexity dataset: ${dataset_path} (run benchmark_suite.sh with -pp/--perplexity to sync it)"
    exit 1
  fi

  echo "python3 ${LOAD_BITSTREAM_PY} ${BOARD_BITSTREAMS_DIR}/${bitstream}.bit" >>"${COMMANDS_FILE}"
  python3 "${LOAD_BITSTREAM_PY}" "${BOARD_BITSTREAMS_DIR}/${bitstream}.bit"
  drop_caches 3

  cd "${BENCHMARK_ROOT}"
  mkdir -p results
  cd "${BENCHMARK_ROOT}/${bin_folder}"

  if [[ ! -x "./${ppl_binary}" ]]; then
    echo "Missing executable: ${BENCHMARK_ROOT}/${bin_folder}/${ppl_binary}"
    exit 1
  fi

  echo "sudo env LD_LIBRARY_PATH=\${PWD}/bin:\${LD_LIBRARY_PATH:-} ./${ppl_binary} -m ${MODEL_DIR}/${model} -f ${dataset_path} -t ${thread} --ctx-size ${PPL_CTX_SIZE} -b ${PPL_BATCH} -ub ${PPL_UBATCH} --chunks ${PPL_CHUNKS} ${FLAGS}" >>"${COMMANDS_FILE}"

  local cmd_status=0
  LD_LIBRARY_PATH="${PWD}/bin:${LD_LIBRARY_PATH:-}" "./${ppl_binary}" -m "${MODEL_DIR}/${model}" -f "${dataset_path}" -t "${thread}" --ctx-size "${PPL_CTX_SIZE}" -b "${PPL_BATCH}" -ub "${PPL_UBATCH}" --chunks "${PPL_CHUNKS}" ${FLAGS} \
    2>&1 | tee "${result_txt}" || cmd_status=$?

  check_cmd_status "${cmd_status}"

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

  echo "Running llama-perplexity on FPGA"
  clear_udma
  python3 "${LOAD_BITSTREAM_PY}" "${HOST_BITSTREAMS_DIR}/${DEFAULT_BITSTREAM_FILE}"

  run_all_benchmarks

  clear_bitstream
  cleanup_temp_files
}

main "$@"
