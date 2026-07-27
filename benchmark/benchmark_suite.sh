#!/bin/bash
set -e

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/.." && pwd)"
cd "${script_dir}"

# ============================================================================
# CONFIG
# ============================================================================

board_user="$(jq -r '.board_user // empty' "${repo_root}/config.json")"
board_hostname="$(jq -r '.board_hostname // empty' "${repo_root}/config.json")"
board_addr="${board_user}@${board_hostname}"
port="$(jq -r '.board_port // empty' "${repo_root}/config.json")"
board_dir="$(jq -r '.board_dir // empty' "${repo_root}/config.json")"
board_sub="benchmark"

# Benchmark registry: benchmark_name -> script_name
# Add new benchmarks by adding entries here
declare -A BENCHMARKS=(
  [llama-cli]="run_llama_cli.sh"
  [llama-bench]="run_llama_bench.sh"
  [synth-bench]="run_synth_bench.sh"
)

# ============================================================================
# GLOBALS
# ============================================================================

result_name=""
time_stamp=$(date +"%Y-%m-%d_%H-%M-%S")
declare -A stage_times
declare -A stage_status

# ============================================================================
# UTILITIES
# ============================================================================

send_pushbullet_notification() {
  local message="$*"
  curl -s -o /dev/null --header 'Access-Token: o.eIEuBUZBIooNKzofTc6WATcyobjqK4TD' \
    --header 'Content-Type: application/json' \
    --data-binary "{\"body\":\"${message}\",\"title\":\"Jude Home (Ubuntu)\",\"type\":\"note\"}" \
    --request POST \
    https://api.pushbullet.com/v2/pushes
  echo "Pushbullet response: $?"
}

log_stage() {
  echo "[SECDA-LLM] $*"
}

usage() {
  cat <<EOF
Usage: $0 [OPTIONS]

Options:
  -n, --name LABEL    Save results with this label (in addition to timestamp)
  -b, --compile       Compile and send binaries to board
  -c, --cli           Run llama-cli
  -l, --llama-bench   Run llama-bench
  -s, --synth-bench   Run synth benchmark (test-backend-ops)
  -p, --parse         Parse and fetch results
  -po, --power        Enable power logging during benchmark runs (default: on)
  --no-power          Disable power logging during benchmark runs
  -t, --threads N     Thread count(s) to run with, comma-separated (default: 1)
  -h, --help          Show this help

Defaults (if no flags specified): compile, llama-bench, parse

Available benchmarks:
EOF
  for bench in "${!BENCHMARKS[@]}"; do
    echo "  - ${bench}"
  done
}

# ============================================================================
# BOARD OPERATIONS
# ============================================================================

compile_and_send() {
  log_stage "Running compile/send..."
  log_stage "Target: ${board_addr}, port: ${port}"
  "${script_dir}/scripts/compile_send_kria.sh" -a "$board_addr" -p "$port" -d "$board_dir" -s "$board_sub"
}

sync_support_scripts() {
  log_stage "Syncing support scripts to board ${board_addr}"
  ssh -p "$port" "$board_addr" "mkdir -p '${board_dir}/${board_sub}/scripts'"
  rsync -avz -e "ssh -p $port" "${script_dir}/scripts/start_power_logging_KRIAv2.sh" "$board_addr":"${board_dir}/${board_sub}/scripts/"
  rsync -avz -e "ssh -p $port" "${script_dir}/scripts/stop_power_logging_KRIAv2.sh" "$board_addr":"${board_dir}/${board_sub}/scripts/"
}

send_and_run_benchmark() {
  local benchmark_name="$1"
  shift  # Remove benchmark name, remaining args are flags for the script
  local script_flags="$@"
  local script_name="${BENCHMARKS[$benchmark_name]}"
  
  if [[ -z "$script_name" ]]; then
    log_stage "Error: Unknown benchmark '$benchmark_name'"
    return 1
  fi

  log_stage "Sending benchmark script: ${benchmark_name}"
  rsync -avz -e "ssh -p $port" "${script_dir}/scripts/${script_name}" "$board_addr":"${board_dir}/"
  rsync -avz -e "ssh -p $port" "${script_dir}/configs/exp_configs.sh" "$board_addr":"${board_dir}/"

  log_stage "Running ${benchmark_name}${script_flags:+ with flags: ${script_flags}}..."
  ssh -T -p "$port" "$board_addr" "chmod +x '${board_dir}/${script_name}'"
  ssh -T -p "$port" "$board_addr" "bash -lc 'source /etc/profile.d/pynq_venv.sh && cd '\''${board_dir}'\'' && ./'\''${script_name}'\'' ${script_flags}'"
}

fetch_and_parse_results() {
  local results_dir="${script_dir}/results/${time_stamp}"
  mkdir -p "$results_dir"

  log_stage "Fetching results from ${board_addr}"
  rsync -avz -e "ssh -p $port" "$board_addr:${board_dir}/${board_sub}/results/*" "$results_dir/" || true
  ssh -t -p $port $board_addr "rm -rf ${board_dir}/${board_sub}/results/*" || true

  log_stage "Parsing results into $results_dir"
  python3 "${script_dir}/scripts/parse_results.py" "$results_dir"
}

# ============================================================================
# REPORTING
# ============================================================================

print_config() {
  cat <<EOF

-----------------------------------------------------------
-- SECDA-LLM Benchmark Suite --
-----------------------------------------------------------
Configurations
--------------
Board User:    ${board_user}
Board Hostname: ${board_hostname}
Board Dir:     ${board_dir}/${board_sub}
Port:          ${port}
Result Name:   ${result_name:-<none>}
-----------------------------------------------------------

EOF
}

print_summary() {
  local total_time=$1
  
  cat <<EOF

========== Timing Summary ==========
EOF
  
  for stage in compile run llama-bench synth-bench parse; do
    local time_val="${stage_times[$stage]}"
    [[ -z "$time_val" ]] && time_val="skipped"
    printf "  %-15s: %s\n" "$stage" "$time_val"
  done
  
  printf "  %-15s: %s\n" "Total" "${total_time}s"
  echo "===================================="
}

save_status() {
  local results_dir="${script_dir}/results/${time_stamp}"
  mkdir -p "$results_dir"
  
  cat >"${results_dir}/status.txt" <<EOF
Timestamp:   $time_stamp
Compile:     ${stage_times[compile]:-skipped}s
Run:         ${stage_times[run]:-skipped}s
Llama-Bench: ${stage_times[llama-bench]:-skipped}s
Synth-Bench: ${stage_times[synth-bench]:-skipped}s
Parse:       ${stage_times[parse]:-skipped}s
Total:       $1s
EOF
  
  log_stage "Timing saved to ${results_dir}/status.txt"
  
  if [[ -n "$result_name" ]]; then
    local clone_path="${script_dir}/results/$result_name"
    rm -rf "$clone_path"
    cp -a "$results_dir/." "$clone_path/"
    log_stage "Results cloned to $clone_path"
    send_pushbullet_notification "$result_name ran in $1s"
  fi
}

# ============================================================================
# MAIN
# ============================================================================

main() {
  local total_start=$SECONDS
  local do_compile=0 do_cli=0 do_llama_bench=0 do_synth_bench=0 do_parse=0 do_power=1 mode_selected=0
  local threads_value=""

  while [[ $# -gt 0 ]]; do
    case "$1" in
      -n | --name)
        result_name="$2"
        shift 2
        ;;
      -b | --compile)
        do_compile=1
        mode_selected=1
        shift
        ;;
      -c | --cli)
        do_cli=1
        mode_selected=1
        shift
        ;;
      -l | --llama-bench)
        do_llama_bench=1
        mode_selected=1
        shift
        ;;
      -s | --synth-bench)
        do_synth_bench=1
        mode_selected=1
        shift
        ;;
      -p | --parse)
        do_parse=1
        mode_selected=1
        shift
        ;;
      -po | --power)
        do_power=1
        shift
        ;;
      --no-power)
        do_power=0
        shift
        ;;
      -t | --threads)
        threads_value="$2"
        shift 2
        ;;
      -h | --help)
        usage
        exit 0
        ;;
      *)
        echo "Unknown option: $1"
        usage
        exit 1
        ;;
    esac
  done

  # Default: run all stages
  if [[ $mode_selected -eq 0 ]]; then
    do_compile=1
    do_llama_bench=1
    do_parse=1
  fi

  # Validate port
  if ! [[ "$port" =~ ^[0-9]+$ ]]; then
    echo "Error: board_port must be numeric"
    exit 1
  fi

  local power_flag=""
  [[ $do_power -eq 1 ]] && power_flag="--power"

  local thread_flag=""
  [[ -n "$threads_value" ]] && thread_flag="--threads ${threads_value}"

  print_config

  # Compile and send
  if [[ $do_compile -eq 1 ]]; then
    local step_start=$SECONDS
    compile_and_send
    stage_times[compile]=$((SECONDS - step_start))
  else
    log_stage "Skipping compile/send"
  fi

  # Sync support scripts
  sync_support_scripts

  # Run llama-cli
  if [[ $do_cli -eq 1 ]]; then
    local step_start=$SECONDS
    send_and_run_benchmark "llama-cli" ${power_flag} ${thread_flag}
    stage_times[run]=$((SECONDS - step_start))
  else
    log_stage "Skipping llama-cli"
  fi

  # Run llama-bench
  if [[ $do_llama_bench -eq 1 ]]; then
    local step_start=$SECONDS
    send_and_run_benchmark "llama-bench" ${power_flag} ${thread_flag}
    stage_times[llama-bench]=$((SECONDS - step_start))
  else
    log_stage "Skipping llama-bench"
  fi

  # Run synth-bench
  if [[ $do_synth_bench -eq 1 ]]; then
    local step_start=$SECONDS
    send_and_run_benchmark "synth-bench" ${power_flag} ${thread_flag}
    stage_times[synth-bench]=$((SECONDS - step_start))
  else
    log_stage "Skipping synth-bench"
  fi

  # Fetch and parse results
  if [[ $do_parse -eq 1 ]]; then
    local step_start=$SECONDS
    fetch_and_parse_results
    stage_times[parse]=$((SECONDS - step_start))
  else
    log_stage "Skipping parse"
  fi

  local total_time=$((SECONDS - total_start))
  print_summary "$total_time"
  save_status "$total_time"
}

main "$@"