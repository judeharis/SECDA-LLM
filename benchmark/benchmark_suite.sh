#!/bin/bash
# set -x
set -e

board_user="$(jq -r '.board_user // empty' "../config.json")"
board_hostname="$(jq -r '.board_hostname // empty' "../config.json")"
board_addr="${board_user}@${board_hostname}"
port="$(jq -r '.board_port // empty' "../config.json")"
board_dir="$(jq -r '.board_dir // empty' "../config.json")"
board_sub="benchmark"



result_name=""

# CLI flags
do_compile=0
do_run=0
do_parse=0
mode_selected=0

send_pushbullet_notification() {
  #local message="$1"
  local message="$*"
  curl -s -o /dev/null --header 'Access-Token: o.eIEuBUZBIooNKzofTc6WATcyobjqK4TD' \
    --header 'Content-Type: application/json' \
    --data-binary "{\"body\":\"${message}\",\"title\":\"Jude Home (Ubuntu)\",\"type\":\"note\"}" \
    --request POST \
    https://api.pushbullet.com/v2/pushes
  push=$?
  echo "Pushbullet response: $push"
}

usage() {
  echo "Usage: $0 [--name LABEL] [-b] [-r] [-p]"
  echo "Defaults: compile=on, run=on, parse=on (when none of -b/-r/-p are provided)"
  echo "Usage: $0 [options]"
  echo "Options:"
  echo "  -n, --name LABEL         Duplicate the timestamped results folder with this name"
  echo "  -b                        Binary generation (compile/send)"
  echo "  -r                        Remote run"
  echo "  -p                        Parse/fetch results"
  echo "  -h, --help               Show this help"
  echo "Defaults: compile=on, run=on, parse=on (when none of -b/-r/-p are provided)"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
  -n | --name)
    if [[ -z "${2:-}" || "${2}" == -* ]]; then
      echo "Error: --name requires a value"
      usage
      exit 1
    fi
    result_name="$2"
    shift 2
    ;;
  -b)
    do_compile=1
    mode_selected=1
    shift
    ;;
  -r)
    do_run=1
    mode_selected=1
    shift
    ;;
  -p)
    do_parse=1
    mode_selected=1
    shift
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

# If no explicit stage is selected, run all stages.
if [ "$mode_selected" -eq 0 ]; then
  do_compile=1
  do_run=1
  do_parse=1
fi

if ! [[ "$port" =~ ^[0-9]+$ ]]; then
  echo "Error: --port must be numeric"
  exit 1
fi

# board_addr="root@rpphost.ddns.net"
# port=2222

total_start=$SECONDS
compile_time="skipped"
run_time="skipped"
parse_time="skipped"
time_stamp=$(date +"%Y-%m-%d_%H-%M-%S")


echo "-----------------------------------------------------------"
echo "-- SECDA-LLM Benchmark Suite --"
echo "-----------------------------------------------------------"
echo "Configurations"
echo "--------------"
echo "Board User: ${board_user}"
echo "Board Hostname: ${board_hostname}"
echo "Board Dir: ${board_dir}/${board_sub}"
echo "Skip Bench: ${skip_bench}"
echo "Bin Gen: ${do_compile}"
echo "Run: ${do_run}"
echo "Parse: ${do_parse}"
echo "Name: ${name}"
echo "-----------------------------------------------------------"

# compile and send to board
if [ "$do_compile" -eq 1 ]; then
  echo "[SECDA-LLM] Running compile/send..."
  echo "[SECDA-LLM] compile target: ${board_addr}, port: ${port}"
  step_start=$SECONDS
  ./scripts/compile_send_kria.sh -a "$board_addr" -p "$port" -d "$board_dir" -s "$board_sub"
  compile_time=$((SECONDS - step_start))
else
  echo "[SECDA-LLM] Skipping compile/send"
fi

# send run_experiment script to board and optionally run it
if [ "$do_run" -eq 1 ]; then
  echo "[SECDA-LLM] Sending experiment scripts to board ${board_addr}"
  step_start=$SECONDS
  rsync -avz -e "ssh -p $port" ./scripts/run_experiment_kria.sh "$board_addr":"${board_dir}/"
  rsync -avz -e "ssh -p $port" ./configs/exp_configs.sh "$board_addr":"${board_dir}/"

  # chmod and run the experiment
  ssh -t -p $port $board_addr 'chmod +x '"${board_dir}"'/run_experiment_kria.sh'
  ssh -t -p $port $board_addr 'source /etc/profile.d/pynq_venv.sh && cd '"${board_dir}"'  && ./run_experiment_kria.sh'
  run_time=$((SECONDS - step_start))
else
  echo "[SECDA-LLM] Skipping remote run"
fi

# collect results and parse
if [ "$do_parse" -eq 1 ]; then
  mkdir -p results/$time_stamp/

  echo "[SECDA-LLM] Fetching results from ${board_addr}"
  step_start=$SECONDS
  rsync -avz -e "ssh -p $port" $board_addr:${board_dir}/$board_sub/results/* results/$time_stamp/ || true
  ssh -t -p $port $board_addr 'rm -rf '${board_dir}'/'$board_sub'/results/*' || true

  echo "[SECDA-LLM] Parsing results into results/$time_stamp/"
  python3 ./scripts/parse_results.py results/$time_stamp/
  parse_time=$((SECONDS - step_start))
else
  echo "[SECDA-LLM] Skipping parsing"
fi

total_time=$((SECONDS - total_start))

# Print timing summary
echo ""
echo "========== Timing Summary =========="
echo "  Compile/Send : ${compile_time}s"
echo "  Remote Run   : ${run_time}s"
echo "  Parse        : ${parse_time}s"
echo "  Total        : ${total_time}s"
echo "===================================="

# Save timing summary to results folder
if [ "$do_parse" -eq 1 ]; then
mkdir -p results/$time_stamp/
cat >results/$time_stamp/status.txt <<EOF
Timestamp: $time_stamp
Compile/Send : ${compile_time}s
Remote Run   : ${run_time}s
Parse        : ${parse_time}s
Total        : ${total_time}s
EOF
echo "[SECDA-LLM] Timing saved to results/$time_stamp/status.txt"

if [[ -n "$result_name" ]]; then
  clone_path="results/$result_name"
  rm -rf "$clone_path"
  cp -a "results/$time_stamp/." "$clone_path/"
  echo "[SECDA-LLM] Results cloned to $clone_path"
  send_pushbullet_notification "$result_name ran in ${total_time}s (compile: ${compile_time}s, run: ${run_time}s, parse: ${parse_time}s)"
fi
fi