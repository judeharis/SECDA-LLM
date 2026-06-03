#!/bin/bash
# set -x
set -e

board_addr="root@jharis.ddns.net"
port=2205
result_name=""

# CLI flags
do_compile=1
do_run=1
do_parse=1

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
  echo "Usage: $0 [--board-addr user@host] [--port N] [--name LABEL] [--no-compile] [--no-run] [--no-parse]"
  echo "Defaults: compile=on, run=on, parse=on"
    echo "Usage: $0 [options]"
    echo "Options:"
    echo "  -a, --board-addr ADDR    Target board ssh address (default: $board_addr)"
    echo "  -p, --port N             SSH port to use (default: $port)"
    echo "  -n, --name LABEL         Duplicate the timestamped results folder with this name"
    echo "  -C, --no-compile         Skip compile/send step"
    echo "  -R, --no-run             Skip remote experiment run"
    echo "  -N, --no-parse           Skip fetching and parsing results"
    echo "  -h, --help               Show this help"
    echo "Defaults: compile=on, run=on, parse=on"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    -a|--board-addr)
      if [[ -z "${2:-}" || "${2}" == -* ]]; then
        echo "Error: --board-addr requires a value"
        usage
        exit 1
      fi
      board_addr="$2"
      shift 2
      ;;
    -p|--port)
      if [[ -z "${2:-}" || "${2}" == -* ]]; then
        echo "Error: --port requires a value"
        usage
        exit 1
      fi
      port="$2"
      shift 2
      ;;
    -n|--name)
      if [[ -z "${2:-}" || "${2}" == -* ]]; then
        echo "Error: --name requires a value"
        usage
        exit 1
      fi
      result_name="$2"
      shift 2
      ;;
    -C|--no-compile)
      do_compile=0
      shift
      ;;
    -R|--no-run)
      do_run=0
      shift
      ;;
    -N|--no-parse)
      do_parse=0
      shift
      ;;
    -h|--help)
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

# compile and send to board (skip if --no-compile)
if [ "$do_compile" -eq 1 ]; then
  echo "[kria_exp_bfpp] Running compile/send..."
  echo "[kria_exp_bfpp] compile target: ${board_addr}, port: ${port}"
  step_start=$SECONDS
  ./compile_send_kria_bfpp.sh -a "$board_addr" -p "$port"
  compile_time=$(( SECONDS - step_start ))
else
  echo "[kria_exp_bfpp] Skipping compile/send (--no-compile)"
fi

# send run_experiment script to board and optionally run it (skip if --no-run)
if [ "$do_run" -eq 1 ]; then
  echo "[kria_exp_bfpp] Sending experiment scripts to board ${board_addr}"
  step_start=$SECONDS
  rsync -avz -e "ssh -p $port" ./scripts/{run_experiment_kria_bfpp.sh,exp_configs.sh} "$board_addr":/home/ubuntu/Workspace/secda_llm/

  # chmod and run the experiment
  ssh -t -p $port $board_addr 'chmod +x /home/ubuntu/Workspace/secda_llm/run_experiment_kria_bfpp.sh'
  ssh -t -p $port $board_addr 'source /etc/profile.d/pynq_venv.sh && cd /home/ubuntu/Workspace/secda_llm/  && ./run_experiment_kria_bfpp.sh'
  run_time=$(( SECONDS - step_start ))
else
  echo "[kria_exp_bfpp] Skipping remote run (--no-run)"
fi

# collect results and parse (skip if --no-parse)
if [ "$do_parse" -eq 1 ]; then
  mkdir -p bfpp_results/$time_stamp/

  echo "[kria_exp_bfpp] Fetching results from ${board_addr}"
  step_start=$SECONDS
  rsync -avz -e "ssh -p $port" $board_addr:/home/ubuntu/Workspace/secda_llm/bfpp/results/* bfpp_results/$time_stamp/ || true
  ssh -t -p $port $board_addr 'rm -rf /home/ubuntu/Workspace/secda_llm/bfpp/results/*' || true

  echo "[kria_exp_bfpp] Parsing results into bfpp_results/$time_stamp/"
  python3 ./scripts/parse_results.py bfpp_results/$time_stamp/
  parse_time=$(( SECONDS - step_start ))
else
  echo "[kria_exp_bfpp] Skipping parsing (--no-parse)"
fi

total_time=$(( SECONDS - total_start ))

# Print timing summary
echo ""
echo "========== Timing Summary =========="
echo "  Compile/Send : ${compile_time}s"
echo "  Remote Run   : ${run_time}s"
echo "  Parse        : ${parse_time}s"
echo "  Total        : ${total_time}s"
echo "===================================="

# Save timing summary to results folder
mkdir -p bfpp_results/$time_stamp/
cat > bfpp_results/$time_stamp/status.txt <<EOF
Timestamp: $time_stamp
Compile/Send : ${compile_time}s
Remote Run   : ${run_time}s
Parse        : ${parse_time}s
Total        : ${total_time}s
EOF
echo "[kria_exp_bfpp] Timing saved to bfpp_results/$time_stamp/status.txt"

if [[ -n "$result_name" ]]; then
  clone_path="bfpp_results/$result_name"
  rm -rf "$clone_path"
  cp -a "bfpp_results/$time_stamp/." "$clone_path/"
  echo "[kria_exp_bfpp] Results cloned to $clone_path"
  send_pushbullet_notification "$result_name ran in ${total_time}s (compile: ${compile_time}s, run: ${run_time}s, parse: ${parse_time}s)"
fi
