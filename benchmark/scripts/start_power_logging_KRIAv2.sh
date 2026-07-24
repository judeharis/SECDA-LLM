#!/usr/bin/env bash

set -euo pipefail

INTERVAL=${1:-0.05}
LOG_FILE=${2:-log.txt}
PID_FILE=${3:-power_logger.pid}

find_ina260_hwmon() {
  local hwmon
  for hwmon in /sys/class/hwmon/hwmon*; do
    if [[ -f "$hwmon/name" ]] && grep -iq "ina260" "$hwmon/name"; then
      echo "$hwmon"
      return 0
    fi
  done
  return 1
}

if [[ -f "$PID_FILE" ]]; then
  existing_pid=$(cat "$PID_FILE" 2>/dev/null || true)
  if [[ -n "${existing_pid}" ]] && kill -0 "$existing_pid" 2>/dev/null; then
    echo "Power logger is already running with PID $existing_pid"
    echo "Use stop_power_logging_KRIAv2.sh $existing_pid"
    exit 1
  fi
fi

HWMON_PATH=$(find_ina260_hwmon || true)
if [[ -z "$HWMON_PATH" ]]; then
  echo "INA260 hwmon device not found."
  exit 1
fi

mkdir -p "$(dirname "$LOG_FILE")"
mkdir -p "$(dirname "$PID_FILE")"

export INTERVAL LOG_FILE HWMON_PATH
nohup bash -c '
  echo "power_uW" > "$LOG_FILE"
  while true; do
    sleep "$INTERVAL"
    POWER=$(cat "$HWMON_PATH/power1_input" 2>/dev/null || true)
    if [[ -n "$POWER" ]]; then
      echo "$POWER" >> "$LOG_FILE"
    fi
  done
' >/dev/null 2>&1 &

LOGGER_PID=$!
echo "$LOGGER_PID" > "$PID_FILE"

# echo "Started power logging"
# echo "PID: $LOGGER_PID"
# echo "INA260 path: $HWMON_PATH"
# echo "Log file: $LOG_FILE"
# echo "PID file: $PID_FILE"