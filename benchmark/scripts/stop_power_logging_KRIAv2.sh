#!/usr/bin/env bash

set -euo pipefail

PID_OR_FILE=${1:-power_logger.pid}
REMOVE_PID_FILE=${2:-power_logger.pid}

resolve_pid() {
  local input=$1
  if [[ "$input" =~ ^[0-9]+$ ]]; then
    echo "$input"
    return 0
  fi

  if [[ -f "$input" ]]; then
    local from_file
    from_file=$(cat "$input" 2>/dev/null || true)
    if [[ "$from_file" =~ ^[0-9]+$ ]]; then
      echo "$from_file"
      return 0
    fi
  fi

  return 1
}

PID=$(resolve_pid "$PID_OR_FILE" || true)
if [[ -z "$PID" ]]; then
  echo "Could not resolve a PID from: $PID_OR_FILE"
  echo "Usage: $0 <pid|pid_file> [pid_file_to_remove]"
  exit 1
fi

if ! kill -0 "$PID" 2>/dev/null; then
  echo "Process $PID is not running."
  if [[ -f "$REMOVE_PID_FILE" ]]; then
    rm -f "$REMOVE_PID_FILE"
  fi
  exit 1
fi

kill "$PID"
if kill -0 "$PID" 2>/dev/null; then
  sleep 0.1
fi

if kill -0 "$PID" 2>/dev/null; then
  echo "PID $PID did not exit after SIGTERM. Sending SIGKILL."
  kill -9 "$PID"
fi

if [[ -f "$REMOVE_PID_FILE" ]]; then
  file_pid=$(cat "$REMOVE_PID_FILE" 2>/dev/null || true)
  if [[ "$file_pid" == "$PID" ]]; then
    rm -f "$REMOVE_PID_FILE"
  fi
fi

echo "Stopped power logger PID $PID"