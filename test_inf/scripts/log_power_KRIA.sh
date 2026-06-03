#!/bin/bash

# Default sampling interval (in seconds)
# INTERVAL=${1:-0.05}

# Default log file name
LOG_FILE=${2:-log.txt}

# Identify hwmon path with ina260
HWMON_PATH=""
for hwmon in /sys/class/hwmon/hwmon*; do
  if grep -iq "ina260" "$hwmon/name"; then
    HWMON_PATH=$hwmon
    break
  fi
done

if [ -z "$HWMON_PATH" ]; then
  echo "❌ INA260 hwmon device not found."
  exit 1
fi

echo "✅ INA260 found at $HWMON_PATH"
# echo "ℹ️  Sampling every $INTERVAL seconds"
echo "📄 Logging to $LOG_FILE"
echo "▶️  Press Enter to START logging..."
read -rp ""

# Start logging loop in background
(
  echo "power_uW" > "$LOG_FILE"
  while true; do
    # TIMESTAMP=$(date +%s.%3N)
    # POWER=$(cat "$HWMON_PATH/power1_input" 2>/dev/null)
    # VOLT=$(cat "$HWMON_PATH/in1_input" 2>/dev/null)
    # CURR=$(cat "$HWMON_PATH/curr1_input" 2>/dev/null)
    # echo "$TIMESTAMP,$POWER,$VOLT,$CURR" >> "$LOG_FILE"
    # sleep "$INTERVAL"
    POWER=$(cat "$HWMON_PATH/power1_input" 2>/dev/null)
    echo "$POWER" >> "$LOG_FILE"
  done
) &
LOGGER_PID=$!

# Wait for Enter key to stop
echo "⏹️  Press Enter to STOP logging..."
read -rp ""
kill "$LOGGER_PID"
wait "$LOGGER_PID" 2>/dev/null

echo "✅ Logging stopped. Results saved in $LOG_FILE"
