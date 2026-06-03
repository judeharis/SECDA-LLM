#!/bin/bash

# Usage: ./record_power_kria.sh <remote_user> <remote_host> <path_to_script.sh> [remote_directory]

# usage example 1: 
#./record_power_kria.sh log_mbv2ImageNet_VMRPP_KRIA_GEMM8_250Mv11_10
#./record_power_kria.sh log_mbv2ImageNet_VMRPP_KRIA_SH_QK_GEMM8_250Mv11_4
#./record_power_kria.sh log_mbv2ImageNet_VMRPP_KRIA_SH_MSQ_OPT_GEMM8_250Mv11_4
#./record_power_kria.sh log_mbv2ImageNet_VMRPP_KRIA_SH_APOT_OPT_GEMM8_250Mv11_4

#./record_power_kria.sh log_incepv1ImageNet_VMRPP_KRIA_GEMM8_250Mv11_10
#./record_power_kria.sh log_incepv1ImageNet_VMRPP_KRIA_SH_QK_GEMM8_250Mv11_4
#./record_power_kria.sh log_incepv1ImageNet_VMRPP_KRIA_SH_MSQ_OPT_GEMM8_250Mv11_4
#./record_power_kria.sh log_incepv1ImageNet_VMRPP_KRIA_SH_APOT_OPT_GEMM8_250Mv11_4

LOG_FILE=${1:-log}  # Default: log
REMOTE_USER=${2:-ubuntu}
REMOTE_HOST=${3:-192.168.1.111}
SCRIPT_PATH=${4:-./log_power_KRIA.sh}  # Default: ./log_power_KRIA.sh
REMOTE_DIR=${5:-/home/ubuntu/power_measurement}  # Default: ~/power_measurement


SCRIPT_NAME=$(basename "$SCRIPT_PATH")

# Check required arguments
if [[ -z "$REMOTE_USER" || -z "$REMOTE_HOST" || -z "$SCRIPT_PATH" ]]; then
  echo "Usage: $0 <remote_user> <remote_host> <path_to_script.sh> [remote_directory]"
  exit 1
fi

# 1. Create remote directory (if not exists)
echo "📦 Creating directory $REMOTE_DIR on $REMOTE_HOST..."
ssh "$REMOTE_USER@$REMOTE_HOST" "mkdir -p '$REMOTE_DIR'"
# ssh -t "$REMOTE_USER@$REMOTE_HOST" "sudo mkdir -p '$REMOTE_DIR'"


# 2. Send the script
echo "🚀 Sending $SCRIPT_NAME to $REMOTE_HOST:$REMOTE_DIR..."
scp "$SCRIPT_PATH" "$REMOTE_USER@$REMOTE_HOST:$REMOTE_DIR/"

# 3. Run the script on the remote host
echo "⚙️  Running script on remote server..."
ssh "$REMOTE_USER@$REMOTE_HOST" "cd '$REMOTE_DIR' && chmod +x '$SCRIPT_NAME' && ./'$SCRIPT_NAME'"

# 4. Retrieve log.txt
echo "📥 Retrieving log.txt from remote..."
scp "$REMOTE_USER@$REMOTE_HOST:$REMOTE_DIR/log.txt" ./log.txt
# 5. rename log.txt to $LOG_FILE
mv log.txt "result/$LOG_FILE.txt"

echo "✅ Done. result/$LOG_FILE.txt downloaded to current directory."
