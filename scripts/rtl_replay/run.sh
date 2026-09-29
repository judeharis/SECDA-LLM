#!/bin/bash
# RTL replay of one bfpp_acc MUL_MAT call in Vivado xsim (see README.md).
#   run.sh <trace.txt> <HLS solution dir> <work dir>
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
trace=$(realpath "$1"); sol=$(realpath "$2"); work=$(realpath -m "$3")
V=${VIVADO_2024:-/mnt/Crucial/Xilinx2024/Vivado/2024.1}
source "$V/settings64.sh" > /dev/null
mkdir -p "$work" && cd "$work"
python3 "$here/prepare.py" "$trace" "$sol" .
hdl="$sol/impl/ip/hdl/verilog"
xvhdl -log ip.log -L floating_point_v7_1_18 "$sol"/impl/ip/hdl/ip/*.vhd > /dev/null
xvhdl -log core.log core_sim/*.vhd > /dev/null
xvhdl -log top.log bfp_acc_top_sim.vhd "$hdl/BFP_Acc_reset_if.vhd" > /dev/null
xvlog -log if.log "$hdl"/*_if.v "$V/data/verilog/src/glbl.v" > /dev/null
xvlog -log tb.log -sv "$here/tb_top.sv" > /dev/null
xelab -log elab.log -debug off -L floating_point_v7_1_18 -L unisims_ver -L unimacro_ver \
  -L secureip -L xpm tb glbl -s tb_top_sim > /dev/null
xsim tb_top_sim -R -log run.log > /dev/null
grep -E "PHASE|RESULT|TIMEOUT|MISMATCH|<-" run.log
