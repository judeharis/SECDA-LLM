# RTL replay of one bfpp_acc MUL_MAT call

This is a cycle-accurate check of the synthesised hardware against the SystemC
simulation, with no board needed. It replays the exact AXI-Stream traffic of one
driver call through the packaged IP's RTL in Vivado xsim, then compares every
output word and reports the cycles between outputs.

It was used for SECDA-LLM TODO 3.7 on 2026-09-29, on bfpp_acc v3
(`BFPP_ACC_KRIA_3_0`), with the single-tile Q2_K case (M=128, N=4, K=256):

- all 512 outputs were bit-identical to the SystemC simulation;
- the RTL took 46 cycles per output, the same as the board (117.77 µs per
  run), while the synced simulation took 16.

1. **Record the stream traffic** in the SystemC simulation. Apply the trace hook
   to SECDA-Core (`git -C <SECDA-Core> apply scripts/rtl_replay/secda_core_dma_trace.patch`),
   rebuild the sim build, and run one call with `SECDA_DMA_TRACE=<trace.txt>`,
   for example:

   ```bash
   SECDA_DMA_TRACE=/tmp/trace.txt secda-test-backend-ops test -b SECDA -t 1 \
     --test-params "29 0 128 4 1 1 16 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 2 10 256 128 1 1 0 0 0 0 0 256 4 1 1 0 0 0 0 Q2_128M_4N_256K"
   ```

   Revert the hook afterwards (`git -C <SECDA-Core> checkout secda-core/secda_integrator/axi4s_engine_generic.sc.h`).
2. **Replay:** `scripts/rtl_replay/run.sh <trace.txt> <HLS solution dir> <work dir>`.
   The solution dir is
   `hardware_automation/generated/<TAG>/<TAG>/<TAG>`, holding `impl/ip` from
   `run.sh 1 0`. The script uses Vivado 2024.1's xsim, for its
   `floating_point_v7_1_18` library; set `VIVADO_2024` to point elsewhere.

**Two simulation-only adjustments,** both made by `prepare.py`:

- **Power-up values.** Every VHDL signal without an initial value is given 0,
  which is how the FPGA configures its registers. Without this,
  `weightloader_A` (a flag with no reset) starts as `U`, WeightLoader_A starts
  reading `din1` straight after reset, and it steals words from the Control
  Unit.
- **Port types.** The IP top declares the core's 1-bit ports as
  `std_logic_vector(0 downto 0)`. Vivado synthesis accepts that, but xsim
  doesn't, so the simulation copy uses `std_logic`.

**Limits.**

- `tb_top.sv` hard-codes the call's shape: 4 phases (input packet 298 words,
  weight opcode 7, 4 x 672 weights, compute opcode 5) and 512 outputs. Adjust
  it for other calls.
- The output DMA is armed after the compute opcode, as `StoreOutputs` does, and
  never stalls.
