# Simulation vs hardware replay (bfpp_acc v3 `driver`)

Checks that the SystemC model computes exactly what the FPGA computes. Capture
real `MUL_MAT` calls on the board, replay their inputs through the simulation
build's `EntryMM`, and compare the outputs bit for bit.

Used for TODO 1.5 on 2026-09-28: 60 calls from the `-b 16 -ub 16` perplexity
run on `kriaB_L` (`BFPP_ACC_KRIA_3_0`) replayed with 0 of 310,464 outputs
different.

1. **Capture.** Apply the capture hook, which is not committed to the driver:
   `git apply scripts/sim_replay/v3_capture.patch`. Rebuild the board build (and
   the sim build, if you want a sim capture too). Then run any workload with
   `SECDA_CAPTURE_DIR=<dir>` and optionally `SECDA_CAPTURE_MAX=<n>` (default 64).
   Each call is written to `<dir>/call_<i>.bin`: an int32 header
   `{M, N, K, wgt_type, inp_stride, wgt_stride, out_stride, layer}`, then the
   q8_K input rows, the weight rows and the output (N x M floats).
   Revert the patch afterwards: `git checkout <driver>/acc_driver.h`.
2. **Build the replay** against the simulation build:

   ```bash
   B=out/build/SECDA-sim-x64/bin
   clang++ -O1 -std=c++17 scripts/sim_replay/replay.cpp -o replay_sim \
     -L$B -lggml-secda -Wl,-rpath,$PWD/$B -Wl,--allow-shlib-undefined
   ```
3. **Replay:** `LD_LIBRARY_PATH=$SYSTEMC_HOME/lib-linux64 ./replay_sim <dir>/call_*.bin`.
   It prints the differing outputs per call (count, max ulp, max abs/rel)
   and a total.

Replaying a capture from the same build must give 0 differences; that checks the
harness itself. The patch applies only to v3's `driver/acc_driver.h`, so the
batches and v4 drivers need the same hook added by hand.
