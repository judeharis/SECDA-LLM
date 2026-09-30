# SECDA-LLM — todo

Follow-ups from the SECDA-Core v6 migration and the Phase 8 fact-check
(2026-09-28). Context for each item is in [STATUS.md](STATUS.md) (Open items) and in
SECDA-DS `docs/secda-llm-migration-plan.md` / `docs/secda-llm-minimal-fork-spec.md`.
Tick items here and record results in the design's `test_status.json`.

Paths below are relative to this repo; `D/` is `srcs/ggml_backend/ggml-secda/acc_dels/bfpp_acc`.

## 1. Verification gaps (do first)

- [x] **1.1 Board perplexity at `-b 16 -ub 16`.** Done 2026-09-28 on `kriaB_L`, recorded
      in v3/v4 `test_status.json`. With `-c 128 --chunks 2 -t 1`:

      | Build | chunk 1 | final |
      |---|---|---|
      | KV260 SECDA (v3, v3b, v4, v4b, all identical) | 3.3273 | 3.3636 +/- 0.60469 |
      | KV260 CPU-only | 3.3414 | 3.3596 |
      | x86 SystemC simulation (all 4 variants) | 3.3146 | 3.3429 +/- 0.59671 |
      | x86 CPU-only | 3.3601 | 3.3637 |

      All 30 MUL_MATs ran on SECDA and preloaded, with no warnings. The board's
      accelerator is within 0.12% of the board CPU. Simulation and board differ
      because their inputs differ, not their accelerator: see 1.5.
- [x] **1.2 Perplexity input in the repo.** `perplexity/ppl_input.txt` (md5
      `b6d844ba5fc667529f6efd3cf3cdd204`, byte-identical to the one used for the
      recorded values). The README and `perplexity/README.md` point at it.
- [x] **1.3 NaN fails a case again.** `secda-test-backend-ops` now fails a case with a
      NaN output, as upstream. `SECDA_TBO_NAN_PASS=1` restores the old pass behaviour.
      - Simulation: the gate suite gives 68/68, identical to the baseline, with no
        NaN in any log.
      - Board: v3/v4, both drivers, give MUL_MAT 51/51 and 55/55, SOFT_MAX 212/212,
        and no NaN.
      - 6.3 remains open if the owner prefers the old default.
- [x] **1.4 v4's SOFT_MAX unit end to end.** Done 2026-09-28, after 2.2.
      - **Before 2.2:** with `secda-llama-cli -fa off`, v4 and v4-batches (60 SECDA
        nodes: 30 MUL_MAT plus 30 SOFT_MAX) generated garbage ("amovereverevere...")
        in simulation, with no `SECDA WARNING`.
      - **After 2.2:** v4 and v4b `-fa off` generate text identical to the CPU-only
        build's, in simulation (vs x86 CPU) and on `kriaB_L` (vs the board CPU
        `-fa off`), with 60 SECDA nodes and no warnings. v3 and v3b match too.
- [x] **1.5 Simulation vs board numerics. Resolved 2026-09-28: the simulation is
      bit-exact; the inputs differ.** At `-b 16` (table in 1.1), simulation sat 0.6%
      below x86 CPU while the board sat 0.12% above its CPU.
      - **Replay.** 60 real `ffn_down` MUL_MAT calls (q3_K, M=576, K=1536,
        N=1/2/16) were captured on `kriaB_L` during the `-b 16` perplexity run and
        replayed through the SystemC model. All 310,464 outputs were bit-identical.
        The tool is in `scripts/sim_replay/`.
      - **Inputs.** Comparing the simulation's own capture with the board's: the
        weights are identical, but the q8_K activations already differ at call 0
        (3 of 12 blocks, in the float scale only). By call 45, 93 of 96 blocks
        differ, with 11,679 int8 values changed. The CPU ops before each MUL_MAT
        (norms, activations) round differently on ARM and x86, and the difference
        compounds through the layers.
      - **Sensitivity.** This 2-chunk perplexity test amplifies such rounding: the
        two CPU-only builds alone differ by 0.5% at `-b 128`. Compare only against
        the CPU build on the same platform.
      - The v3 `driver` was replayed; `driver_batches` and v4 share the MUL_MAT unit
        but were not replayed.

## 2. Driver fixes (spec Phase 2, H1/H2)

These bugs were kept for parity during the migration (minimal-fork spec §6, risk 4).
2.1 and 2.2 were done together on 2026-09-28, since 2.2 needs 2.1's bounds checks.

- [x] **2.1 H1, driver hygiene** (all 4 driver variants; `acc_container.h` and
      `acc_driver_mt.h` are identical in all four):
      - `reset()` uses `layer_preloaded.assign(500, false)`. `resize` had kept the
        previous plan's flags set.
      - `is_preloaded(l)`, with bounds checks (`l < 0` means unplanned), replaces
        `alloced_layers >= layer && layer_preloaded[layer]` in `LoadWeights` and
        the `EntryMM` flag reads.
      - The tile maps use `.at(layer).at(m)`.
- [x] **2.2 H2, explicit layers.** The bug: the planner numbered SOFT_MAX nodes
      too, while the driver pushed one tile map per preloaded MUL_MAT and then
      looked it up by that shared number. With `-fa off`, MUL_MAT *k* therefore
      used another layer's weights.
      - **Driver:** `setLayer(int)` in the connector and `driver_interface.h`
        (backend `setLayer_T`). The tile maps are stored by layer
        (`set_tile_maps`), and `alloc_layer` refuses `l < 0`.
      - **Planner:** each planned MUL_MAT's layer is its MUL_MAT ordinal, which is
        also what `preload_weights_alloc` gets. `secda_planner_set_layer`, which
        replaces `check_node`, calls `setLayer` before each MUL_MAT. SOFT_MAX no
        longer touches layers. The positional mirror is gone, and with it the
        "layer desync" warnings.
      - **Unplanned nodes:** a MUL_MAT that isn't in the plan, or whose weights
        moved since planning, gets layer -1 (weights sent with the call) and one
        warning. This was chosen over the spec's "replan", because nothing in
        llama.cpp's current flow moves weights between plans.
      - **What it fixes:** the tile-map shift with `-fa off`. By construction, a
        zero-row SOFT_MAX or an eval-callback early exit can no longer shift a
        MUL_MAT's layer; neither case was run.
      - **Verified:**
        - the gate suite gives 68/68, identical to the baseline;
        - 1.4 passes in simulation and on `kriaB_L`;
        - board tbo 51/51 and 55/55, SOFT_MAX 212/212;
        - `-b 16` perplexity 3.3636, unchanged, in all four variants.
- [ ] **2.3 (optional) H3:** the `SECDA_GRAPH_STATS` writer.

## 3. Hardware flow

- [x] **3.1 `sched-sync` on v3 and v4.** Done 2026-09-28.
      - **Cause:** SECDA-Core's cross-check, not the design. `LoadWeights` Loop
        1.1's six nested loops (`loop_A`-`loop_G`, one per weight type) sit on
        `if (wgt_types == ...)` branches. HLS schedules them into shared cycles,
        and their summed costs (37) exceed the parent's span (27). The check
        failed the whole design on that one loop.
      - **SECDA-Core fix (`48da6eb`):** such a loop is now `overlapping_children`,
        with no derivable own overhead, and is reported as unverified. Labelled
        loop rows in the HLS report are now also checked.
      - **Suite-wide:** of the 28 saved schedules in SECDA-DS, the 14 that failed
        now read (bfpp_acc v3/v4, every MM2IM, FCGEMM 1/2, TCONV Z1). The 14 that
        passed are unchanged.
      - **Applied** (`SECDA_SCHED_SYNC=apply ./run.sh 1 0`, converged on pass 2):
        19 `DWAIT` sites in v3 and 23 in v4, in the generated `acc_schedule.h`
        (with `acc_schedule.json`). Loops HLS gives no single figure for
        (`--outer`, `--variable`) were left unmodelled.
      - **Largest change:** `BFPP_UNIT` Compute's inner loop costs 15 cycles per
        `vec_dot` in hardware, where simulation used 1. A `llama-cli -n 4` run's
        `HWC_X1_Compute` went from 3.8M to 61.2M cycles.
      - **Gate suite:** re-baselined; the old baseline is kept as
        `step0_pre_sched_sync`. Against the old baseline, only
        `fpga_compute_cycles`/`fpga_weight_transfer_cycles` differ (16 checks);
        pass counts, perplexity, text and warnings are identical.
- [x] **3.2 Monitor target states.** Driver-only, done 2026-09-28:
      - `Weight_Transfer_B-D` (5-7) now target 1, like A;
      - `HWC_X1_Compute` (12) targets 2 (`computeS` computing) and now reads real
        compute cycles;
      - `prf.csv`'s `fpga_compute_cycles` reads it instead of the Scheduler at
        state 31, a one-cycle handshake (1,200 cycles for a whole run).
      - **Left as they are (owner's call):** `Weight_Transfer_A-D` enter state 1
        at their first weight and never leave it, so they count cycles since then.
        `WeightLoader_A-D` (8-11) only ever report state 0. Making either
        meaningful needs `HWC_SIG` busy/idle writes and a bitstream rebuild.
- [x] **3.3 Bitstreams tracked**: `BFPP_ACC_KRIA_{3,4}_0.{bit,hwh}` and the
      `CPU_1_0` reset pair (a copy of SECDA-Sandboxed's), force-added past the
      `.gitignore` pattern as Sandboxed does. The md5s are in STATUS.md.
- [x] **3.4 `./secda load` and `run-on-board --bin` work on this repo.**
      2026-09-28 on `kriaB_L`:
      - `./secda load bfpp_acc/v3` checked the six addresses against the `.hwh`,
        deployed the pair and loaded it;
      - `./secda run-on-board bfpp_acc/v3 --bin <secda-test-backend-ops>
        --deploy <bins>:bins --env LD_LIBRARY_PATH=bins --check "grep -q '51/51
        tests passed'" -- test -b SECDA -o MUL_MAT` passed, reset the board to
        `CPU_1_0`, and left no u-dma-buf.
- [x] **3.5 Board check of section 3.** Done 2026-09-29 on `kriaB_L`, after a
      reboot. The first attempt, on 2026-09-28, hit `cma_alloc: alloc failed`
      and the board went down.
      - **Checks:** all four variants pass test-backend-ops (51/51, 55/55,
        SOFT_MAX 212/212). llama-cli text with flash attention auto and `-fa off`
        equals the board CPU's `-fa off` text. No warnings, no u-dma-buf left.
      - **Counters** for `llama-cli -n 4 "what is my name?"`, board vs synced
        simulation:
        - `HWC_X1_Compute` 80.3M vs 61.2M, board/sim **1.31** (before the sync,
          simulation had 3.8M: 21x low);
        - Load_Unit 5.35M vs 3.63M, 1.47;
        - the free-running counters about 1.41.
      - **`fpga_compute_cycles` is right on hardware:** 401,711 µs, 98% of the
        board's wall-clock compute wait (409,749 µs).
- [x] **3.6 Everything the HLS report gives is now modelled.** Done 2026-09-29.
      - **SECDA-Core `ad29364`**, two model fixes, each with a test:
        - `--variable max/min` applies to leaf loops only. On `LoadWeights` Loop 1
          ("83 ~ 1051", nested loops included) it would have written
          `DWAIT(1051)`.
        - A pipelined child loop's fill/drain (latency - II) is charged to its
          parent, once per run of the child.
      - **Applied** with `SECDA_SCHED_ARGS="--outer --variable max"`, converged on
        pass 2: 9 more constants in v3 and 11 in v4. The largest is
        `SCHED_BFPP_UNIT_Compute_L1_1` = 29, per row, since HLS flattened the
        `m` and `k` loops into one pipeline (II 16, latency 44).
      - **Not modellable:** Control_Unit, Load_Unit and Softmax_Unit L1 have
        waits inside conditionals; `LoadWeights` L1.1 has overlapping children.
      - **Effect on the board comparison: 0.2%.** For `llama-cli -n 4`,
        `HWC_X1_Compute` board/sim is 1.310 (was 1.313). The gate suite changed
        only in the cycle checks and was re-baselined (the previous baseline is
        kept as `step0_pre_outer`).
- [x] **3.7 The last 30 cycles per output.** Done 2026-09-30.
      - **RTL replay** (2026-09-29, `scripts/rtl_replay`): the packaged IP's RTL
        in xsim, fed the exact stream traffic recorded from the SystemC
        simulation of the single-tile Q2_K case. All 512 outputs were
        bit-identical to simulation and came out **46 cycles apart**, like the
        board: the cost is in the design, not the DMA or AXI.
      - **Cause.** HLS flattened the `n` and `m` loops into Loop 1.1 (attributed
        to line 91, the `n` loop), so Loop 1.1 iterates once per output, paying
        the `k` pipeline's fill each time: `(kb-1)·16 + 44 + 2`. Sched-sync put
        `SCHED_BFPP_UNIT_Compute_L1_1` (29) in the `n` body, once per row.
      - **Fix, SECDA-Core `facb566`:** when source loops between an HLS loop and
        its HLS child have no region of their own (and aren't unrolled), the
        `DWAIT` goes in the innermost of them; `--apply` removes the old one.
        Of the 28 saved schedules in SECDA-DS, only this loop in v3/v4 moves.
      - **Applied:** the `DWAIT` is now at the end of the `m` body in v3 and v4.
        No HLS rerun needed for the hardware (`DWAIT` is erased under synthesis).
      - **Board comparison:** `llama-cli -n 4` `HWC_X1_Compute` sim 79.7M vs board
        80.3M, **1.008** (was 1.310). Single-tile Q2_K compute 115.29 us vs
        117.77 us (was 41.55 us).
      - **Gate suite:** only the cycle counters differ (16 of 65 checks);
        re-baselined, previous kept as `step0_pre_flatten`. Since the sync, the
        `-b 16` perplexity runs on `SECDA-sim-x64` only (`out/baseline_mf`, local).
- [x] **3.8 Weight-transfer ratios.** Done 2026-09-30: the design is exact, the
      gap was driver time inside the counter, and the driver now keeps most of
      it out.
      - **What the counter measures.** `fpga_weight_transfer_cycles` is
        `Load_Unit` (monitor 1) in state 4: from reading the weight opcode until
        all four WeightLoaders finish.
      - **RTL replay** (`scripts/rtl_replay`, single-tile Q2_K, v3): the weight
        phase (4 x 672 words) takes 681 cycles in the RTL, 674 in simulation.
        With the streams fed one word a cycle, the hardware matches the model.
      - **Cause.** On the board, `Load_Unit` enters state 4 as soon as the opcode
        arrives on DMA 0, and then waits while the ARM programs DMAs 1-3 over
        MMIO. `LoadWeights_Inference` starts DMA 0 (opcode + its weights) first;
        `LoadWeights_Preloaded` sends the opcode, waits for it, then sets four
        offsets and starts four DMAs. In SystemC the driver takes no simulated
        time, so simulation never sees that window.
      - **Evidence: a fixed cost per call**, not a bandwidth ratio. The ratio
        falls with transfer size, as a constant does:

        | Case | Path | Calls | Sim | Board | Extra per call |
        |---|---|---|---|---|---|
        | test-backend-ops, 51 tests (all M=16, K=256) | Inference | 51 | ~37 us | 388 us | ~6.9 us |
        | single Q2_K tile | Inference | 1 | 3.39 us | ~9.2 us | ~5.8 us |
        | `llama-cli -n 4` | Preloaded | 600 | 3.63M cyc | 5.35M cyc | ~14.3 us |

        The board figures are from the 2026-09-28/29 `kriaB_L` runs
        (`hardware_automation/results/bfpp_acc_v3/20260928T212011Z`, TODO 3.5);
        the single-tile board value is only kept as a ratio. The call count is
        from a traced sim run (600 sends on DMA 1).
      - **Driver fix (`5695980`):** DMAs 1-3 are started before the opcode DMA,
        on both weight paths, in all four drivers. kria2, old vs new driver in
        one session: every check passes (MUL_MAT 51/51 and 55/55, SOFT_MAX
        212/212 on v4) and `llama-cli` text is identical. Load_Unit cycles:

        | Run | Old | New | Sim |
        |---|---|---|---|
        | MUL_MAT, v3 driver | 87,355 | 38,181 | ~7,400 (estimate) |
        | MUL_MAT, v3 batches | 98,505 | 43,398 | |
        | `llama-cli -n 4`, v3 driver | 5,451,961 | 5,029,119 | 3,632,400 |
        | `llama-cli -n 4`, v4 batches | 5,392,652 | 5,027,229 | 3,632,400 |

        Wall-clock time is unchanged beyond run-to-run noise: the ARM does the
        same work in a different order, so the ~8.6 ms saving predicted above
        didn't materialise. `HWC_X1_Compute` is 80,342,232 in every run.
- [ ] **3.9 The rest of the preloaded weight gap (optional).** `llama-cli`
      board/sim is still 1.38 (~9.3 us per load). DMA 0 carries the opcode and
      then tile A as two transfers, and a simple-mode AXI DMA takes no second
      transfer until the first completes, so the ARM waits for the opcode and
      programs A while the Load_Unit counts. Closing it needs the opcode and
      tile A in one transfer (a header slot before each preloaded tile A in the
      preload layout), or the opcode on its own stream (a hardware change).
      Affects only the counter, not the output.

## 4. Benchmark suite — done 2026-09-30

- [x] **4.1 Config shape.** `benchmark_suite.sh` reads `boards.KRIA` (or
      `boards.$SECDA_BOARD`), the entry `./secda load` uses.
- [x] **4.2 Root on the board.** A non-root `board_user` gets its run scripts
      started with `sudo bash -lc` (passwordless sudo needed); root runs as is.
- [x] **4.3 Runtime entries.** `configs/runtimes/runtime_dict_v6.json`:
      `bfpp.v3`, `bfpp.v3b`, `bfpp.v4`, `bfpp.v4b`, `cpu.v6`, tagged
      `<hw>_KRIA_<version>_<opt>` as `parse_results.py` expects;
      `configs/exp_configs/MobileLLMQ2_v6.json` selects them and
      `e2e_exp_config_gen.py` now loads the v6 dictionary.
- [x] **4.4 Pushbullet token** removed from `benchmark_suite.sh`: read from
      `config.json`'s `push_bullet_token` or `$PUSHBULLET_TOKEN`, and skipped if
      neither is set. (It is still in history: 6.1.)
- [x] **4.5 One full suite run**, on kria2 (not `kriaB_L`), `-b -c -l -p` with
      `SECDA_DMA_IN_BUF_MB=64`: `llama_perf.csv` for all five runtimes from
      both `llama-cli` and `llama-bench`; 30/30 nodes preloaded, no warnings.
      Results in `benchmark/results/v6_migration/` (gitignored):

      | Runtime | `llama-bench` t/s | J | `llama-cli` s |
      |---|---|---|---|
      | v3 driver | 6.43 | 224.7 | 1.40 |
      | v3 batches | 6.42 | 224.3 | 1.39 |
      | v4 driver | 6.35 | 229.9 | 1.42 |
      | v4 batches | 6.29 | 232.2 | 1.45 |
      | CPU | 5.90 | 222.6 | 1.26 |

      Also fixed on the way: the run scripts now take their board path from
      their own location (was `/home/ubuntu/Workspace/secda_llm`), reset with
      `CPU_1_0` (the tracked pair; was `CPU_KRIA_1_0`), put the runtime folder
      on `LD_LIBRARY_PATH` (the deploy puts the libraries there, not in
      `bin/`), and the suite syncs `hardware_automation/bitstreams/KRIA/` to
      the board. Models are read from `<board_dir>/models`.

## 5. Build and repo hygiene

- [x] **5.1 Ignore the run outputs.** `secda_profile.json` and `dma*.csv` are in
      `.gitignore` and in the fork excludes `setup.sh` writes. Done 2026-09-30.
- [ ] **5.2 Upstream presets** (`x64-linux-gcc-*` and the others). Configure one to
      confirm it stops at "GGML_SECDA needs a design". Then either set
      `GGML_SECDA=OFF` in them or hide them.
- [x] **5.3 KV260 CMA headroom.** Done 2026-09-30 as a run-time option. The
      first suite run failed at init: four 192 MB input buffers didn't fit a
      fragmented CMA (949 MB free, the fourth `create udmabuf6` refused).
      `SECDA_DMA_IN_BUF_MB` now overrides `DMA_IN_BUF_SIZE` in all four
      drivers (`dma_in_buf_size()`), and preloading stops at whatever fits
      (`alloc_layer` checks `dma_wgt_size()`); the suite passes it through.
      MobileLLM-125M preloads 30/30 with 16 MB, same text. The default stays
      192 MB for the bigger models. Sizing it from the model automatically
      would need the plan before `initACC`; not done.
- [ ] **5.4 armv7 / Z1 preset.** `SECDA-armv7-debug` hasn't been built since the v6
      port and there is no Z1 hardware variant. Build it once or remove it.
- [ ] **5.5 Phase 7, SECDA-Core CMake tooling (optional, in SECDA-Core).** Add a
      `build_system: "cmake"` config key, so that `./secda build` runs
      `cmake --build --preset` and `./secda list` reads `hw_params.json` without
      Bazel; `revamp_scan.py` should skip `.bazelrc` for CMake repos.

## 6. Owner decisions

- [ ] **6.1 Rotate the Pushbullet tokens:** the one in `benchmark_suite.sh`'s history
      and the one in the local `config.json`.
- [ ] **6.2 Branch.** SECDA-DS's `CLAUDE.md` names `v3_core_upgrade`, but the work is
      on `v3_core_upgrade_wip`. Fast-forward `v3_core_upgrade`, or update the docs.
- [ ] **6.3 NaN semantics** (1.3): upstream's failure is now the default, with `SECDA_TBO_NAN_PASS=1` as the opt-out. Confirm, or flip the default.
