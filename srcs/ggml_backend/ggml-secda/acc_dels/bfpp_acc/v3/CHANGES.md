# bfpp_acc/v3 — changes

## 2026-09-30 — weight DMAs 1-3 started before the opcode DMA

**Why.** SECDA-LLM TODO 3.8: `fpga_weight_transfer_cycles` (the Load_Unit in
state 4) starts when the weight opcode arrives on DMA 0. Both weight paths
started DMA 0 first, so on the board the counter also ran while the ARM
programmed DMAs 1-3. The RTL replay showed the design itself matches
simulation.

| Where | Change |
|---|---|
| `driver*/acc_driver_mt.h` `LoadWeights_Inference` | DMAs 1-3 started before DMA 0 (opcode + A weights + compute opcode) |
| `driver*/acc_driver_mt.h` `LoadWeights_Preloaded` | DMAs 1-3 programmed and started before the opcode send; profiler slots 7/8 for that segment |

**Verified.** Simulation: the gate suite matches the baseline (SystemC charges
no time to the driver). kria2, old vs new driver in one session: every check
passes, `llama-cli` text is identical, and the Load_Unit counter falls by about
half for test-backend-ops and 8% for `llama-cli -n 4` (board/sim 1.50 to 1.38).
Wall-clock time doesn't change beyond run-to-run noise: the CPU does the same
work, only in a different order. The rest of the preloaded gap is DMA 0 itself,
which has to finish the opcode before the ARM can program tile A.

## 2026-09-30 — flattened Compute loop charged per output

**Why.** SECDA-LLM TODO 3.7: the RTL replay showed outputs 46 cycles apart, as on
the board. HLS flattens Compute's `n` and `m` loops into Loop 1.1, so one Loop 1.1
iteration is one output, but sched-sync had put `SCHED_BFPP_UNIT_Compute_L1_1` in
the `n` body, charging it once per row. SECDA-Core `facb566` places a flattened
loop's `DWAIT` in the innermost flattened source loop.

| Where | Change |
|---|---|
| `accelerator/bfp_processor_threads.sc.h` | `DWAIT(SCHED_BFPP_UNIT_Compute_L1_1)` moved from the `n` body to the end of the `m` body (value unchanged, 29) |
| `accelerator/acc_schedule.{h,json}` | regenerated; timestamps only |

**Verified.** Simulation: the gate suite differs from its previous baseline only in
the cycle counters; text, pass counts and perplexity are identical (re-baselined,
the old one kept as `step0_pre_flatten`). Against the board, `llama-cli -n 4`
`HWC_X1_Compute` is 79.7M in simulation vs 80.3M (board/sim **1.008**, was 1.310),
and the single-tile Q2_K compute is 115.29 us vs 117.77 us (was 41.55 us).
Synthesis is unaffected (`DWAIT` is erased under `__SYNTHESIS__`), so the
bitstreams stand.

## 2026-09-29 — sched-sync --outer --variable max

**Why.** SECDA-LLM TODO 3.6: the first sync modelled only loops with a fixed
per-iteration figure. SECDA-Core `ad29364` makes the outer and variable cases
safe: the variable range applies to leaf loops only, and pipelined children's
fill/drain is charged to the parent.

| Where | Change |
|---|---|
| `accelerator/*.sc.h`, `acc_schedule.{h,json}` | `SECDA_SCHED_ARGS="--outer --variable max"`, converged on pass 2: 9 more constants, e.g. `SCHED_BFPP_UNIT_Compute_L1_1` 29 (per row of the flattened `m`/`k` pipeline, II 16, latency 44) |

**Verified.** Simulation: the gate suite differs from its previous baseline only
in the cycle counters (re-baselined). The board comparison barely moves: for
`llama-cli -n 4`, `HWC_X1_Compute` board/sim is 1.310 (was 1.313). The rest is
about 30 cycles per output that the HLS schedule doesn't show (TODO 3.7).

## 2026-09-28 — simulated timing synced to HLS; monitor targets; fpga_compute_cycles

**Why.** SECDA-Core's `sched-sync` refused this design (a false error, fixed in
SECDA-Core `48da6eb`), so the simulation's `DWAIT` timing had never been checked
against HLS. `prf.csv`'s `fpga_compute_cycles` read the Scheduler monitor at
state 31, a one-cycle handshake, and monitors 5-7 and 12 had no target state.

| Where | Change |
|---|---|
| `accelerator/*.sc.h` | `sched_sync.py --apply` (converged on pass 2): 19 `DWAIT(SCHED_...)` sites; generated `accelerator/acc_schedule.h` and `acc_schedule.json`, included from `acc_config.sc.h`. Erased under `__SYNTHESIS__`: the hardware is unchanged |
| `driver*/driver_interface.h` | target states: `Weight_Transfer_B-D` 1, `HWC_X1_Compute` 2 (computing) |
| `driver*/acc_driver.h` | `fpga_compute_cycles` = `HWC_X1_Compute` (monitor 12), was the Scheduler (monitor 3) |

**Effect.** Simulated compute is about 16x what it was: HLS takes 15 cycles per
`vec_dot` iteration, where simulation used 1 (a `llama-cli -n 4` run's
`HWC_X1_Compute` went from 3.8M to 61.2M cycles). `fpga_compute_cycles` in
`prf.csv` is not comparable with earlier runs.

**Verified.** Simulation: the gate suite differs from its previous baseline only
in `fpga_compute_cycles`/`fpga_weight_transfer_cycles` (pass counts, perplexity,
generated text and warnings identical), and was re-baselined. KV260 (`kriaB_L`, 2026-09-29): test-backend-ops and llama-cli
unchanged. For `llama-cli -n 4`, `HWC_X1_Compute` is 80.3M cycles, 1.31x the
synced simulation's 61.2M (it was 21x before the sync), and `fpga_compute_cycles`
is 98% of the wall-clock compute wait.

## 2026-09-28 — explicit MUL_MAT layers (driver hygiene)

**Why.** Shared with v4, where SOFT_MAX nodes between the MUL_MATs shifted the
driver's positional layer counter and MUL_MATs ran with another layer's
weights (SECDA-LLM TODO 2.1, 2.2). v3 has no SOFT_MAX, so its results don't
change; it gets the same explicit layers and bounds checks.

| Where | Change |
|---|---|
| `driver*/acc_container.h` | `reset()` clears `layer_preloaded` (`assign`; `resize` kept stale flags); `is_preloaded(l)` with bounds checks; `set_tile_maps(l, ...)` stores tile maps by layer; `alloc_layer` refuses `l < 0` |
| `driver*/driver_interface.h` | `preloadWeights` files the tile maps under its layer (was `push_back`); new `setLayer(int)` |
| `driver*/acc_driver_mt.h` | `LoadWeights` uses `is_preloaded(layer)` (was `alloced_layers >= layer && layer_preloaded[layer]`); tile maps read with `.at(layer).at(m)` |
| `driver*/acc_driver.h` | the `EntryMM` flag reads use `is_preloaded` |
| `driver*/acc_driver_connector.{h,cc}` | `setLayer` exported |
| backend `secda_planner.cpp`, `ops_support.*`, `ggml-secda.cpp` | each planned MUL_MAT's layer is its MUL_MAT ordinal; `secda_planner_set_layer` calls `setLayer_T` before each MUL_MAT (-1, weights sent with the call, if unplanned or its weights moved) |

**Verified.** Simulation: the gate suite is unchanged (68/68). KV260
(`kriaB_L`): MUL_MAT 51/51 (55/55), `-b 16` perplexity 3.3636 unchanged,
llama-cli text (flash attention auto and off) equal to the board's CPU-only
`-fa off`, in both drivers.

## 2026-09-28 — secda_profile.json (SECDA-Core profiler)

**Why.** SECDA-Core's profile parsers read `secda_profile.json`; SECDA-LLM only
wrote `prf.csv` and read two of the 13 hardware counters, by index.

| Where | Change |
|---|---|
| `driver*/driver_interface.h` | the 13 counters are named (`name_hwc`), in register order |
| `driver*/acc_driver.h` | under `ACC_PROFILE`, every counter is sampled after each accelerator call (after the driver-time measurement) and summed |
| `driver*/acc_container.h` | at exit, next to `prf.csv`: `secda_profile.json` with the same timings plus the named counter totals (SECDA-Core `write_secda_profile_us`, `551f407`; `$SECDA_PROFILE_PATH` overrides the path) |

**Verified.** Simulation: the gate suite is unchanged (68/68 against the
pre-migration baseline); the JSON's timings equal `prf.csv`. KV260 (`kriaB_L`):
the same, with real counter totals (Load_Unit cycles x 5 ns = prf.csv's
`fpga_weight_transfer_cycles`).

## 2026-09-28 — SECDA-Core axi_support v6, HWC order, first KV260 build

**Why.** SECDA-LLM moved onto the suite's SECDA-Core (SECDA-DS
docs/secda-llm-migration-plan.md). v5's multi_dma took fixed buffer addresses;
v6 allocates every buffer through u-dma-buf. The HWC reset was declared before
the monitors, so SECDA-Core v6's hwc_ctrl read the wrong registers.

| Where | Change |
|---|---|
| `accelerator/acc_config.sc.h`, `driver*/acc_container.h`, `driver*/systemc_binding.h` | `axi_support/v5/axi_api_v5.h` -> `axi_support/v6/axi_api.h`; fixed `dma_in*`/`dma_out*` removed |
| `driver*/driver_interface.h`, `driver*/acc_driver.h` | `s_mdma mdma1(4, dma_addrs)` + `mdma1.alloc_buffers(DMA_IN_BUF_SIZE, DMA_OUT_BUF_SIZE)` (after `sysC_binder` in simulation); `dma_addrs_in/out` removed |
| `accelerator/acc.sc.h` | `HWC_Reset` declared after the 13 monitors |
| `hw_params.json` | `BFPP_ACCv3_0_KRIA`: KRIA_dma_4_hp_4_ctrl_hwc.tcl, 200 MHz |
| SECDA-Core `e1c1abc` | v6 regains `dma_sync_mem`, `get_send/recv_bandwidth`, `profile_reset`, which these drivers call |

**Verified.**
- Simulation reproduces the v5 baseline exactly (test-backend-ops, perplexity 3.3449, llama-cli tokens and cycle counters), driver and driver_batches.
- KV260 (`kriaB_L`), timing-tolerated WNS -0.307 ns (6.1%): MUL_MAT 51/51; perplexity 3.3625, identical to the board's CPU-only build; 30/30 nodes preloaded.
