# bfpp_acc/v3 — changes

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
