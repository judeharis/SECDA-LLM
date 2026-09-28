# bfpp_acc/v4 — changes

## 2026-09-28 — SECDA-Core axi_support v6, HWC order, first KV260 build

**Why.** SECDA-LLM moved onto the suite's SECDA-Core (SECDA-DS
docs/secda-llm-migration-plan.md). v5's multi_dma took fixed buffer addresses;
v6 allocates every buffer through u-dma-buf. The HWC reset was declared before
the monitors, so SECDA-Core v6's hwc_ctrl read the wrong registers.

| Where | Change |
|---|---|
| `accelerator/acc_config.sc.h`, `driver*/acc_container.h`, `driver*/systemc_binding.h` | `axi_support/v5/axi_api_v5.h` -> `axi_support/v6/axi_api.h`; fixed `dma_in*`/`dma_out*` removed |
| `driver*/driver_interface.h`, `driver*/acc_driver.h` | `s_mdma mdma1(4, dma_addrs)` + `mdma1.alloc_buffers(DMA_IN_BUF_SIZE, DMA_OUT_BUF_SIZE)` (after `sysC_binder` in simulation); `dma_addrs_in/out` removed |
| `accelerator/acc.sc.h` | `HWC_Reset` declared after the 14 monitors |
| `hw_params.json` | `BFPP_ACCv4_0_KRIA`: KRIA_dma_4_hp_4_ctrl_hwc.tcl, 200 MHz |
| SECDA-Core `e1c1abc` | v6 regains `dma_sync_mem`, `get_send/recv_bandwidth`, `profile_reset`, which these drivers call |

**Verified.**
- Simulation reproduces the v5 baseline exactly (test-backend-ops, perplexity 3.3449, llama-cli tokens and cycle counters), driver and driver_batches.
- KV260 (`kriaB_L`), timing-tolerated WNS -0.217 ns (4.3%): MUL_MAT 51/51 SOFT_MAX 212/212; perplexity 3.3625, identical to the board's CPU-only build; 30/30 nodes preloaded.
