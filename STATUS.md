# SECDA-LLM — Status

**Objective:** integrate llama.cpp with the SECDA design methodology, so that LLM
accelerators can be designed, simulated and deployed inside a real inference
framework. See [README.md](README.md).

**Last checkpoint:** 2026-09-28

---

## Headline

**SECDA-LLM now builds against the suite's SECDA-Core (axi_support v6).** Both
bfpp_acc v3 and v4 are verified on the KV260, with bitstreams built through
SECDA-Core's flow. On `kriaB_L`:
- `secda-test-backend-ops` passes MUL_MAT, and on v4 SOFT_MAX;
- `secda-llama-cli` runs all 30 MUL_MAT nodes on the accelerator, and its greedy
  text is identical to simulation's.

Perplexity at `-b 16 -ub 16`, with all 30 MUL_MATs on SECDA, gives 3.3636 on the
board, within 0.12% of the board's CPU-only 3.3596. At `-b 128` only 1 of the 30
runs on SECDA. Simulation at `-b 16` gives 3.3429 (x86 CPU 3.3637). That gap is
not the model: replaying 60 real board MUL_MAT calls through the SystemC model
gives bit-identical outputs (310,464 of 310,464). The activations differ between
ARM and x86 from the first layer, through CPU-op rounding ([TODO.md](TODO.md)
1.5). With `-fa off`, v4 now runs SOFT_MAX on the accelerator end to end: the
generated text equals the CPU-only build's, in simulation and on the board (TODO
2.2 fixed the layer shift that had produced garbage).

**The llama.cpp fork is down to backend registration:** 3 files and 17 lines
over upstream `06938ac12`.

**All the work is committed and pushed** on `v3_core_upgrade_wip`
(GitHub at `ce9d85c` on 2026-09-28). The
previous checkpoint's risks are closed: the work was untracked, had no remote copy,
and v3 was mid-restructure.

---

## Where things stand

| Area | State |
|---|---|
| Branch | `v3_core_upgrade_wip`, pushed (`origin` at `ce9d85c` on 2026-09-28). `v3_core_upgrade` (`a9b6ab5`) and `main` (`183376a`) have not moved; both are ancestors of the WIP branch. |
| bfpp_acc v3 (`driver`, `driver_batches`) | On axi_support v6, with `HWC_Reset` declared after the 13 monitors. `BFPP_ACCv3_0_KRIA` is built at 200 MHz, timing-tolerated (WNS -0.307 ns, 6.1%), and `verified_accurate` on the KV260. `secda_profile.json` names all 13 counters, but the driver sets target states only for monitors 0-4, so only those have meaningful totals. |
| bfpp_acc v4 (v3 plus a SOFT_MAX unit) | The same. `BFPP_ACCv4_0_KRIA` is timing-tolerated (WNS -0.217 ns, 4.3%). 14 named counters; targets are set for 0-4 and 13 (`Softmax_Unit`). |
| legacy v1, v2 | Retired to `acc_dels/bfpp_acc/legacy/` (`9c6758f`). Still axi_support v5 and not built; configure refuses them. `v3/accelerator_alt` was dropped. |
| `example_acc/bfp_softmm` | Moved to SECDA-Sandboxed `experiments/bfp_softmm`, which tracks it there (`1627502`; v6 port `bf3cc99`). `example_acc/` keeps three perplexity logs. |
| llama.cpp fork | `625ec1d28` on `v3_core_upgrade`, pushed. `scripts/check_fork_markers.py` passes. |
| Backend | Plans weight preloads from the scheduler's `graph_optimize` calls (`secda_planner.{h,cpp}`), with no llama.cpp hook. |
| Tools | `secda-test-backend-ops`, `secda-llama-bench` and `secda-llama-cli` are built from upstream sources plus `srcs/tools/patches`. `compile_send_kria.sh` deploys them under the old board names. |
| SECDA-Core | `third_party/secda_core` is pinned at `551f407`. Inside SECDA-DS the suite's `SECDA-Core/` is used instead. |
| Bitstreams | `hardware_automation/bitstreams/KRIA/BFPP_ACC_KRIA_{3,4}_0.{bit,hwh}` and `CPU_1_0.{bit,hwh}`, tracked (force-added). |
| Z1 / armv7 | The `SECDA-armv7-debug` preset is kept but has not been built since the v6 port. The Z1 is out of scope: LLM weights don't fit its 128 MB CMA. |

---

## What was verified

All records are dated 2026-09-28 and kept in each design's `test_status.json`
(append-only) and `CHANGES.md`, under
`srcs/ggml_backend/ggml-secda/acc_dels/bfpp_acc/{v3,v4}/`.

- **Simulation, v6 port against the v5 baseline.** The baseline was taken first, on
  the same WIP sources. The results are identical to it:
  - `test-backend-ops -b SECDA`: MUL_MAT 51/51 (`driver_batches` 55/55), and on v4
    SOFT_MAX 212/212;
  - `llama-perplexity`, MobileLLM-125M-HF Q2_K (`-c 128 --chunks 2 -t 1 -b 128`):
    3.3449 +/- 0.59780, equal to the x86 CPU-only build. The input was the first
    4096 bytes of llama.cpp's `README.md` at `06938ac12`, not wikitext-2 (see
    README, *Checks used for the v6 migration*);
  - `llama-cli` greedy tokens and the simulated cycle counters.
- **Pass counts.** They come from `test-backend-ops` with the owner's changes
  (`secda-test-backend-ops` since the minimal fork), which reports a NaN output
  but doesn't fail it. The simulation logs have no `NaN at index` lines. The
  board logs were not kept.
- **Minimal fork, planner and tools.** All 68 checks of the simulation gate suite
  (Gates A-C of the minimal-fork spec) are identical to the pre-change baseline.
  They were unchanged again after the profiling change. The checks include
  perplexity at `-b 16 -ub 16`: 30/30 MUL_MAT nodes on SECDA and preloaded, with
  3.3429 +/- 0.59671 in all four variants. The gate outputs are in the gitignored
  `out/baseline_mf/` on this machine.
- **KV260 (`kriaB_L`): v3 and v4, `driver` and `driver_batches`:**
  - MUL_MAT 51/51 (55/55), and on v4 SOFT_MAX 212/212;
  - perplexity 3.3625 +/- 0.60376, identical to the board's CPU-only build (x86
    gives 3.3449 because CPU-op rounding differs between ARM and x86). At `-b 128`
    the driver's size check (`N x kb <= SUP_KNB`, 512) accepts only 1 of the 30
    MUL_MAT nodes per chunk, so this mostly compares CPU with CPU. The
    `-b 16 -ub 16` run puts all 30 on SECDA: 3.3636 +/- 0.60469 in all four
    variants, against 3.3596 for the board CPU (see `test_status.json`);
  - `llama-cli` generated text identical to simulation, with 30/30 MUL_MAT nodes
    on SECDA and preloaded. The token check is against simulation, not against
    the board's CPU. The board's CPU-only build diverges from both at the first
    generated token. The migration plan's Phase 5 criterion was "tokens match the
    CPU run";
  - v4's SOFT_MAX unit is covered only by test-backend-ops. In the model runs,
    auto flash-attention removes SOFT_MAX, and v4 plans the same 30 MUL_MAT nodes
    as v3;
  - `llama_perf.csv` and `tbo.csv` written, and no u-dma-buf left behind.
- **The bitstreams.** The `.hwh` addresses match `acc_config.sc.h`, and the IO and
  utilization gates pass.
- **Profiling.** `secda_profile.json` timings equal `prf.csv`.
  - The totals of the monitors with target states (0-4) are real on the board:
    Load_Unit's 9.6e6 cycles x 5 ns equals `fpga_weight_transfer_cycles`.
  - Monitors 5-12 have no target state. In the v3 board profile, `HWC_X1_Compute`
    reads 0 and `Weight_Transfer_B`-`D` read 5857 each, against 1.08e8 for
    `Weight_Transfer_A`.

The board tests used ad hoc scripts, which are not in the repo. The plan and
verification detail is in SECDA-DS `docs/secda-llm-migration-plan.md` and
`docs/secda-llm-minimal-fork-spec.md`.

---

## Open items

The actionable follow-ups are in [TODO.md](TODO.md).

- **Pushbullet tokens.** `benchmark/benchmark_suite.sh` hard-codes a token. It has
  been in history since `4769480` (2026-06-03) and is already on `origin/main`. A
  different token was in the local `config.json`. Rotate both, then take the token
  out of the script.
- **KV260 CMA fragments after long uptime.**
  - The KV260 build takes 4 x (192 MB + 16 MB) = 832 MB of the board's ~1 GB CMA.
  - After `kriaB_L` had been up 38 h, the u-dma-bufs failed to allocate, even for
    a binary that had passed hours earlier. A reboot fixed it.
  - Smaller per-DMA buffers (`DMA_IN_BUF_SIZE` in `acc_config.sc.h`) would make
    this less likely.
- **Accelerator vs CPU near-ties.** The accelerator's greedy text can differ from
  the x86 CPU-only build by one token at near-ties, because of its BFP arithmetic.
  It happens identically in simulation and on the board, before and after the
  migration, so it is not a migration regression. The board's own CPU-only build
  is a different case. Its greedy text diverges from both the accelerator and x86
  at the first generated token. That is presumably the same ARM vs x86 CPU rounding
  seen in perplexity. So on the board, tokens are checked against simulation.
- **SECDA-Core tooling for CMake (optional).** `./secda build`, `list` and
  `gen-vscode` still call Bazel.
  `hw-gen`, the HLS/HLX builds, `sched-sync`, `load` and `run-on-board --bin`
  all work for this repo (TODO section 3).
- **Simulated timing is synced to HLS** (TODO 3.1, 3.5). HLS takes 15 cycles per
  `vec_dot`, where simulation used 1. For `llama-cli -n 4`, the board's compute
  counter is 1.31x simulation's, down from 21x before the sync; weight transfer
  is 1.47x. The loops the sync couldn't model account for the rest (TODO 3.6).
- **Monitors 4-11 aren't busy counters.** `Weight_Transfer_A-D` never leave
  their busy state after the first weight; `WeightLoader_A-D` never report one.
  Left as they are: fixing them needs a bitstream rebuild (TODO 3.2).
- **The benchmark suite can't run the new builds yet.** None of the migration's
  board tests used it. Before the first suite run:
  - **Runtime entries.** The configs name pre-migration bitstreams, e.g.
    `BFPB_Q3_v1` in `benchmark/configs/exp_configs.sh` and `configs/runtimes/`. Add
    entries for `BFPP_ACC_KRIA_3_0` and `BFPP_ACC_KRIA_4_0`.
  - **Config shape.** `benchmark_suite.sh` reads a pre-SECDA-Core `config.json`
    shape: `board_user`, `board_hostname`, `board_port` and `board_dir` at the
    top level. `config.json` keeps them under `boards.KRIA`, so the suite stops
    with "Error: board_port must be numeric". Read `.boards.KRIA.*` instead, and
    update `benchmark/README.md`.
  - **Root.** The suite starts the run scripts over ssh as `board_user` (`ubuntu`)
    without sudo. The scripts load bitstreams, write `/dev/u-dma-buf-mgr` and run
    the binaries without sudo, so they need root.
- **Bitstreams are tracked** (TODO 3.3), force-added past the `.gitignore`
  pattern as SECDA-Sandboxed does. They are timing-tolerated builds, and a
  rebuild won't be bit-identical.
  - md5 of the pairs:
    - `BFPP_ACC_KRIA_3_0`: `.bit` `cac1729e940a7cd8e3f04f315d7bc60e`, `.hwh`
      `35c8cdcae15cfcc15b0104534f0d3dd4`;
    - `BFPP_ACC_KRIA_4_0`: `.bit` `86ca50aa85185c5e109f8728000ab8a2`, `.hwh`
      `4100a5471843759366b5ff9540cbee97`.
  - `CPU_1_0` is a copy of SECDA-Sandboxed's reset pair (`.bit`
    `ba1345689058644eaa950d167e1cdd25`), the one used on the board.
- **Explicit layers don't replan.** A planned MUL_MAT whose weights moved since
  planning is sent with the call, with a warning, rather than replanned (TODO
  2.2). Nothing in llama.cpp's current flow moves weights between plans.
- **Small:**
  - `secda_profile.json` and `dma<N>.csv` are in neither `.gitignore` nor the
    fork excludes that `setup.sh` writes.
  - The upstream-style presets (`x64-linux-gcc-*`) select no design, so their
    configure should stop at "GGML_SECDA needs a design". This is read from the
    CMake, not run.

---

## Next session

1. Work through [TODO.md](TODO.md): section 3 (hardware flow) or 4 (benchmark
   suite).
2. Rotate the Pushbullet tokens.
3. Decide which branch carries the work. SECDA-DS's `CLAUDE.md` names
   `v3_core_upgrade`, but the work is on `v3_core_upgrade_wip`.
   `v3_core_upgrade` can fast-forward to it.
4. Fix the benchmark suite (see Open items), then run it on the KV260:
   - add runtime entries for `BFPP_ACC_KRIA_{3,4}_0`;
   - make `benchmark_suite.sh` read `.boards.KRIA.*` from `config.json`;
   - run the board-side scripts as root.
5. Consider smaller per-DMA buffers for the KV260.

---

## Key files

| Path | Role |
|---|---|
| [README.md](README.md) | Layout, setup, presets, KV260 flow and fork policy |
| [TODO.md](TODO.md) | Follow-up work from the migration, by priority |
| `srcs/ggml_backend/ggml-secda/` | The backend, including `secda_planner.{h,cpp}` |
| `srcs/ggml_backend/ggml-secda/acc_dels/bfpp_acc/{v3,v4}/` | The designs, `hw_params.json`, `test_status.json`, `CHANGES.md` |
| `srcs/tools/` | The `secda-*` tools: patches and `refresh_patch.sh` |
| `scripts/check_fork_markers.py`, `scripts/llama_fork_base` | Fork policy check |
| `CMakePresets.json` | The SECDA presets |
| `benchmark/scripts/compile_send_kria.sh` | KV260 build and deploy |
| `plan.md`, `docs/` | The owner's design notes |

---

## Session log

_Most recent 10 entries. Older entries roll into `status-archive/<YYYY>.md`._

### 2026-09-27/28 — SECDA-Core v6 migration, KV260 bring-up, minimal fork
Carried out SECDA-DS's migration plan, following the owner's four decisions of
2026-09-27:
- **WIP.** Committed the WIP as six labelled snapshots (`9889372`..`fd8515b`).
  Retired v1/v2 to `legacy/` and dropped `v3/accelerator_alt` (`9c6758f`).
- **v6 port.** Ported v3 and v4, both drivers each, to axi_support v6 against v5
  baselines taken first (`ee990bb`). Moved `HWC_Reset` after the monitors
  (`bea4143`).
- **Hardware.** Added `hw_params.json` (`084e37b`), built `BFPP_ACC_KRIA_{3,4}_0`,
  and verified both on `kriaB_L` against simulation and test-backend-ops
  (`9070914`, `fa22f3a`).
- **Fork.** Cut the llama.cpp fork to backend registration (`625ec1d28`). Planning
  moved into the backend (`875e34b`, `46e761f`) and the tool changes into
  `srcs/tools` patches (`6353559`). `setup.sh` and the marker checker came in
  `79bed9d`.
- **Profiling.** Added `secda_profile.json` with named counters (`6b5c410`).
- **bfp_softmm.** Moved it to SECDA-Sandboxed (`6727d25`).

`kriaB_L` needed a reboot after 38 h up (CMA fragmentation). README and STATUS were
rewritten to match.

### 2026-09-10 — first checkpoint (inspection only)
Created this file as the first use of the `project-checkpoint` skill on a repo other than AMD.
Written from `plan.md`, the working tree and `git log` — nothing here was observed running. The
finding that matters: 49 dirty files with the active work (`bfpp_acc/v4/`, `example_acc/`,
`plan.md`) entirely untracked and no remote copy.
