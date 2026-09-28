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
- [ ] **1.4 v4's SOFT_MAX unit end to end. Blocked on 2.2.** Checked 2026-09-28 in
      simulation with `secda-llama-cli -fa off`:
      - v3 generates text identical to the CPU-only build.
      - v4 and v4-batches (60 SECDA nodes: 30 MUL_MAT plus 30 SOFT_MAX) generate
        garbage ("amovereverevere...") with **no** `SECDA WARNING`, so the desync
        check misses it.
      - *Done when:* after 2.2, the v4 `-fa off` tokens equal the CPU-only build's, in
        simulation and on the board.
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

These bugs were kept for parity during the migration (minimal-fork spec §6, risk 4). The
planner reports three of them as `SECDA WARNING: layer desync`. Make each fix a
separate commit, and re-run Gate A (a)-(c) after each one.

- [ ] **2.1 H1, driver hygiene** (host-side only; all 4 driver variants):
      - `acc_container.h`: `layer_preloaded.assign(500,false)` → size it from the plan;
      - add `is_preloaded(l)` with a bounds check, and use it at the `EntryMM` flag
        reads instead of `alloced_layers >= layer && ...`;
      - `acc_driver_mt.h`: `.at(m)` instead of `[m]`.
      Fixes stale `layer_preloaded` flags and the unbounded read at 500+ layers.
- [ ] **2.2 H2, explicit layers:**
      - the connector gets `setLayer(int)`;
      - the planner gives each MUL_MAT its ordinal and calls `setLayer` before each
        one;
      - replan and warn if `src0->data` differs from the planned layer.
      Fixes the v4 SOFT_MAX tile-map shift with `-fa off`, the zero-row SOFT_MAX desync
      and the eval-callback early-break desync.
      *Extra gate:* 1.4.
- [ ] **2.3 (optional) H3:** the `SECDA_GRAPH_STATS` writer.

## 3. Hardware flow

- [ ] **3.1 `sched-sync` fails on v3 and v4.** The error is "BFPP_UNIT_LoadWeights
      Loop 1.1: own overhead computes as -10 cycles (span 27, nested 37)", in
      `hardware_automation/generated/BFPP_ACC_KRIA_{3,4}_0/sched_apply_1.log`.
      - Decide first whether SECDA-Core's cross-check misreads the HLS report (nested >
        span points to a pipelined or flattened child loop) or the design's loop
        structure is at fault.
      - If it is the check, fix it in `SECDA-DS/SECDA-Core` on `main`.
      - *Done when:* `acc_schedule.h` is generated for both designs and the gate
        suite still passes.
- [ ] **3.2 Monitor target states.** Only monitors 0-4 have target states (v4 also
      13, `Softmax_Unit`). Set targets for 5-12 in `D/v{3,4}/accelerator/acc.sc.h` or
      document why they have none; for example, `HWC_X1_Compute` reads 0 on the
      board. This is the migration plan's Phase 2 item.
      - It needs a bitstream rebuild if the monitors change (at most 2 HLS and 2 HLX
        runs at once).
      - Re-test on the board afterwards.
- [ ] **3.3 Track the bitstreams.** The plan's Phase 4 asked for it, and Sandboxed
      does it. Track `hardware_automation/bitstreams/KRIA/BFPP_ACC_KRIA_{3,4}_0.{bit,hwh}`,
      plus the KV260 `CPU_1_0` reset pair, with a `.gitignore` exception.
      - The pairs are timing-tolerated builds that can't be reproduced bit for bit.
        Their md5s are in STATUS.md.
      - Alternatively, write down why they stay untracked.
- [ ] **3.4 Try `./secda load` and `run-on-board --bin` on this repo.** The board
      tests used `load_bitstream.py`. They need `secda_init_path` and `bitstream_dir`
      set in `config.json`.

## 4. Benchmark suite (can't run the new builds today)

- [ ] **4.1 Config shape.** `benchmark/benchmark_suite.sh` reads `board_user`,
      `board_hostname`, `board_port` and `board_dir` from the top level of
      `config.json`, but those keys now live under `boards.KRIA`. Read
      `.boards.KRIA.*` and update `benchmark/README.md`.
- [ ] **4.2 Root on the board.** The suite starts the run scripts over ssh as
      `board_user` without sudo. The scripts load bitstreams, write
      `/dev/u-dma-buf-mgr` and run the binaries, all without sudo. Start them with sudo
      from the suite, or document that the board user must be root.
- [ ] **4.3 Runtime entries.** `benchmark/configs/exp_configs.sh` and
      `configs/runtimes/*.json` name pre-migration bitstreams (e.g. `BFPB_Q3_v1`). Add
      runtimes for `BFPP_ACC_KRIA_3_0` and `BFPP_ACC_KRIA_4_0`, both drivers each.
- [ ] **4.4 Remove the hard-coded Pushbullet token** from `benchmark_suite.sh`. Read
      it from `config.json` or the environment. This only stops new copies; it is
      already in public history.
- [ ] **4.5 One full suite run** on `kriaB_L` with the new runtimes. *Done when:*
      `llama_perf.csv` is produced for every runtime.

## 5. Build and repo hygiene

- [ ] **5.1 Ignore the run outputs.** Add `secda_profile.json` and `dma*.csv` to
      `.gitignore` and to the fork excludes that `setup.sh` writes.
- [ ] **5.2 Upstream presets** (`x64-linux-gcc-*` and the others). Configure one to
      confirm it stops at "GGML_SECDA needs a design". Then either set
      `GGML_SECDA=OFF` in them or hide them.
- [ ] **5.3 KV260 CMA headroom.** 4 x (192 MB + 16 MB) = 832 MB of about 1 GB of CMA
      fragments after long uptime.
      - Size `DMA_IN_BUF_SIZE` from the model: MobileLLM-125M needs far less.
      - Or make it a CMake option per runtime.
      - Or allocate one input buffer per DMA only as big as the plan needs.
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
