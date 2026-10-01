# SECDA-LLM

SECDA-LLM integrates llama.cpp with the SECDA design methodology. The ggml backend
`ggml-secda` offloads graph nodes to a SystemC accelerator design. Every design runs
K-quant `MUL_MAT` (Q2_K to Q6_K); bfpp_acc v4 also runs `SOFT_MAX`. The same design
runs as a SystemC simulation inside llama.cpp on x86-64, and as an FPGA bitstream
on the AMD Kria KV260.

SECDA-LLM builds on SECDA-Core (axi_support v6, the profiler, the hardware
automation flow) and is part of the SECDA-DS suite. Current state:
[STATUS.md](STATUS.md). What has been verified for each design version is in its
`test_status.json` and `CHANGES.md`.

## Repository layout

| Path | Contents |
|---|---|
| `srcs/ggml_backend/ggml-secda/` | The backend: `ggml-secda.cpp` (registration, op dispatch), `ops_support.{h,cpp}`, `secda_planner.{h,cpp}` (plans weight preloads from the scheduler's `graph_optimize` calls). `setup.sh` links this directory into llama.cpp. |
| `srcs/ggml_backend/ggml-secda.h` | The backend's public header, exported by the `ggml-secda` target. |
| `srcs/ggml_backend/ggml-secda/acc_dels/bfpp_acc/v3/` | BFPP accelerator v3. Contains `accelerator/` (SystemC/HLS source), `accelerator/driver/`, `accelerator/driver_batches/` (a batch-capable driver), `hw_params.json`, `test_status.json` and `CHANGES.md`. |
| `.../bfpp_acc/v4/` | v3 plus a hardware SOFT_MAX unit. Same layout. |
| `.../bfpp_acc/legacy/{v1,v2}/` | Retired designs, still on axi_support v5. Configure stops if `SECDA_BFPP_ACC_V1` or `SECDA_BFPP_ACC_V2` is set. |
| `srcs/tools/` | The `secda-*` tool variants (see *Tools* below): `patches/`, `secda_perf/` (the `llama_perf.csv` writer), `cmake/secda_apply_patch.cmake` and `refresh_patch.sh`. |
| `cmake/` | `FindSYSC.cmake` (SystemC, found through `SYSTEMC_HOME`) and `FindSECDA_CORE.cmake` (a fallback for an out-of-tree SECDA-Core build). |
| `scripts/` | `check_fork_markers.py` and `llama_fork_base` (the fork policy), `install_systemc.sh`, and `sim_replay/` (replay board MUL_MAT calls through the SystemC model, bit for bit). |
| `hardware_automation/` | Output of SECDA-Core's hardware flow: `generated/<TAG>/` projects and `bitstreams/KRIA/*.bit`/`.hwh`. All of it is gitignored. |
| `benchmark/` | The board benchmark flow: `benchmark_suite.sh`, `run_e2e.sh`, `configs/` and `scripts/` (including `compile_send_kria.sh`). See `benchmark/README.md`. |
| `perplexity/` | wikitext-2 data for `llama-perplexity`. |
| `test/`, `docs/`, `plan.md` | Op dumps and reports, notes and plans. |
| `example_acc/` | Perplexity logs from earlier board runs. The `bfp_softmm` experiment moved to SECDA-Sandboxed (`experiments/bfp_softmm`). |
| `llama.cpp/` | Submodule: the llama.cpp fork (`judeharis/llama.cpp`, branch `v3_core_upgrade`). |
| `third_party/secda_core/` | Submodule: SECDA-Core, used only outside SECDA-DS. |
| `secda` | Entry point for SECDA-Core's CLI (`./secda hw-gen`, `./secda load`, ...). |
| `config.example.json` | Template for the machine-local `config.json` (untracked): Vivado paths, boards. |

### What the llama.cpp fork contains

The fork differs from upstream `06938ac12` (recorded in `scripts/llama_fork_base`) in
three files, 17 added lines:

- `ggml/CMakeLists.txt` adds the `GGML_SECDA` option.
- `ggml/src/CMakeLists.txt` adds `ggml_add_backend(SECDA)`.
- `ggml/src/ggml-backend-reg.cpp` includes `ggml-secda.h` and registers the backend.

Everything else lives in this repo. That covers the backend (linked in as
`llama.cpp/ggml/src/ggml-secda` by `setup.sh`), the Find modules, the presets and
the tool changes. The backend is registered statically, so `GGML_BACKEND_DL` must
be OFF. Configure stops if it is ON.

## Setup

You need:
- CMake 3.24 or later (`--fresh`; presets version 3 alone needs 3.21), Ninja and
  `patch`;
- clang for the x86-64 presets;
- SystemC 2.3.3 for simulation;
- `aarch64-linux-gnu-gcc`/`g++` in `/usr/bin` for the KV260 presets.

The submodule URLs are SSH (`git@github.com:judeharis/...`).

**Standalone clone:**

```bash
git clone -b v3_core_upgrade https://github.com/judeharis/SECDA-LLM.git
cd SECDA-LLM
git submodule update --init          # llama.cpp and third_party/secda_core
./setup.sh
```

**Inside SECDA-DS:** leave `third_party/secda_core` unchecked-out. The build uses
the suite's `SECDA-Core/` instead.

```bash
cd SECDA-DS/SECDA-LLM
git submodule update --init llama.cpp
./setup.sh
```

**What `setup.sh` does.** It symlinks `srcs/ggml_backend/ggml-secda` to
`llama.cpp/ggml/src/ggml-secda`. It then adds that link to the fork's
`info/exclude`, with `/.vscode/`, `/results/` and the run outputs `tbo.csv`,
`llama_perf.csv`, `prf.csv`, `/_gstats/` and `/_plans/`. It does not exclude
`secda_profile.json`, which every SECDA preset writes, or the `dma<N>.csv` files a
KV260 build writes. A run from inside `llama.cpp/` therefore leaves untracked files
in the fork; run from a scratch directory. Re-run `setup.sh` after any checkout of
`llama.cpp` that removes the link. If the link is missing, configure stops and
tells you to run it.

**Where SECDA-Core comes from.** The build and `./secda` resolve it in the same order:
1. `$SECDA_CORE_DIR`, if set;
2. `../SECDA-Core`, when this repo sits inside SECDA-DS (the parent's
   `.gitmodules` lists `path = SECDA-Core`);
3. `third_party/secda_core`.

The result is cached as `SECDA_LLM_SECDA_CORE_DIR`. If it changes, reconfigure with
`--fresh`.

**SystemC.** The simulation presets need SystemC. `FindSYSC` looks in
`$SYSTEMC_HOME/include` and in `$SYSTEMC_HOME/lib-linux64` (or `lib`).
`scripts/install_systemc.sh` builds 2.3.3 into `/opt/tools/systemc-2.3.3` and adds
`SYSTEMC_HOME` to `~/.bashrc`.

## Configure, build, test

Each SECDA preset builds in `out/build/<preset>` and sets `SECDA_ACC_PROFILE=ON`,
so the driver writes `prf.csv` and `secda_profile.json`.

| Preset | Builds for | Design | Notes |
|---|---|---|---|
| `SECDA-sim-x64` | x86-64, SystemC simulation | v3 `driver` | RelWithDebInfo, clang, Q2_K-Q6_K, `ACC_PRELOAD` |
| `SECDA-sim-x64-batches` | x86-64, simulation | v3 `driver_batches` | |
| `SECDA-sim-x64-v4` | x86-64, simulation | v4 `driver` | |
| `SECDA-nosim-x64` | x86-64, CPU only | none | Debug, SECDA off: the CPU reference |
| `SECDA-aarch64-kria` | KV260 | v3 `driver` | Release, `BUILD_KRIA`, `ACC_PRELOAD`: the board build |
| `SECDA-aarch64-debug` | KV260 | v3 `driver` | Debug |
| `SECDA-armv7-debug` | PYNQ-Z1 | v3 `driver` | Uses the `/opt/gcc-arm-8.3-2019.03-...` toolchain. Not built since the v6 port, and there is no Z1 hardware variant. |

With `ACC_PRELOAD`, the planner preloads MUL_MAT weights into the DMA input buffers
when it plans a graph, as far as they fit (`DMA_WGT_SIZE`). Weights that don't fit
are sent on each call.

**Other design combinations.** Pass the options on the configure line and use a
separate build directory. When both are set, V4 takes priority over V3.

```bash
cmake --preset SECDA-aarch64-kria -B out/build/SECDA-aarch64-kria-v4 -DSECDA_BFPP_ACC_V4=ON
cmake --preset SECDA-sim-x64-v4 -B out/build/SECDA-sim-x64-v4-batches -DSECDA_BFPP_ACC_V4_DRIVER_BATCHES=ON
cmake --preset SECDA-aarch64-kria -B out/build/SECDA-aarch64-kria-v3b -DSECDA_BFPP_ACC_V3_DRIVER_BATCHES=ON
```

**Simulation and CPU.** The same commands work for `SECDA-sim-x64-batches`,
`SECDA-sim-x64-v4` and `SECDA-nosim-x64`:

```bash
cmake --preset SECDA-sim-x64
cmake --build --preset SECDA-sim-x64 -j
ctest --preset SECDA-sim-x64
```

**KV260 cross-compile.** There is no build preset for this one, so build the
directory directly:

```bash
cmake --preset SECDA-aarch64-kria
cmake --build out/build/SECDA-aarch64-kria -j --target secda-llama-cli secda-test-backend-ops secda-llama-bench llama-perplexity
```

**The upstream llama.cpp presets** (`x64-linux-gcc-*` and the others) select no
design. SECDA is on by default, so their configure is expected to stop at
"GGML_SECDA needs a design". For a CPU-only build, use `SECDA-nosim-x64`.

### Tools

`srcs/tools` builds three SECDA variants of llama.cpp tools. Each is the pinned
llama.cpp's own source with a patch from `srcs/tools/patches/` applied at build
time, so the fork carries none of these changes. They go into `bin/` next to the
unchanged upstream tools. `SECDA_LLM_BUILD_TOOLS=OFF` skips them.

| Target | Built from | Adds |
|---|---|---|
| `secda-test-backend-ops` | `tests/test-backend-ops.cpp` + `test-backend-ops.patch` | `-t`, `--test-params`, `tbo.csv`, and the SECDA plan for perf runs |
| `secda-llama-bench` | `tools/llama-bench` + `llama-bench.patch` | writes `llama_perf.csv` |
| `secda-llama-cli` | `tools/cli` + `cli.patch` | prints the perf summary and writes `llama_perf.csv` |

`secda-test-backend-ops` differs from upstream's in a few more ways. Support is
checked on the `out` node only, and a perf run lasts at least 20 s instead of 1 s.
A NaN in the output fails the case, as upstream, and is printed as
`NaN at index ...`. `SECDA_TBO_NAN_PASS=1` restores the owner's older behaviour,
where a NaN was printed but passed.

**Checks used for the v6 migration.** Run them from a scratch directory: the
backend and the tools write `prf.csv`, `secda_profile.json`, `llama_perf.csv` and
`tbo.csv` into the current directory. Compare the results with the same run from
`SECDA-nosim-x64`.

```bash
B=<SECDA-LLM>/out/build/SECDA-sim-x64/bin
$B/secda-test-backend-ops -b SECDA -o MUL_MAT          # v4: also -o SOFT_MAX
$B/llama-perplexity -m MobileLLM-125M-HF.Q2_K.gguf -f <SECDA-LLM>/perplexity/ppl_input.txt -c 128 --chunks 2 -t 1 -b 128
$B/llama-perplexity -m MobileLLM-125M-HF.Q2_K.gguf -f <SECDA-LLM>/perplexity/ppl_input.txt -c 128 --chunks 2 -t 1 -b 16 -ub 16
```

The perplexity input, `perplexity/ppl_input.txt`, is the first 4096 bytes of
llama.cpp's `README.md` at `06938ac12` (md5 `b6d844ba5fc667529f6efd3cf3cdd204`),
not `perplexity/wikitext-2-raw`. That input gave the recorded 3.3449 +/- 0.59780
(simulation and x86 CPU) and 3.3625 +/- 0.60376 (KV260).

The `-b 128` run mostly exercises the CPU. The driver's size check
(`N x kb <= SUP_KNB`, 512) accepts only 1 of the 30 MUL_MAT nodes per chunk, and the
rest run on the CPU. The `-b 16 -ub 16` run puts all 30 on SECDA, for every design
variant: 3.3429 +/- 0.59671 in simulation (x86 CPU-only: 3.3637) and 3.3636 +/-
0.60469 on the KV260 (board CPU-only: 3.3596). The gap between simulation and
board is not the accelerator. The SystemC model reproduces the hardware bit for bit
on replayed board inputs (`scripts/sim_replay`), but the CPU ops that produce those
inputs round differently on ARM and x86, and this 2-chunk test amplifies that.
Compare perplexity only against the CPU build on the same platform. Beyond
perplexity, the accelerator's end-to-end coverage comes from
`secda-llama-cli`, which plans 30/30 MUL_MAT nodes on SECDA, and from
`secda-test-backend-ops`.

## Running on the KV260

### Bitstreams

Each design's `hw_params.json` holds its hardware variant: `BFPP_ACCv3_0_KRIA` or
`BFPP_ACCv4_0_KRIA`. Both use `KRIA_dma_4_hp_4_ctrl_hwc.tcl` (4 DMAs) at 200 MHz.

First, create `config.json`: run `./secda setup-config`, or copy
`config.example.json`, then set these in it:
- the Vivado paths;
- the `KRIA` board (the `boards` section, which `setup-config` doesn't prompt for);
- `secda_init_path`: this repo's absolute path;
- `bitstream_dir`: `<repo>/hardware_automation/bitstreams`.

`hw-gen` builds its source paths from `secda_init_path`, and `load` looks for the
pair under `bitstream_dir`. Both ship as `CHANGE_ME` placeholders. Keep
`path_to_solutions` as `srcs/ggml_backend/ggml-secda/acc_dels`. Then:

```bash
./secda hw-gen bfpp_acc/v3 --no-run     # writes hardware_automation/generated/BFPP_ACC_KRIA_3_0/
cd hardware_automation/generated/BFPP_ACC_KRIA_3_0
./run.sh 1 0                            # HLS only
./run.sh 0 1                            # HLX only
```

When its build gates pass, `run.sh` copies the pair to
`hardware_automation/bitstreams/KRIA/BFPP_ACC_KRIA_3_0.{bit,hwh}`. Its timing gate
refuses any negative WNS.

After HLS, `run.sh` also syncs the simulation's timing to the HLS schedule
(SECDA-Core `sched_sync.py`). By default (`SECDA_SCHED_SYNC=apply`) it rewrites
the `DWAIT` arguments in the accelerator source and the generated
`accelerator/acc_schedule.h`, and resynthesises until nothing changes. The
hardware is unaffected, but the simulated cycle counts change. Use
`SECDA_SCHED_SYNC=report` to leave the source alone.

The current v3 and v4 builds missed setup by 0.307 and 0.217 ns at 200 MHz. Under
the SECDA-DS 30% rule, their pairs were copied by hand from
`generated/<TAG>/generated_files/` (see `test_status.json`).

### Building and deploying

`benchmark/scripts/compile_send_kria.sh` builds and deploys each runtime listed in
`benchmark/configs/exp_configs.sh`:

- **Build.** It configures with the KV260 cross-compile flags that
  `SECDA-aarch64-kria` also uses, plus each runtime's flags from `exp_configs.sh`
  (`bin_flags_array`: design, quant types).
  - Every accelerated runtime also gets `-DACC_PRELOAD=ON`, which `exp_configs.sh`
    can't turn off.
  - A runtime whose `bin_dir` is `bin_cpu` is built as a CPU runtime, without it.

  It builds into `out/build/Kria_SECDA/<bin_dir>` (CPU runtimes:
  `out/build/Kria_CPU/<bin_dir>`).
- **Deploy.** It copies the tools to `<board_path>/<board_sub>/<bin_dir>/` under the
  names the run scripts use: `secda-llama-cli` as `<name>`,
  `secda-test-backend-ops` as `<name>_tbo`, `secda-llama-bench` as `<name>_bench`
  and `llama-perplexity` as `<name>_perplexity`.
- **Shared libraries.** It also copies the whole `bin/`, because the binaries need
  the shared libraries built alongside them.

```bash
benchmark/scripts/compile_send_kria.sh -a <user>@<host> -p <port> -d <board_path> -s benchmark
```

`benchmark/benchmark_suite.sh -b` calls it with the board settings from
`config.json`'s `boards.KRIA` entry (or `boards.$SECDA_BOARD`); see
`benchmark/README.md`.

### On the board

- **Load the bitstream as a `.bit` + `.hwh` pair**, through PYNQ, as root.
  - The migration's board tests loaded them with `load_bitstream.py`, a PYNQ
    `Overlay` wrapper. The run scripts expect it at `~/load_bitstream.py` on the
    board. SECDA-LLM doesn't ship it; SECDA-TFLite's `scripts/load_bitstream.py` is
    the same file.
  - `./secda load bfpp_acc/v3` checks the driver's addresses against the `.hwh`,
    deploys the pair and loads it, leaving it loaded.
  - `./secda run-on-board bfpp_acc/v3 --bin <binary> --deploy <bin dir>:bins
    --env LD_LIBRARY_PATH=bins -- <args>` loads, runs, collects the output under
    `hardware_automation/results/` and resets the board. Add `--check "<cmd>"` for
    a PASS/FAIL verdict.
  - After a run, reset the board to the CPU-only `CPU_1_0` pair (tracked in
    `hardware_automation/bitstreams/KRIA/`, copied from SECDA-Sandboxed).
- **Run the binaries as root**, with `LD_LIBRARY_PATH` pointing at the deployed
  `bin/`: `sudo env LD_LIBRARY_PATH=${PWD}/bin ./<binary> ...`.
  - The run scripts in `benchmark/scripts/` call the binaries directly with
    `LD_LIBRARY_PATH=${PWD}/bin:...`, with no sudo. They also load bitstreams and
    write `/dev/u-dma-buf-mgr` without sudo, so the scripts themselves have to be
    started as root.
  - The `sudo env ...` line in their commands log is only a replay line.
- **DMA buffers.** axi_support v6 allocates every DMA buffer through u-dma-buf
  (`/dev/u-dma-buf-mgr`), which the kernel must provide. The KV260 build takes
  4 x (192 MB in + 16 MB out) = 832 MB of the board's ~1 GB CMA
  (`DMA_IN_BUF_SIZE`/`DMA_OUT_BUF_SIZE` in `accelerator/acc_config.sc.h`).
- **Leftover buffers.** The buffers are deleted at a normal exit, but a killed run
  leaves `udmabufN` behind. List them with `ls /sys/class/u-dma-buf/` and remove
  each with `echo 'delete udmabufN' > /dev/u-dma-buf-mgr`.
  `benchmark/scripts/run_llama_bench.sh` does this before it starts.
- **CMA fragmentation.** After long uptime, allocation can fail ("Failed to create
  udmabufN") even though `CmaFree` shows enough. Dropping caches does not help; a
  reboot does.

## Fork policy (llama.cpp)

- **Only what can't live here goes in the fork.** Anything that can live in
  SECDA-LLM lives here. Tool changes go in as patches under `srcs/tools/patches`,
  not as fork edits.
- **Every added or edited region in the fork is bracketed by markers**, even a
  single line (use `#` instead of `//` in CMake and `.gitignore`):

  ```c
  // Jude: Added - <why>
  ...
  // Jude: Added end
  ```

  Changed upstream lines go inside a `// Jude: Edited - <what>` ... `// Jude: Edited end`
  pair.
- **What the checker enforces.** `scripts/check_fork_markers.py` diffs `llama.cpp`
  against `scripts/llama_fork_base`. It fails on:
  - any change outside a marker pair;
  - upstream lines removed outside an `Edited` region;
  - a deleted upstream file;
  - an added symlink or binary;
  - a mode change.

  By default it checks the fork's `HEAD`; `--worktree` checks uncommitted edits.
  Run it before every fork commit:

  ```bash
  scripts/check_fork_markers.py
  ```

**Bumping llama.cpp:**

1. Rebase or merge the fork's `v3_core_upgrade` onto the new upstream commit,
   keeping only the marked regions. Write the new base into
   `scripts/llama_fork_base` and run `scripts/check_fork_markers.py`.
2. Re-run `./setup.sh`, then reconfigure with `--fresh`.
3. If a tool patch no longer applies, the build stops with "... does not apply ...
   see srcs/tools/refresh_patch.sh". Apply the old patch to the new upstream file
   (`patch --merge`, or by hand), fix it up, then regenerate the patch:

   ```bash
   srcs/tools/refresh_patch.sh <test-backend-ops|llama-bench|cli> <edited file>
   ```

   The edited file must start from the new upstream file. A patched copy from an
   older build would silently revert upstream changes.
4. Check that planning still works. The backend depends on the scheduler calling
   `graph_optimize` for each split. If a bump loses that, the backend prints
   `SECDA WARNING: split not recorded by graph_optimize` and falls back to
   planning per split. That is still correct, but it re-preloads the weights for
   every split.
