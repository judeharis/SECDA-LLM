# Benchmark Suite

This folder contains the SECDA-LLM benchmark flow for building binaries, running experiments on the board, and parsing the collected results.

The scripts now treat the SECDA-LLM repository root as the CMake project entrypoint. They no longer assume `llama.cpp` is the top-level project directory.

## Main Entry Point

Use `benchmark_suite.sh` to run the benchmark pipeline from either the repository root or the `benchmark/` directory.

The script reads the board from `../config.json` using `jq`: the
`boards.KRIA` entry (set `SECDA_BOARD=<key>` for another), the same entry
`./secda load` and `./secda run-on-board` use:

- `board_user`
- `board_hostname`
- `board_port`
- `board_dir`

**Root on the board.** The run scripts load bitstreams, write
`/dev/u-dma-buf-mgr` and drop caches. With `board_user` root they run as is;
any other user needs passwordless sudo, and the suite starts them with
`sudo bash -lc`.

**Bitstreams** come from `hardware_automation/bitstreams/KRIA/` (`.bit` and
`.hwh`, including the `CPU_1_0` reset pair) and are synced to
`${board_dir}/bitstreams/` on every run. **Models** are read from
`${board_dir}/models/`; put the `.gguf` files there, or link an existing
directory.

**DMA buffers.** Each run allocates four DMA input buffers of 192 MB from
CMA, which can fail once the board's CMA has fragmented. Set
`SECDA_DMA_IN_BUF_MB` when calling the suite to pass a smaller size to the
board runs (MobileLLM-125M preloads all its layers with 16). Layers that don't
fit are sent per call instead of preloaded.

It combines those values into the remote target and then runs up to three stages:

- `-b`: build and send binaries to the board
- `-r`: copy experiment scripts and run the benchmark on the board
- `-l` or `--llama-bench`: copy scripts and run FPGA `llama-bench` on the board
- `-pp` or `--perplexity`: sync the wikitext-2 dataset and run FPGA `llama-perplexity` on the board
- `-p`: fetch raw results and parse them locally

If you run the script without `-b`, `-r`, or `-p`, it runs all three stages.

### Examples

Run the full flow:

```bash
./benchmark_suite.sh
```

Only build and deploy binaries:

```bash
./benchmark_suite.sh -b
```

Only run experiments on the board:

```bash
./benchmark_suite.sh -r
```

Only run FPGA llama-bench on the board:

```bash
./benchmark_suite.sh -l
```

Only run FPGA llama-perplexity on the board:

```bash
./benchmark_suite.sh -pp
```

Only fetch and parse results:

```bash
./benchmark_suite.sh -p
```

Save a named copy of the parsed result folder:

```bash
./benchmark_suite.sh -n my_run
```

## `llama-perplexity` (`-pp`)

Unlike `llama-bench`/`llama-cli`, `llama-perplexity` runs against a real text
corpus (`perplexity/wikitext-2-raw/wiki.test.raw`, auto-synced to
`${board_dir}/datasets/` the first time `-pp` is used) rather than synthetic
prompt/gen lengths, so its four tunables (`PPL_CHUNKS`, `PPL_BATCH`,
`PPL_UBATCH`, `PPL_CTX_SIZE` env vars, see `scripts/run_llama_perplexity.sh`)
behave differently than the equivalent `llama-bench` knobs.

This runner only collects the PPL score itself — no power logging, no
`prf.csv`, no `llama_perf.csv`. Each run writes a single
`perplexity_<model>_<threads>_<tag>.txt` stdout capture, and
`parse_results.py` reads the `Final estimate: PPL = X +/- Y` line straight
out of that file into `perplexity_results.csv`.

- **`PPL_CTX_SIZE` (`-c`/`--ctx-size`) affects both duration and the
  measured PPL number, more directly than any other knob.** It sets the
  window length each chunk covers; `llama-perplexity` only scores the
  second half of each window (`first = n_ctx/2` in `perplexity.cpp`), so a
  smaller context means every scored token has less preceding context to
  condition on, which generally raises (worsens) reported PPL — "PPL at
  ctx=512" and "PPL at ctx=2048" for the same model are not comparable
  numbers, independent of anything SECDA-specific. Default (`512`) matches
  `llama-perplexity`'s own built-in default, set before arg parsing at
  `tools/perplexity/perplexity.cpp:2016` — this overrides the generic
  llama.cpp default of "0 = model's trained context" specifically for this
  tool, and is the traditional wikitext-2 PPL@512 benchmarking convention.
  With `PPL_CHUNKS` fixed, duration scales ~linearly with `PPL_CTX_SIZE`
  (each chunk does a longer forward pass). It does not affect whether SECDA
  engages — same as `PPL_BATCH` below, only `PPL_UBATCH` controls that.
- **`PPL_UBATCH` (`-ub`) gates whether SECDA engages at all.** SECDA's
  `DimCheck` (`srcs/ggml_backend/ggml-secda/acc_dels/bfpp_acc/v3/accelerator/driver/acc_driver.h`)
  rejects any `MUL_MAT` where `ubatch_size * ceil(K/256) > 512` (`K` being a
  layer's reduction dimension) and falls back to CPU for it — since
  `llama_decode` always splits work into `ubatch_size`-token chunks per
  compute call, this is the real per-call `N` the accelerator sees,
  regardless of `-b`. The default (`16`) keeps every layer of
  `tiny-llama-1.1B` (largest `K` = 5632, the FFN down-proj) under the limit
  (`N_max = floor(131072 / K) ≈ 23`); recompute `N_max` before adding a
  larger model to `configs/exp_configs.sh`.
- **`PPL_CHUNKS` affects both run duration and the measured PPL number.**
  Each chunk is an independent, non-overlapping window of the corpus, so
  duration scales ~linearly with chunk count, and the reported PPL (and its
  `+/-` error bar) is an average over however much text got scored — fewer
  chunks means a smaller sample and a noisier number, not just a faster
  run. Keep `PPL_CHUNKS` fixed across configs being compared (it already is,
  since it's one env var applied to every model/binary in the run).
- **`PPL_BATCH` (`-b`) affects duration but should not affect measured
  PPL.** It only controls how many chunks get scheduled together per
  KV-cache-clear cycle (throughput/overhead), not which tokens get scored or
  how their NLL is computed — `-ub`, not `-b`, determines the actual
  per-call token count seen by SECDA's `DimCheck`.

## Configuration Notebook

`exp_config_gen.ipynb` is a helper notebook for generating the shell config used by the benchmark scripts.

Its main job is to write:

```bash
configs/exp_configs.sh
```

That generated shell file contains the arrays consumed by the benchmark scripts, including:

- models
- model names
- runtimes
- binaries
- accelerator tags
- bitstreams
- build flags

## How the Notebook Works

The notebook loads runtime definitions from:

- `configs/runtimes/runtime_dict_og.json`
- `configs/runtimes/runtime_dict_1.json`
- `configs/runtimes/runtime_dict_2.json`
- `configs/runtimes/runtime_dict_3.json`
- `configs/runtimes/runtime_dict_v6.json`: the runtimes for the SECDA-Core
  migration: `bfpp.v3`, `bfpp.v3b`, `bfpp.v4`, `bfpp.v4b` (`BFPP_ACC_KRIA_3_0`
  and `_4_0`, `driver` and `driver_batches`) and `cpu.v6`. The other
  dictionaries name pre-migration bitstreams that are no longer in the repo.

`scripts/e2e_exp_config_gen.py --exp_config_name MobileLLMQ2_v6` writes
`configs/exp_configs.sh` for those five runtimes without the notebook.

It loads model definitions from:

- `configs/models/models_dict.json`

Then it supports two common workflows:

1. Load a prebuilt experiment config from `configs/exp_configs/*.json`
2. Manually choose `models` and `runtimes` in the notebook and regenerate `configs/exp_configs.sh`

## Typical Notebook Usage

### Option 1: Use a predefined experiment config

In the notebook, load a JSON config such as:

```python
models, runtimes = load_exp_config("configs/exp_configs/Mamba790Q2.json")
generate_config(runtime_dict, models_dict, runtimes, models)
```

### Option 2: Create a custom experiment config

Edit the model and runtime lists in the notebook, then regenerate:

```python
models = ["TinyLlama1Q2"]
runtimes = ["q2q3q4q5q6v2.0A", "cpuv1.0"]
generate_config(runtime_dict, models_dict, runtimes, models)
```

After running the cell, `configs/exp_configs.sh` is updated and the benchmark scripts will use the new configuration.

## Related Scripts

- `scripts/compile_send_kria.sh`: builds and deploys binaries to the board from the SECDA-LLM root project
- `scripts/run_experiment_kria.sh`: runs the benchmark on the board
- `scripts/parse_results.py`: parses downloaded benchmark results

## Build and Test Flow

The benchmark pipeline follows the root wrapper project and the top-level `CMakePresets.json`.

1. Generate or refresh `configs/exp_configs.sh` from the notebook.
2. Run `./benchmark_suite.sh -b` to compile and send binaries from the SECDA-LLM root project.
3. Run `./benchmark_suite.sh -r` to execute the experiments on the board.
4. Optional: run `./benchmark_suite.sh -l` to execute FPGA `llama-bench` on the board.
5. Run `./benchmark_suite.sh -p` to fetch and parse the benchmark outputs locally.

## Output

Parsed benchmark outputs are written under:

```bash
results/<timestamp>/
```

That folder includes the parsed results and a `status.txt` timing summary for the run.