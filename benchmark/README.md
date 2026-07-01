# Benchmark Suite

This folder contains the SECDA-LLM benchmark flow for building binaries, running experiments on the board, and parsing the collected results.

## Main Entry Point

Use `benchmark_suite.sh` to run the benchmark pipeline.

The script reads default board settings from `../config.json` using `jq`:

- `board_user`
- `board_hostname`
- `board_port`
- `board_dir`

It combines those values into the remote target and then runs up to three stages:

- `-b`: build and send binaries to the board
- `-r`: copy experiment scripts and run the benchmark on the board
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

Only fetch and parse results:

```bash
./benchmark_suite.sh -p
```

Save a named copy of the parsed result folder:

```bash
./benchmark_suite.sh -n my_run
```

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

- `scripts/compile_send_kria.sh`: builds and deploys binaries to the board
- `scripts/run_experiment_kria.sh`: runs the benchmark on the board
- `scripts/parse_results.py`: parses downloaded benchmark results

## Output

Parsed benchmark outputs are written under:

```bash
results/<timestamp>/
```

That folder includes the parsed results and a `status.txt` timing summary for the run.