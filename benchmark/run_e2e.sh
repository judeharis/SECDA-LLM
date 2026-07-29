#!/bin/bash
set -eo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ============================================================================
# CONFIG - edit these to control what run_exp does for every experiment below
# ============================================================================

synth_config_name="synth_params"
suite_flags="-l -p -po" #  llama-bench + parse (compile/synth-bench skipped by default)

# ============================================================================
# run_exp <exp_config_name>
#
# Regenerates configs/exp_configs.sh from configs/exp_configs/<name>.json via
# scripts/e2e_exp_config_gen.py, then runs the benchmark suite against it.
# ============================================================================

run_exp() {
  local exp_name="$1"

  echo "[run_e2e] === Running experiment: ${exp_name} ==="
  python3 "${script_dir}/scripts/e2e_exp_config_gen.py" \
    --exp_config_name "${exp_name}" \
    --synth_config_name "${synth_config_name}"

  "${script_dir}/benchmark_suite.sh" -n "${exp_name}" ${suite_flags}
}

# ============================================================================
# EXPERIMENTS - comment/uncomment or add lines to control what runs
# ============================================================================

# run_exp GPT2Q2

# run_exp Granite4M350Q2
# run_exp Granite4M350Q3S
# run_exp Granite4M350Q3M
# run_exp Granite4M350Q3L
# run_exp Granite4M350Q4S
# run_exp Granite4M350Q4M
# run_exp Granite4M350Q5S
# run_exp Granite4M350Q5M
# run_exp Granite4M350Q6

# run_exp Llama3B1Q2
# run_exp Llama3B1Q3S
# run_exp Llama3B1Q3M
# run_exp Llama3B1Q3L
# run_exp Llama3B1Q4S
# run_exp Llama3B1Q4M
# run_exp Llama3B1Q5S
# run_exp Llama3B1Q5M
# run_exp Llama3B1Q6

# run_exp llama600M
# run_exp Mamba130Q3S

# run_exp Mamba790Q2
# run_exp Mamba790Q3L
# run_exp Mamba790Q3M
# run_exp Mamba790Q3S
# run_exp Mamba790Q4M
# run_exp Mamba790Q4S
# run_exp Mamba790Q5M
# run_exp Mamba790Q5S
# run_exp Mamba790Q6

# run_exp MobileLLaMAQ2
# run_exp MobileLLMQ2
# run_exp NanoMistralQ3S

# run_exp TinyLlama1Q2

# run_exp TinyLlama2Q2
# run_exp TinyLlama2Q3L
# run_exp TinyLlama2Q3M
# run_exp TinyLlama2Q3S
# run_exp TinyLlama2Q4M
# run_exp TinyLlama2Q4S
# run_exp TinyLlama2Q5M
# run_exp TinyLlama2Q5S
# run_exp TinyLlama2Q6
