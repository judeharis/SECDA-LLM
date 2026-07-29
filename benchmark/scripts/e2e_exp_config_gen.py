import argparse
import importlib.util
import json
import sys
from itertools import product
from pathlib import Path

sys.dont_write_bytecode = True

SCRIPT_DIR = Path(__file__).resolve().parent
CONFIGS_DIR = SCRIPT_DIR.parent / "configs"

# ---------------------------------------------------------------------------
# Load generate_mul_mat_suite module (same directory as this script)
# ---------------------------------------------------------------------------
_SUITE_SCRIPT = SCRIPT_DIR / "generate_mul_mat_suite.py"
_spec = importlib.util.spec_from_file_location("generate_mul_mat_suite", _SUITE_SCRIPT)
_suite_mod = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = _suite_mod
_spec.loader.exec_module(_suite_mod)


def declare_array(f, name, list):
    f.write("declare -a {}_array=(\n".format(name))
    for i in list:
        f.write('  "{}" \n'.format(i))
    f.write(")\n")


def generate_synth_bench(synth_params):
    """Expand synth_params (type_a x type_b x m x n x k) into test names/lines."""
    type_map = _suite_mod.load_type_map_from_header(_suite_mod.DEFAULT_GGML_H)

    names = []
    lines = []
    for type_a, type_b, m, n, k in product(
        synth_params["type_a"],
        synth_params["type_b"],
        synth_params["m"],
        synth_params["n"],
        synth_params["k"],
    ):
        type_a_id = _suite_mod.parse_type(str(type_a), type_map)
        type_b_id = _suite_mod.parse_type(str(type_b), type_map)
        name = f"{type_a}_{m}M_{n}N_{k}K".replace("_K", "").replace("GGML_TYPE_", "")
        line = _suite_mod.build_mul_mat_line(
            type_a=type_a_id,
            type_b=type_b_id,
            m=int(m),
            n=int(n),
            k=int(k),
            name=name,
        )
        names.append(name)
        lines.append(line)

    return names, lines


def generate_exp_config(exp_config_name, synth_config_name="synth_params"):
    with open(CONFIGS_DIR / "runtimes" / "runtime_dict_og.json", "r") as jf:
        runtime_dict = json.load(jf)

    for suffix in ("1", "2", "3"):
        with open(CONFIGS_DIR / "runtimes" / f"runtime_dict_{suffix}.json", "r") as jf:
            runtime_dict.update(json.load(jf))

    with open(CONFIGS_DIR / "models" / "models_dict.json", "r") as jf:
        models_dict = json.load(jf)

    with open(CONFIGS_DIR / "synth_configs" / f"{synth_config_name}.json", "r") as jf:
        synth_params = json.load(jf)

    with open(CONFIGS_DIR / "exp_configs" / f"{exp_config_name}.json", "r") as jf:
        exp_config = json.load(jf)

    models = exp_config["models"]
    runtimes = exp_config["runtimes"]

    bitstreams_list = [runtime_dict[runtime]["bitstreams"] for runtime in runtimes]
    binaries_list = [runtime_dict[runtime]["binaries"] for runtime in runtimes]
    acc_tags_list = [runtime_dict[runtime]["acc_tag"] for runtime in runtimes]
    bins_list = [runtime_dict[runtime]["bins"] for runtime in runtimes]
    accelerated_list = [runtime_dict[runtime]["accelerated"] for runtime in runtimes]
    models_list = [models_dict[model] for model in models]
    model_names_list = list(models)
    bin_flags_list = [runtime_dict[runtime]["bin_flags"] for runtime in runtimes]

    synth_test_names, synth_test_lines = generate_synth_bench(synth_params)

    with open(CONFIGS_DIR / "exp_configs.sh", "w+") as f:
        declare_array(f, "models", models_list)
        declare_array(f, "model_names", model_names_list)
        declare_array(f, "bitstreams", bitstreams_list)
        declare_array(f, "binaries", binaries_list)
        declare_array(f, "acc_tags", acc_tags_list)
        declare_array(f, "bins", bins_list)
        declare_array(f, "accelerated", accelerated_list)
        declare_array(f, "bin_flags", bin_flags_list)
        declare_array(f, "runtimes", runtimes)
        declare_array(f, "synth_test_names", synth_test_names)
        declare_array(f, "synth_test_lines", synth_test_lines)


def main():
    parser = argparse.ArgumentParser(description="Generate experiment configuration")
    parser.add_argument(
        "--exp_config_name",
        type=str,
        default="Mamba790Q2",
        help="Name of the experiment configuration",
    )
    parser.add_argument(
        "--synth_config_name",
        type=str,
        default="synth_params",
        help="Name of the synth params configuration (configs/synth_configs/<name>.json)",
    )
    args = parser.parse_args()
    generate_exp_config(args.exp_config_name, args.synth_config_name)


if __name__ == "__main__":
    main()
