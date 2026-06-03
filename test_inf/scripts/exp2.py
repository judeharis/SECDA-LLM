import argparse
import json


def generate_exp_config(exp_config_name):

    def declare_array(f, name, list):
        f.write("declare -a {}_array=(\n".format(name))
        for i in list:
            f.write('  "{}" \n'.format(i))
        f.write(")\n")

    with open("configs/runtime_dict_og.json", "r") as jf:
        runtime_dict = json.load(jf)

    with open("configs/runtime_dict_1.json", "r") as jf:
        runtime_dict_1 = json.load(jf)

    with open("configs/runtime_dict_2.json", "r") as jf:
        runtime_dict_2 = json.load(jf)

    with open("configs/runtime_dict_3.json", "r") as jf:
        runtime_dict_3 = json.load(jf)

    runtime_dict.update(runtime_dict_1)
    runtime_dict.update(runtime_dict_2)
    runtime_dict.update(runtime_dict_3)

    with open("configs/models_dict.json", "r") as jf:
        models_dict = json.load(jf)
    # models_dict = {
    #     "TinyLlama": "tiny-llama-miniguanaco-1.5t.q2_k.gguf",
    #     "MobileLLaMA": "MobileLLaMA-1.4B-Base-Q2_K.gguf",
    #     "GPT2": "Cryptography_GPT_2_v1.0.0.Q2_K.gguf",
    #     "MobileLLM": "MobileLLM-125M-HF.Q2_K.gguf",
    #     "llama600M": "llama-600M-rus.Q3_K.gguf",
    #     "NanoMistral": "nano-mistral-q3_k_s.gguf",
    #     "TinyLlama2Q2": "tinyllama-2-1b-miniguanaco.Q2_K.gguf",
    #     "TinyLlama2Q3L": "tinyllama-2-1b-miniguanaco.Q3_K_L.gguf",
    #     "TinyLlama2Q3M": "tinyllama-2-1b-miniguanaco.Q3_K_M.gguf",
    #     "TinyLlama2Q3S": "tinyllama-2-1b-miniguanaco.Q3_K_S.gguf",
    #     "TinyLlama2Q4M": "tinyllama-2-1b-miniguanaco.Q4_K_M.gguf",
    #     "TinyLlama2Q4S": "tinyllama-2-1b-miniguanaco.Q4_K_S.gguf",
    #     "TinyLlama2Q5M": "tinyllama-2-1b-miniguanaco.Q5_K_M.gguf",
    #     "TinyLlama2Q5S": "tinyllama-2-1b-miniguanaco.Q5_K_S.gguf",
    #     "TinyLlama2Q6": "tinyllama-2-1b-miniguanaco.Q6_K.gguf",
    #     "Mamba790Q2": "mamba-790m-hf.Q2_K.gguf",
    #     "Mamba790Q3L": "mamba-790m-hf.Q3_K_L.gguf",
    #     "Mamba790Q3M": "mamba-790m-hf.Q3_K_M.gguf",
    #     "Mamba790Q3S": "mamba-790m-hf.Q3_K_S.gguf",
    #     "Mamba790Q4M": "mamba-790m-hf.Q4_K_M.gguf",
    #     "Mamba790Q4S": "mamba-790m-hf.Q4_K_S.gguf",
    #     "Mamba790Q5M": "mamba-790m-hf.Q5_K_M.gguf",
    #     "Mamba790Q5S": "mamba-790m-hf.Q5_K_S.gguf",
    #     "Mamba790Q6": "mamba-790m-hf.Q6_K.gguf",
    # }

    exp_config_path = "configs/exp_configs/" + exp_config_name + ".json"
    with open(exp_config_path, "r") as jf:
        exp_config = json.load(jf)

    models = exp_config["models"]
    runtimes = exp_config["runtimes"]

    bitstreams_list = [runtime_dict[runtime]["bitstreams"] for runtime in runtimes]
    binaries_list = [runtime_dict[runtime]["binaries"] for runtime in runtimes]
    acc_tags_list = [runtime_dict[runtime]["acc_tag"] for runtime in runtimes]
    bins_list = [runtime_dict[runtime]["bins"] for runtime in runtimes]
    accelerated_list = [runtime_dict[runtime]["accelerated"] for runtime in runtimes]
    models_list = [models_dict[model] for model in models]
    model_names_list = [model for model in models]
    bin_flags_list = [runtime_dict[runtime]["bin_flags"] for runtime in runtimes]

    f = open(f"exp_configs.sh", "w+")
    declare_array(f, "models", models_list)
    declare_array(f, "model_names", model_names_list)
    declare_array(f, "bitstreams", bitstreams_list)
    declare_array(f, "binaries", binaries_list)
    declare_array(f, "acc_tags", acc_tags_list)
    declare_array(f, "bins", bins_list)
    declare_array(f, "accelerated", accelerated_list)
    declare_array(f, "bin_flags", bin_flags_list)
    declare_array(f, "runtimes", runtimes)

    f.close()


def main():
    parser = argparse.ArgumentParser(description="Generate experiment configuration")
    parser.add_argument(
        "--exp_config_name",
        type=str,
        default="Mamba790Q2",
        help="Name of the experiment configuration",
    )
    args = parser.parse_args()
    generate_exp_config(args.exp_config_name)


if __name__ == "__main__":
    main()
