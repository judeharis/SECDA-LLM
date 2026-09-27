declare -a models_array=(
  "tiny-llama-miniguanaco-1.5t.q2_k.gguf" 
)
declare -a model_names_array=(
  "TinyLlama1Q2" 
)
declare -a bitstreams_array=(
  "BFPB_Q3_v1" 
  "CPU_KRIA_1_0" 
)
declare -a binaries_array=(
  "llama_cli_v1A_secda_q3" 
  "llama_cli_v1_cpu" 
)
declare -a acc_tags_array=(
  "BFPB_KRIA_v1.0A_q3" 
  "CPU_KRIA_v1.0_cpu" 
)
declare -a bins_array=(
  "bin_bfpb_q3" 
  "bin_cpu" 
)
declare -a accelerated_array=(
  "true" 
  "false" 
)
declare -a bin_flags_array=(
  "-DSECDA_QK3=ON -DSECDA_BFPP_ACC_V3=ON" 
  "-DNOPERF=ON -DGGML_SECDA=OFF" 
)
declare -a runtimes_array=(
  "bfpbp.q3v1.0A" 
  "cpuv1.0" 
)
