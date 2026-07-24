declare -a models_array=(
  "tiny-llama-miniguanaco-1.5t.q2_k.gguf" 
)
declare -a model_names_array=(
  "TinyLlama1Q2" 
)
declare -a bitstreams_array=(
  "BFP_Q2Q3Q4Q5Q6_v2" 
  "CPU_KRIA_1_0" 
)
declare -a binaries_array=(
  "llama_cli_v2A_secda_q2q3q4q5q6" 
  "llama_cli_v1_cpu" 
)
declare -a acc_tags_array=(
  "BFPP_KRIA_v2.0A_q2q3q4q5q6" 
  "CPU_KRIA_v1.0_cpu" 
)
declare -a bins_array=(
  "bin_q2q3q4q5q6" 
  "bin_cpu" 
)
declare -a accelerated_array=(
  "true" 
  "false" 
)
declare -a bin_flags_array=(
  "-DSECDA_QK2=ON -DSECDA_QK3=ON -DSECDA_QK4=ON -DSECDA_QK5=ON -DSECDA_QK6=ON -DSECDA_BFPP_ACC_V3=ON" 
  "-DNOPERF=ON -DGGML_SECDA=OFF" 
)
declare -a runtimes_array=(
  "q2q3q4q5q6v2.0A" 
  "cpuv1.0" 
)
declare -a synth_test_names_array=(
  "Q2_128M_8N_256K" 
)
declare -a synth_test_lines_array=(
  "29 0 128 8 1 1 16 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 2 10 256 128 1 1 0 0 0 0 0 256 8 1 1 0 0 0 0 Q2_128M_8N_256K" 
)
