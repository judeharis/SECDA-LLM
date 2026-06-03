declare -a models_array=(
  "granite-4.0-350m-Q6_K.gguf" 
)
declare -a model_names_array=(
  "Granite4M350Q6" 
)
declare -a bitstreams_array=(
  "BFP_Q2Q3Q4Q5Q6_v3" 
)
declare -a binaries_array=(
  "llama_cli_v3A_secda_q2q3q4q5q6" 
)
declare -a acc_tags_array=(
  "BFPP_KRIA_v3.0A_q2q3q4q5q6" 
)
declare -a bins_array=(
  "bin_q2q3q4q5q6" 
)
declare -a accelerated_array=(
  "true" 
)
declare -a bin_flags_array=(
  "-DSECDA_QK2=ON -DSECDA_QK3=ON -DSECDA_QK4=ON -DSECDA_QK5=ON -DSECDA_QK6=ON -DSECDA_BFPP_ACC_V3=ON" 
)
declare -a runtimes_array=(
  "q2q3q4q5q6v3.0A" 
)
