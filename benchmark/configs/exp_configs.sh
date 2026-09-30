declare -a models_array=(
  "MobileLLM-125M-HF.Q2_K.gguf" 
)
declare -a model_names_array=(
  "MobileLLMQ2" 
)
declare -a bitstreams_array=(
  "BFPP_ACC_KRIA_3_0" 
  "BFPP_ACC_KRIA_3_0" 
  "BFPP_ACC_KRIA_4_0" 
  "BFPP_ACC_KRIA_4_0" 
  "CPU_1_0" 
)
declare -a binaries_array=(
  "llama_cli_bfpp_v3" 
  "llama_cli_bfpp_v3b" 
  "llama_cli_bfpp_v4" 
  "llama_cli_bfpp_v4b" 
  "llama_cli_cpu" 
)
declare -a acc_tags_array=(
  "BFPP_KRIA_v3.0_driver" 
  "BFPP_KRIA_v3.0_batches" 
  "BFPP_KRIA_v4.0_driver" 
  "BFPP_KRIA_v4.0_batches" 
  "CPU_KRIA_v1.0_cpu" 
)
declare -a bins_array=(
  "bin_bfpp_v3" 
  "bin_bfpp_v3b" 
  "bin_bfpp_v4" 
  "bin_bfpp_v4b" 
  "bin_cpu" 
)
declare -a accelerated_array=(
  "true" 
  "true" 
  "true" 
  "true" 
  "false" 
)
declare -a bin_flags_array=(
  "-DSECDA_QK2=ON -DSECDA_QK3=ON -DSECDA_QK4=ON -DSECDA_QK5=ON -DSECDA_QK6=ON -DSECDA_BFPP_ACC_V3=ON" 
  "-DSECDA_QK2=ON -DSECDA_QK3=ON -DSECDA_QK4=ON -DSECDA_QK5=ON -DSECDA_QK6=ON -DSECDA_BFPP_ACC_V3=ON -DSECDA_BFPP_ACC_V3_DRIVER_BATCHES=ON" 
  "-DSECDA_QK2=ON -DSECDA_QK3=ON -DSECDA_QK4=ON -DSECDA_QK5=ON -DSECDA_QK6=ON -DSECDA_BFPP_ACC_V4=ON" 
  "-DSECDA_QK2=ON -DSECDA_QK3=ON -DSECDA_QK4=ON -DSECDA_QK5=ON -DSECDA_QK6=ON -DSECDA_BFPP_ACC_V4=ON -DSECDA_BFPP_ACC_V4_DRIVER_BATCHES=ON" 
  "-DNOPERF=ON -DGGML_SECDA=OFF" 
)
declare -a runtimes_array=(
  "bfpp.v3" 
  "bfpp.v3b" 
  "bfpp.v4" 
  "bfpp.v4b" 
  "cpu.v6" 
)
declare -a synth_test_names_array=(
  "Q2_128M_4N_256K" 
  "Q3_128M_4N_256K" 
  "Q4_128M_4N_256K" 
  "Q5_128M_4N_256K" 
  "Q6_128M_4N_256K" 
)
declare -a synth_test_lines_array=(
  "29 0 128 4 1 1 16 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 2 10 256 128 1 1 0 0 0 0 0 256 4 1 1 0 0 0 0 Q2_128M_4N_256K" 
  "29 0 128 4 1 1 16 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 2 11 256 128 1 1 0 0 0 0 0 256 4 1 1 0 0 0 0 Q3_128M_4N_256K" 
  "29 0 128 4 1 1 16 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 2 12 256 128 1 1 0 0 0 0 0 256 4 1 1 0 0 0 0 Q4_128M_4N_256K" 
  "29 0 128 4 1 1 16 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 2 13 256 128 1 1 0 0 0 0 0 256 4 1 1 0 0 0 0 Q5_128M_4N_256K" 
  "29 0 128 4 1 1 16 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 2 14 256 128 1 1 0 0 0 0 0 256 4 1 1 0 0 0 0 Q6_128M_4N_256K" 
)
