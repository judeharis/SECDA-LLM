#!/bin/bash
eval "$(conda shell.bash hook)"
conda activate test

send_pushbullet_notification() {
  #local message="$1"
  local message="$*"
  curl -s -o /dev/null --header 'Access-Token: o.eIEuBUZBIooNKzofTc6WATcyobjqK4TD' \
    --header 'Content-Type: application/json' \
    --data-binary "{\"body\":\"${message}\",\"title\":\"Jude Home (Ubuntu)\",\"type\":\"note\"}" \
    --request POST \
    https://api.pushbullet.com/v2/pushes
  push=$?
  echo "Pushbullet response: $push"
}

run_exp() {
  local exp_name="$1"
  pushd scripts
  conda activate test
  python exp2.py --exp_config_name "$exp_name"
  popd
  conda activate base
  # ./kria_exp_bfpp.sh --name "$exp_name" -C
  ./kria_exp_bfpp.sh -C
  # send_pushbullet_notification "$exp_name"
}

# run_exp Mamba790Q2
# run_exp Mamba790Q3L
# run_exp Mamba790Q3M
# run_exp Mamba790Q3S
# run_exp Mamba790Q4M
# run_exp Mamba790Q4S
# run_exp Mamba790Q5M
# run_exp Mamba790Q5S
# run_exp Mamba790Q6
# run_exp TinyLlama2Q2
# run_exp TinyLlama2Q3L
# run_exp TinyLlama2Q3M
# run_exp TinyLlama2Q3S
# run_exp TinyLlama2Q4M
# run_exp TinyLlama2Q4S
# run_exp TinyLlama2Q5M
# run_exp TinyLlama2Q5S
# run_exp TinyLlama2Q6



# run_exp Granite4M350Q2
run_exp Granite4M350Q3S
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


# run_exp MobileLLaMAQ2
# run_exp TinyLlama1Q2
# run_exp GPT2Q2

# run_exp TinyLlama2Q2
# run_exp Mamba790Q2
