#!/bin/bash
# set -x
set -e

source ./exp_configs.sh

models=("${models_array[@]}")
model_names=("${model_names_array[@]}")
bitstreams=("${bitstreams_array[@]}")
binaries=("${binaries_array[@]}")
acc_tag=("${acc_tags_array[@]}")
bins=("${bins_array[@]}")
accelerated=("${accelerated_array[@]}")

threads=(1)
# threads=(1)

# board_path
board_path="/home/ubuntu/Workspace/secda_llm"
board_sub="benchmark"

# create accelerated or not accelerated array

# create a loop for each model
run_arg='--single-turn --no-warmup --temp 0 -s 1712523969 -p "what is my name?" --log-file "${mn}_${thread}_${tag}"'
num_runs=20
temp=10

rm -f ./commands.txt
touch ./commands.txt

echo "Running Experiment for Kria"

function error_exit {
  echo "error at line $1 in $2"

  echo "-----------------------------------------------------------"
  echo "Clearing Bitstream"
  echo "-----------------------------------------------------------"
  bash -i -c 'cd /home/ubuntu/Workspace/secda_llm/bitstreams && python3 ~/load_bitstream.py -q CPU_KRIA_1_0.bit'

  rm -f sds_trace_data.dat
  exit 1
}

trap 'error_exit ${LINENO} ${BASH_SOURCE}' ERR

echo "-----------------------------------------------------------"
echo "Clear UDMA"
echo "-----------------------------------------------------------"
                          
cat /proc/meminfo | grep -i cma
clear_udmabuf_if_exists() {
  local idx="$1"
  if [ -e "/dev/udmabuf${idx}" ]; then
    echo "delete udmabuf${idx}" > /dev/u-dma-buf-mgr
    echo "Cleared /dev/udmabuf${idx}"
  fi
}
clear_all_udmabuf() {
  local dev idx

  for dev in /dev/udmabuf*; do
    [ -e "$dev" ] || continue
    idx="${dev#/dev/udmabuf}"
    [ -n "$idx" ] || continue
    clear_udmabuf_if_exists "$idx"
  done
}
clear_all_udmabuf
cat /proc/meminfo | grep -i cma
bash -i -c 'cd /home/ubuntu/bitstreams/ && python3 ~/load_bitstream.py CPU_1_0.bit'

mn_count=0
for model in "${models[@]}"; do
  mn=${model_names[$mn_count]}
  mn_count=$((mn_count + 1))
  for thread in "${threads[@]}"; do
    count=0
    for binary in "${binaries[@]}"; do
      acc=${accelerated[$count]}
      bin_folder=${bins[$count]}
      tag=${acc_tag[$count]}
      bitstream=${bitstreams[$count]}
      count=$((count + 1))

      # echo "Model: ${model}"
      # echo "Thread: ${thread}"
      # echo "Binary: ${binary}"
      # echo "Bin Folder: ${bin_folder}"
      # echo "Accelerated: $acc"
      # echo "Tag: $tag"

      echo "========================================"

      echo "Running Experiment for ${model}_${thread}_${tag}"
      # Run ACC
      if [ $acc == true ]; then
        # Map ACC
        echo "python3 ~/load_bitstream.py $board_path/bitstreams/$bitstream.bit" >>commands.txt
        python3 ~/load_bitstream.py $board_path/bitstreams/$bitstream.bit
        sudo sh -c "/bin/echo 3 > /proc/sys/vm/drop_caches" && sleep 3
      fi
      # Run ACC and pipe to file
      cd ${board_path}/${board_sub}/
      mkdir -p results
      # chmod +x log_power_KRIA.sh
      # ./log_power_KRIA.sh ../results/${mn}_${thread}_${tag}_power.txt &
      cd ${board_path}/${board_sub}/${bin_folder}

      chmod +x ./${binary}
      echo sudo ./${binary} -m ${board_path}/models/${model} -n ${num_runs} -t ${thread} ${run_arg} >>../commands.txt
      ./${binary} -m ${board_path}/models/${model} -n ${num_runs} -t ${thread} --single-turn --no-warmup --temp ${temp} -s 1712523969 -p "what is my name?" --log-file "${mn}_${thread}_${tag}" 2>&1 | tee ../results/${mn}_${thread}_${tag}.txt
      echo  "cd ${board_path}/$board_sub/${bin_folder} && ./${binary} -m ${board_path}/models/${model} -n ${num_runs} -t ${thread} ${run_arg} >../results/${model}-${thread}-${binary}.txt 2>&1" >>../commands.txt
      
      # this geneates a prf.csv file and a llama_perf.csv file, move it to the results folder
      if [ $acc == true ]; then
        mv -f prf.csv ../results/${mn}_${thread}_${tag}_prf.csv
      fi

      if [ -f llama_perf.csv ]; then
        echo "llama_perf.csv found, moving to results folder"
        mv -f llama_perf.csv ../results/${mn}_${thread}_${tag}_llama_perf.csv
      fi

      if [ -f ../log.txt ]; then
        mv -f ../log.txt ../results/${mn}_${thread}_${tag}_power.txt
      fi

      sudo sh -c "/bin/echo 3 > /proc/sys/vm/drop_caches" && sleep 1
      echo "========================================"

    done
  done
done

echo "-----------------------------------------------------------"
echo "Clearing Bitstream"
echo "-----------------------------------------------------------"
bash -i -c 'cd /home/ubuntu/Workspace/secda_llm/bitstreams && python3 ~/load_bitstream.py -q CPU_KRIA_1_0.bit'

rm -f sds_trace_data.dat

#================================================================================================