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

# path
path="/home/ubuntu/Workspace/secda_llm"

# create accelerated or not accelerated array

# create a loop for each model
run_arg='--single-turn --no-warmup --temp 0 -s 1712523969 -p "what is my name?" --log-file "${mn}_${thread}_${tag}"'
num_runs=10

rm -f ./commands.txt
touch ./commands.txt

# source "/usr/local/share/pynq-venv/bin/activate"
# export PYNQ_JUPYTER_NOTEBOOKS=/root/jupyter_notebooks
# export BOARD=KV260
# export XILINX_XRT=/usr
# export PATH=$PATH:/usr/local/share/pynq-venv/bin/microblazeel-xilinx-elf/bin/
# python3 /usr/local/share/pynq-venv/pynq-dts/insert_dtbo.py
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

trap 'error_exit "An error occurred. Exiting."' ERR

echo "-----------------------------------------------------------"
echo "Clear UDMA"
echo "-----------------------------------------------------------"
/home/ubuntu/udma.sh

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
        echo "python3 ~/load_bitstream.py /home/ubuntu/Workspace/secda_llm/bitstreams/$bitstream.bit" >>commands.txt
        python3 ~/load_bitstream.py /home/ubuntu/Workspace/secda_llm/bitstreams/$bitstream.bit
        sudo sh -c "/bin/echo 3 > /proc/sys/vm/drop_caches" && sleep 3
      fi
      # Run ACC and pipe to file
      cd ${path}/bfpp/
      chmod +x log_power_KRIA.sh
      ./log_power_KRIA.sh ../results/${mn}_${thread}_${tag}_power.txt &
      cd ${path}/bfpp/${bin_folder}

      chmod +x ./${binary}
      echo sudo ./${binary} -m ${path}/models/${model} -n ${num_runs} -t ${thread} ${run_arg} >>../commands.txt
      # sudo ./${binary} -m ${path}/models/${model} -n ${num_runs} -t ${thread} ${run_arg} >../results/${model}-${thread}-${binary}.txt 2>&1
      ./${binary} -m ${path}/models/${model} -n ${num_runs} -t ${thread} --single-turn --no-warmup --temp 0 -s 1712523969 -p "what is my name?" --log-file "${mn}_${thread}_${tag}" 2>&1 | tee ../results/${mn}_${thread}_${tag}.txt
      echo  "cd ${path}/bfpp/${bin_folder} && ./${binary} -m ${path}/models/${model} -n ${num_runs} -t ${thread} ${run_arg} >../results/${model}-${thread}-${binary}.txt 2>&1" >>../commands.txt
      
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
# echo "perf stat -r 1 -x, -o perf-results-tmp.csv -e ${PEVENTS_ALL} ./${binary} -m ${path}/models/${model} -n ${num_runs} -t ${thread} ${run_arg} 2>&1 | tee ../results/${model}-${thread}-${binary}.txt"
# sudo perf stat -r 1 -x, -o perf-results-tmp.csv -e ${PEVENTS_ALL} ./${binary} -m ${path}/models/${model} -n ${num_runs} -t ${thread} ${run_arg} 2>&1 | tee ../results/${model}-${thread}-${binary}.txt
