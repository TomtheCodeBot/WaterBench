#!/bin/bash

# Define the directory path
directory="/home/duy/WaterBench/hyperparameter_tuning/onebitsparsenormalhash"

# List all folders in the directory
# folders=$(ls -d $directory/*/)
folders="/home/duy/WaterBench/hyperparameter_tuning/onebitsparsenormalhash/llama2-7b-chat-4k_onebitsparsenormalhash-!-#-\$-''-(-)-,-LRB-RRB-.-:-?-\`\`_g0.1_d15.0_hard /home/duy/WaterBench/hyperparameter_tuning/onebitsparsenormalhash/llama2-7b-chat-4k_onebitsparsenormalhash-MD-VB-VP_g0.1_d15.0_hard"
# Iterate through the folders
for folder in $folders; do
    full_path=$(realpath $folder)
    echo "Folder: $full_path"
    python eval.py \
    --input_dir $full_path 
done