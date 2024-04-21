#!/bin/bash

# Define the directory path
directory="/home/duy/WaterBench/hyperparameter_tuning/onebitsparse"

# List all folders in the directory
folders=$(ls -d $directory/*0.25*0.0/)
#folders="/home/duy/WaterBench/hyperparameter_tuning/onebitsparse/llama2-7b-chat-4k_onebitsparse_g0.7_d5.0 /home/duy/WaterBench/hyperparameter_tuning/onebitsparse/llama2-7b-chat-4k_onebitsparse_g0.7_d10.0 /home/duy/WaterBench/hyperparameter_tuning/onebitsparse/llama2-7b-chat-4k_onebitsparse_g0.7_d15.0"
threshold=4.0
# Iterate through the folders
for folder in $folders; do
    full_path=$(realpath $folder)
    echo "Folder: $full_path"
    python detect.py \
    --input_dir $full_path --threshold $threshold
done