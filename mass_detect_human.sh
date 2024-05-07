#!/bin/bash

# Define the directory path
directory="/home/duy/WaterBench/hyperparameter_tuning/onebitsparsenormalhash"

# List all folders in the directory
folders=$(ls -d $directory/*/)
threshold=4.0
# Iterate through the folders
for folder in $folders; do
    full_path=$(realpath $folder)
    echo "Folder: $full_path"
    
    python detect_human.py \
        --reference_dir  $full_path \
        --detect_dir llama2-7b-chat-4k_no_g0.2_d10.0_hard --threshold $threshold
done