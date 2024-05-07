#!/bin/bash

# Define the directory path
directory="/home/duy/WaterBench/hyperparameter_tuning/onebitsparsenormalhash"

# List all folders in the directory
folders=$(ls -d $directory/*/)
# Iterate through the folders
for folder in $folders; do
    full_path=$(realpath $folder)
    echo "Folder: $full_path"
    python eval.py \
    --input_dir $full_path 
done