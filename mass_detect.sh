#!/bin/bash
#SBATCH -c 4 # request two cores 
#SBATCH -p preempt
#SBATCH -o logs/detect_notagcap.out
#SBATCH -e logs/error_detect_notagcap.out
#SBATCH --mem=48G
#SBATCH --time=1-00:00:00
#SBATCH --job-name=detect_notagcap
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:a100:1



# Define the directory path
directory="/cluster/tufts/laolab/kdoan02/WaterBench/hyperparameter_tuning/ewd"

# List all folders in the directory
# folders=$(ls -d $directory/*/)
#folders="/cluster/tufts/laolab/kdoan02/WaterBench/hyperparameter_tuning/onebitsparsenormalhash/llama2-7b-chat-4k_onebitsparsenormalhash-DT-EX-PDT-WDT_g0.05_d5.0_hard"
#folders="/cluster/tufts/laolab/kdoan02/WaterBench/hyperparameter_tuning/notagsparse/llama2-7b-chat-4k_notagsparse_g0.1_d4.0_hard"
#folders="/cluster/tufts/laolab/kdoan02/WaterBench/hyperparameter_tuning/notagsparse/llama2-7b-chat-4k_sweetsparse_g0.1_d3.0_hard"
folders="/cluster/tufts/laolab/kdoan02/WaterBench/hyperparameter_tuning/notaglsh/llama2-7b-chat-4k_notaglsh_g0.05_d6.0_hard"
threshold=4.0
# Iterate through the folders
for folder in $folders; do
    full_path=$(realpath $folder)
    echo "Folder: $full_path"
    python detect.py \
    --input_dir $full_path --threshold $threshold
done