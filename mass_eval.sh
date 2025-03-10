#!/bin/bash
#SBATCH -c 4 # request two cores 
#SBATCH -p laolab
#SBATCH -o logs/eval_notagcap.out
#SBATCH -e logs/error_eval_notagcap.out
#SBATCH --mem=48G
#SBATCH --time=1-00:00:00
#SBATCH --job-name=eval_notagcap
#SBATCH --ntasks-per-node=1


# Define the directory path
directory="/cluster/tufts/laolab/kdoan02/WaterBench/hyperparameter_tuning-phi-3-mini-4k-instruct/notagsparse"

# List all folders in the directory
folders=$(ls -d $directory/*/)
folders="/cluster/tufts/laolab/kdoan02/WaterBench/hyperparameter_tuning/notagcap/llama2-7b-chat-4k_notagcap_g0.25_d6.0_hard /cluster/tufts/laolab/kdoan02/WaterBench/hyperparameter_tuning-phi-3-mini-4k-instruct/notagcap/phi-3-mini-4k-instruct_notagcap_g0.1_d7.0_hard"
# Iterate through the folders
for folder in $folders; do
    full_path=$(realpath $folder)
    echo "Folder: $full_path"
    python eval.py \
    --input_dir $full_path 
done