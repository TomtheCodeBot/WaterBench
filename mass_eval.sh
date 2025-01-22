#!/bin/bash
#SBATCH -c 4 # request two cores 
#SBATCH -p laolab
#SBATCH -o logs/eval_notag.out
#SBATCH -e logs/error_eval_notag.out
#SBATCH --mem=48G
#SBATCH --time=1-00:00:00
#SBATCH --job-name=Evalold
#SBATCH --ntasks-per-node=1


# Define the directory path
directory="/cluster/tufts/laolab/kdoan02/WaterBench/hyperparameter_tuning/notagsparse"

# List all folders in the directory
folders=$(ls -d $directory/*/)
#folders="/cluster/tufts/laolab/kdoan02/WaterBench/hyperparameter_tuning_sweet/sweet/llama2-7b-chat-4k_sweet_g0.5_d10.0"
# Iterate through the folders
for folder in $folders; do
    full_path=$(realpath $folder)
    echo "Folder: $full_path"
    python eval.py \
    --input_dir $full_path 
done