#!/bin/bash
#SBATCH -c 4 # request two cores 
#SBATCH -p preempt
#SBATCH -o perplexity_calc.out
#SBATCH -e error_perplexity_calc.out
#SBATCH --mem=96G
#SBATCH --time=1-00:00:00
#SBATCH --job-name=SparseWatermark
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:a100:1

# Define the directory path
directory="/cluster/tufts/laolab/kdoan02/selected_phi_results"

# List all folders in the directory
#folders=$(ls -d $directory/*/)
#folders="/cluster/tufts/laolab/kdoan02/selected_results/llama2-7b-chat-4k_no_g0.2_d10.0_hard"
# Define values for iteration/cluster/tufts/laolab/kdoan02/selected_phi_results
model="llama2-13b"
key="pred"
mode="hf"


for folder in $folders; do
    full_path=$(realpath $folder)
    # Iterate through gamma values
    python perplexity_calc.py $full_path $model $key $mode
done
