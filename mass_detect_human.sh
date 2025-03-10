#!/bin/bash
#SBATCH -c 4 # request two cores 
#SBATCH -p preempt
#SBATCH -o logs/human_notagcap.out
#SBATCH -e logs/error_human_notagcap.out
#SBATCH --mem=48G
#SBATCH --time=1-00:00:00
#SBATCH --job-name=Humanold
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:a100:1


nvidia-smi 


# Define the directory path
directory="/cluster/tufts/laolab/kdoan02/selected_phi_results/sweet-ewd-phi"
#folders="$(ls -d $directory/*/)"

# List all folders in the directory
folders="/cluster/tufts/laolab/kdoan02/WaterBench/hyperparameter_tuning/notaglsh/llama2-7b-chat-4k_notaglsh_g0.05_d6.0_hard"

threshold=4.0
# Iterate through the folders
for folder in $folders; do
    full_path=$(realpath $folder)
    echo "Folder: $full_path"
    
    python detect_human.py \
        --reference_dir  $full_path \
        --detect_dir human_generation --threshold $threshold
        #--detect_dir llama2-7b-chat-4k_no_g0.2_d10.0_hard --threshold $threshold
done