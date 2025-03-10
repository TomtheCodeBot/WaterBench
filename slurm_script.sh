#!/bin/bash
#SBATCH -c 4 # request two cores 
#SBATCH -p laolab
#SBATCH -o logs/notagnewwordlsh.out
#SBATCH -e logs/error_notagnewwordlsh.out
#SBATCH --mem=48G
#SBATCH --time=1-00:00:00
#SBATCH --job-name=notagnewwordlsh
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:l40s:1


# Define values for iteration
gamma_values="0.05"
delta_values="6" # In the case of no tag sparse watermark, delta indicates the modular number
bl_type="hard"
#bl_type="soft"
#mode="gpt"
#dataset="multi_news"   
#datasets="longform_qa"   
datasets="longform_qa finance_qa multi_news qmsum" 
# datasets="longform_qa" 
mode_list="notagnewwordlsh"
# Iterate through gamma values
for dataset in $datasets; do
    for mode in $mode_list; do
        for gamma in $gamma_values; do
            # Iterate through delta values
            for delta in $delta_values; do
                # Set variables for bl_type and mode

                # Execute the command with the specified parameters
                python pred.py \
                    --mode $mode \
                    --gamma $gamma \
                    --delta $delta \
                    --bl_type $bl_type \
                    --dataset $dataset \
                    --model llama2-7b-chat-4k \
                    --hyper_parameter_dir \
                    #--pos_tag NN NP
                    #--pos_tag "!" "#" "$" "''" "(" ")" "," "LRB" "RRB" "." ":" "?" "\`\`"\
                    
                # Add any additional commands here if needed
                
            done
        done
    done
done