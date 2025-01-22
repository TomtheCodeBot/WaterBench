#!/bin/bash
#SBATCH -c 4 # request two cores 
#SBATCH -p laolab
#SBATCH -o logs/notag_7_8_9_Phi.out
#SBATCH -e logs/error_notag_7_8_9_Phi.out
#SBATCH --mem=48G
#SBATCH --time=1-00:00:00
#SBATCH --job-name=notag_7_8_9_Phi
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:l40s:1


# Define values for iteration
gamma_values="0.1 0.25 0.5"
delta_values="7 8 9" # In the case of no tag sparse watermark, delta indicates the modular number
bl_type="hard"
#bl_type="soft"
#mode="gpt"
#dataset="multi_news"   
#datasets="longform_qa"   
#datasets="finance_qa multi_news qmsum"   
datasets="longform_qa finance_qa multi_news qmsum" 
mode_list="notagsparse"
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
                    --model phi-3-mini-4k-instruct \
                    --hyper_parameter_dir \
                    #--pos_tag NN NP
                    #--pos_tag "!" "#" "$" "''" "(" ")" "," "LRB" "RRB" "." ":" "?" "\`\`"\
                    
                # Add any additional commands here if needed
                
            done
        done
    done
done