#!/bin/bash

# Define values for iteration
gamma_values="0.05"
delta_values="15"
bl_type="hard"
#bl_type="soft"
#mode="gpt"
#dataset="multi_news"   
datasets="finance_qa longform_qa multi_news qmsum"   
mode_list="onebitsparsenormalhash"
cuda=6
# Iterate through gamma values
for dataset in $datasets; do
    for mode in $mode_list; do
        for gamma in $gamma_values; do
            # Iterate through delta values
            for delta in $delta_values; do
                # Set variables for bl_type and mode

                # Execute the command with the specified parameters
                CUDA_VISIBLE_DEVICES=$cuda python pred.py \
                    --mode $mode \
                    --gamma $gamma \
                    --delta $delta \
                    --bl_type $bl_type \
                    --dataset $dataset \
                    --model llama2-7b-chat-4k \
                    --hyper_parameter_dir \
                    --pos_tag NN
                    #--pos_tag "!" "#" "$" "''" "(" ")" "," "LRB" "RRB" "." ":" "?" "\`\`"\
                    
                # Add any additional commands here if needed
                
            done
        done
    done
done