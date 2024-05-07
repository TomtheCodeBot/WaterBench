#!/bin/bash

# Define values for iteration
gamma_values="0.1"
delta_values="5"
bl_type="hard"
#mode="gpt"
#dataset="multi_news"   
datasets="finance_qa longform_qa"   
mode_list="onebitsparsenormalhash"
cuda=5
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
                    --pos_tag JJ\
                # Add any additional commands here if needed
                
            done
        done
    done
done