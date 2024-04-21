#!/bin/bash

# Define values for iteration
gamma_values="0.25"
delta_values="5"
bl_type="hard"
#mode="gpt"
dataset="finance_qa"   
mode_list="old onebitsparse"
cuda=2
# Iterate through gamma values
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
            # Add any additional commands here if needed
            
        done
    done
done