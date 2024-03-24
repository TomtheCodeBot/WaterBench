export CUDA_VISIBLE_DEVICES=6

#python detect.py \
#    --input_dir ./pred/llama2-7b-chat-4k_v2_g0.25_d15.0_hard --threshold 4.5\
#
#python detect.py \
#    --input_dir ./pred/llama2-7b-chat-4k_old_g0.25_d4.5_hard --threshold 4.5\
#    
#python detect.py \
#    --input_dir ./pred/llama2-7b-chat-4k_gpt_g0.1_d10.0_hard --threshold 4.2\
#
#python eval.py \
#    --input_dir ./pred/llama2-7b-chat-4k_v2_g0.25_d15.0_hard  --threshold 4.5
#
#python eval.py \
#    --input_dir ./pred/llama2-7b-chat-4k_old_g0.25_d4.5_hard --threshold 4.5
#
#python eval.py \
#    --input_dir ./pred/llama2-7b-chat-4k_gpt_g0.1_d10.0_hard --threshold 4.2
#
#python detect_human.py \
#    --reference_dir llama2-7b-chat-4k_gpt_g0.1_d10.0_hard  \
#    --detect_dir llama2-7b-chat-4k_no_g0.2_d10.0_hard --threshold 4.2\
#
#python detect_human.py \
#    --reference_dir llama2-7b-chat-4k_v2_g0.25_d15.0_hard  \
#    --detect_dir llama2-7b-chat-4k_no_g0.2_d10.0_hard --threshold 4.5\
#
#python detect_human.py \
#    --reference_dir  llama2-7b-chat-4k_old_g0.25_d4.5_hard \
#    --detect_dir llama2-7b-chat-4k_no_g0.2_d10.0_hard --threshold 4.5\

python detect.py \
    --input_dir /home/duy/WaterBench/all_sub/llama2-7b-chat-4k_v2_g0.25_d15.0_hard --threshold 4.5\

python detect.py \
   --input_dir /home/duy/WaterBench/all_sub/llama2-7b-chat-4k_old_g0.25_d4.5_hard --threshold 4.5\

python detect.py \
   --input_dir /home/duy/WaterBench/all_sub/llama2-7b-chat-4k_gpt_g0.1_d10.0_hard --threshold 4.5\


python detect_human.py \
    --reference_dir  llama2-7b-chat-4k_sparsev2_g0.25_d4.5_hard \
    --detect_dir llama2-7b-chat-4k_no_g0.2_d10.0_hard --threshold 4.5\