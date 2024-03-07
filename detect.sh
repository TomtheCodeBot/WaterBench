export CUDA_VISIBLE_DEVICES=0

python detect.py \
    --input_dir ./pred/vicuna-v1.5-7b-16k_gpt_g0.5_d10.0_hard \

python detect.py \
    --input_dir ./pred/vicuna-v1.5-7b-16k_v2_g0.5_d10.0_hard \
    
python detect.py \
    --input_dir ./pred/vicuna-v1.5-7b-16k_old_g0.5_d10.0_hard \

python detect.py \
    --input_dir ./pred/vicuna-v1.5-7b-16k_new_g0.5_d10.0_hard \

python detect.py \
    --input_dir ./pred/vicuna-v1.5-7b-16k_sparse_g0.5_d10.0_hard