# change whatever you need
export CUDA_VISIBLE_DEVICES=0 

python pred.py \
    --mode sparse \
    --gamma 0.25 \
    --delta 4.5 \
    --bl_type hard --dataset multi_news \
    --model llama2-7b-chat-4k 

python pred.py \
    --mode old \
    --gamma 0.25 \
    --delta 4.5 \
    --bl_type hard --dataset multi_news \
    --model llama2-7b-chat-4k 

python pred.py \
    --mode old \
    --gamma 0.1 \
    --delta 10 \
    --bl_type soft --dataset multi_news \
    --model llama2-7b-chat-4k 


python pred.py \
    --mode gpt \
    --gamma 0.1 \
    --delta 10 \
    --bl_type hard --dataset multi_news \
    --model llama2-7b-chat-4k 

python pred.py \
    --mode v2 \
    --gamma 0.25 \
    --delta 15 \
    --bl_type hard --dataset multi_news \
    --model llama2-7b-chat-4k 


python pred.py \
    --mode sparse \
    --gamma 0.25 \
    --delta 4.5 \
    --bl_type hard --dataset qmsum \
    --model llama2-7b-chat-4k 

python pred.py \
    --mode old \
    --gamma 0.25 \
    --delta 4.5 \
    --bl_type hard --dataset qmsum \
    --model llama2-7b-chat-4k 

python pred.py \
    --mode old \
    --gamma 0.1 \
    --delta 10 \
    --bl_type soft --dataset qmsum \
    --model llama2-7b-chat-4k 


python pred.py \
    --mode gpt \
    --gamma 0.1 \
    --delta 10 \
    --bl_type hard --dataset qmsum \
    --model llama2-7b-chat-4k 

python pred.py \
    --mode v2 \
    --gamma 0.25 \
    --delta 15 \
    --bl_type hard --dataset qmsum \
    --model llama2-7b-chat-4k 
