# change whatever you need
export CUDA_VISIBLE_DEVICES=3
export TORCH_COMPILE_DEBUG=1
#python pred.py \
#    --mode sparse \
#    --gamma 0.25 \
#    --delta 4.5 \
#    --bl_type hard --dataset qmsum \
#    --model llama2-7b-chat-4k 
#
#python pred.py \
#    --mode old \
#    --gamma 0.25 \
#    --delta 4.5 \
#    --bl_type hard --dataset qmsum \
#    --model llama2-7b-chat-4k 

#python pred.py \
#    --mode old \
#    --gamma 0.1 \
#    --delta 10 \
#    --bl_type soft --dataset multi_news \
#    --model llama2-7b-chat-4k 
#



#python pred.py \
#    --mode no \
#    --gamma 0.2 \S
#    --delta 10 \
#    --bl_type hard --dataset multi_news \
#    --model llama2-7b-chat-4k 
#
#python pred.py \
#    --mode no \
#    --gamma 0.2 \
#    --delta 10 \
#    --bl_type hard --dataset qmsum \
#    --model llama2-7b-chat-4k 
#python pred.py \
#    --mode sparsev2 \
#    --gamma 0.25 \
#    --delta 4.5 \
#    --bl_type hard --dataset qmsum \
#    --model llama2-7b-chat-4k 

#python pred.py \
#    --mode ogv2 \
#    --gamma 0.25 \
#    --delta 15 \
#    --bl_type hard --dataset longform_qa \
#    --model llama2-7b-chat-4k
#
#python pred.py \
#    --mode ogv2 \
#    --gamma 0.25 \
#    --delta 15 \
#    --bl_type hard --dataset finance_qa \
#    --model llama2-7b-chat-4k

#python pred.py \
#    --mode sparsev2 \
#    --gamma 0.25 \
#    --delta 4.5 \
#    --bl_type hard --dataset longform_qa \
#    --model llama2-7b-chat-4k \
#    --random_bit_String

#python pred.py \
#    --mode sparsev2 \
#    --gamma 0.25 \
#    --delta 4.5 \
#    --bl_type hard --dataset finance_qa \
#    --model llama2-7b-chat-4k \
#    --random_bit_String

#python pred.py \
#    --mode sparsev2 \
#    --gamma 0.25 \
#    --delta 4.5 \
#    --bl_type hard --dataset multinews \
#    --model llama2-7b-chat-4k \
#    --random_bit_String

#python pred.py \
#    --mode sparsev2 \
#    --gamma 0.25 \
#    --delta 4.5 \
#    --bl_type hard --dataset qmsum \
#    --model llama2-7b-chat-4k \
#    --random_bit_String

python pred.py \
    --mode onebitsparse \
    --gamma 0.2 \
    --delta 4.5 \
    --bl_type hard --dataset multi_news \
    --model llama2-7b-chat-4k 
