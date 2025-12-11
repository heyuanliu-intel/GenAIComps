python3 opea_text2video_microservice.py &

port=30006
rank=4
export FA3_Q_CHUNK=8192
export FA3_KV_CHUNK=8192
export USE_SP=1
export PT_HPU_LAZY_MODE=1

deepspeed --num_nodes 1 \
    --num_gpus $rank \
    --no_local_rank \
    --master_port $port \
    /home/user/comps/text2video/src/text_to_video_generation.py \
    --use_habana \
    --dtype bf16 \
    --context_parallel_size $rank