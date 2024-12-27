#!/bin/bash

export CUDA_VISIBLE_DEVICES=0,1,2,3
pip install /mnt/afs/user/wuxianye/test/auto_api/lightllm-api/zzm/fschat-0.2.36-py3-none-any.whl
pip install aiofiles openai
pip install /mnt/afs/user/wuxianye/test/auto_api/lightllm-api/zzm/opencv_python_headless-4.10.0.84-cp37-abi3-manylinux_2_17_x86_64.manylinux2014_x86_64.whl


MODEL_PATH=/mnt/afs/user/wuxianye/share_data/vit6b_qwen2_5_72b_stage3_v8.2_stage4_dpo_20241205/
VIT_ENV=vit6b+aspect_v5+vit
MAX_PATCH_NUM=-1
IMAGE_TOKEN_NUM=256
gpus=4
eos_id=151645
dataset_id=107
model_name=vit6b_qwen2_5_72b_stage3_v8.2_stage4_dpo_20241205
model_id=2123
system_prompt=''
max_new_tokens=4096
temperature=0.5
top_k=20
top_p=0.25
repetition_penalty=1.05
do_sample=True
stop_sequences=['<|im_end|>']
skip_special_tokens=True



model_path=$MODEL_PATH  # Replace with your model path
VIT_ENV=$VIT_ENV DISABLE_CHECK_MAX_LEN_INFER=1 MAX_PATCH_NUM=$MAX_PATCH_NUM IMAGE_TOKEN_NUM=$IMAGE_TOKEN_NUM LOADWORKER=4 /opt/conda/bin/python -m lightllm.server.api_server     --host 0.0.0.0     --port 8080     --tp $gpus     --eos_id $eos_id     --max_req_input_len 32000     --max_req_total_len 34000     --max_total_token_num 80000     --model_dir $model_path     --mode triton_gqa_flashdecoding     --trust_remote_code     --tokenizer_mode fast     --enable_multimodal     --nccl_port 27888     --data_type bf16 &

check_port() {
    local host="$1"
    local port="$2"
    local retries="$3"
    local delay="$4"

    for i in $(seq 1 "$retries"); do
        python3 -c "import socket; sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM); exit(0) if sock.connect_ex((\"$host\", $port)) == 0 else exit(1)"
        if [ $? -eq 0 ]; then
            echo "Port $port is available."
            return 0
        else
            echo "Waiting for port $port to become available... Attempt $i/$retries"
            sleep "$delay"
        fi
    done
    return 1
}

if ! check_port "0.0.0.0" 8080 60 10; then
    echo "8080 is not available"
    exit 1
fi

python3 -m fastchat.serve.controller --host 0.0.0.0  --port 21001 &

if ! check_port "0.0.0.0" 21001 30 10; then
    echo "21001 is not available"
    exit 1
fi

python3 -m fastchat.serve.lightllm_worker_internvl     --host 0.0.0.0     --port 21390     --api-url http://0.0.0.0:8080     --conv-template internlm2-chat-v3     --controller-address http://0.0.0.0:21001     --img-token-number $IMAGE_TOKEN_NUM     --eos-id $eos_id     --model-name $model_name  & 

if ! check_port "0.0.0.0" 21390 30 10; then
    echo "21390 is not available"
    exit 1
fi

python3 -m fastchat.serve.internvl_openai_api_server     --host 0.0.0.0 --port 8000      --controller-address http://0.0.0.0:21001 &

if ! check_port "0.0.0.0" 8000 30 10; then
    echo "8000 is not available"
    exit 1
fi
SYSTEM_PROMPT_ARG=""
if [[ -n "$system_prompt" ]]; then
SYSTEM_PROMPT_ARG="--system-prompt "$system_prompt""
fi
python /mnt/afs/user/zhengzhimeng/FastChat/test_dataset/test_async.py      --dataset-id $dataset_id      --output-file  /mnt/afs/user/zhengzhimeng/FastChat/test_dataset/output.json        --max_concurrent_tasks  10      --openai_server_url  http://0.0.0.0:8000/v1/      --model_id $model_id     $SYSTEM_PROMPT_ARG      --temperature $temperature      --max_tokens $max_new_tokens     --top_k $top_k      --top_p $top_p      --repetition_penalty $repetition_penalty      --do_sample $do_sample      --stop_sequences $stop_sequences      --skip_special_tokens $skip_special_tokens      