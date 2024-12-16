#!/bin/bash


pip install /mnt/afs/user/zhengzhimeng/FastChat/dist/fschat-0.2.36-py3-none-any.whl
pip install aiofiles openai
pip install /mnt/afs/user/zhengzhimeng/FastChat/test_dataset/opencv_python_headless-4.10.0.84-cp37-abi3-manylinux_2_17_x86_64.manylinux2014_x86_64.whl

eos_id=92542
model_path="/mnt/afs/user/zhengzhimeng/models/vit0.3b_internlm7b_stage3_v6.8.1_lr2e-5_20240911_v1/" 

VIT_ENV="vit0.3b+aspect_v5+internchat" \
LOADWORKER=4 /opt/conda/bin/python -m lightllm.server.api_server \
    --host 0.0.0.0                 \
    --port 8080                    \
    --tp 1                         \
    --eos_id $eos_id                 \
    --max_req_input_len 32000      \
    --max_req_total_len 34000   \
    --max_total_token_num 80000    \
    --model_dir $model_path        \
    --mode triton_gqa_flashdecoding \
    --trust_remote_code            \
    --tokenizer_mode fast          \
    --enable_multimodal            \
    --nccl_port 27888              \
    --data_type bf16 &

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

python3 -m fastchat.serve.lightllm_worker_internvl  \
    --host 0.0.0.0  \
    --port 21390   \
    --api-url http://0.0.0.0:8080  \
    --conv-template internlm2-chat-v3  \
    --controller-address http://0.0.0.0:21001  \
    --img-token-number 64     \
    --eos-id $eos_id \
    --model-name test1210  & 

if ! check_port "0.0.0.0" 21390 30 10; then
    echo "21390 is not available"
    exit 1
fi

python3 -m fastchat.serve.internvl_openai_api_server \
    --host 0.0.0.0 --port 8000  \
    --controller-address http://0.0.0.0:21001 &

if ! check_port "0.0.0.0" 8000 30 10; then
    echo "8000 is not available"
    exit 1
fi

python ./test_async.py  \
    --dataset-id 124  \
    --output-file  ./test/output.json    \
    --max_concurrent_tasks  10  \
    --openai_server_url  http://0.0.0.0:8000/v1/