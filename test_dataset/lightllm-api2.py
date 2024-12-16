import requests
import json
from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from utils import *

import subprocess
from typing import Tuple, List, Dict, Any
from fastapi.testclient import TestClient
import uvicorn
import argparse


app = FastAPI()
# SCO_ID_FILE = "/mnt/afs/user/wuxianye/test/auto_api/lightllm-api/scoid.json"
SCO_ID_FILE = "./scoid.json"


@app.post("/send", response_model=ResponseModel)
def send(request: RequestModel):
    """
    参数：
    model_path=$1
    VIT_ENV=$2
    max_num=$3
    image_token_num=$4
    gpus=$5
    # 工作空间
    workspace_name=$6
    # 分区
    aec2_name=$7
    # 镜像
    container_image_uri=$8
    # eos_id
    eos_id=${9}
    dataset_id=${10}
    model_name=${11}
    """
    

    sh_path = "./lightllm_deploy4.sh"
    with open(SCO_ID_FILE, "r", encoding="utf-8") as file: # 读取需求，转换为sco id
        sco_id = json.load(file)


    # Model Config
    model_path = request.model.path
    model_name = request.model.model_name
    default_system_prompt = request.model.default_system_prompt
    vit = request.model.config["vision"].model_type.value
    if vit == VisionModelType.v_0_3b:
        vit_env = "internchat"
    else:
        vit_env = "vit"
    
    VIT_ENV = vit+ "+" + sco_id["backend"]["lightllm"]["param"]["dynamic"][request.model.config["vision"].dynamic_preprocess_version.value] + "+" + vit_env
    # print("VIT_ENV", VIT_ENV)
    MAX_PATCH_NUM = request.model.config["vision"].max_patch_num
    IMAGE_TOKEN_NUM = request.model.config["vision"].image_token_num.value
    
    gpus = request.model.gpu_num
    eos_id = request.model.eos_id
    

    # Sampling Settings
    temperature = request.sampling.temperature
    top_k = request.sampling.top_k
    top_p = request.sampling.top_p
    repetition_penalty = request.sampling.repetition_penalty
    max_new_tokens = request.sampling.max_new_tokens
    do_sample = request.sampling.do_sample
    stop_sequences = request.sampling.stop_sequences
    skip_special_tokens = request.sampling.skip_special_tokens

    # Dataset Setting
    dataset_id = request.data.dataset_id


    # System Setting
    workspace = list(sco_id["cluter"][request.system.cluster.value]["workspace"].values())[0]
    pool = sco_id["cluter"][request.system.cluster.value]["pool"][request.system.pool.value]
    container_image = sco_id["backend"][request.system.backend.value]["container_image"][request.system.cluster.value]
    
    
    print("workspace", workspace)
    print("pool", pool)
    print("container_image", container_image)

    # 写infer.sh
    # 定义脚本内容
    infer_script = f"""#!/bin/bash


pip install /mnt/afs/user/wuxianye/test/auto_api/lightllm-api/zzm/fschat-0.2.36-py3-none-any.whl
pip install aiofiles openai
pip install /mnt/afs/user/wuxianye/test/auto_api/lightllm-api/zzm/opencv_python_headless-4.10.0.84-cp37-abi3-manylinux_2_17_x86_64.manylinux2014_x86_64.whl


MODEL_PATH={model_path}
VIT_ENV={VIT_ENV}
MAX_PATCH_NUM={MAX_PATCH_NUM}
IMAGE_TOKEN_NUM={IMAGE_TOKEN_NUM}
gpus={gpus}
eos_id={eos_id}
dataset_id={dataset_id}
model_name={model_name}
system_prompt={default_system_prompt}
max_new_tokens={max_new_tokens}
temperature={temperature}
top_k={top_k}
top_p={top_p}
repetition_penalty={repetition_penalty}
do_sample={do_sample}
stop_sequences={stop_sequences}
skip_special_tokens={skip_special_tokens}



model_path=$MODEL_PATH  # Replace with your model path
VIT_ENV=$VIT_ENV \
MAX_PATCH_NUM=$MAX_PATCH_NUM \
IMAGE_TOKEN_NUM=$IMAGE_TOKEN_NUM \
LOADWORKER=4 /opt/conda/bin/python -m lightllm.server.api_server \
    --host 0.0.0.0 \
    --port 8080 \
    --tp $gpus \
    --eos_id $eos_id \
    --max_req_input_len 32000 \
    --max_req_total_len 34000 \
    --max_total_token_num 80000 \
    --model_dir $model_path \
    --mode triton_gqa_flashdecoding \
    --trust_remote_code \
    --tokenizer_mode fast \
    --enable_multimodal \
    --nccl_port 27888 \
    --data_type bf16 &

check_port() {{
    local host="$1"
    local port="$2"
    local retries="$3"
    local delay="$4"

    for i in $(seq 1 "$retries"); do
        python3 -c "import socket; sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM); exit(0) if sock.connect_ex((\\"$host\\", $port)) == 0 else exit(1)"
        if [ $? -eq 0 ]; then
            echo "Port $port is available."
            return 0
        else
            echo "Waiting for port $port to become available... Attempt $i/$retries"
            sleep "$delay"
        fi
    done
    return 1
}}

if ! check_port "0.0.0.0" 8080 60 10; then
    echo "8080 is not available"
    exit 1
fi

python3 -m fastchat.serve.controller --host 0.0.0.0  --port 21001 &

if ! check_port "0.0.0.0" 21001 30 10; then
    echo "21001 is not available"
    exit 1
fi

python3 -m fastchat.serve.lightllm_worker_internvl \
    --host 0.0.0.0 \
    --port 21390 \
    --api-url http://0.0.0.0:8080 \
    --conv-template internlm2-chat-v3 \
    --controller-address http://0.0.0.0:21001 \
    --img-token-number $IMAGE_TOKEN_NUM \
    --eos-id $eos_id \
    --model-name $model_name  & 

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

python /mnt/afs/user/zhengzhimeng/FastChat/test_dataset/test_async.py  \
    --dataset-id $dataset_id  \
    --output-file  /mnt/afs/user/zhengzhimeng/FastChat/test_dataset/output.json    \
    --max_concurrent_tasks  10  \
    --openai_server_url  http://0.0.0.0:8000/v1/  \
    --system-prompt $system_prompt  \
    --temperature $temperature  \
    --max_tokens $max_new_tokens \
    --top_k $top_k  \
    --top_p $top_p  \
    --repetition_penalty $repetition_penalty  \
    --do_sample $do_sample  \
    --stop_sequences $stop_sequences  \
    --skip_special_tokens $skip_special_tokens  \
"""
    print(infer_script)
    # 写入文件
    filename = "/mnt/afs/user/zhengzhimeng/FastChat/test_dataset/infer1.sh"
    with open(filename, "w", encoding="utf-8") as file:
        file.write(infer_script)

    print(f"脚本已写入到 {filename}")

    try:
        subprocess.run([sh_path, str(gpus), workspace, pool, container_image])
    except Exception as e:
        print("failed to run")
        return {"code": 500, "message": "infer failed:{}".format(e)}

    print("success")
    return {"code": 200, "message": "success"}


if __name__ == "__main__":
    # client = TestClient(app)
    # request_model = RequestModel(
    #     model=Model(
    #         path="/mnt/afs/user/zhengzhimeng/models/vit0.3b_internlm7b_stage3_v6.8.1_lr2e-5_20240911_v1/",
    #         model_name="test_1213vqa",
    #         default_system_prompt="你是商汤科技开发的日日新多模态大模型",
    #         config={
    #             "vision": VisionConfig(
    #                 model_type="vit_0.3b",
    #                 dynamic_preprcrocess_version="v3",
    #                 max_patch_num=12,
    #                 use_thumbnail=True,
    #                 image_token_num=256,
    #             ),
    #             "llm": LLMConfig(model_type="internlm2_7b"),
    #         },
    #     ),
    #     system=SystemModel(cluster="public", pool="foundation"),
    #     sampling=SamplingModel(
    #         temperature=0.5,
    #         top_k=20,
    #         top_p=0.25,
    #         repetition_penalty=1.05,
    #         max_new_tokens=4096,
    #         do_sample=True,
    #         stop_sequences=["<|im_end|>"],
    #         skip_special_tokens=True,
    #     ),
    #     data=DataModel(dataset_id=124),
    # )
    # response = client.post("/send", json=request_model.model_dump())
    # print(response.json())
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default="21039", help="port to run the server on")
    args = parser.parse_args()
    uvicorn.run(app, host="0.0.0.0", port=args.port)