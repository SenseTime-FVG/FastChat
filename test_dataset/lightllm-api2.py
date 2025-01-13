import requests
import json
from pathlib import Path
import re

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from utils import *

import subprocess
from typing import Tuple, List, Dict, Any
from fastapi.testclient import TestClient
import uvicorn
import argparse
from fastapi.middleware.cors import CORSMiddleware

system_url = "http://103.237.29.236:10069/apis"
app = FastAPI()
# SCO_ID_FILE = "/mnt/afs/user/wuxianye/test/auto_api/lightllm-api/scoid.json"
SCO_ID_FILE = "./scoid.json"
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 允许的来源
    allow_credentials=True,  # 是否允许发送 cookies
    allow_methods=["*"],  # 允许的 HTTP 方法
    allow_headers=["*"],  # 允许的 HTTP 请求头
)
import threading
class JobStorage:
    """
    A class to map model_id to its associated job_id and workspace_name.
    """
    def __init__(self):
        self.lock = threading.Lock()
        self.mapping = {}

    def save_mapping(self, model_id, job_id, workspace_name):
        with self.lock:
            self.mapping[model_id] = {
                "job_id": job_id,
                "workspace_name": workspace_name,
            }

    def get_mapping(self, model_id):
        with self.lock:
            return self.mapping.get(model_id)

    def delete_mapping(self, model_id):
        with self.lock:
            if model_id in self.mapping:
                del self.mapping[model_id]

job_storage = JobStorage()


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
    # Dataset Setting
    dataset_id = request.data.dataset_id
    url = f"{system_url}/model"
    payload = json.dumps({
    "dataset_id": dataset_id, 
    "model_name": model_name,
    })
    headers = {
    'Content-Type': 'application/json'
    }
    response_model_id = requests.request("POST", url, headers=headers, data=payload)
    model_id = response_model_id.json()["model_id"]
    print("model_id: ", model_id)
    try:
        default_system_prompt = f"'{str(request.model.default_system_prompt)}'" if str(request.model.default_system_prompt) else "''"
        vit = request.model.config["vision"].model_type.value
        if vit == VisionModelType.v_0_3b:
            vit_env = "internchat"
        else:
            vit_env = "vit"
        if vit == VisionModelType.v_6b:
            vit = "vit6b"
        
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
        skip_special_tokens = str(request.sampling.skip_special_tokens)

        
        

        # System Setting
        workspace = list(sco_id["cluter"][request.system.cluster.value]["workspace"].values())[0]
        pool = sco_id["cluter"][request.system.cluster.value]["pool"][request.system.pool.value]
        container_image = sco_id["backend"][request.system.backend.value]["container_image"][request.system.cluster.value]
        
        
        
        print("workspace", workspace)
        print("pool", pool)
        print("container_image", container_image)

        # 写infer.sh
        # 定义脚本内容
        if request.system.cluster.value == "m":
            infer_script = f"""#!/bin/bash

pip install /mnt/afs/user/zhengzhimeng/FastChat/dist/fschat-0.2.36-py3-none-any.whl
pip install aiofiles openai
pip install /mnt/afs/user/zhengzhimeng/FastChat/test_dataset/opencv_python_headless-4.10.0.84-cp37-abi3-manylinux_2_17_x86_64.manylinux2014_x86_64.whl


MODEL_PATH={model_path}
VIT_ENV={VIT_ENV}
MAX_PATCH_NUM={MAX_PATCH_NUM}
IMAGE_TOKEN_NUM={IMAGE_TOKEN_NUM}
gpus={gpus}
eos_id={eos_id}
dataset_id={dataset_id}
model_name={model_name}
model_id={model_id}
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
DISABLE_CHECK_MAX_LEN_INFER=1 \
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
    --worker-address http://0.0.0.0:21390 \
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
SYSTEM_PROMPT_ARG=""
if [[ -n "$system_prompt" ]]; then
SYSTEM_PROMPT_ARG="--system-prompt \"$system_prompt\""
fi
python /mnt/afs/user/zhengzhimeng/FastChat/test_dataset/test_async.py  \
    --dataset-id $dataset_id  \
    --output-file  /mnt/afs/user/zhengzhimeng/FastChat/test_dataset/$model_id.json    \
    --max_concurrent_tasks  10  \
    --openai_server_url  http://0.0.0.0:8000/v1/  \
    --model_id $model_id \
    $SYSTEM_PROMPT_ARG  \
    --temperature $temperature  \
    --max_tokens $max_new_tokens \
    --top_k $top_k  \
    --top_p $top_p  \
    --repetition_penalty $repetition_penalty  \
    --do_sample $do_sample  \
    --stop_sequences $stop_sequences  \
    --skip_special_tokens $skip_special_tokens  \
"""
        else:
            infer_script = f"""#!/bin/bash

pip install /mnt/afs/user/zhengzhimeng/FastChat/dist/fschat-0.2.36-py3-none-any.whl
pip install aiofiles openai
pip install /mnt/afs/user/zhengzhimeng/FastChat/test_dataset/opencv_python_headless-4.10.0.84-cp37-abi3-manylinux_2_17_x86_64.manylinux2014_x86_64.whl


MODEL_PATH={model_path}
VIT_ENV={VIT_ENV}
MAX_PATCH_NUM={MAX_PATCH_NUM}
IMAGE_TOKEN_NUM={IMAGE_TOKEN_NUM}
gpus={gpus}
eos_id={eos_id}
dataset_id={dataset_id}
model_name={model_name}
model_id={model_id}
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
DISABLE_CHECK_MAX_LEN_INFER=1 \
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
    --worker-address http://0.0.0.0:21390 \
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
SYSTEM_PROMPT_ARG=""
if [[ -n "$system_prompt" ]]; then
SYSTEM_PROMPT_ARG="--system-prompt \"$system_prompt\""
fi
python /mnt/afs/user/zhengzhimeng/FastChat/test_dataset/test_async.py  \
    --dataset-id $dataset_id  \
    --output-file  /mnt/afs/user/zhengzhimeng/FastChat/test_dataset/$model_id.json    \
    --max_concurrent_tasks  10  \
    --openai_server_url  http://0.0.0.0:8000/v1/  \
    --model_id $model_id \
    $SYSTEM_PROMPT_ARG  \
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
            file.flush()

        print(f"脚本已写入到 {filename}")
        import pdb
        pdb.set_trace()
        try:
            result = subprocess.run([sh_path, str(gpus), workspace, pool, container_image],
            stdout=subprocess.PIPE,  # Capture standard output
            stderr=subprocess.PIPE,
            text=True
        )   
            job_id_match = re.search(r"job id\s*:\s*(\S+)", result.stdout)
            job_id = job_id_match.group(1)
            job_storage.save_mapping(model_id, job_id, workspace)
            print(f"Extracted job id: {job_id}")
            print("success")
            print(job_storage.mapping)
            # task = asyncio.create_task(get_job_state_async(workspace, job_id, model_id))
            return {"code": 200, "message": "success", "job_id": job_id}
        except Exception as e:
            print("failed to run")
            return {"code": 500, "message": "infer failed:{}".format(e)}

        
    except Exception as e:
        url = f"{system_url}/update_model_status"
        payload = json.dumps({
        "model_id": model_id,
        "status": 2,
        })
        headers = {
        'Content-Type': 'application/json'
        }
        response = requests.request("POST", url, headers=headers, data=payload)
        return {"code": 500, "message": "infer failed:{}".format(e)}


class CancelJobByModelRequest(BaseModel):
    model_id: int

def get_job_state(workspace, job_id, model_id):
    url = f"{system_url}/update_model_status"
    payload = json.dumps({
    "model_id": model_id,
    "status": 3,
    })
    headers = {
    'Content-Type': 'application/json'
    }
    response = requests.request("POST", url, headers=headers, data=payload)
    with open(SCO_ID_FILE, "r", encoding="utf-8") as file:
        sco_id = json.load(file)
        workspace_name = list(sco_id["cluter"][workspace]["workspace"].values())[0]
    try:
        while True:
            result = subprocess.run(
                ["sco", "acp", "jobs", "describe", f"--workspace-name={workspace_name}", job_id, "--format=json"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )

            if result.returncode != 0:
                error_message = f"Failed to describe job.\nError: {result.stderr.strip()}\nOutput: {result.stdout.strip()}"
                print(error_message)

        
            job_info = json.loads(result.stdout)
            state = job_info.get("state", "UNKNOWN")    # RUNNING, STARTING
            if state == "STARTING":
                time.sleep(10)
                continue
            elif state == "RUNNING":
                url = f"{system_url}/update_model_status"
                payload = json.dumps({
                "model_id": model_id,
                "status": 0,
                })
                headers = {
                'Content-Type': 'application/json'
                }
                response = requests.request("POST", url, headers=headers, data=payload)
                break
            print(f"Job state: {state}")
            return {"code": 200, "message": state, "job_id": job_id}

    except Exception as e:
        return {"code": 500, "message": "infer failed:{}".format(e)}

@app.delete("/delete_model", response_model=ResponseModel)
def delete_model(request: CancelJobByModelRequest):
    try:
        model_id = request.model_id
        mapping = job_storage.get_mapping(model_id)
        if not mapping:
            return {"code": 404, "message": f"Model ID '{model_id}' not found"}
        
        job_storage.delete_mapping(model_id)
        return {"code": 200, "message": f"Model ID '{model_id}' deleted successfully"}
    except Exception as e:
        return {"code": 500, "message": f"Failed to delete model ID: {e}"}

@app.post("/cancel_job", response_model=ResponseModel)
def cancel_job(request: CancelJobByModelRequest):
    try:
        model_id = request.model_id
        mapping = job_storage.get_mapping(model_id)
        if not mapping:
            return {"code": 404, "message": f"Model ID '{model_id}' not found"}
        workspace_name = mapping["workspace_name"]
        job_id = mapping["job_id"]

        print("Executing command:")
        print(
            ["sco", "acp", "jobs", "delete", f"--workspace-name={workspace_name}", job_id]
        )
        result = subprocess.run(
            ["sco", "acp", "jobs", "delete", f"--workspace-name={workspace_name}", job_id],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        if result.returncode != 0:
            error_message = f"Failed to delete job.\nError: {result.stderr.strip()}\nOutput: {result.stdout.strip()}"
            print(error_message)
        return {"code": 200, "message": "success delete", "job_id": job_id}
    
    except Exception as e:
        return {"code": 500, "message": "delete failed:{}".format(e)}



if __name__ == "__main__":
    client = TestClient(app)
    request_model = RequestModel(
        model=Model(
            path="/mnt/afs/user/zhengzhimeng/models/vit0.3b_qwen2_5_14b_stage3_v6.9.3.5_20241219_v1/",
            model_name="test_0113v7",
            default_system_prompt="",
            config={
                "vision": VisionConfig(
                    model_type="vit_0.3b",
                    dynamic_preprcrocess_version="v3",
                    max_patch_num=12,
                    use_thumbnail=True,
                    image_token_num=256,
                ),
                "llm": LLMConfig(model_type="qwen2.5_14b"),
            },
        ),
        system=SystemModel(cluster="m", pool="AMP"),
        sampling=SamplingModel(
            temperature=0.5,
            top_k=20,
            top_p=0.25,
            repetition_penalty=1.05,
            max_new_tokens=4096,
            do_sample=True,
            stop_sequences=["<|im_end|>"],
            skip_special_tokens=True,
        ),
        data=DataModel(dataset_id=271),
    )
    response = client.post("/send", json=request_model.model_dump())
    print(response.json())
    # parser = argparse.ArgumentParser()
    # parser.add_argument("--port", type=int, default="21039", help="port to run the server on")
    # args = parser.parse_args()
    # uvicorn.run(app, host="0.0.0.0", port=args.port)