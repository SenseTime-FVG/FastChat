"""
A model worker that executes the model based on LightLLM.

See documentations at docs/lightllm_integration.md
"""

import argparse
import asyncio
import json
import os
import torch
import uvicorn

from transformers import AutoConfig

from typing import List

from fastapi import FastAPI, Request, BackgroundTasks
from fastapi.responses import StreamingResponse, JSONResponse

from fastchat.serve.base_model_worker import BaseModelWorker
from fastchat.serve.model_worker import (
    logger,
    worker_id,
)
import requests

app = FastAPI()


class LightLLMWorker(BaseModelWorker):
    def __init__(
        self,
        controller_addr: str,
        worker_addr: str,
        worker_id: str,
        model_path: str,
        model_names: List[str],
        limit_worker_concurrency: int,
        no_register: bool,
        conv_template: str,
        api_url: str,
        max_num: int,
        special_token: dict,
        img_processor: str,
    ):
        super().__init__(
            controller_addr,
            worker_addr,
            worker_id,
            model_path,
            model_names,
            limit_worker_concurrency,
            conv_template,
        )

        logger.info(
            f"Loading the model {self.model_names} on worker {worker_id}, worker type: LightLLM worker..."
        )
        self.api_url=api_url
        self.max_num=max_num
        self.special_token = special_token
        self.conv_template = conv_template
        self.img_processor = img_processor

        if not no_register:
            self.init_heart_beat()

    async def generate_stream(self, params):
        self.call_ct += 1

        prompt = params.pop("prompt")
        temperature = float(params.get("temperature", 0.5))
        top_p = float(params.get("top_p", 0.25))
        top_k = params.get("top_k", 20)
        repetition_penalty = float(params.get("repetition_penalty", 1.05))
        max_new_tokens = params.get("max_new_tokens", 1024)
        
        parameters = {
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
            "repetition_penalty": repetition_penalty,
            "max_new_tokens": max_new_tokens,
            "stop_sequences": ["<|im_end|>"],
        }

        import pdb
        pdb.set_trace()
        images_data = []
        for image in params.get("images", []):
            images_data.append({"type": "base64", "data": image})

        data = {
            "inputs": prompt,
            "parameters": parameters,
            "multimodal_params": {"images": images_data},
        }
        headers = {"Content-Type": "application/json"}
        response = requests.post(self.api_url + "/generate_stream", headers=headers, data=json.dumps(data), stream=True)
        for chunk in response.iter_lines():
            if chunk:
                yield json.loads(chunk.decode("utf-8")[5:])

        

    async def generate(self, params):
        generate_text= ""
        async for x in self.generate_stream(params):
            generate_text+=x["token"]["text"]
            pass

        return {"text": generate_text, "error_code": 0}
        # TODO: return class

def release_worker_semaphore():
    worker.semaphore.release()


def acquire_worker_semaphore():
    if worker.semaphore is None:
        worker.semaphore = asyncio.Semaphore(worker.limit_worker_concurrency)
    return worker.semaphore.acquire()



@app.post("/worker_generate_stream")
async def api_generate_stream(request: Request):
    params = await request.json()
    await acquire_worker_semaphore()
    params["request"] = request
    generator = worker.generate_stream(params)
    return StreamingResponse(generator, background=background_tasks)


@app.post("/worker_generate")
async def api_generate(request: Request):
    params = await request.json()
    await acquire_worker_semaphore()
    params["request"] = request
    output = await worker.generate(params)
    release_worker_semaphore()
    return JSONResponse(output)


@app.post("/worker_get_status")
async def api_get_status(request: Request):
    return worker.get_status()


@app.post("/count_token")
async def api_count_token(request: Request):
    params = await request.json()
    return worker.count_token(params)


@app.post("/worker_get_conv_template")
async def api_get_conv(request: Request):
    return worker.get_conv_template()


@app.post("/model_details")
async def api_model_details(request: Request):
    return {"context_length": worker.context_len}

@app.post("/worker_details") #TODO
async def api_model_details(request: Request):
    config = {
        "context_length": worker.context_len,
        "special_token": worker.special_token,
        "conv_template": worker.conv_template,
        "img_processor": worker.img_processor,
        "max-num" : worker.max_num,
        

    }
    return config

if __name__ == "__main__":
    torch.multiprocessing.set_start_method("spawn")
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="localhost")
    parser.add_argument("--port", type=int, default=21390)
    parser.add_argument(
        "--api-url", type=str, default="http://10.119.19.117:8080"
    )
    parser.add_argument("--worker-address", type=str, default="http://localhost:21390")
    parser.add_argument(
        "--controller-address", type=str, default="http://localhost:21001"
    )
    parser.add_argument(
        "--conv-template", type=str, default=None, help="Conversation prompt template."
    )
    parser.add_argument(
        "--model-names",
        type=lambda s: s.split(","),
        help="Optional display comma separated names",
    )
    parser.add_argument("--limit-worker-concurrency", type=int, default=1024)
    parser.add_argument("--no-register", action="store_true")

    args = parser.parse_args()

    # model_config = AutoConfig.from_pretrained(args.model_dir)
    # context_length = get_context_length(model_config
    context_length = 1024 * 32


    worker = LightLLMWorker(
        args.controller_address,
        args.worker_address,
        worker_id,
        None,
        args.model_names,
        args.limit_worker_concurrency,
        args.no_register,
        args.conv_template,
        args.api_url,
        max_num=6,
        special_token={"img": "<img></img>\n",
                        "video": "<img></img>\n"},
        img_processor="v3",
    )

    uvicorn.run(app, host=args.host, port=args.port, log_level="info")