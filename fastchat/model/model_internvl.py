"""Inference for FastChat models."""
import abc
import gc
import json
import math
import os
import sys
import time
from typing import Iterable, Optional, Dict
import torch

from transformers.generation.logits_process import (
    LogitsProcessorList,
    RepetitionPenaltyLogitsProcessor,
    TemperatureLogitsWarper,
    TopKLogitsWarper,
    TopPLogitsWarper,
)
from transformers import TextIteratorStreamer
from threading import Thread
from fastchat.utils import is_partial_stop, is_sentence_complete, get_context_length


def prepare_logits_processor(
    temperature: float, repetition_penalty: float, top_p: float, top_k: int
) -> LogitsProcessorList:
    processor_list = LogitsProcessorList()
    # TemperatureLogitsWarper doesn't accept 0.0, 1.0 makes it a no-op so we skip two cases.
    if temperature >= 1e-5 and temperature != 1.0:
        processor_list.append(TemperatureLogitsWarper(temperature))
    if repetition_penalty > 1.0:
        processor_list.append(RepetitionPenaltyLogitsProcessor(repetition_penalty))
    if 1e-8 <= top_p < 1.0:
        processor_list.append(TopPLogitsWarper(top_p))
    if top_k > 0:
        processor_list.append(TopKLogitsWarper(top_k))
    return processor_list

# zzm
def get_buffer_from_url(url, mode="image"):
    """
    根据不同的URL格式返回buffer。
    支持 file, base64, http/https。
    """
    if url.startswith("file://"):
        file_path = url[7:]  # 去掉 'file://' 前缀
        with open(file_path, "rb") as f:
            buffer = f.read()
    elif url.startswith("http://") or url.startswith("https://"):
        response = requests.get(url)
        response.raise_for_status()
        buffer = response.content
    elif url.startswith("data:"):
        # 处理以data:开头的base64编码数据
        base64_data = url.split(",")[1]  # 只取base64部分
        buffer = base64.b64decode(base64_data)
    else:
        raise ValueError("不支持的URL格式")

def bytes_sha256(data):
    import hashlib

    sha256 = hashlib.sha256()
    sha256.update(data)
    return sha256.hexdigest()

@torch.inference_mode()
def generate_stream_internvl(
    model,
    tokenizer,
    params: Dict,
    device: str,
    context_len: int,
    stream_interval: int = 2,
    judge_sent_end: bool = False,
):
    if hasattr(model, "device"):
        device = model.device
    
    prompt = params["prompt"]
    import base64
    from PIL import Image
    import io

    all_pixel_values = []
    if params.get("images"):
        for image in params["images"]:
            if image.startswith("data:"):
                image_obj = image.split(";", 1)[1]
                image_obj = base64.b64decode(image_obj[7:])
            else:
                image_obj = base64.b64decode(image)
            image_obj = Image.open(io.BytesIO(image_obj))
            image = image_obj.convert("RGB")
            from fastchat.model.image_utils import dynamic_preprocess_v3, build_transform
            #需要传入的参数 self.image_size self.max_num self.use_thumbnail
            processed_images = dynamic_preprocess_v3(
                                image,
                                max_num=6,    #max_num,
                                image_size=448,  #image_size,
                                use_thumbnail=True,    # use_thumbnail,
            )
            transform=build_transform(is_train=False, image_size=448)
            pixel_value = [
                transform(image) for image in processed_images
            ]
            replacement = (
                    "<|img_start|>"
                    + "<IMG_CONTEXT>" * 256 * len(pixel_value)
                    + "<|img_end|>\n"
                )
            prompt = prompt.replace("<|img_start|><|img_end|>\n", replacement, 1)
            all_pixel_values.extend(pixel_value)   
        if all_pixel_values:
            all_pixel_values = torch.stack(all_pixel_values).to(torch.bfloat16).cuda()
        else:
            all_pixel_values = None
    else:
        all_pixel_values = None
    img_context_token_id = tokenizer.convert_tokens_to_ids("<IMG_CONTEXT>")
    model.img_context_token_id = img_context_token_id
    eos_token_id = [
            tokenizer.eos_token_id,
            tokenizer.convert_tokens_to_ids(["<|im_end|>"])[0],
        ]
    
    # Read parameters
    
    len_prompt = len(prompt)
    temperature = float(params.get("temperature", 0.5))
    repetition_penalty = float(params.get("repetition_penalty", 1.05))
    top_p = float(params.get("top_p", 0.25))
    top_k = int(params.get("top_k", 20))  # -1 means disable
    max_new_tokens = int(params.get("max_new_tokens", 1024))
    stop_str = params.get("stop", None)
    stop_token_ids = params.get("stop_token_ids", None) or []
    do_sample = params.get("do_sample", True)
    if tokenizer.eos_token_id not in stop_token_ids:
        stop_token_ids.append(tokenizer.eos_token_id)

    # logits_processor = prepare_logits_processor(
    #     temperature, repetition_penalty, top_p, top_k
    # )
    model_inputs = tokenizer(
            prompt, return_tensors="pt", add_special_tokens=False
        )
    input_ids = model_inputs["input_ids"].cuda()
    attention_mask = model_inputs["attention_mask"].cuda()
    
    max_src_len = context_len - max_new_tokens - 1

    input_ids = input_ids[-max_src_len:]
    # output_ids = list(input_ids)
    # input_echo_len = len(input_ids)


    # start_ids = torch.as_tensor([input_ids], device=device)

    # past_key_values = out = None
    # token_logprobs = [None]  # The first token has no logprobs.
    # sent_interrupt = False
    # finish_reason = None
    # stopped = False
    streamer = TextIteratorStreamer(
            tokenizer, skip_prompt=True, skip_special_tokens=True
        )
    generation_config = dict(
            top_p=top_p,
            top_k=top_k,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature,
            repetition_penalty=repetition_penalty,
            eos_token_id=eos_token_id,
            streamer=streamer,
        )
    generation_kwargs = dict(
                pixel_values=all_pixel_values,
                input_ids=input_ids,
                attention_mask=attention_mask,
                **generation_config,
                
            )
    thread = Thread(target=model.generate, kwargs=generation_kwargs)
    thread.start()
    output = ""
    for i, new_text in enumerate(streamer):
        output += new_text
        if i % stream_interval == 0:
            rfind_start = 0

            partially_stopped = False
            if stop_str:
                if isinstance(stop_str, str):
                    pos = output.rfind(stop_str, rfind_start)
                    if pos != -1:
                        output = output[:pos]
                    else:
                        partially_stopped = is_partial_stop(output, stop_str)
                elif isinstance(stop_str, Iterable):
                    for each_stop in stop_str:
                        pos = output.rfind(each_stop, rfind_start)
                        if pos != -1:
                            output = output[:pos]
                            break
                        else:
                            partially_stopped = is_partial_stop(output, each_stop)
                            if partially_stopped:
                                break
                else:
                    raise ValueError("Invalid stop field type.")

            # prevent yielding partial stop sequence
            if not partially_stopped:
                yield {
                    "text": output,
                    "usage": {
                        # "prompt_tokens": input_echo_len,
                        "completion_tokens": i,
                        # "total_tokens": input_echo_len + i,
                    },
                    "finish_reason": None,
                }
    output = output.strip()

    # finish stream event, which contains finish reason
    if i == max_new_tokens - 1:
        finish_reason = "length"
    elif partially_stopped:
        finish_reason = None
    else:
        finish_reason = "stop"

    yield {
        "text": output,
        "usage": {
            # "prompt_tokens": input_echo_len,
            "completion_tokens": i,
            # "total_tokens": input_echo_len + i,
        },
        "finish_reason": finish_reason,
    }

    # clean
    gc.collect()
    torch.cuda.empty_cache()
    if device == "xpu":
        torch.xpu.empty_cache()
    if device == "npu":
        torch.npu.empty_cache()