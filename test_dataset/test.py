import requests
from PIL import Image
import base64
from io import BytesIO
import asyncio
import aiohttp
import openai
import aiofiles
import json
def encode_image(image):
    if image.startswith("http://") or image.startswith("https://"):
        response = requests.get(image)
        image = Image.open(BytesIO(response.content)).convert("RGB")
    else:
        image = Image.open(image).convert("RGB")

    buffered = BytesIO()
    image.save(buffered, format="PNG")
    img_b64_str = base64.b64encode(buffered.getvalue()).decode("utf-8")

    return img_b64_str


# print(encode_image("https://mllm-dataset.aoss-internal.cn-sh-01.sensecoreapi-oss.cn/image/2024-12-13T10:55:55.894607.jpg"))

async def process_entry(session, model, system_prompt, parameters,entry):
    question = entry.get("question", "")
    content_list = [{"type": "text", "text": question}]

    for img in entry.get("imgs", []):
        oss_key = img.get("oss_key", "")
        image_b64 = encode_image(oss_key)
        if image_b64:
            content_list.append({"type": "image_url", "image_url": {"url": image_b64}})
    messages = [
        {
            "role": "user",
            "content": content_list,
        }
    ]
    if system_prompt:
        messages = [
            {
                "role": "system",
                "content": system_prompt,
            }
        ] + messages

    payload = {
            "model": model, 
            "messages": messages, 
            "temperature": parameters["temperature"],
            "top_p": parameters["top_p"],
            "top_k": parameters["top_k"],
            "repetition_penalty": parameters["repetition_penalty"],
            "max_tokens": parameters["max_tokens"],
            "do_sample": parameters["do_sample"],
            "stop_sequences": parameters["stop_sequences"],
            "skip_special_tokens": parameters["skip_special_tokens"],
            }
    
    async with session.post(openai.base_url + f"chat/completions", json=payload) as response:
        result = await response.json()

    answer = result.get("choices", [{}])[0].get("message", {}).get("content", "")

    entry["result"] = {
        "answer": answer,
        "system_prompt": system_prompt,
    }
    return entry

semaphore = asyncio.Semaphore(10)
async with aiohttp.ClientSession() as session:
    tasks = []
    for group in data:
        group_tasks = []
        for entry in group:
            async with semaphore:
                group_tasks.append(process_entry(session, model_name, system_prompt, parameters, entry))
        tasks.append(asyncio.gather(*group_tasks))
    processed_groups = await asyncio.gather(*tasks)

    async with aiofiles.open("/mnt/afs/user/zhengzhimeng/FastChat/test_dataset/2233.json", mode="w") as f_out:
        await f_out.write(json.dumps(processed_groups, ensure_ascii=False, indent=4))
 