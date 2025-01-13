import asyncio
import base64
from io import BytesIO
from PIL import Image
import requests
import aiohttp
import openai
import aiofiles
from loguru import logger
import json
import argparse

system_url = "http://103.237.29.236:10069/apis"

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


def fetch_data(api_url):
    try:
        response = requests.get(api_url)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        logger.error(f"Error fetching data: {e}")
        exit()

def test_list_models():
    model_list = openai.models.list()
    names = [x.id for x in model_list.data]
    return names

async def worker(semaphore, task_queue, session, model_name, system_prompt, parameters, processed_results):
    while True:
        line = await task_queue.get()
        async with semaphore:
            processed_data = await process_entry(session, model_name, system_prompt, parameters, line)
            if processed_data:
                processed_results.append(processed_data)
        task_queue.task_done()

async def process_entry(session, model, system_prompt, parameters, line):
    history = []
    for entry in line:
        question = entry.get("question", "")
        content_list = [{"type": "text", "text": question}]

        for img in entry.get("imgs", []):
            oss_key = img.get("oss_key", "")
            image_b64 = encode_image(oss_key)
            if image_b64:
                content_list.append({"type": "image_url", "image_url": {"url": image_b64}})
        messages = history + [
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

        test = debug_message(messages)
        async with session.post(openai.base_url + f"chat/completions", json=payload) as response:
            result = await response.json()

        answer = result.get("choices", [{}])[0].get("message", {}).get("content", "")
        print(answer)
        entry["result"] = {
            "answer": answer,
            "system_prompt": system_prompt,
        }

        if len(line) > 1:
            history.append(
                {
                    "role": "user",
                    "content": content_list,
                }
            )
            history.append(
                {
                    "role": "assistant",
                    "content": answer,
                }
            )
    
    return line

# from typing import List, Any, Union, Literal
# from pydantic import BaseModel
# class Message(BaseModel):
#     role: Literal["user", "assistant", "system"]
#     content: Union[str, List[Any]]

def debug_message(messages, max_length=20, shift=26):
    import copy

    print_message = copy.deepcopy(messages)
    for message in print_message:

        if message["role"] != "user" or isinstance(message["content"], str):
            continue
        for cpart in message["content"]:
            cpart_type = cpart["type"]
            if "url" in cpart_type:
                cpart[cpart_type]["url"] = cpart[cpart_type]["url"][
                    shift : shift + max_length
                ]
            elif "audio" == cpart_type:
                cpart[cpart_type]["url"] = cpart[cpart_type]["url"][
                    shift : shift + max_length
                ]
    return json.dumps(
        [msg for msg in print_message], ensure_ascii=False
    )


async def main(dataset_id: str, output_file: str, system_prompt: str ,parameters: dict, model_id: int,max_concurrent_tasks: int=10):
    api_url = f"{system_url}/download_dataset/{dataset_id}"
    try:
        response = requests.get(api_url)
        response.raise_for_status()
        data = response.json()
    except requests.exceptions.RequestException as e:
        logger.error(f"Error fetching data: {e}")
        exit()
    model_name = test_list_models()[0]
    print(data)
    
    with open("/mnt/afs/user/zhengzhimeng/FastChat/test_dataset/data.json", mode="w") as f_out:
        f_out.write(json.dumps(data, ensure_ascii=False, indent=4))
    try:
        semaphore = asyncio.Semaphore(max_concurrent_tasks)
        task_queue: asyncio.Queue[str] = asyncio.Queue()
        for group in data:
            task_queue.put_nowait(group)
        processed_results = []

        async with aiohttp.ClientSession() as session:
            tasks = []
            for _ in range(max_concurrent_tasks):
                task = asyncio.create_task(
                worker(semaphore, task_queue, session, model_name, system_prompt, parameters, processed_results)
            )
                tasks.append(task)
            await task_queue.join()
            for task in tasks:
                task.cancel()
        async with aiofiles.open(output_file, mode="w") as f_out:
            # wrapped_results = [[item] for item in processed_results]
            await f_out.write(json.dumps(processed_results, ensure_ascii=False, indent=4))
          
        logger.info(f"Processed data saved to {output_file}")
        url = f"{system_url}/upload_result_to_model"
        payload = {'model_id': model_id}
        files=[
        ('file',('dataset({dataset_id}).json', open(output_file,'rb'),'application/octet-stream'))
        ]
        response = requests.request("POST", url, headers={}, data=payload, files=files)
        response.raise_for_status()
        logger.info("Upload successful!")
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

        logger.error("Upload failed.: {}".format(e))
        exit()
    

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-id", type=str, default="124", help="API URL to fetch data")
    parser.add_argument("--output-file", type=str, default="/mnt/afs/user/zhengzhimeng/FastChat/test_dataset/output.json", help="File to save output")
    parser.add_argument("--openai_server_url", type=str, default="http://0.0.0.0:8000/v1/", help="Openai Serve base URL")
    parser.add_argument("--system-prompt", type=str, default="", help="System prompt")
    parser.add_argument("--max_concurrent_tasks", type=int, default=10, help="Max concurrent tasks")
    parser.add_argument("--model_id", type=int, required=True, help="model_id to fetch")
    parser.add_argument("--temperature", type=float, default=0.5, help="Temperature")
    parser.add_argument("--max_tokens", type=int, default=4096, help="Max tokens")
    parser.add_argument("--top_p", type=float, default=0.25, help="Top p")
    parser.add_argument("--top_k", type=int, default=20, help="Top k")
    parser.add_argument("--repetition_penalty", type=float, default=1.05, help="Repetition penalty")
    parser.add_argument("--do_sample", type=bool, default=True, help="Do sample")
    parser.add_argument("--stop_sequences", type=list, default=["<|im_end|>"], help="Stop sequences")
    parser.add_argument("--skip_special_tokens", type=bool, default=True, help="Skip special tokens")
    args = parser.parse_args()
    openai.api_key = "EMPTY"
    openai.base_url = args.openai_server_url
    parameters = {
        "temperature": args.temperature,
        "top_k": args.top_k,
        "top_p": args.top_p,
        "repetition_penalty": args.repetition_penalty,
        "max_tokens": args.max_tokens,
        "do_sample": args.do_sample,
        "stop_sequences": args.stop_sequences,
        "skip_special_tokens": args.skip_special_tokens,
    }
    asyncio.run(main(args.dataset_id, args.output_file, args.system_prompt, parameters, args.model_id, args.max_concurrent_tasks))