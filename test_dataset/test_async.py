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


async def main(dataset_id: str, output_file: str, system_prompt: str ,parameters: dict, max_concurrent_tasks: int=10):
    api_url = f"{system_url}/download_dataset/{dataset_id}"
    try:
        response = requests.get(api_url)
        response.raise_for_status()
        data = response.json()
    except requests.exceptions.RequestException as e:
        logger.error(f"Error fetching data: {e}")
        exit()
    model_name = test_list_models()[0]
    try:
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
    except requests.exceptions.RequestException as e:
        logger.error("Cannot Post Model_id")
        return ""
    try:
        semaphore = asyncio.Semaphore(max_concurrent_tasks)
        async with aiohttp.ClientSession() as session:
            tasks = []
            for group in data:
                group_tasks = []
                for entry in group:
                    async with semaphore:
                        group_tasks.append(process_entry(session, model_name, system_prompt, parameters,entry))
                tasks.append(asyncio.gather(*group_tasks))
            processed_groups = await asyncio.gather(*tasks)

            async with aiofiles.open(output_file, mode="w") as f_out:
                await f_out.write(json.dumps(processed_groups, ensure_ascii=False, indent=4))
            logger.info(f"Processed data saved to {output_file}")
        
        url = f"{system_url}/upload_result_to_model"
        payload = {'model_id': model_id}
        files=[
        ('file',('dataset({dataset_id}).json', open(output_file,'rb'),'application/octet-stream'))
        ]
        response = requests.request("POST", url, headers={}, data=payload, files=files)
        response.raise_for_status()
        logger.info("Upload successful!")
    except:
        url = f"{system_url}/update_model_status"
        payload = json.dumps({
        "model_id": model_id,
        "status": 2,
        })
        headers = {
        'Content-Type': 'application/json'
        }
        response = requests.request("POST", url, headers=headers, data=payload)

        logger.error("Upload failed.")
        exit()
    

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-id", type=str, default="124", help="API URL to fetch data")
    parser.add_argument("--output-file", type=str, default="/mnt/afs/user/zhengzhimeng/FastChat/test_dataset/output.json", help="File to save output")
    parser.add_argument("--openai_server_url", type=str, default="http://0.0.0.0:8000/v1/", help="Openai Serve base URL")
    parser.add_argument("--system-prompt", type=str, default="", help="System prompt")
    parser.add_argument("--max_concurrent_tasks", type=int, default=10, help="Max concurrent tasks")
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
    asyncio.run(main(args.dataset_id, args.output_file, args.system_prompt, parameters, args.max_concurrent_tasks))