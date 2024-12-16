# import requests
# import base64

# api_url = "http://101.230.144.192:10069/apis/download_dataset/1"


# def encode_image(image):
    
#     from io import BytesIO
#     import requests

#     from PIL import Image

#     if image.startswith("http://") or image.startswith("https://"):
#         response = requests.get(image)
#         image = Image.open(BytesIO(response.content)).convert("RGB")
#     else:
#         image = Image.open(image).convert("RGB")

#     buffered = BytesIO()
#     image.save(buffered, format="PNG")
#     img_b64_str = base64.b64encode(buffered.getvalue()).decode("utf-8")

#     return img_b64_str

# try:
#     response = requests.get(api_url)
#     response.raise_for_status()  # 检查请求是否成功
#     data = response.json()  # 解析 JSON 数据
# except requests.exceptions.RequestException as e:
#     print(f" {e}")
#     exit()


# result_list = []
# for group in data:
#     for item in group:
#         question = item.get("question", "")
#         for img in item.get("imgs", []):
#             local_path = img.get("local_path", "")
#             result_list.append({"text": question, "image_path": local_path})
# messages = [
#         {
#             "role": "user",
#             "content": [
#                 {"type": "text", "text": result_list[0]["text"]},
#                 {"type": "image_url", "image_url": {"url": encode_image(result_list[0]["image_path"])}},
#             ],
#         }
#     ]


# import openai
# openai.api_key = "EMPTY"  # Not support yet
# openai.base_url = "http://localhost:12003/v1/"

# def test_list_models():
#     model_list = openai.models.list()
#     names = [x.id for x in model_list.data]
#     return names


# models = test_list_models()
# for model in models:
#     completion = openai.chat.completions.create(
#         model=model,
#         messages=messages,
#         max_tokens=4096
#     )
#     print(completion.choices[0])
#     print(completion.choices[0].message.content)
#     data = {"model": model, "messages": messages}
#     response = requests.post("http://localhost:12003/v1/chat/prompt", json=data)
#     print(response.json())
#     print("=" * 25)
