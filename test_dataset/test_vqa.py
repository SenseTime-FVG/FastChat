from utils import *
import requests
request_model = RequestModel(
        model=Model(
            path="/mnt/afs/user/zhengzhimeng/models/vit0.3b_internlm7b_stage3_v6.8.1_lr2e-5_20240911_v1/",
            model_name="test_1213vqav4",
            default_system_prompt="你是商汤科技开发的日日新多模态大模型",
            config={
                "vision": VisionConfig(
                    model_type="vit_0.3b",
                    dynamic_preprcrocess_version="v3",
                    max_patch_num=12,
                    use_thumbnail=True,
                    image_token_num=256,
                ),
                "llm": LLMConfig(model_type="internlm2_7b"),
            },
        ),
        system=SystemModel(cluster="public", pool="foundation"),
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
        data=DataModel(dataset_id=124),
    )
print(request_model.model_dump_json(indent=4))
# response = requests.post("http://10.119.30.22:21039/send", json=request_model.model_dump())
response = requests.post("http://103.237.29.236:10069/fastchat/send", json=request_model.model_dump())
print(response.json())