import requests


# url = "http://10.119.30.84:10039/send" 
url = "http://101.230.144.192:10069/lightllm_autosco/send" 

payload = {
    "model":{
        "MODEL_PATH":"/mnt/afs/dailinjun/workdir/transfer_model/vit0.3b_qwen2.5_0.5b_stage3_v6.8.8_general_grounding_fromstage2_d241121_v3_3700_hf/",
        "model_type":{
            "vit_type":"vit0.3b",         # 可以为""
            "llm_type":"internlm2_7b",
            "audio_type":"internlm2_7b",  # 可以为""
        },
        "vit":{  # vit_type为空，则vit取值均为空
            # "mode_type":"grounding",
            "dynamic":"v3",                   # vit_type为空，则为空
            "special_label":"",               # 是否添加"本轮用户上传..."
            "MAX_PATCH_NUM":"12",             # -1表示动态；>=0表示固定长度，按此值进行切patch
            "use_thumbnail":"True",           # 切patch是否使用缩略图，默认为True；调用api时在system_param中传入
            "IMAGE_TOKEN_NUM":"256",          # 默认256
        }
        # 此为vit所用； audio有另外的api及其调用
        
    },
    "param": {
        "gpus": "1",
        # 集群
        "cluter":"sensecore",
        # 工作空间
        # "workspace":"ndation_workspace",
        "workspace":"deploy",
        # 分区
        # "pool":"foundation",
        "pool":"server",
        # 副本数
        "instance_replica":"1",
        # eos_id
        "eos_id":"151645",  
        # 在scoid.json中改镜像    
        "dataset_id":"124"          
    },
    "system_param":{
        "temperature": 0.5,
            "top_k": 20,
            "top_p": 0.25,
            "repetition_penalty": 1.05,
            "max_new_tokens": 1024,
            "do_sample": True,
            "stop_sequences": ["<|im_end|>"],
            "skip_special_tokens": True,  # 是否跳过special_token,默认为True
    }
}

headers = {
    "Content-Type": "application/json"
}

response = requests.post(url, json=payload, headers=headers)
print("success")
