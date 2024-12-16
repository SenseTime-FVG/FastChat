from typing import List, Literal, Union
import requests
from pydantic import (
    BaseModel,
    Field,
    field_validator,
    computed_field,
    ValidationError,
    model_validator,
)
from enum import Enum
from fastapi import FastAPI
from fastapi.testclient import TestClient

app = FastAPI()


# ----------------------------- Model Setting


class MLLMType(str, Enum):
    vision = "vision"
    audio = "audio"
    llm = "llm"


## VisionConfig
class VisionModelType(str, Enum):
    v0_3b = "vit_0.3b"
    v0_6b = "vit_0.6b"
    v1b = "vit_1b"


class DynamicPreprocessVersion(str, Enum):
    v1 = "v1"
    v2 = "v2"
    v3 = "v3"


class ImageTokenNum(int, Enum):
    VALUE_64 = 64
    VALUE_256 = 256


class VisionConfig(BaseModel):
    model_type: VisionModelType = Field(description="Vision模型类型")
    dynamic_preprcrocess_version: DynamicPreprocessVersion = Field(
        DynamicPreprocessVersion.v3, description="动态预处理版本"
    )
    max_patch_num: int = Field(
        -1,
        description=r"最大patch数量，-1表示动态，>=0表示固定大小。动态策略为单图12，(1,6]为6，>6为0",
    )
    use_thumbnail: bool = Field(True, description="是否使用缩略图")
    image_token_num: ImageTokenNum = Field(
        ImageTokenNum.VALUE_256, description="图片token数量"
    )
    # TODO add video config


## AudioConfig
class AudioModelType(str, Enum):
    a06b = "whisper_0.6b"


class AudioConfig(BaseModel):
    model_type: AudioModelType = Field(description="音频模型类型")


## LLMConfig
class LLMModelType(str, Enum):
    l_internlm2_7b = "internlm2_7b"
    l_internlm2_20b = "internlm2_20b"
    l_qwen2_5_0_5b = "qwen2.5_0.5b"
    l_qwen2_5_1b = "qwen2.5_1b"
    l_qwen2_5_7b = "qwen2.5_7b"
    l_qwen2_5_14b = "qwen2.5_14b"
    l_qwen2_5_72b = "qwen2.5_72b"


class LLMConfig(BaseModel):
    model_type: LLMModelType = Field(description="LLM模型类型")


ModelCofig = Union[VisionConfig, AudioConfig, LLMConfig]


class Model(BaseModel):
    path: str = Field(description="模型路径")
    default_system_prompt: str = Field(
        "你是商汤科技开发的日日新多模态大模型，英文名叫SenseChat，是一个有用无害的人工智能助手",
        description="默认系统提示",
    )
    config: dict = Field(
        {_model_type: ModelCofig for _model_type in MLLMType},
        description="模型配置",
    )
        

    @computed_field
    @property
    def gpu_num(self) -> int:
        total_size = sum(
            int(config.model_type.split("_")[-1][:-1])
            for config in self.config.values()
            if config.model_type.split("_")[-1][:-1].isdigit()
        )
        return 4 if total_size >= 72 else 1  # TODO Need to do more granular calculation

    @computed_field
    @property
    def eos_id(self) -> int:
        llm_model_type = self.config.get(MLLMType.llm).model_type
        if llm_model_type.startswith("internlm2"):
            return 92542
        elif llm_model_type.startswith("qwen"):
            return 151645
        else:
            raise ValueError(f"Invalid LLM model type: {llm_model_type}")


# ----------------------------- System Setting
class ClusterType(str, Enum):
    m = "m"
    public = "public"


class WorkspaceType(str, Enum):
    AMP = "AMP"
    amplarge = "amplarge"
    ndation_workspace = "ndation_workspace"
    debug = "debug"

    @classmethod
    def get_cluster(cls, workspace):
        if workspace in [cls.AMP, cls.amplarge]:
            return ClusterType.m
        elif workspace in [cls.ndation_workspace, cls.debug]:
            return ClusterType.public
        else:
            raise ValueError("Invalid workspace type")


class SystemModel(BaseModel):
    cluster: ClusterType = Field(ClusterType.public, description="集群")
    workspace: WorkspaceType = Field(
        WorkspaceType.ndation_workspace, description="工作空间"
    )

    @model_validator(mode="after")
    def validate_system(self):
        valid_cluster = WorkspaceType.get_cluster(self.workspace)
        if valid_cluster != self.cluster:
            raise ValueError(
                f"Workspace '{self.workspace}' is not valid for cluster '{self.cluster}'"
            )
        return self


# ----------------------------- Sampling Setting


class SamplingModel(BaseModel):
    temperature: float = Field(0.5, description="温度系数")
    top_k: int = Field(20, description="top k")
    top_p: float = Field(0.25, ge=0, le=1, description="top p")
    repetition_penalty: float = Field(1.05, ge=0, description="重复惩罚")
    max_new_tokens: int = Field(4096, ge=1, description="最大新token数量")
    do_sample: bool = Field(True, description="是否采样")
    stop_sequences: List[str] = Field(["<|im_end|>"], description="停止序列")
    skip_special_tokens: bool = Field(True, description="是否跳过特殊token，默认为True")


class DataModel(BaseModel):
    dataset_id: int = Field(description="数据集ID")


class RequestModel(BaseModel):
    model: Model = Field(description="模型配置")
    system: SystemModel = Field(SystemModel(), description="系统配置")
    sampling: SamplingModel = Field(SamplingModel(), description="采样配置")
    data: DataModel = Field(description="数据配置")


class ResponseModel(BaseModel):
    code: int = Field(description="返回码")
    message: str = Field(description="返回信息")



@app.post("/send", response_model=ResponseModel)
def send(request: RequestModel):
    print(request.model.config)
    return {"code": 0, "message": "success"}


@app.get("/sampling_params")
def sampling_params():
    return SamplingModel().model_dump()

# 集群 m / public
# 工作空间 AMP / amplarge / ndation_workspace / debug


if __name__ == "__main__":
    client = TestClient(app)
    request_model = RequestModel(
        model=Model(
            path="/mnt/afs/dailinjun/workdir/transfer_model/vit0.3b_qwen2.5_0.5b_stage3_v6.8.8_general_grounding_fromstage2_d241121_v3_3700_hf/",
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
        system=SystemModel(cluster="m", workspace="AMP"),
        data=DataModel(dataset_id=124),
    )
    response = client.post("/send", json=request_model.model_dump())
    
    print(response.json())
    response = client.get("/sampling_params")
    print(response.json())