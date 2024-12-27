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
from functools import cached_property


# ----------------------------- Model Setting


class MLLMType(str, Enum):
    vision = "vision"
    audio = "audio"
    llm = "llm"


## VisionConfig
class VisionModelType(str, Enum):
    v_0_3b = "vit0.3b"
    v_0_6b = "vit0.6b"
    v_1b = "vit_1b"
    v_6b = "vit_6b"


class DynamicPreprocessVersion(str, Enum):
    v1 = "v1"
    v2 = "v2"
    v3 = "v3"


class ImageTokenNum(int, Enum):
    VALUE_64 = 64
    VALUE_256 = 256


class VisionConfig(BaseModel):
    model_type: VisionModelType = Field(description="Vision模型类型")
    dynamic_preprocess_version: DynamicPreprocessVersion = Field(
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
    a_0_6b = "whisper_0.6b"


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
    model_name: str = Field(description="模型名称")
    default_system_prompt: str = Field(
        default="",
        description="默认系统提示",
    )
    config: dict[MLLMType, ModelCofig] = Field(
        default_factory=dict,
        description="模型配置",
    )

    @computed_field
    @cached_property
    def gpu_num(self) -> int:
        total_size = sum(
            int(config.model_type.split("_")[-1][:-1])
            for config in self.config.values()
            if config.model_type.split("_")[-1][:-1].isdigit()
        )
        return 4 if total_size >= 72 else 1  # TODO Need to do more granular calculation

    @computed_field
    @cached_property
    def eos_id(self) -> int:
        llm_model_type = self.config[MLLMType.llm].model_type
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

    @classmethod
    def get_pool(cls, cluster):
        if cluster == cls.m:
            return [PoolType.AMP, PoolType.amplarge]
        elif cluster == cls.public:
            return [PoolType.foundation, PoolType.debug]
        else:
            raise ValueError("Invalid cluster type")


class PoolType(str, Enum):
    AMP = "AMP"
    amplarge = "amplarge"
    foundation = "foundation"
    debug = "debug"

    @classmethod
    def get_cluster(cls, pool):
        if pool in [cls.AMP, cls.amplarge]:
            return ClusterType.m
        elif pool in [cls.foundation, cls.debug]:
            return ClusterType.public
        else:
            raise ValueError("Invalid pool type")
        
class BackendType(str, Enum):
    transformer = "transformer"
    lightllm = "lightllm"
    vllm = "vllm"

class SystemModel(BaseModel):
    cluster: ClusterType = Field(ClusterType.public, description="集群")
    pool: PoolType = Field(PoolType.foundation, description="算力池")
    backend: BackendType = Field(BackendType.lightllm, description="后端")

    @model_validator(mode="after")
    def validate_system(self):
        valid_cluster = PoolType.get_cluster(self.pool)
        if valid_cluster != self.cluster:
            raise ValueError(
                f"Pool '{self.pool}' is not valid for cluster '{self.cluster}'"
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
    job_id: str = Field(description="任务ID", default=None)