"""
配置模块 - 全局配置和超参数设置
Configuration Module - Global settings and hyperparameters
"""

from dataclasses import dataclass, field
from typing import Optional, List
from pathlib import Path
import os


@dataclass
class ModelConfig:
    """
    模型配置类
    Model configuration class

    Attributes:
        model_name: 基础模型名称或路径
        lora_r: LoRA秩（低秩矩阵维度）
        lora_alpha: LoRA缩放参数
        lora_dropout: LoRA dropout率
        target_modules: 需要应用LoRA的目标模块
        max_length: 最大序列长度
        device: 运行设备
    """
    model_name: str = "meta-llama/Llama-2-7b-hf"
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    target_modules: List[str] = field(default_factory=lambda: ["q_proj", "v_proj", "k_proj", "o_proj"])
    max_length: int = 2048
    device: str = "cuda"
    torch_dtype: str = "float16"


@dataclass
class TrainingConfig:
    """
    训练配置类
    Training configuration class

    Attributes:
        output_dir: 模型输出目录
        num_train_epochs: 训练轮数
        per_device_train_batch_size: 单设备训练批次大小
        per_device_eval_batch_size: 单设备评估批次大小
        learning_rate: 学习率
        warmup_steps: 预热步数
        logging_steps: 日志记录间隔
        save_steps: 模型保存间隔
        eval_steps: 评估间隔
    """
    output_dir: str = "./output"
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 4
    per_device_eval_batch_size: int = 4
    learning_rate: float = 2e-4
    warmup_steps: int = 100
    logging_steps: int = 10
    save_steps: int = 500
    eval_steps: int = 500
    gradient_accumulation_steps: int = 4
    max_grad_norm: float = 1.0
    weight_decay: float = 0.01


@dataclass
class InferenceConfig:
    """
    推理配置类
    Inference configuration class

    Attributes:
        max_new_tokens: 生成的最大token数
        temperature: 生成温度
        top_p: nucleus采样参数
        top_k: top-k采样参数
        do_sample: 是否使用采样
        repetition_penalty: 重复惩罚系数
    """
    max_new_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 50
    do_sample: bool = True
    repetition_penalty: float = 1.1


@dataclass
class APIConfig:
    """
    API配置类
    API configuration class

    Attributes:
        host: 服务主机地址
        port: 服务端口
        debug: 调试模式
        workers: 工作进程数
    """
    host: str = "0.0.0.0"
    port: int = 8000
    debug: bool = True
    workers: int = 1


class Config:
    """
    全局配置管理类
    Global configuration manager

    统一管理所有配置项，支持从环境变量覆盖

    Usage:
        config = Config()
        print(config.model.model_name)
    """

    def __init__(self):
        self.model = ModelConfig()
        self.training = TrainingConfig()
        self.inference = InferenceConfig()
        self.api = APIConfig()
        self._load_from_env()

    def _load_from_env(self):
        """
        从环境变量加载配置
        Load configuration from environment variables
        """
        # 模型配置
        if os.getenv("MODEL_NAME"):
            self.model.model_name = os.getenv("MODEL_NAME")
        if os.getenv("LORA_R"):
            self.model.lora_r = int(os.getenv("LORA_R"))
        if os.getenv("MAX_LENGTH"):
            self.model.max_length = int(os.getenv("MAX_LENGTH"))

        # API配置
        if os.getenv("API_HOST"):
            self.api.host = os.getenv("API_HOST")
        if os.getenv("API_PORT"):
            self.api.port = int(os.getenv("API_PORT"))

    @property
    def base_dir(self) -> Path:
        """项目根目录"""
        return Path(__file__).parent.parent.parent


# 全局配置实例
config = Config()