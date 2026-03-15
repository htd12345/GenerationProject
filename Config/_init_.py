"""
配置模块初始化
Configuration module initialization
"""

from .settings import Config, config, ModelConfig, TrainingConfig, InferenceConfig, APIConfig

__all__ = [
    "Config",
    "config",
    "ModelConfig",
    "TrainingConfig",
    "InferenceConfig",
    "APIConfig"
]