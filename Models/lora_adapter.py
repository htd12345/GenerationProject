"""
LoRA微调适配器模块
LoRA Fine-tuning Adapter Module

实现LLaMA2模型的LoRA (Low-Rank Adaptation) 微调
支持垂直领域专业数据的适配训练
"""

import torch
from torch import nn
from typing import Optional, List, Dict, Any, Union
from dataclasses import dataclass
from pathlib import Path
import logging

# 配置日志
logger = logging.getLogger(__name__)


@dataclass
class LoRAConfig:
    """
    LoRA配置数据类
    LoRA configuration dataclass

    Attributes:
        r: LoRA秩 (rank), 低秩矩阵的维度
        lora_alpha: LoRA缩放参数, 实际缩放因子为 lora_alpha / r
        lora_dropout: Dropout概率
        target_modules: 需要应用LoRA的模块名称列表
        bias: bias处理方式, 可选 'none', 'all', 'lora_only'
        modules_to_save: 除了LoRA层外需要训练的模块
    """
    r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    target_modules: List[str] = None
    bias: str = "none"
    modules_to_save: Optional[List[str]] = None

    def __post_init__(self):
        if self.target_modules is None:
            self.target_modules = ["q_proj", "v_proj", "k_proj", "o_proj"]


class LoRALinear(nn.Module):
    """
    LoRA线性层
    LoRA Linear Layer

    在原始线性层旁路添加低秩分解矩阵
    公式: W' = W + BA, 其中 B∈R^{d×r}, A∈R^{r×k}

    Args:
        original_layer: 原始线性层
        r: LoRA秩
        lora_alpha: LoRA缩放参数
        lora_dropout: Dropout概率
    """

    def __init__(
            self,
            original_layer: nn.Linear,
            r: int = 16,
            lora_alpha: int = 32,
            lora_dropout: float = 0.05
    ):
        super().__init__()
        self.original_layer = original_layer
        self.r = r
        self.lora_alpha = lora_alpha
        self.scaling = lora_alpha / r

        # 获取原始层的维度
        in_features = original_layer.in_features
        out_features = original_layer.out_features

        # 冻结原始层参数
        for param in self.original_layer.parameters():
            param.requires_grad = False

        # 创建LoRA矩阵 A 和 B
        # A: 降维矩阵 (r × in_features), 使用kaiming初始化
        # B: 升维矩阵 (out_features × r), 初始化为0
        self.lora_A = nn.Parameter(torch.randn(r, in_features) * 0.01)
        self.lora_B = nn.Parameter(torch.zeros(out_features, r))

        # Dropout层
        self.lora_dropout = nn.Dropout(p=lora_dropout) if lora_dropout > 0 else nn.Identity()

        logger.info(f"LoRA layer created: in={in_features}, out={out_features}, r={r}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        Forward pass

        Args:
            x: 输入张量 (batch_size, seq_len, in_features)

        Returns:
            输出张量 (batch_size, seq_len, out_features)
        """
        # 原始层输出
        original_output = self.original_layer(x)

        # LoRA旁路: x @ A^T @ B^T * scaling
        lora_output = self.lora_dropout(x) @ self.lora_A.T @ self.lora_B.T * self.scaling

        return original_output + lora_output

    def merge_weights(self) -> None:
        """
        合并LoRA权重到原始层
        Merge LoRA weights into original layer

        用于推理时减少计算开销
        """
        with torch.no_grad():
            # W' = W + BA * scaling
            self.original_layer.weight.data += (self.lora_B @ self.lora_A) * self.scaling
            # 清零LoRA参数
            self.lora_A.data.zero_()
            self.lora_B.data.zero_()

        logger.info("LoRA weights merged into original layer")


class LLaMA2LoRAAdapter:
    """
    LLaMA2 LoRA适配器
    LLaMA2 LoRA Adapter

    负责加载LLaMA2模型并应用LoRA微调

    Args:
        model_name: 模型名称或路径
        lora_config: LoRA配置
        device: 运行设备
        torch_dtype: 数据类型

    Example:
        >>> adapter = LLaMA2LoRAAdapter("meta-llama/Llama-2-7b-hf")
        >>> model = adapter.load_model()
        >>> tokenizer = adapter.load_tokenizer()
    """

    def __init__(
            self,
            model_name: str,
            lora_config: Optional[LoRAConfig] = None,
            device: str = "cuda",
            torch_dtype: torch.dtype = torch.float16
    ):
        self.model_name = model_name
        self.lora_config = lora_config or LoRAConfig()
        self.device = device
        self.torch_dtype = torch_dtype
        self.model = None
        self.tokenizer = None

        logger.info(f"LoRA Adapter initialized for model: {model_name}")

    def load_tokenizer(self):
        """
        加载分词器
        Load tokenizer

        Returns:
            transformers.PreTrainedTokenizer
        """
        try:
            from transformers import AutoTokenizer
        except ImportError:
            raise ImportError("请安装transformers: pip install transformers")

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=True,
            use_fast=False
        )

        # 设置pad_token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        logger.info(f"Tokenizer loaded, vocab size: {len(self.tokenizer)}")
        return self.tokenizer

    def load_model(self, apply_lora: bool = True):
        """
        加载模型并应用LoRA
        Load model and apply LoRA

        Args:
            apply_lora: 是否应用LoRA

        Returns:
            transformers.PreTrainedModel
        """
        try:
            from transformers import AutoModelForCausalLM
        except ImportError:
            raise ImportError("请安装transformers: pip install transformers")

        logger.info(f"Loading model from {self.model_name}...")

        # 加载基础模型
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=self.torch_dtype,
            device_map="auto" if self.device == "cuda" else None,
            trust_remote_code=True,
            low_cpu_mem_usage=True
        )

        # 应用LoRA
        if apply_lora:
            self._apply_lora()

        # 打印可训练参数统计
        self._print_trainable_parameters()

        return self.model

    def _apply_lora(self) -> None:
        """
        将LoRA应用到模型的指定模块
        Apply LoRA to specified modules in the model
        """
        logger.info(f"Applying LoRA to modules: {self.lora_config.target_modules}")

        for name, module in self.model.named_modules():
            # 检查是否为目标模块
            for target in self.lora_config.target_modules:
                if target in name and isinstance(module, nn.Linear):
                    # 获取父模块和属性名
                    parts = name.rsplit('.', 1)
                    if len(parts) == 2:
                        parent_name, attr_name = parts
                        parent = self.model.get_submodule(parent_name)
                    else:
                        parent = self.model
                        attr_name = name

                    # 创建LoRA层并替换
                    lora_layer = LoRALinear(
                        module,
                        r=self.lora_config.r,
                        lora_alpha=self.lora_config.lora_alpha,
                        lora_dropout=self.lora_config.lora_dropout
                    )
                    setattr(parent, attr_name, lora_layer)
                    logger.info(f"LoRA applied to: {name}")

    def _print_trainable_parameters(self) -> None:
        """
        打印可训练参数统计
        Print trainable parameters statistics
        """
        trainable = 0
        total = 0

        for param in self.model.parameters():
            total += param.numel()
            if param.requires_grad:
                trainable += param.numel()

        percentage = 100 * trainable / total
        logger.info(f"Trainable parameters: {trainable:,} / {total:,} ({percentage:.2f}%)")

    def save_lora_weights(self, save_path: Union[str, Path]) -> None:
        """
        保存LoRA权重
        Save LoRA weights

        Args:
            save_path: 保存路径
        """
        save_path = Path(save_path)
        save_path.mkdir(parents=True, exist_ok=True)

        lora_state_dict = {}
        for name, module in self.model.named_modules():
            if isinstance(module, LoRALinear):
                lora_state_dict[f"{name}.lora_A"] = module.lora_A.data.cpu()
                lora_state_dict[f"{name}.lora_B"] = module.lora_B.data.cpu()

        torch.save(lora_state_dict, save_path / "lora_weights.bin")
        logger.info(f"LoRA weights saved to {save_path}")

    def load_lora_weights(self, load_path: Union[str, Path]) -> None:
        """
        加载LoRA权重
        Load LoRA weights

        Args:
            load_path: 权重文件路径
        """
        load_path = Path(load_path)
        lora_state_dict = torch.load(load_path / "lora_weights.bin")

        for name, module in self.model.named_modules():
            if isinstance(module, LoRALinear) and f"{name}.lora_A" in lora_state_dict:
                module.lora_A.data = lora_state_dict[f"{name}.lora_A"].to(self.device)
                module.lora_B.data = lora_state_dict[f"{name}.lora_B"].to(self.device)

        logger.info(f"LoRA weights loaded from {load_path}")

    def get_trainable_parameters(self) -> List[nn.Parameter]:
        """
        获取所有可训练参数
        Get all trainable parameters

        Returns:
            可训练参数列表
        """
        return [p for p in self.model.parameters() if p.requires_grad]


class DomainAdapter:
    """
    垂直领域适配器
    Domain-specific Adapter

    针对特定垂直领域进行模型微调和推理

    Args:
        model_name: 基础模型名称
        domain_name: 领域名称
        lora_config: LoRA配置

    Example:
        >>> adapter = DomainAdapter("meta-llama/Llama-2-7b-hf", "medical")
        >>> adapter.load_model()
        >>> output = adapter.inference("患者症状为...")
    """

    def __init__(
            self,
            model_name: str,
            domain_name: str = "general",
            lora_config: Optional[LoRAConfig] = None
    ):
        self.model_name = model_name
        self.domain_name = domain_name
        self.lora_adapter = LLaMA2LoRAAdapter(
            model_name=model_name,
            lora_config=lora_config
        )
        self.is_loaded = False

    def load_model(self, lora_weights_path: Optional[str] = None) -> None:
        """
        加载模型，可选加载预训练的LoRA权重

        Args:
            lora_weights_path: LoRA权重路径（可选）
        """
        self.lora_adapter.load_tokenizer()
        self.lora_adapter.load_model(apply_lora=True)

        if lora_weights_path:
            self.lora_adapter.load_lora_weights(lora_weights_path)

        self.is_loaded = True
        logger.info(f"Domain adapter loaded for domain: {self.domain_name}")

    def inference(
            self,
            prompt: str,
            max_new_tokens: int = 512,
            temperature: float = 0.7,
            top_p: float = 0.9,
            **kwargs
    ) -> str:
        """
        执行推理

        Args:
            prompt: 输入提示文本
            max_new_tokens: 最大生成token数
            temperature: 生成温度
            top_p: nucleus采样参数

        Returns:
            生成的文本
        """
        if not self.is_loaded:
            raise RuntimeError("模型未加载，请先调用 load_model()")

        # 编码输入
        inputs = self.lora_adapter.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=self.lora_adapter.lora_config.r * 128
        ).to(self.lora_adapter.device)

        # 生成
        with torch.no_grad():
            outputs = self.lora_adapter.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=True,
                pad_token_id=self.lora_adapter.tokenizer.eos_token_id,
                **kwargs
            )

        # 解码输出
        generated_text = self.lora_adapter.tokenizer.decode(
            outputs[0],
            skip_special_tokens=True
        )

        return generated_text

    def save(self, save_path: str) -> None:
        """保存领域适配器"""
        self.lora_adapter.save_lora_weights(save_path)
        logger.info(f"Domain adapter saved to {save_path}")
