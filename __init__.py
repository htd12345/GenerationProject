"""
Backend Package
后端主包

LLaMA2 + LoRA 垂直领域推理系统

模块结构:
- config: 配置管理
- models: 模型定义（LoRA适配器）
- core: 核心功能模块
  - unified_semantic: 统一语义建模
  - cross_modal_alignment: 跨模态对齐
  - query_decomposition: 复杂查询分解
  - multi_hop_reasoning: 多跳推理
  - explainability: 答案可解释性
- api: FastAPI服务接口
- utils: 工具函数

使用示例:
    from backend import DomainAdapter
    from backend.core import MultiHopReasoner
    # 加载模型
    adapter = DomainAdapter("meta-llama/Llama-2-7b-hf")
    adapter.load_model()
    # 执行推理
    result = adapter.inference("问题文本")
"""

__version__ = "1.0.0"
__author__ = "LLaMA2 LoRA Project"

from .config import Config, config
from .models import (
    LoRAConfig,
    LLaMA2LoRAAdapter,
    DomainAdapter
)
from .core import (
    # Unified Semantic
    UnifiedSemanticModel,
    TextEncoder,
    TableEncoder,

    # Cross-Modal Alignment
    CrossModalAligner,

    # Query Decomposition
    QueryDecomposer,
    QueryType,

    # Multi-Hop Reasoning
    MultiHopReasoner,
    ReasoningChain,

    # Explainability
    ExplainabilityEngine,
    ExplanationLevel,
    ExplanationFormat
)
from .api import app, create_app
from .utils import set_seed, get_device

__all__ = [
    # config
    "config",
    "config",

    # models
    "LoRAConfig",
    "LLaMA2LoRAAdapter",
    "DomainAdapter",

    # core - Semantic
    "UnifiedSemanticModel",
    "TextEncoder",
    "TableEncoder",

    # core - Alignment
    "CrossModalAligner",

    # core - Decomposition
    "QueryDecomposer",
    "QueryType",

    # core - Reasoning
    "MultiHopReasoner",
    "ReasoningChain",

    # core - Explainability
    "ExplainabilityEngine",
    "ExplanationLevel",
    "ExplanationFormat",

    # api
    "app",
    "create_app",

    # utils
    "set_seed",
    "get_device"
]
