# Backend 目录说明

LLaMA2 + LoRA 垂直领域推理系统后端

## 项目结构

```
backend/
├── config/                 # 配置模块
│   ├── __init__.py
│   └── settings.py        # 全局配置和超参数
│
├── models/                 # 模型模块
│   ├── __init__.py
│   └── lora_adapter.py    # LoRA微调适配器实现
│
├── core/                   # 核心功能模块
│   ├── __init__.py
│   ├── unified_semantic.py          # 统一语义建模
│   ├── cross_modal_alignment.py     # 跨模态对齐
│   ├── query_decomposition.py       # 复杂查询分解
│   ├── multi_hop_reasoning.py       # 多跳推理
│   └── explainability.py            # 答案可解释性
│
├── api/                    # API服务模块
│   ├── __init__.py
│   └── main.py            # FastAPI服务接口
│
├── utils/                  # 工具模块
│   ├── __init__.py
│   └── helpers.py         # 辅助函数和工具类
│
├── __init__.py            # 包初始化
├── run_server.py          # 服务启动脚本
└── requirements.txt       # 依赖列表
```

## 模块说明

### 1. 配置模块 (config/)
- `ModelConfig`: 模型配置（LoRA参数、设备等）
- `TrainingConfig`: 训练配置
- `InferenceConfig`: 推理配置
- `APIConfig`: API服务配置

### 2. 模型模块 (models/)
- `LoRAConfig`: LoRA配置数据类
- `LoRALinear`: LoRA线性层实现
- `LLaMA2LoRAAdapter`: LLaMA2 LoRA适配器
- `DomainAdapter`: 垂直领域适配器

### 3. 核心模块 (core/)

#### 3.1 统一语义建模 (unified_semantic.py)
- `TextEncoder`: 文本编码器
- `TableEncoder`: 表格编码器
- `UnifiedSemanticModel`: 统一语义模型
- `SemanticIndex`: 语义索引

#### 3.2 跨模态对齐 (cross_modal_alignment.py)
- `ProjectionHead`: 投影头
- `ContrastiveLoss`: 对比学习损失
- `CrossModalAttention`: 跨模态注意力
- `CrossModalAligner`: 跨模态对齐器

#### 3.3 复杂查询分解 (query_decomposition.py)
- `QueryClassifier`: 查询分类器
- `ComplexityEstimator`: 复杂度评估器
- `QueryDecomposer`: 查询分解器
- `SubQuery`: 子查询数据类

#### 3.4 多跳推理 (multi_hop_reasoning.py)
- `KnowledgeStore`: 知识存储
- `ReasoningPathPlanner`: 推理路径规划器
- `InferenceEngine`: 推理引擎
- `MultiHopReasoner`: 多跳推理器

#### 3.5 答案可解释性 (explainability.py)
- `ExplainabilityEngine`: 可解释性引擎
- `ExplanationFormatter`: 解释格式化器
- `ConfidenceCalibrator`: 置信度校准器

### 4. API模块 (api/)
- `/infer`: 通用推理接口
- `/infer/text`: 纯文本推理
- `/infer/table`: 表格推理
- `/decompose`: 查询分解
- `/train`: LoRA微调训练

## 快速开始

```bash
# 安装依赖
pip install -r requirements.txt

# 启动服务
python run_server.py --host 0.0.0.0 --port 8000

# 访问文档
# http://localhost:8000/docs
```

## 使用示例

from backend import DomainAdapter, MultiHopReasoner, QueryDecomposer

# 1. 加载模型
adapter = DomainAdapter("meta-llama/Llama-2-7b-hf")
adapter.load_model()

# 2. 查询分解
decomposer = QueryDecomposer()
result = decomposer.decompose("比较A公司和B公司的营收")

# 3. 多跳推理
reasoner = MultiHopReasoner(adapter.model, adapter.tokenizer)
chain = reasoner.reason("问题")

# 4. 解释输出
print(chain.final_answer)
