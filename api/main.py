"""
FastAPI服务接口模块
FastAPI Service Interface Module

提供REST API接口，支持便捷的推理调用
"""

from fastapi import FastAPI, HTTPException, UploadFile, File, Form, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Any, Union
from enum import Enum
import logging
import json
import asyncio
from pathlib import Path

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ==================== Pydantic模型 ====================

class InputType(str, Enum):
    """输入类型枚举"""
    TEXT = "text"
    TABLE = "table"
    MIXED = "mixed"


class ExplanationLevel(str, Enum):
    """解释详细程度枚举"""
    BRIEF = "brief"
    STANDARD = "standard"
    DETAILED = "detailed"
    TECHNICAL = "technical"


class OutputFormat(str, Enum):
    """输出格式枚举"""
    TEXT = "text"
    MARKDOWN = "markdown"
    JSON = "json"
    HTML = "html"


class InferenceRequest(BaseModel):
    """
    推理请求模型
    Inference Request Model
    """
    query: str = Field(..., description="查询文本", min_length=1)
    input_type: InputType = Field(default=InputType.TEXT, description="输入类型")
    table_data: Optional[Dict[str, Any]] = Field(default=None, description="表格数据(字典格式)")
    context: Optional[List[str]] = Field(default=None, description="上下文信息列表")
    knowledge: Optional[Dict[str, str]] = Field(default=None, description="额外知识字典")
    max_hops: int = Field(default=5, ge=1, le=10, description="最大推理跳数")
    enable_decomposition: bool = Field(default=True, description="是否启用查询分解")
    explanation_level: ExplanationLevel = Field(default=ExplanationLevel.STANDARD, description="解释详细程度")
    output_format: OutputFormat = Field(default=OutputFormat.MARKDOWN, description="输出格式")

    class Config:
        json_schema_extra = {
            "example": {
                "query": "比较A公司和B公司在2023年的营收情况",
                "input_type": "text",
                "context": ["A公司是一家科技公司", "B公司是一家传统企业"],
                "max_hops": 3,
                "enable_decomposition": True,
                "explanation_level": "standard",
                "output_format": "markdown"
            }
        }


class TextOnlyRequest(BaseModel):
    """纯文本推理请求"""
    text: str = Field(..., description="输入文本", min_length=1)
    max_new_tokens: int = Field(default=512, ge=1, le=2048, description="最大生成token数")
    temperature: float = Field(default=0.7, ge=0.0, le=2.0, description="生成温度")

    class Config:
        json_schema_extra = {
            "example": {
                "text": "请解释什么是机器学习？",
                "max_new_tokens": 256,
                "temperature": 0.7
            }
        }


class TableInferenceRequest(BaseModel):
    """表格推理请求"""
    query: str = Field(..., description="查询问题")
    table: Dict[str, Any] = Field(..., description="表格数据")
    format: str = Field(default="markdown", description="表格格式")

    class Config:
        json_schema_extra = {
            "example": {
                "query": "这个表格显示的趋势是什么？",
                "table": {
                    "headers": ["年份", "销售额", "增长率"],
                    "rows": [
                        ["2021", "100万", "10%"],
                        ["2022", "120万", "20%"],
                        ["2023", "150万", "25%"]
                    ]
                },
                "format": "markdown"
            }
        }


class DecompositionRequest(BaseModel):
    """查询分解请求"""
    query: str = Field(..., description="待分解的查询")
    strategy: str = Field(default="hybrid", description="分解策略")

    class Config:
        json_schema_extra = {
            "example": {
                "query": "比较A公司和B公司在2023年的营收，并分析增长趋势",
                "strategy": "hybrid"
            }
        }


class TrainingRequest(BaseModel):
    """训练请求"""
    dataset_path: str = Field(..., description="数据集路径")
    output_dir: str = Field(default="./output", description="输出目录")
    num_epochs: int = Field(default=3, ge=1, le=100, description="训练轮数")
    batch_size: int = Field(default=4, ge=1, le=32, description="批次大小")
    learning_rate: float = Field(default=2e-4, ge=1e-6, le=1e-2, description="学习率")
    lora_r: int = Field(default=16, ge=1, le=64, description="LoRA秩")

    class Config:
        json_schema_extra = {
            "example": {
                "dataset_path": "/data/train.json",
                "output_dir": "./output/model",
                "num_epochs": 3,
                "batch_size": 4,
                "learning_rate": 0.0002,
                "lora_r": 16
            }
        }


class InferenceResponse(BaseModel):
    """推理响应模型"""
    success: bool = Field(..., description="是否成功")
    answer: str = Field(..., description="答案")
    confidence: float = Field(..., description="置信度")
    explanation: str = Field(..., description="解释")
    reasoning_steps: List[Dict[str, Any]] = Field(default_factory=list, description="推理步骤")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="元数据")


class DecompositionResponse(BaseModel):
    """分解响应模型"""
    success: bool
    original_query: str
    query_type: str
    complexity_score: float
    sub_queries: List[Dict[str, Any]]
    execution_plan: List[List[str]]


class HealthResponse(BaseModel):
    """健康检查响应"""
    status: str
    model_loaded: bool
    version: str


# ==================== FastAPI应用 ====================

def create_app() -> FastAPI:
    """
    创建FastAPI应用实例
    Create FastAPI application instance

    Returns:
        FastAPI应用
    """
    app = FastAPI(
        title="LLaMA2 LoRA 推理服务",
        description="""
## 垂直领域LLaMA2+LoRA推理服务

### 功能特性
- **统一语义建模**: 支持文本、表格及混合输入的统一语义表示
- **跨模态对齐**: 文本与表格模态的对比学习对齐
- **复杂查询分解**: 将复杂问题自动分解为子问题
- **多跳推理**: 支持链式多步骤推理
- **答案可解释性**: 提供详细的推理过程解释

### 使用说明
1. `/infer` - 通用推理接口
2. `/infer/text` - 纯文本推理
3. `/infer/table` - 表格推理
4. `/decompose` - 查询分解
5. `/train` - LoRA微调训练
        """,
        version="1.0.0",
        docs_url="/docs",
        redoc_url="/redoc"
    )

    # CORS中间件
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    return app


app = create_app()


# ==================== 全局状态 ====================

class AppState:
    """应用状态管理"""

    def __init__(self):
        self.model_loaded = False
        self.model = None
        self.tokenizer = None
        self.domain_adapter = None
        self.semantic_model = None
        self.aligner = None
        self.decomposer = None
        self.reasoner = None
        self.explainability_engine = None

    def initialize(self):
        """初始化模型和组件"""
        logger.info("Initializing models and components...")
        # 实际使用时加载模型
        # self._load_model()
        self.model_loaded = True
        logger.info("Initialization complete")

    def _load_model(self):
        """加载模型（实际实现）"""
        # from backend.models import DomainAdapter
        # from backend.core import (
        #     UnifiedSemanticModel,
        #     CrossModalAligner,
        #     QueryDecomposer,
        #     MultiHopReasoner,
        #     ExplainabilityEngine
        # )
        pass


state = AppState()


# ==================== API路由 ====================

@app.on_event("startup")
async def startup_event():
    """应用启动事件"""
    state.initialize()
    logger.info("Application started")


@app.get("/", response_model=HealthResponse, tags=["系统"])
async def root():
    """
    根路径 - 服务状态检查
    """
    return HealthResponse(
        status="healthy",
        model_loaded=state.model_loaded,
        version="1.0.0"
    )


@app.get("/health", response_model=HealthResponse, tags=["系统"])
async def health_check():
    """
    健康检查接口
    Health check endpoint
    """
    return HealthResponse(
        status="healthy",
        model_loaded=state.model_loaded,
        version="1.0.0"
    )


@app.post("/infer", response_model=InferenceResponse, tags=["推理"])
async def inference(request: InferenceRequest):
    """
    通用推理接口
    General inference endpoint

    支持文本、表格、混合输入的推理

    Args:
        request: 推理请求参数

    Returns:
        InferenceResponse: 推理结果
    """
    try:
        logger.info(f"Received inference request: {request.query[:50]}...")

        # 模拟推理过程
        # 实际实现时调用相应模块

        # 1. 查询分解（如果启用）
        sub_queries = []
        if request.enable_decomposition:
            # decomposition = state.decomposer.decompose(request.query)
            # sub_queries = [{"id": sq.id, "text": sq.text} for sq in decomposition.sub_queries]
            sub_queries = [
                {"id": "0", "text": "理解查询意图"},
                {"id": "1", "text": "检索相关信息"},
                {"id": "2", "text": "综合分析得出结论"}
            ]

        # 2. 语义建模
        # semantic_rep = state.semantic_model.forward(text=request.query)

        # 3. 多跳推理
        # reasoning_chain = state.reasoner.reason(request.query, request.context)

        # 4. 生成解释
        answer = f"基于分析，对于问题「{request.query}」的回答..."
        confidence = 0.85

        # 生成解释
        explanation = _generate_explanation(
            request.query,
            answer,
            sub_queries,
            request.explanation_level,
            request.output_format
        )

        return InferenceResponse(
            success=True,
            answer=answer,
            confidence=confidence,
            explanation=explanation,
            reasoning_steps=sub_queries,
            metadata={
                "input_type": request.input_type.value,
                "max_hops": request.max_hops,
                "decomposition_enabled": request.enable_decomposition
            }
        )

    except Exception as e:
        logger.error(f"Inference error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/infer/text", response_model=InferenceResponse, tags=["推理"])
async def text_inference(request: TextOnlyRequest):
    """
    纯文本推理接口
    Text-only inference endpoint

    Args:
        request: 文本推理请求

    Returns:
        InferenceResponse: 推理结果
    """
    try:
        logger.info(f"Text inference: {request.text[:50]}...")

        # 模拟生成
        answer = f"对于输入文本的分析结果..."
        confidence = 0.9

        return InferenceResponse(
            success=True,
            answer=answer,
            confidence=confidence,
            explanation=f"对文本「{request.text[:100]}...」进行了分析和推理。",
            reasoning_steps=[],
            metadata={"temperature": request.temperature, "max_tokens": request.max_new_tokens}
        )

    except Exception as e:
        logger.error(f"Text inference error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/infer/table", response_model=InferenceResponse, tags=["推理"])
async def table_inference(request: TableInferenceRequest):
    """
    表格推理接口
    Table inference endpoint

    Args:
        request: 表格推理请求

    Returns:
        InferenceResponse: 推理结果
    """
    try:
        logger.info(f"Table inference for query: {request.query[:50]}...")

        # 处理表格数据
        table = request.table

        # 模拟推理
        answer = f"根据表格数据分析，{request.query}"
        confidence = 0.82

        # 生成表格摘要
        table_summary = _summarize_table(table)

        return InferenceResponse(
            success=True,
            answer=answer,
            confidence=confidence,
            explanation=f"表格包含 {len(table.get('rows', []))} 行数据。\n{table_summary}",
            reasoning_steps=[
                {"id": "0", "text": "解析表格结构"},
                {"id": "1", "text": "提取关键信息"},
                {"id": "2", "text": "回答查询问题"}
            ],
            metadata={"table_format": request.format}
        )

    except Exception as e:
        logger.error(f"Table inference error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/decompose", response_model=DecompositionResponse, tags=["查询处理"])
async def decompose_query(request: DecompositionRequest):
    """
    查询分解接口
    Query decomposition endpoint

    将复杂查询分解为多个子查询

    Args:
        request: 分解请求

    Returns:
        DecompositionResponse: 分解结果
    """
    try:
        logger.info(f"Decomposing query: {request.query[:50]}...")

        # 模拟分解过程
        # 实际实现:
        # decomposition = state.decomposer.decompose(request.query, strategy=request.strategy)

        # 简化示例
        sub_queries = [
            {"id": "0", "text": "识别查询中的关键实体", "dependencies": []},
            {"id": "1", "text": "检索实体相关信息", "dependencies": ["0"]},
            {"id": "2", "text": "整合信息得出结论", "dependencies": ["1"]}
        ]

        return DecompositionResponse(
            success=True,
            original_query=request.query,
            query_type="multi_hop",
            complexity_score=0.7,
            sub_queries=sub_queries,
            execution_plan=[["0"], ["1"], ["2"]]
        )

    except Exception as e:
        logger.error(f"Decomposition error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/train", tags=["训练"])
async def train_model(
        request: TrainingRequest,
        background_tasks: BackgroundTasks
):
    """
    LoRA微调训练接口
    LoRA fine-tuning training endpoint

    Args:
        request: 训练请求参数
        background_tasks: 后台任务

    Returns:
        训练任务信息
    """
    try:
        logger.info(f"Starting training: {request.dataset_path}")

        # 添加后台训练任务
        background_tasks.add_task(
            _train_model_task,
            request.dict()
        )

        return {
            "success": True,
            "message": "Training started in background",
            "config": {
                "dataset_path": request.dataset_path,
                "output_dir": request.output_dir,
                "num_epochs": request.num_epochs,
                "batch_size": request.batch_size,
                "learning_rate": request.learning_rate,
                "lora_r": request.lora_r
            }
        }

    except Exception as e:
        logger.error(f"Training error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/upload/table", tags=["数据"])
async def upload_table(file: UploadFile = File(...)):
    """
    上传表格文件
    Upload table file

    支持 CSV, Excel 格式

    Args:
        file: 上传的文件

    Returns:
        解析后的表格数据
    """
    try:
        content = await file.read()
        filename = file.filename.lower()

        if filename.endswith('.csv'):
            # 解析CSV
            import csv
            from io import StringIO

            text = content.decode('utf-8')
            reader = csv.reader(StringIO(text))
            rows = list(reader)

            if rows:
                headers = rows[0]
                data_rows = rows[1:]

                return {
                    "success": True,
                    "table": {
                        "headers": headers,
                        "rows": data_rows
                    },
                    "row_count": len(data_rows)
                }

        elif filename.endswith(('.xlsx', '.xls')):
            # 解析Excel（需要openpyxl）
            return {
                "success": False,
                "message": "Excel解析需要安装openpyxl库"
            }
        else:
            raise HTTPException(status_code=400, detail="不支持的文件格式")

    except Exception as e:
        logger.error(f"File upload error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


# ==================== 辅助函数 ====================

def _generate_explanation(
        query: str,
        answer: str,
        steps: List[Dict],
        level: ExplanationLevel,
        format: OutputFormat
) -> str:
    """
    生成解释文本
    Generate explanation text
    """
    if format == OutputFormat.JSON:
        import json
        return json.dumps({
            "query": query,
            "answer": answer,
            "steps": steps,
            "level": level.value
        }, ensure_ascii=False, indent=2)

    elif format == OutputFormat.MARKDOWN:
        lines = [
            f"## 推理过程\n",
            f"**问题**: {query}\n",
            f"**答案**: {answer}\n",
            "\n### 推理步骤\n"
        ]
        for i, step in enumerate(steps, 1):
            lines.append(f"{i}. {step.get('text', step.get('id', ''))}")
        return "\n".join(lines)

    else:  # TEXT
        lines = [
            f"问题: {query}",
            f"答案: {answer}",
            "推理步骤:"
        ]
        for i, step in enumerate(steps, 1):
            lines.append(f"  {i}. {step.get('text', step.get('id', ''))}")
        return "\n".join(lines)


def _summarize_table(table: Dict) -> str:
    """
    生成表格摘要
    Summarize table
    """
    headers = table.get("headers", [])
    rows = table.get("rows", [])

    if not headers or not rows:
        return "空表格"

    summary = f"表格包含字段: {', '.join(str(h) for h in headers)}"
    summary += f"\n共 {len(rows)} 条数据记录"

    return summary


async def _train_model_task(config: Dict):
    """
    后台训练任务
    Background training task
    """
    logger.info(f"Training task started with config: {config}")

    # 模拟训练过程
    # 实际实现:
    # from backend.models import LLaMA2LoRAAdapter
    # from transformers import Trainer, TrainingArguments
    # ...

    import time
    for epoch in range(config.get("num_epochs", 3)):
        logger.info(f"Epoch {epoch + 1}/{config['num_epochs']}")
        await asyncio.sleep(1)  # 模拟训练

    logger.info("Training completed")


# ==================== 运行入口 ====================

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        workers=1
    )
