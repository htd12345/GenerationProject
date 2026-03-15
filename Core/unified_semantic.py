"""
统一语义建模模块
Unified Semantic Modeling Module

负责将文本和表格数据转换为统一的语义表示
支持多种模态输入的统一编码
"""

import torch
import torch.nn as nn
from torch import Tensor
from typing import Dict, List, Optional, Union, Tuple, Any
from dataclasses import dataclass
from abc import ABC, abstractmethod
import logging
import re

logger = logging.getLogger(__name__)


@dataclass
class SemanticRepresentation:
    """
    语义表示数据类
    Semantic representation dataclass

    Attributes:
        embeddings: 语义向量表示 (batch_size, seq_len, hidden_size)
        attention_mask: 注意力掩码
        modality_type: 模态类型 ('text', 'table', 'mixed')
        tokens: 原始token列表
        metadata: 额外元数据
    """
    embeddings: Tensor
    attention_mask: Tensor
    modality_type: str
    tokens: Optional[List[str]] = None
    metadata: Optional[Dict[str, Any]] = None

    def to(self, device: str) -> 'SemanticRepresentation':
        """移动到指定设备"""
        self.embeddings = self.embeddings.to(device)
        self.attention_mask = self.attention_mask.to(device)
        return self


class BaseEncoder(ABC):
    """
    编码器基类
    Base encoder class

    定义所有编码器的通用接口
    """

    @abstractmethod
    def encode(self, inputs: Any) -> SemanticRepresentation:
        """
        编码输入数据

        Args:
            inputs: 输入数据

        Returns:
            SemanticRepresentation: 语义表示对象
        """
        pass

    @abstractmethod
    def get_hidden_size(self) -> int:
        """获取隐藏层维度"""
        pass


class TextEncoder(BaseEncoder):
    """
    文本编码器
    Text Encoder

    将纯文本转换为语义向量表示

    Args:
        model: 预训练语言模型
        tokenizer: 分词器
        max_length: 最大序列长度
        device: 运行设备

    Example:
        >>> encoder = TextEncoder(model, tokenizer)
        >>> semantic_rep = encoder.encode("这是一个文本输入")
        >>> print(semantic_rep.embeddings.shape)
    """

    def __init__(
            self,
            model: nn.Module,
            tokenizer: Any,
            max_length: int = 2048,
            device: str = "cuda"
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.device = device
        self.hidden_size = model.config.hidden_size

        logger.info(f"TextEncoder initialized, hidden_size={self.hidden_size}")

    def encode(self, text: Union[str, List[str]]) -> SemanticRepresentation:
        """
        编码文本

        Args:
            text: 输入文本或文本列表

        Returns:
            SemanticRepresentation: 文本的语义表示
        """
        # 处理单个文本和文本列表
        is_single = isinstance(text, str)
        if is_single:
            text = [text]

        # Tokenize
        encoded = self.tokenizer(
            text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_length
        )

        input_ids = encoded["input_ids"].to(self.device)
        attention_mask = encoded["attention_mask"].to(self.device)

        # 获取模型输出
        with torch.no_grad():
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                return_dict=True
            )

        # 使用最后一层hidden state
        embeddings = outputs.hidden_states[-1]

        # 获取tokens
        tokens = [self.tokenizer.convert_ids_to_tokens(ids) for ids in input_ids]

        return SemanticRepresentation(
            embeddings=embeddings,
            attention_mask=attention_mask,
            modality_type="text",
            tokens=tokens[0] if is_single else tokens,
            metadata={"input_texts": text}
        )

    def get_hidden_size(self) -> int:
        return self.hidden_size


class TableEncoder(BaseEncoder):
    """
    表格编码器
    Table Encoder

    将表格数据转换为语义向量表示
    支持多种表格格式：DataFrame、字典、二维列表

    Args:
        model: 预训练语言模型
        tokenizer: 分词器
        max_length: 最大序列长度
        device: 运行设备
        table_format: 表格序列化格式 ('linear', 'markdown', 'html')

    Example:
        import pandas as pd
        df = pd.DataFrame({"A": [1, 2], "B": [3, 4]})
        encoder = TableEncoder(model, tokenizer)
        semantic_rep = encoder.encode(df)
    """

    def __init__(
            self,
            model: nn.Module,
            tokenizer: Any,
            max_length: int = 2048,
            device: str = "cuda",
            table_format: str = "markdown"
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.device = device
        self.table_format = table_format
        self.hidden_size = model.config.hidden_size

        logger.info(f"TableEncoder initialized, format={table_format}")

    def serialize_table(self, table: Any) -> str:
        """
        将表格序列化为文本

        Args:
            table: 表格数据 (DataFrame, dict, list)

        Returns:
            序列化后的文本
        """
        # 支持pandas DataFrame
        try:
            import pandas as pd
            if isinstance(table, pd.DataFrame):
                if self.table_format == "markdown":
                    return self._dataframe_to_markdown(table)
                elif self.table_format == "html":
                    return table.to_html()
                else:
                    return self._dataframe_to_linear(table)
        except ImportError:
            pass

        # 支持字典格式
        if isinstance(table, dict):
            return self._dict_to_text(table)

        # 支持二维列表
        if isinstance(table, list):
            return self._list_to_text(table)

        raise ValueError(f"不支持的表格类型: {type(table)}")

    def _dataframe_to_markdown(self, df) -> str:
        """将DataFrame转换为Markdown格式"""
        lines = []

        # 表头
        headers = df.columns.tolist()
        lines.append("| " + " | ".join(str(h) for h in headers) + " |")
        lines.append("| " + " | ".join("---" for _ in headers) + " |")

        # 数据行
        for _, row in df.iterrows():
            values = [str(v) for v in row.values]
            lines.append("| " + " | ".join(values) + " |")

        return "\n".join(lines)

    def _dataframe_to_linear(self, df) -> str:
        """将DataFrame转换为线性文本格式"""
        parts = []

        for col in df.columns:
            values = df[col].tolist()
            parts.append(f"{col}: {', '.join(str(v) for v in values)}")

        return " | ".join(parts)

    def _dict_to_text(self, table_dict: dict) -> str:
        """将字典转换为文本"""
        if "headers" in table_dict and "rows" in table_dict:
            # 标准格式: {"headers": [...], "rows": [[...], ...]}
            headers = table_dict["headers"]
            rows = table_dict["rows"]
            lines = ["| " + " | ".join(str(h) for h in headers) + " |"]
            lines.append("| " + " | ".join("---" for _ in headers) + " |")
            for row in rows:
                lines.append("| " + " | ".join(str(v) for v in row) + " |")
            return "\n".join(lines)
        else:
            # 键值对格式
            return " | ".join(f"{k}: {v}" for k, v in table_dict.items())

    def _list_to_text(self, table_list: list) -> str:
        """将二维列表转换为文本"""
        if not table_list:
            return ""

        # 假设第一行是表头
        headers = table_list[0]
        rows = table_list[1:]

        lines = ["| " + " | ".join(str(h) for h in headers) + " |"]
        lines.append("| " + " | ".join("---" for _ in headers) + " |")
        for row in rows:
            lines.append("| " + " | ".join(str(v) for v in row) + " |")

        return "\n".join(lines)

    def encode(self, table: Any) -> SemanticRepresentation:
        """
        编码表格

        Args:
            table: 表格数据

        Returns:
            SemanticRepresentation: 表格的语义表示
        """
        # 序列化表格
        serialized = self.serialize_table(table)

        # 使用文本编码器编码序列化后的文本
        text_encoder = TextEncoder(
            self.model,
            self.tokenizer,
            self.max_length,
            self.device
        )

        result = text_encoder.encode(serialized)
        result.modality_type = "table"
        result.metadata = {"serialized_table": serialized}

        return result

    def get_hidden_size(self) -> int:
        return self.hidden_size


class UnifiedSemanticModel(nn.Module):
    """
    统一语义建模模型
    Unified Semantic Model

    整合文本和表格编码，提供统一的语义表示

    Args:
        model: 基础语言模型
        tokenizer: 分词器
        hidden_size: 隐藏层维度
        dropout: Dropout概率
        device: 运行设备

    Example:
        >>> usm = UnifiedSemanticModel(model, tokenizer)
        >>>
        >>> # 纯文本输入
        >>> text_rep = usm.encode_text("这是一个问题")
        >>>
        >>> # 表格输入
        >>> table_rep = usm.encode_table(dataframe)
        >>>
        >>> # 混合输入
        >>> mixed_rep = usm.encode_mixed("问题", dataframe)
    """

    def __init__(
            self,
            model: nn.Module,
            tokenizer: Any,
            hidden_size: int = 4096,
            dropout: float = 0.1,
            device: str = "cuda"
    ):
        super().__init__()
        self.device = device
        self.hidden_size = hidden_size

        # 初始化编码器
        self.text_encoder = TextEncoder(model, tokenizer, device=device)
        self.table_encoder = TableEncoder(model, tokenizer, device=device)

        # 模态融合层
        self.modal_fusion = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size)
        )

        # 模态类型嵌入
        self.modal_type_embedding = nn.Embedding(3, hidden_size)  # 0: text, 1: table, 2: mixed

        logger.info(f"UnifiedSemanticModel initialized, hidden_size={hidden_size}")

    def encode_text(self, text: Union[str, List[str]]) -> SemanticRepresentation:
        """
        编码纯文本

        Args:
            text: 输入文本

        Returns:
            文本的语义表示
        """
        return self.text_encoder.encode(text)

    def encode_table(self, table: Any) -> SemanticRepresentation:
        """
        编码表格

        Args:
            table: 表格数据

        Returns:
            表格的语义表示
        """
        return self.table_encoder.encode(table)

    def encode_mixed(
            self,
            text: str,
            table: Any,
            fusion_strategy: str = "concat"
    ) -> SemanticRepresentation:
        """
        编码混合输入（文本 + 表格）

        Args:
            text: 文本输入
            table: 表格输入
            fusion_strategy: 融合策略 ('concat', 'attention', 'mean')

        Returns:
            混合语义表示
        """
        # 分别编码
        text_rep = self.encode_text(text)
        table_rep = self.encode_table(table)

        # 获取池化表示
        text_pooled = self._mean_pooling(text_rep.embeddings, text_rep.attention_mask)
        table_pooled = self._mean_pooling(table_rep.embeddings, table_rep.attention_mask)

        # 融合
        if fusion_strategy == "concat":
            combined = torch.cat([text_pooled, table_pooled], dim=-1)
            fused = self.modal_fusion(combined)
        elif fusion_strategy == "mean":
            fused = (text_pooled + table_pooled) / 2
        else:
            fused = text_pooled + table_pooled

        # 扩展回序列维度
        batch_size = fused.size(0)
        fused = fused.unsqueeze(1).expand(-1, max(
            text_rep.embeddings.size(1),
            table_rep.embeddings.size(1)
        ), -1)

        # 创建新的语义表示
        return SemanticRepresentation(
            embeddings=fused,
            attention_mask=torch.ones(fused.size()[:2], device=self.device),
            modality_type="mixed",
            metadata={
                "text_rep": text_rep.metadata,
                "table_rep": table_rep.metadata,
                "fusion_strategy": fusion_strategy
            }
        )

    def _mean_pooling(self, embeddings: Tensor, attention_mask: Tensor) -> Tensor:
        """
        平均池化

        Args:
            embeddings: (batch, seq_len, hidden)
            attention_mask: (batch, seq_len)

        Returns:
            池化后的表示 (batch, hidden)
        """
        mask = attention_mask.unsqueeze(-1).expand(embeddings.size()).float()
        sum_embeddings = torch.sum(embeddings * mask, dim=1)
        sum_mask = torch.clamp(mask.sum(dim=1), min=1e-9)
        return sum_embeddings / sum_mask

    def forward(
            self,
            text: Optional[str] = None,
            table: Optional[Any] = None,
            input_type: str = "auto"
    ) -> SemanticRepresentation:
        """
        前向传播

        Args:
            text: 可选的文本输入
            table: 可选的表格输入
            input_type: 输入类型 ('text', 'table', 'mixed', 'auto')

        Returns:
            语义表示
        """
        if input_type == "auto":
            # 自动判断输入类型
            if text is not None and table is not None:
                input_type = "mixed"
            elif table is not None:
                input_type = "table"
            else:
                input_type = "text"

        if input_type == "text":
            return self.encode_text(text)
        elif input_type == "table":
            return self.encode_table(table)
        else:
            return self.encode_mixed(text, table)


class SemanticIndex:
    """
    语义索引
    Semantic Index

    用于存储和检索语义表示，支持相似度搜索

    Args:
        dimension: 向量维度
        metric: 相似度度量 ('cosine', 'euclidean', 'dot')
    """

    def __init__(
            self,
            dimension: int = 4096,
            metric: str = "cosine"
    ):
        self.dimension = dimension
        self.metric = metric
        self.vectors = []
        self.metadata = []

        logger.info(f"SemanticIndex created, dimension={dimension}, metric={metric}")

    def add(self, representation: SemanticRepresentation, metadata: Dict = None) -> None:
        """
        添加语义表示到索引

        Args:
            representation: 语义表示
            metadata: 关联的元数据
        """
        # 池化并存储
        pooled = representation.embeddings.mean(dim=1).squeeze().cpu().numpy()
        self.vectors.append(pooled)
        self.metadata.append(metadata or {})

    def search(
            self,
            query: SemanticRepresentation,
            top_k: int = 5
    ) -> List[Tuple[float, Dict]]:
        """
        搜索相似项

        Args:
            query: 查询语义表示
            top_k: 返回前k个结果

        Returns:
            [(相似度分数, 元数据), ...]
        """
        import numpy as np

        query_vec = query.embeddings.mean(dim=1).squeeze().cpu().numpy()

        # 计算相似度
        scores = []
        for vec in self.vectors:
            if self.metric == "cosine":
                score = np.dot(query_vec, vec) / (
                        np.linalg.norm(query_vec) * np.linalg.norm(vec) + 1e-9
                )
            elif self.metric == "dot":
                score = np.dot(query_vec, vec)
            else:  # euclidean
                score = -np.linalg.norm(query_vec - vec)
            scores.append(score)

        # 排序并返回top_k
        indices = np.argsort(scores)[::-1][:top_k]
        return [(scores[i], self.metadata[i]) for i in indices]
