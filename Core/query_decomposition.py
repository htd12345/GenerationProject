"""
复杂查询分解模块
Complex Query Decomposition Module

将复杂的多跳问题分解为多个简单的子问题
支持递归分解和并行分解策略
"""

import torch
import torch.nn as nn
from torch import Tensor
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
from enum import Enum
import re
import logging
from collections import defaultdict

logger = logging.getLogger(__name__)


class QueryType(Enum):
    """
    查询类型枚举
    Query Type Enum

    定义不同类型的查询
    """
    SIMPLE = "simple"  # 简单查询，单跳
    MULTI_HOP = "multi_hop"  # 多跳查询，需要多步推理
    COMPARATIVE = "comparative"  # 比较查询，涉及多个实体对比
    AGGREGATION = "aggregation"  # 聚合查询，需要计算统计值
    BOOLEAN = "boolean"  # 布尔查询，是/否问题
    TEMPORAL = "temporal"  # 时序查询，涉及时间维度


class DecompositionStrategy(Enum):
    """
    分解策略枚举
    Decomposition Strategy Enum
    """
    SEQUENTIAL = "sequential"  # 顺序分解，子问题有依赖关系
    PARALLEL = "parallel"  # 并行分解，子问题相互独立
    HYBRID = "hybrid"  # 混合分解，结合顺序和并行


@dataclass
class SubQuery:
    """
    子查询数据类
    Sub-Query Dataclass

    Attributes:
        id: 子查询ID
        text: 子查询文本
        query_type: 查询类型
        dependencies: 依赖的子查询ID列表
        variables: 查询变量绑定
        metadata: 额外元数据
    """
    id: str
    text: str
    query_type: QueryType
    dependencies: List[str] = field(default_factory=list)
    variables: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __hash__(self):
        return hash(self.id)

    def __eq__(self, other):
        return isinstance(other, SubQuery) and self.id == other.id


@dataclass
class QueryDecomposition:
    """
    查询分解结果数据类
    Query Decomposition Result Dataclass

    Attributes:
        original_query: 原始查询文本
        query_type: 原始查询类型
        sub_queries: 分解后的子查询列表
        execution_order: 执行顺序（拓扑排序结果）
        strategy: 分解策略
        complexity_score: 复杂度评分
    """
    original_query: str
    query_type: QueryType
    sub_queries: List[SubQuery]
    execution_order: List[str]
    strategy: DecompositionStrategy
    complexity_score: float = 0.0

    def get_execution_plan(self) -> List[List[str]]:
        """
        获取执行计划（并行层级）

        Returns:
            执行层级列表，每层可以并行执行
        """
        # 构建依赖图
        in_degree = defaultdict(int)
        graph = defaultdict(list)

        for sq in self.sub_queries:
            if sq.id not in in_degree:
                in_degree[sq.id] = 0
            for dep in sq.dependencies:
                graph[dep].append(sq.id)
                in_degree[sq.id] += 1

        # 拓扑排序分层
        levels = []
        current_level = [sq.id for sq in self.sub_queries if in_degree[sq.id] == 0]

        while current_level:
            levels.append(current_level)
            next_level = []
            for node in current_level:
                for neighbor in graph[node]:
                    in_degree[neighbor] -= 1
                    if in_degree[neighbor] == 0:
                        next_level.append(neighbor)
            current_level = next_level

        return levels


class QueryClassifier(nn.Module):
    """
    查询分类器
    Query Classifier

    将输入查询分类到不同的查询类型

    Args:
        hidden_size: 隐藏层维度
        num_classes: 类别数量

    Example:
        >>> classifier = QueryClassifier(4096, len(QueryType))
        >>> query_type = classifier.classify(query_embedding)
    """

    def __init__(
            self,
            hidden_size: int = 4096,
            num_classes: int = 6
    ):
        super().__init__()

        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size // 2, num_classes)
        )

        self.query_types = list(QueryType)

        logger.info(f"QueryClassifier initialized with {num_classes} classes")

    def forward(self, query_embedding: Tensor) -> Tensor:
        """
        前向传播

        Args:
            query_embedding: 查询嵌入 (batch, hidden_size)

        Returns:
            分类logits (batch, num_classes)
        """
        return self.classifier(query_embedding)

    def classify(self, query_embedding: Tensor) -> QueryType:
        """
        分类查询

        Args:
            query_embedding: 查询嵌入

        Returns:
            查询类型
        """
        with torch.no_grad():
            logits = self.forward(query_embedding)
            pred_idx = logits.argmax(dim=-1).item()
            return self.query_types[pred_idx]


class ComplexityEstimator(nn.Module):
    """
    复杂度评估器
    Complexity Estimator

    评估查询的复杂程度，决定是否需要分解

    Args:
        hidden_size: 隐藏层维度
    """

    def __init__(self, hidden_size: int = 4096):
        super().__init__()

        self.estimator = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, 1),
            nn.Sigmoid()
        )

        logger.info("ComplexityEstimator initialized")

    def forward(self, query_embedding: Tensor) -> Tensor:
        """
        评估复杂度

        Args:
            query_embedding: 查询嵌入

        Returns:
            复杂度分数 (batch, 1)，范围[0, 1]
        """
        return self.estimator(query_embedding)

    def estimate(self, query_embedding: Tensor) -> float:
        """
        获取复杂度分数

        Args:
            query_embedding: 查询嵌入

        Returns:
            复杂度分数
        """
        with torch.no_grad():
            return self.forward(query_embedding).item()


class DecompositionModel(nn.Module):
    """
    分解模型
    Decomposition Model

    基于Transformer的查询分解生成模型

    Args:
        hidden_size: 隐藏层维度
        num_heads: 注意力头数
        num_layers: 层数
        max_sub_queries: 最大子查询数量
    """

    def __init__(
            self,
            hidden_size: int = 4096,
            num_heads: int = 8,
            num_layers: int = 2,
            max_sub_queries: int = 5
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.max_sub_queries = max_sub_queries

        # Transformer解码器层
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_size,
            nhead=num_heads,
            dim_feedforward=hidden_size * 4,
            dropout=0.1,
            batch_first=True
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers)

        # 输出投影
        self.output_proj = nn.Linear(hidden_size, hidden_size)

        logger.info(f"DecompositionModel: layers={num_layers}, max_sub={max_sub_queries}")

    def forward(
            self,
            query_embedding: Tensor,
            memory: Optional[Tensor] = None
    ) -> Tensor:
        """
        前向传播

        Args:
            query_embedding: 查询嵌入 (batch, seq_len, hidden)
            memory: 记忆张量（之前生成的子查询）

        Returns:
            分解特征 (batch, seq_len, hidden)
        """
        if memory is None:
            # 初始分解，使用查询本身作为memory
            memory = query_embedding

        # 解码生成分解
        output = self.decoder(query_embedding, memory)
        return self.output_proj(output)


class QueryDecomposer:
    """
    查询分解器
    Query Decomposer

    主要的查询分解协调类

    Args:
        model: 语言模型（用于生成式分解）
        tokenizer: 分词器
        hidden_size: 隐藏层维度
        complexity_threshold: 复杂度阈值，超过则分解

    Example:
        >>> decomposer = QueryDecomposer(model, tokenizer)
        >>> result = decomposer.decompose("比较A公司和B公司在2023年的营收")
        >>> for sq in result.sub_queries:
        ...     print(sq.text)
    """

    def __init__(
            self,
            model: nn.Module = None,
            tokenizer: Any = None,
            hidden_size: int = 4096,
            complexity_threshold: float = 0.5
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.hidden_size = hidden_size
        self.complexity_threshold = complexity_threshold

        # 初始化子模块
        self.classifier = QueryClassifier(hidden_size)
        self.complexity_estimator = ComplexityEstimator(hidden_size)
        self.decomposition_model = DecompositionModel(hidden_size)

        # 分解模板
        self.templates = self._init_templates()

        logger.info("QueryDecomposer initialized")

    def _init_templates(self) -> Dict[str, str]:
        """
        初始化分解提示模板

        Returns:
            模板字典
        """
        return {
            "decompose_prompt": """请将以下复杂问题分解为多个简单的子问题。
每个子问题应该是可以独立回答的简单问题。
请按照以下格式输出：
1. [子问题1]
2. [子问题2]
...

原始问题：{query}

分解结果：""",

            "dependency_prompt": """分析以下子问题之间的依赖关系：
{sub_queries}

请指出哪些子问题需要其他子问题的答案才能回答。
格式：子问题ID -> 依赖的子问题ID列表""",

            "multi_hop_template": [
                "首先，{step1}",
                "然后，{step2}",
                "最后，{step3}"
            ],

            "comparison_template": [
                "查询{entity1}的{attribute}",
                "查询{entity2}的{attribute}",
                "比较两者的{attribute}"
            ],

            "aggregation_template": [
                "获取所有{entity}的{attribute}",
                "计算{operation}"
            ]
        }

    def decompose(
            self,
            query: str,
            strategy: DecompositionStrategy = DecompositionStrategy.HYBRID
    ) -> QueryDecomposition:
        """
        分解查询

        Args:
            query: 原始查询文本
            strategy: 分解策略

        Returns:
            QueryDecomposition: 分解结果
        """
        logger.info(f"Decomposing query: {query}")

        # 编码查询
        query_embedding = self._encode_query(query)

        # 分类查询类型
        query_type = self.classifier.classify(query_embedding)
        logger.info(f"Query type: {query_type.value}")

        # 评估复杂度
        complexity = self.complexity_estimator.estimate(query_embedding)
        logger.info(f"Complexity score: {complexity:.3f}")

        # 判断是否需要分解
        if complexity < self.complexity_threshold or query_type == QueryType.SIMPLE:
            # 简单查询，不需要分解
            return QueryDecomposition(
                original_query=query,
                query_type=query_type,
                sub_queries=[SubQuery(
                    id="0",
                    text=query,
                    query_type=query_type
                )],
                execution_order=["0"],
                strategy=strategy,
                complexity_score=complexity
            )

        # 根据查询类型选择分解方法
        if query_type == QueryType.MULTI_HOP:
            sub_queries = self._decompose_multi_hop(query)
        elif query_type == QueryType.COMPARATIVE:
            sub_queries = self._decompose_comparison(query)
        elif query_type == QueryType.AGGREGATION:
            sub_queries = self._decompose_aggregation(query)
        else:
            sub_queries = self._decompose_general(query)

        # 确定执行顺序
        execution_order = self._topological_sort(sub_queries)

        return QueryDecomposition(
            original_query=query,
            query_type=query_type,
            sub_queries=sub_queries,
            execution_order=execution_order,
            strategy=strategy,
            complexity_score=complexity
        )

    def _encode_query(self, query: str) -> Tensor:
        """编码查询"""
        if self.model is None:
            # 返回随机嵌入用于测试
            return torch.randn(1, self.hidden_size)

        inputs = self.tokenizer(query, return_tensors="pt")
        with torch.no_grad():
            outputs = self.model(**inputs, output_hidden_states=True)
        return outputs.hidden_states[-1].mean(dim=1)

    def _decompose_multi_hop(self, query: str) -> List[SubQuery]:
        """
        分解多跳查询

        Args:
            query: 原始查询

        Returns:
            子查询列表
        """
        # 使用规则+模板分解
        sub_queries = []

        # 检测查询中的实体和关系
        entities = self._extract_entities(query)

        # 生成子查询
        for i, entity in enumerate(entities):
            sq = SubQuery(
                id=str(i),
                text=f"查询{entity}的相关信息",
                query_type=QueryType.SIMPLE,
                dependencies=[]
            )
            sub_queries.append(sq)

        # 添加最终整合查询
        final_sq = SubQuery(
            id=str(len(entities)),
            text=f"根据上述信息回答原始问题",
            query_type=QueryType.SIMPLE,
            dependencies=[str(i) for i in range(len(entities))]
        )
        sub_queries.append(final_sq)

        return sub_queries

    def _decompose_comparison(self, query: str) -> List[SubQuery]:
        """
        分解比较查询

        Args:
            query: 原始查询

        Returns:
            子查询列表
        """
        sub_queries = []

        # 提取比较对象和属性
        entities = self._extract_entities(query)
        attributes = self._extract_attributes(query)

        # 为每个实体创建属性查询
        entity_ids = []
        for i, entity in enumerate(entities):
            entity_id = str(i)
            entity_ids.append(entity_id)

            for attr in attributes:
                sq = SubQuery(
                    id=f"{i}_{attr}",
                    text=f"查询{entity}的{attr}",
                    query_type=QueryType.SIMPLE,
                    dependencies=[]
                )
                sub_queries.append(sq)

        # 添加比较查询
        compare_sq = SubQuery(
            id="compare",
            text=f"比较{'和'.join(entities)}的{', '.join(attributes)}",
            query_type=QueryType.COMPARATIVE,
            dependencies=[sq.id for sq in sub_queries if sq.id != "compare"]
        )
        sub_queries.append(compare_sq)

        return sub_queries

    def _decompose_aggregation(self, query: str) -> List[SubQuery]:
        """
        分解聚合查询

        Args:
            query: 原始查询

        Returns:
            子查询列表
        """
        sub_queries = []

        # 提取聚合目标
        entities = self._extract_entities(query)
        operations = self._extract_operations(query)

        # 数据收集查询
        collect_sq = SubQuery(
            id="collect",
            text=f"收集所有{entities[0] if entities else '相关'}的数据",
            query_type=QueryType.SIMPLE,
            dependencies=[]
        )
        sub_queries.append(collect_sq)

        # 聚合操作查询
        for i, op in enumerate(operations):
            agg_sq = SubQuery(
                id=f"agg_{i}",
                text=f"计算{op}",
                query_type=QueryType.AGGREGATION,
                dependencies=["collect"]
            )
            sub_queries.append(agg_sq)

        return sub_queries

    def _decompose_general(self, query: str) -> List[SubQuery]:
        """
        通用分解方法

        Args:
            query: 原始查询

        Returns:
            子查询列表
        """
        # 使用LLM进行分解
        if self.model is not None:
            return self._llm_decompose(query)

        # 后备：基于关键词的简单分解
        return self._keyword_decompose(query)

    def _llm_decompose(self, query: str) -> List[SubQuery]:
        """使用LLM进行分解"""
        prompt = self.templates["decompose_prompt"].format(query=query)

        # 这里应该调用模型生成
        # 简化实现：返回单子查询
        return [SubQuery(
            id="0",
            text=query,
            query_type=QueryType.SIMPLE
        )]

    def _keyword_decompose(self, query: str) -> List[SubQuery]:
        """基于关键词的分解"""
        # 按句号、分号等分割
        parts = re.split(r'[，。；]', query)
        parts = [p.strip() for p in parts if p.strip()]

        if len(parts) <= 1:
            return [SubQuery(id="0", text=query, query_type=QueryType.SIMPLE)]

        sub_queries = []
        for i, part in enumerate(parts):
            sq = SubQuery(
                id=str(i),
                text=part,
                query_type=QueryType.SIMPLE,
                dependencies=[str(j) for j in range(i)] if i > 0 else []
            )
            sub_queries.append(sq)

        return sub_queries

    def _extract_entities(self, query: str) -> List[str]:
        """提取查询中的实体"""
        # 简化实现：使用正则匹配
        # 实际应用中应使用NER模型
        patterns = [
            r'([^的和与比较]+)(?:和|与|和|vs)',
            r'比较([^的和与]+)和([^的和与]+)'
        ]

        entities = []
        for pattern in patterns:
            matches = re.findall(pattern, query)
            entities.extend([m for m in matches if isinstance(m, str)])

        return entities[:5]  # 限制数量

    def _extract_attributes(self, query: str) -> List[str]:
        """提取查询中的属性"""
        # 简化实现
        attr_keywords = ['营收', '利润', '收入', '支出', '增长率', '市值', '规模']

        attributes = []
        for attr in attr_keywords:
            if attr in query:
                attributes.append(attr)

        return attributes or ['相关信息']

    def _extract_operations(self, query: str) -> List[str]:
        """提取聚合操作"""
        op_keywords = ['总和', '平均', '最大', '最小', '计数', '排序']

        operations = []
        for op in op_keywords:
            if op in query:
                operations.append(op)

        return operations or ['统计值']

    def _topological_sort(self, sub_queries: List[SubQuery]) -> List[str]:
        """
        拓扑排序

        Args:
            sub_queries: 子查询列表

        Returns:
            排序后的子查询ID列表
        """
        # 构建依赖图
        in_degree = {sq.id: len(sq.dependencies) for sq in sub_queries}
        graph = defaultdict(list)

        for sq in sub_queries:
            for dep in sq.dependencies:
                graph[dep].append(sq.id)

        # Kahn算法
        queue = [sq.id for sq in sub_queries if in_degree[sq.id] == 0]
        result = []

        while queue:
            node = queue.pop(0)
            result.append(node)

            for neighbor in graph[node]:
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)

        return result

    def get_execution_plan(
            self,
            decomposition: QueryDecomposition
    ) -> List[List[SubQuery]]:
        """
        获取可并行执行的层级计划

        Args:
            decomposition: 分解结果

        Returns:
            执行层级列表，每层可并行
        """
        levels = decomposition.get_execution_plan()
        id_to_sq = {sq.id: sq for sq in decomposition.sub_queries}

        return [[id_to_sq[sqid] for sqid in level] for level in levels]
