"""
多跳推理模块
Multi-Hop Reasoning Module

实现多步骤推理链，支持复杂的链式推理
包含推理路径规划和推理状态管理
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Dict, List, Optional, Tuple, Any, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict
import logging

logger = logging.getLogger(__name__)


class ReasoningState(Enum):
    """
    推理状态枚举
    Reasoning State Enum
    """
    PENDING = "pending"  # 待执行
    IN_PROGRESS = "in_progress"  # 执行中
    COMPLETED = "completed"  # 已完成
    FAILED = "failed"  # 失败


class ReasoningType(Enum):
    """
    推理类型枚举
    Reasoning Type Enum
    """
    DEDUCTIVE = "deductive"  # 演绎推理
    INDUCTIVE = "inductive"  # 归纳推理
    ABDUCTIVE = "abductive"  # 溯因推理
    ANALOGICAL = "analogical"  # 类比推理


@dataclass
class ReasoningStep:
    """
    推理步骤数据类
    Reasoning Step Dataclass

    Attributes:
        step_id: 步骤ID
        description: 步骤描述
        input_facts: 输入事实/知识
        output_fact: 输出结论
        reasoning_type: 推理类型
        confidence: 置信度
        state: 执行状态
        dependencies: 依赖的步骤ID
        evidence: 支持证据
    """
    step_id: str
    description: str
    input_facts: List[str] = field(default_factory=list)
    output_fact: str = ""
    reasoning_type: ReasoningType = ReasoningType.DEDUCTIVE
    confidence: float = 0.0
    state: ReasoningState = ReasoningState.PENDING
    dependencies: List[str] = field(default_factory=list)
    evidence: List[str] = field(default_factory=list)

    def __hash__(self):
        return hash(self.step_id)


@dataclass
class ReasoningChain:
    """
    推理链数据类
    Reasoning Chain Dataclass

    Attributes:
        chain_id: 链ID
        query: 原始查询
        steps: 推理步骤列表
        final_answer: 最终答案
        overall_confidence: 总体置信度
        reasoning_path: 推理路径（步骤ID序列）
    """
    chain_id: str
    query: str
    steps: List[ReasoningStep]
    final_answer: str = ""
    overall_confidence: float = 0.0
    reasoning_path: List[str] = field(default_factory=list)

    def get_step(self, step_id: str) -> Optional[ReasoningStep]:
        """根据ID获取步骤"""
        for step in self.steps:
            if step.step_id == step_id:
                return step
        return None

    def get_ordered_steps(self) -> List[ReasoningStep]:
        """按推理路径顺序获取步骤"""
        step_map = {s.step_id: s for s in self.steps}
        return [step_map[sid] for sid in self.reasoning_path if sid in step_map]


class KnowledgeStore:
    """
    知识存储
    Knowledge Store

    存储推理过程中的中间知识和事实

    Args:
        max_facts: 最大事实数量
    """

    def __init__(self, max_facts: int = 10000):
        self.max_facts = max_facts
        self.facts: Dict[str, Dict] = {}  # fact_id -> fact_info
        self.fact_embeddings: Optional[Tensor] = None
        self.fact_index: Dict[str, List[str]] = defaultdict(list)  # 关键词索引

        logger.info(f"KnowledgeStore initialized, max_facts={max_facts}")

    def add_fact(
            self,
            fact_id: str,
            content: str,
            embedding: Optional[Tensor] = None,
            metadata: Optional[Dict] = None
    ) -> None:
        """
        添加事实

        Args:
            fact_id: 事实ID
            content: 事实内容
            embedding: 事实嵌入向量
            metadata: 元数据
        """
        if len(self.facts) >= self.max_facts:
            # 简单的FIFO策略
            oldest_key = next(iter(self.facts))
            del self.facts[oldest_key]

        self.facts[fact_id] = {
            "content": content,
            "embedding": embedding,
            "metadata": metadata or {}
        }

        # 更新索引
        keywords = self._extract_keywords(content)
        for kw in keywords:
            self.fact_index[kw].append(fact_id)

    def get_fact(self, fact_id: str) -> Optional[Dict]:
        """获取事实"""
        return self.facts.get(fact_id)

    def search_facts(
            self,
            query: str,
            top_k: int = 5
    ) -> List[Tuple[str, float, Dict]]:
        """
        搜索相关事实

        Args:
            query: 查询文本
            top_k: 返回数量

        Returns:
            [(fact_id, score, fact_info), ...]
        """
        # 基于关键词的简单搜索
        keywords = self._extract_keywords(query)
        candidate_ids = set()

        for kw in keywords:
            candidate_ids.update(self.fact_index.get(kw, []))

        # 计算相关性分数（简化版）
        results = []
        for fid in candidate_ids:
            if fid in self.facts:
                # 计算关键词重叠分数
                fact_keywords = self._extract_keywords(self.facts[fid]["content"])
                overlap = len(set(keywords) & set(fact_keywords))
                score = overlap / max(len(keywords), 1)
                results.append((fid, score, self.facts[fid]))

        # 排序并返回top_k
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]

    def _extract_keywords(self, text: str) -> List[str]:
        """提取关键词"""
        # 简化实现：按空格和标点分割
        import re
        words = re.findall(r'\w+', text.lower())
        return [w for w in words if len(w) > 1]

    def clear(self) -> None:
        """清空知识库"""
        self.facts.clear()
        self.fact_index.clear()


class ReasoningPathPlanner:
    """
    推理路径规划器
    Reasoning Path Planner

    规划从问题到答案的最优推理路径

    Args:
        max_hops: 最大推理跳数
        beam_width: 束搜索宽度
    """

    def __init__(
            self,
            max_hops: int = 5,
            beam_width: int = 3
    ):
        self.max_hops = max_hops
        self.beam_width = beam_width

        logger.info(f"ReasoningPathPlanner: max_hops={max_hops}, beam_width={beam_width}")

    def plan(
            self,
            query: str,
            knowledge_store: KnowledgeStore,
            initial_facts: Optional[List[str]] = None
    ) -> List[ReasoningStep]:
        """
        规划推理路径

        Args:
            query: 查询文本
            knowledge_store: 知识存储
            initial_facts: 初始已知事实

        Returns:
            推理步骤列表
        """
        steps = []

        # 搜索相关事实
        relevant_facts = knowledge_store.search_facts(query, top_k=self.beam_width)

        # 构建推理步骤
        hop = 0
        current_query = query
        visited_facts = set()

        while hop < self.max_hops and relevant_facts:
            fact_id, score, fact_info = relevant_facts[0]

            if fact_id in visited_facts:
                relevant_facts.pop(0)
                continue

            visited_facts.add(fact_id)

            # 创建推理步骤
            step = ReasoningStep(
                step_id=f"hop_{hop}",
                description=f"基于事实进行推理 (hop {hop})",
                input_facts=[fact_info["content"]],
                reasoning_type=ReasoningType.DEDUCTIVE,
                confidence=score
            )
            steps.append(step)

            # 更新查询（提取新问题）
            # 这里简化处理，实际应使用模型生成
            current_query = f"基于上述事实，回答: {query}"

            # 搜索下一跳的事实
            relevant_facts = knowledge_store.search_facts(current_query, top_k=self.beam_width)
            hop += 1

        return steps


class InferenceEngine(nn.Module):
    """
    推理引擎
    Inference Engine

    执行实际的推理计算

    Args:
        hidden_size: 隐藏层维度
        model: 语言模型
        tokenizer: 分词器
    """

    def __init__(
            self,
            hidden_size: int = 4096,
            model: nn.Module = None,
            tokenizer: Any = None
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.model = model
        self.tokenizer = tokenizer

        # 推理层
        self.reasoning_layer = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, hidden_size)
        )

        # 置信度估计器
        self.confidence_estimator = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, 1),
            nn.Sigmoid()
        )

        logger.info("InferenceEngine initialized")

    def infer(
            self,
            premise: str,
            hypothesis: str = None,
            evidence: List[str] = None
    ) -> Tuple[str, float]:
        """
        执行推理

        Args:
            premise: 前提/已知信息
            hypothesis: 假设（可选）
            evidence: 证据列表

        Returns:
            (结论, 置信度)
        """
        # 构建推理提示
        if hypothesis:
            prompt = f"已知：{premise}\n问题：{hypothesis}是否成立？\n推理："
        else:
            prompt = f"已知：{premise}\n结论："

        if evidence:
            prompt = f"证据：\n" + "\n".join(f"- {e}" for e in evidence) + "\n" + prompt

        # 使用模型生成
        if self.model is not None:
            conclusion = self._generate(prompt)
        else:
            conclusion = f"基于前提的推理结论"

        # 估计置信度
        confidence = self._estimate_confidence(premise, conclusion)

        return conclusion, confidence

    def _generate(self, prompt: str) -> str:
        """使用模型生成文本"""
        inputs = self.tokenizer(prompt, return_tensors="pt")

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=256,
                temperature=0.7,
                do_sample=True
            )

        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

    def _estimate_confidence(self, premise: str, conclusion: str) -> float:
        """估计推理置信度"""
        # 简化实现：基于规则的置信度估计
        # 实际应使用神经网络

        # 长度相关
        length_factor = min(len(conclusion) / 100, 1.0)

        # 关键词相关
        certainty_words = ['确定', '必然', '一定', '证明']
        uncertainty_words = ['可能', '也许', '大概', '或许']

        certainty_count = sum(1 for w in certainty_words if w in conclusion)
        uncertainty_count = sum(1 for w in uncertainty_words if w in conclusion)

        keyword_factor = 0.5 + 0.1 * (certainty_count - uncertainty_count)
        keyword_factor = max(0.1, min(1.0, keyword_factor))

        return (length_factor + keyword_factor) / 2

    def forward(
            self,
            premise_embedding: Tensor,
            evidence_embedding: Tensor
    ) -> Tuple[Tensor, Tensor]:
        """
        前向传播

        Args:
            premise_embedding: 前提嵌入
            evidence_embedding: 证据嵌入

        Returns:
            (推理结果嵌入, 置信度)
        """
        combined = torch.cat([premise_embedding, evidence_embedding], dim=-1)
        output = self.reasoning_layer(combined)
        confidence = self.confidence_estimator(output)

        return output, confidence


class MultiHopReasoner:
    """
    多跳推理器
    Multi-Hop Reasoner

    协调多跳推理过程的主要类

    Args:
        model: 语言模型
        tokenizer: 分词器
        hidden_size: 隐藏层维度
        max_hops: 最大推理跳数

    Example:
        >>> reasoner = MultiHopReasoner(model, tokenizer)
        >>> chain = reasoner.reason("问题", knowledge_store)
        >>> print(chain.final_answer)
    """

    def __init__(
            self,
            model: nn.Module = None,
            tokenizer: Any = None,
            hidden_size: int = 4096,
            max_hops: int = 5
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.hidden_size = hidden_size
        self.max_hops = max_hops

        # 初始化子组件
        self.knowledge_store = KnowledgeStore()
        self.path_planner = ReasoningPathPlanner(max_hops=max_hops)
        self.inference_engine = InferenceEngine(hidden_size, model, tokenizer)

        logger.info(f"MultiHopReasoner initialized, max_hops={max_hops}")

    def reason(
            self,
            query: str,
            context: Optional[List[str]] = None,
            knowledge: Optional[Dict[str, str]] = None
    ) -> ReasoningChain:
        """
        执行多跳推理

        Args:
            query: 查询问题
            context: 上下文信息
            knowledge: 额外知识 {fact_id: content}

        Returns:
            ReasoningChain: 推理链结果
        """
        logger.info(f"Starting reasoning for: {query[:50]}...")

        # 添加知识到知识库
        if knowledge:
            for fid, content in knowledge.items():
                self.knowledge_store.add_fact(fid, content)

        # 规划推理路径
        steps = self.path_planner.plan(
            query,
            self.knowledge_store,
            initial_facts=context
        )

        # 执行推理步骤
        chain_id = f"chain_{hash(query) % 10000}"
        reasoning_path = []
        accumulated_knowledge = context or []

        for step in steps:
            step.state = ReasoningState.IN_PROGRESS

            # 执行推理
            conclusion, confidence = self.inference_engine.infer(
                premise="\n".join(accumulated_knowledge + step.input_facts),
                evidence=step.evidence
            )

            step.output_fact = conclusion
            step.confidence = confidence
            step.state = ReasoningState.COMPLETED

            reasoning_path.append(step.step_id)
            accumulated_knowledge.append(conclusion)

            # 将结论添加到知识库
            self.knowledge_store.add_fact(
                f"{chain_id}_{step.step_id}",
                conclusion
            )

        # 生成最终答案
        final_answer = self._synthesize_answer(query, accumulated_knowledge)
        overall_confidence = self._compute_overall_confidence(steps)

        chain = ReasoningChain(
            chain_id=chain_id,
            query=query,
            steps=steps,
            final_answer=final_answer,
            overall_confidence=overall_confidence,
            reasoning_path=reasoning_path
        )

        logger.info(f"Reasoning completed with {len(steps)} steps, confidence={overall_confidence:.2f}")

        return chain

    def reason_with_decomposition(
            self,
            query: str,
            decomposition: Any,  # QueryDecomposition from query_decomposition module
            context: Optional[List[str]] = None
    ) -> ReasoningChain:
        """
        基于查询分解结果进行推理

        Args:
            query: 原始查询
            decomposition: 查询分解结果
            context: 上下文

        Returns:
            推理链
        """
        steps = []
        step_results = {}

        # 获取执行计划
        execution_plan = decomposition.get_execution_plan()

        for level, level_steps in enumerate(execution_plan):
            for sub_query in level_steps:
                # 获取依赖步骤的结果
                dep_results = [step_results[d] for d in sub_query.dependencies if d in step_results]

                # 执行推理
                premise = "\n".join(dep_results + (context or []))
                conclusion, confidence = self.inference_engine.infer(
                    premise=premise,
                    hypothesis=sub_query.text
                )

                step = ReasoningStep(
                    step_id=sub_query.id,
                    description=sub_query.text,
                    input_facts=dep_results,
                    output_fact=conclusion,
                    reasoning_type=ReasoningType.DEDUCTIVE,
                    confidence=confidence,
                    state=ReasoningState.COMPLETED,
                    dependencies=sub_query.dependencies
                )

                steps.append(step)
                step_results[sub_query.id] = conclusion

        # 生成最终答案
        final_answer = self._synthesize_answer(query, list(step_results.values()))
        overall_confidence = self._compute_overall_confidence(steps)

        return ReasoningChain(
            chain_id=f"chain_decomp_{hash(query) % 10000}",
            query=query,
            steps=steps,
            final_answer=final_answer,
            overall_confidence=overall_confidence,
            reasoning_path=[s.step_id for s in steps]
        )

    def _synthesize_answer(
            self,
            query: str,
            knowledge: List[str]
    ) -> str:
        """
        综合推理结果生成最终答案

        Args:
            query: 原始查询
            knowledge: 推理过程中积累的知识

        Returns:
            最终答案
        """
        if self.model is not None:
            prompt = f"""问题：{query}

推理过程：
{chr(10).join(f"- {k}" for k in knowledge)}

请根据推理过程，给出最终答案："""
            return self.inference_engine._generate(prompt)

        # 简化实现
        return knowledge[-1] if knowledge else "无法得出结论"

    def _compute_overall_confidence(
            self,
            steps: List[ReasoningStep]
    ) -> float:
        """
        计算总体置信度

        Args:
            steps: 推理步骤列表

        Returns:
            总体置信度
        """
        if not steps:
            return 0.0

        # 使用几何平均（对链式推理更合理）
        confidences = [s.confidence for s in steps if s.confidence > 0]

        if not confidences:
            return 0.0

        import math
        return math.exp(sum(math.log(c) for c in confidences) / len(confidences))

    def explain(self, chain: ReasoningChain) -> str:
        """
        生成推理过程的可解释性说明

        Args:
            chain: 推理链

        Returns:
            解释文本
        """
        lines = [
            f"问题: {chain.query}",
            "",
            "推理过程:"
        ]

        for i, step in enumerate(chain.get_ordered_steps(), 1):
            lines.append(f"  步骤{i}: {step.description}")
            lines.append(f"    输入: {step.input_facts}")
            lines.append(f"    结论: {step.output_fact}")
            lines.append(f"    置信度: {step.confidence:.2%}")
            lines.append("")

        lines.extend([
            f"最终答案: {chain.final_answer}",
            f"总体置信度: {chain.overall_confidence:.2%}"
        ])

        return "\n".join(lines)
