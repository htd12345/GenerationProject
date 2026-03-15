"""
答案可解释性模块
Answer Explainability Module

提供推理过程的可视化和解释能力
生成人类可理解的推理路径说明
"""

import torch
from torch import Tensor
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
from enum import Enum
import logging
import json

logger = logging.getLogger(__name__)


class ExplanationLevel(Enum):
    """
    解释详细程度枚举
    Explanation Level Enum
    """
    BRIEF = "brief"  # 简要解释
    STANDARD = "standard"  # 标准解释
    DETAILED = "detailed"  # 详细解释
    TECHNICAL = "technical"  # 技术细节


class ExplanationFormat(Enum):
    """
    解释格式枚举
    Explanation Format Enum
    """
    TEXT = "text"  # 纯文本
    MARKDOWN = "markdown"  # Markdown格式
    HTML = "html"  # HTML格式
    JSON = "json"  # JSON格式
    GRAPH = "graph"  # 图结构


@dataclass
class Evidence:
    """
    证据数据类
    Evidence Dataclass

    Attributes:
        id: 证据ID
        content: 证据内容
        source: 证据来源
        relevance_score: 相关性分数
        citation: 引用信息
    """
    id: str
    content: str
    source: str = ""
    relevance_score: float = 0.0
    citation: str = ""


@dataclass
class ReasoningNode:
    """
    推理节点数据类
    Reasoning Node Dataclass

    用于构建推理图

    Attributes:
        id: 节点ID
        content: 节点内容
        node_type: 节点类型 ('fact', 'inference', 'conclusion')
        confidence: 置信度
        evidence: 支持证据
        parent_ids: 父节点ID列表
        children_ids: 子节点ID列表
    """
    id: str
    content: str
    node_type: str = "fact"
    confidence: float = 0.0
    evidence: List[Evidence] = field(default_factory=list)
    parent_ids: List[str] = field(default_factory=list)
    children_ids: List[str] = field(default_factory=list)


@dataclass
class ExplanationResult:
    """
    解释结果数据类
    Explanation Result Dataclass

    Attributes:
        query: 原始查询
        answer: 最终答案
        reasoning_nodes: 推理节点列表
        reasoning_path: 推理路径
        confidence: 总体置信度
        summary: 摘要说明
        evidence_list: 证据列表
        metadata: 额外元数据
    """
    query: str
    answer: str
    reasoning_nodes: List[ReasoningNode]
    reasoning_path: List[str]
    confidence: float
    summary: str = ""
    evidence_list: List[Evidence] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_text(self, level: ExplanationLevel = ExplanationLevel.STANDARD) -> str:
        """转换为文本格式"""
        return ExplanationFormatter.format_text(self, level)

    def to_markdown(self, level: ExplanationLevel = ExplanationLevel.STANDARD) -> str:
        """转换为Markdown格式"""
        return ExplanationFormatter.format_markdown(self, level)

    def to_json(self) -> str:
        """转换为JSON格式"""
        return ExplanationFormatter.format_json(self)

    def to_html(self, level: ExplanationLevel = ExplanationLevel.STANDARD) -> str:
        """转换为HTML格式"""
        return ExplanationFormatter.format_html(self, level)


class ExplanationFormatter:
    """
    解释格式化器
    Explanation Formatter

    提供多种格式的解释输出
    """

    @staticmethod
    def format_text(
            result: ExplanationResult,
            level: ExplanationLevel = ExplanationLevel.STANDARD
    ) -> str:
        """
        格式化为纯文本

        Args:
            result: 解释结果
            level: 详细程度

        Returns:
            文本格式解释
        """
        lines = []

        # 标题
        lines.append("=" * 50)
        lines.append("【问答解释】")
        lines.append("=" * 50)
        lines.append("")

        # 问题和答案
        lines.append(f"问题: {result.query}")
        lines.append(f"答案: {result.answer}")
        lines.append(f"置信度: {result.confidence:.1%}")
        lines.append("")

        # 摘要
        if result.summary:
            lines.append("【摘要】")
            lines.append(result.summary)
            lines.append("")

        # 推理过程
        if level != ExplanationLevel.BRIEF:
            lines.append("【推理过程】")
            for i, node_id in enumerate(result.reasoning_path, 1):
                node = next((n for n in result.reasoning_nodes if n.id == node_id), None)
                if node:
                    lines.append(f"  {i}. {node.content}")
                    if level in [ExplanationLevel.DETAILED, ExplanationLevel.TECHNICAL]:
                        lines.append(f"     置信度: {node.confidence:.1%}")
                        if node.evidence:
                            lines.append(f"     证据: {len(node.evidence)}条")
                    lines.append("")

        # 证据
        if level == ExplanationLevel.DETAILED or level == ExplanationLevel.TECHNICAL:
            if result.evidence_list:
                lines.append("【支持证据】")
                for ev in result.evidence_list:
                    lines.append(f"  - {ev.content}")
                    if ev.source:
                        lines.append(f"    来源: {ev.source}")
                    lines.append("")

        # 技术细节
        if level == ExplanationLevel.TECHNICAL:
            lines.append("【技术细节】")
            lines.append(f"  推理节点数: {len(result.reasoning_nodes)}")
            lines.append(f"  证据数量: {len(result.evidence_list)}")
            if result.metadata:
                lines.append("  元数据:")
                for k, v in result.metadata.items():
                    lines.append(f"    {k}: {v}")

        lines.append("=" * 50)
        return "\n".join(lines)

    @staticmethod
    def format_markdown(
            result: ExplanationResult,
            level: ExplanationLevel = ExplanationLevel.STANDARD
    ) -> str:
        """
        格式化为Markdown

        Args:
            result: 解释结果
            level: 详细程度

        Returns:
            Markdown格式解释
        """
        lines = []

        # 标题
        lines.append("# 📋 问答解释")
        lines.append("")

        # 问题和答案
        lines.append("## 🎯 问答")
        lines.append(f"- **问题**: {result.query}")
        lines.append(f"- **答案**: {result.answer}")
        lines.append(f"- **置信度**: `{result.confidence:.1%}`")
        lines.append("")

        # 摘要
        if result.summary:
            lines.append("## 📝 摘要")
            lines.append(result.summary)
            lines.append("")

        # 推理过程
        if level != ExplanationLevel.BRIEF:
            lines.append("## 🔍 推理过程")
            lines.append("")
            lines.append("```mermaid")
            lines.append("graph TD")

            for i, node_id in enumerate(result.reasoning_path):
                node = next((n for n in result.reasoning_nodes if n.id == node_id), None)
                if node:
                    label = node.content[:30] + "..." if len(node.content) > 30 else node.content
                    lines.append(f"    {node_id}[\"{label}\"]")
                    if i > 0:
                        prev_id = result.reasoning_path[i - 1]
                        lines.append(f"    {prev_id} --> {node_id}")

            lines.append("```")
            lines.append("")

            # 详细步骤
            lines.append("### 步骤详情")
            for i, node_id in enumerate(result.reasoning_path, 1):
                node = next((n for n in result.reasoning_nodes if n.id == node_id), None)
                if node:
                    lines.append(f"{i}. **{node.content}**")
                    if level in [ExplanationLevel.DETAILED, ExplanationLevel.TECHNICAL]:
                        lines.append(f"   - 置信度: {node.confidence:.1%}")
                    lines.append("")

        # 证据
        if level in [ExplanationLevel.DETAILED, ExplanationLevel.TECHNICAL]:
            if result.evidence_list:
                lines.append("## 📚 支持证据")
                lines.append("")
                for i, ev in enumerate(result.evidence_list, 1):
                    lines.append(f"{i}. {ev.content}")
                    if ev.source:
                        lines.append(f"   - *来源: {ev.source}*")
                    lines.append("")

        return "\n".join(lines)

    @staticmethod
    def format_json(result: ExplanationResult) -> str:
        """
        格式化为JSON

        Args:
            result: 解释结果

        Returns:
            JSON格式解释
        """
        data = {
            "query": result.query,
            "answer": result.answer,
            "confidence": result.confidence,
            "summary": result.summary,
            "reasoning_path": result.reasoning_path,
            "reasoning_nodes": [
                {
                    "id": node.id,
                    "content": node.content,
                    "type": node.node_type,
                    "confidence": node.confidence,
                    "parent_ids": node.parent_ids,
                    "children_ids": node.children_ids,
                    "evidence_count": len(node.evidence)
                }
                for node in result.reasoning_nodes
            ],
            "evidence": [
                {
                    "id": ev.id,
                    "content": ev.content,
                    "source": ev.source,
                    "relevance_score": ev.relevance_score
                }
                for ev in result.evidence_list
            ],
            "metadata": result.metadata
        }

        return json.dumps(data, ensure_ascii=False, indent=2)

    @staticmethod
    def format_html(
            result: ExplanationResult,
            level: ExplanationLevel = ExplanationLevel.STANDARD
    ) -> str:
        """
        格式化为HTML

        Args:
            result: 解释结果
            level: 详细程度

        Returns:
            HTML格式解释
        """
        html = f"""
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <title>问答解释</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
            max-width: 800px;
            margin: 0 auto;
            padding: 20px;
            background: #f5f5f5;
        }}
        .container {{
            background: white;
            border-radius: 8px;
            padding: 30px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }}
        h1 {{
            color: #333;
            border-bottom: 2px solid #4CAF50;
            padding-bottom: 10px;
        }}
        h2 {{
            color: #555;
            margin-top: 30px;
        }}
        .qa-box {{
            background: #e8f5e9;
            padding: 15px;
            border-radius: 6px;
            margin: 15px 0;
        }}
        .question {{
            font-weight: bold;
            color: #2e7d32;
        }}
        .answer {{
            margin-top: 10px;
            color: #1b5e20;
        }}
        .confidence {{
            display: inline-block;
            background: #4CAF50;
            color: white;
            padding: 3px 10px;
            border-radius: 12px;
            font-size: 0.9em;
        }}
        .reasoning-step {{
            background: #fff3e0;
            padding: 10px 15px;
            margin: 10px 0;
            border-left: 3px solid #ff9800;
            border-radius: 0 6px 6px 0;
        }}
        .evidence {{
            background: #e3f2fd;
            padding: 10px 15px;
            margin: 10px 0;
            border-radius: 6px;
        }}
        .summary {{
            background: #fce4ec;
            padding: 15px;
            border-radius: 6px;
            font-style: italic;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>📋 问答解释</h1>

        <div class="qa-box">
            <div class="question">问题: {result.query}</div>
            <div class="answer">答案: {result.answer}</div>
            <span class="confidence">置信度: {result.confidence:.1%}</span>
        </div>
"""

        if result.summary:
            html += f"""
        <div class="summary">
            <strong>摘要:</strong> {result.summary}
        </div>
"""

        if level != ExplanationLevel.BRIEF and result.reasoning_path:
            html += """
        <h2>🔍 推理过程</h2>
"""
            for i, node_id in enumerate(result.reasoning_path, 1):
                node = next((n for n in result.reasoning_nodes if n.id == node_id), None)
                if node:
                    html += f"""
        <div class="reasoning-step">
            <strong>步骤 {i}:</strong> {node.content}
            {f'<br><small>置信度: {node.confidence:.1%}</small>' if level in [ExplanationLevel.DETAILED, ExplanationLevel.TECHNICAL] else ''}
        </div>
"""

        if level in [ExplanationLevel.DETAILED, ExplanationLevel.TECHNICAL] and result.evidence_list:
            html += """
        <h2>📚 支持证据</h2>
"""
            for ev in result.evidence_list:
                html += f"""
        <div class="evidence">
            {ev.content}
            {f'<br><small>来源: {ev.source}</small>' if ev.source else ''}
        </div>
"""

        html += """
    </div>
</body>
</html>
"""
        return html


class ExplainabilityEngine:
    """
    可解释性引擎
    Explainability Engine

    生成推理过程的可解释性输出

    Args:
        model: 语言模型
        tokenizer: 分词器

    Example:
        >>> engine = ExplainabilityEngine(model, tokenizer)
        >>> explanation = engine.explain(query, answer, reasoning_chain)
        >>> print(explanation.to_markdown())
    """

    def __init__(
            self,
            model=None,
            tokenizer=None
    ):
        self.model = model
        self.tokenizer = tokenizer

        logger.info("ExplainabilityEngine initialized")

    def explain(
            self,
            query: str,
            answer: str,
            reasoning_chain: Any = None,
            evidence_list: List[Evidence] = None,
            confidence: float = 0.0,
            level: ExplanationLevel = ExplanationLevel.STANDARD
    ) -> ExplanationResult:
        """
        生成解释

        Args:
            query: 查询
            answer: 答案
            reasoning_chain: 推理链（来自multi_hop_reasoning模块）
            evidence_list: 证据列表
            confidence: 置信度
            level: 解释详细程度

        Returns:
            ExplanationResult: 解释结果
        """
        # 构建推理节点
        reasoning_nodes = []
        reasoning_path = []

        if reasoning_chain is not None:
            reasoning_nodes, reasoning_path = self._build_reasoning_nodes(reasoning_chain)

        # 生成摘要
        summary = self._generate_summary(query, answer, reasoning_nodes)

        return ExplanationResult(
            query=query,
            answer=answer,
            reasoning_nodes=reasoning_nodes,
            reasoning_path=reasoning_path,
            confidence=confidence,
            summary=summary,
            evidence_list=evidence_list or [],
            metadata={
                "level": level.value,
                "node_count": len(reasoning_nodes),
                "evidence_count": len(evidence_list or [])
            }
        )

    def _build_reasoning_nodes(
            self,
            reasoning_chain
    ) -> Tuple[List[ReasoningNode], List[str]]:
        """
        从推理链构建推理节点

        Args:
            reasoning_chain: 推理链对象

        Returns:
            (节点列表, 路径)
        """
        nodes = []
        path = []

        # 处理推理链中的步骤
        if hasattr(reasoning_chain, 'steps'):
            for i, step in enumerate(reasoning_chain.steps):
                node = ReasoningNode(
                    id=step.step_id if hasattr(step, 'step_id') else f"step_{i}",
                    content=step.description if hasattr(step, 'description') else str(step),
                    node_type="inference",
                    confidence=step.confidence if hasattr(step, 'confidence') else 0.0,
                    parent_ids=step.dependencies if hasattr(step, 'dependencies') else [],
                    evidence=[
                        Evidence(id=f"ev_{i}_{j}", content=fact)
                        for j, fact in enumerate(getattr(step, 'input_facts', []))
                    ]
                )
                nodes.append(node)
                path.append(node.id)

        # 添加结论节点
        if hasattr(reasoning_chain, 'final_answer'):
            conclusion_node = ReasoningNode(
                id="conclusion",
                content=reasoning_chain.final_answer,
                node_type="conclusion",
                confidence=reasoning_chain.overall_confidence if hasattr(reasoning_chain, 'overall_confidence') else 0.0
            )
            nodes.append(conclusion_node)
            path.append("conclusion")

        return nodes, path

    def _generate_summary(
            self,
            query: str,
            answer: str,
            reasoning_nodes: List[ReasoningNode]
    ) -> str:
        """
        生成摘要

        Args:
            query: 查询
            answer: 答案
            reasoning_nodes: 推理节点

        Returns:
            摘要文本
        """
        if self.model is not None:
            prompt = f"""请用一两句话总结以下问答的推理过程：

问题: {query}
答案: {answer}
推理步骤: {' → '.join(n.content[:50] for n in reasoning_nodes[:3])}

摘要:"""
            return self._generate(prompt)

        # 简化实现
        step_count = len(reasoning_nodes)
        if step_count == 0:
            return f"直接回答了问题。"
        elif step_count <= 2:
            return f"通过{step_count}步推理得出答案。"
        else:
            return f"经过{step_count}步推理分析后得出答案，推理过程逻辑清晰。"

    def _generate(self, prompt: str) -> str:
        """使用模型生成文本"""
        inputs = self.tokenizer(prompt, return_tensors="pt")

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=100,
                temperature=0.7,
                do_sample=True
            )

        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

    def format_output(
            self,
            result: ExplanationResult,
            format: ExplanationFormat = ExplanationFormat.MARKDOWN,
            level: ExplanationLevel = ExplanationLevel.STANDARD
    ) -> str:
        """
        格式化输出

        Args:
            result: 解释结果
            format: 输出格式
            level: 详细程度

        Returns:
            格式化后的字符串
        """
        if format == ExplanationFormat.TEXT:
            return result.to_text(level)
        elif format == ExplanationFormat.MARKDOWN:
            return result.to_markdown(level)
        elif format == ExplanationFormat.JSON:
            return result.to_json()
        elif format == ExplanationFormat.HTML:
            return result.to_html(level)
        else:
            return result.to_text(level)


class ConfidenceCalibrator:
    """
    置信度校准器
    Confidence Calibrator

    校准和细化置信度分数

    Args:
        temperature: 温度参数
    """

    def __init__(self, temperature: float = 1.0):
        self.temperature = temperature
        self.calibration_history = []

        logger.info(f"ConfidenceCalibrator initialized, temperature={temperature}")

    def calibrate(
            self,
            raw_confidence: float,
            evidence_count: int = 0,
            reasoning_steps: int = 0
    ) -> float:
        """
        校准置信度

        Args:
            raw_confidence: 原始置信度
            evidence_count: 证据数量
            reasoning_steps: 推理步数

        Returns:
            校准后的置信度
        """
        import math

        # 温度缩放
        calibrated = 1 / (1 + math.exp(-(math.log(raw_confidence / (1 - raw_confidence + 1e-9)) / self.temperature)))

        # 证据调整
        evidence_factor = min(1.0, 0.8 + 0.02 * evidence_count)

        # 推理步数调整（过多步骤可能降低可靠性）
        if reasoning_steps > 0:
            step_penalty = 1.0 - 0.02 * max(0, reasoning_steps - 3)
            step_penalty = max(0.7, step_penalty)
        else:
            step_penalty = 1.0

        final_confidence = calibrated * evidence_factor * step_penalty

        # 记录历史
        self.calibration_history.append({
            "raw": raw_confidence,
            "calibrated": final_confidence,
            "evidence_count": evidence_count,
            "reasoning_steps": reasoning_steps
        })

        return final_confidence

    def get_statistics(self) -> Dict:
        """获取校准统计"""
        if not self.calibration_history:
            return {}

        raw_avg = sum(h["raw"] for h in self.calibration_history) / len(self.calibration_history)
        cal_avg = sum(h["calibrated"] for h in self.calibration_history) / len(self.calibration_history)

        return {
            "total_calibrations": len(self.calibration_history),
            "raw_confidence_avg": raw_avg,
            "calibrated_confidence_avg": cal_avg,
            "adjustment_ratio": cal_avg / (raw_avg + 1e-9)
        }
