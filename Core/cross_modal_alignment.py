"""
跨模态对齐模块
Cross-Modal Alignment Module

实现文本模态与表格模态之间的对齐和映射
支持对比学习和跨模态检索
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class AlignmentResult:
    """
    对齐结果数据类
    Alignment result Dataclass

    Attributes:
        text_features: 文本特征向量
        table_features: 表格特征向量
        alignment_scores: 对齐分数矩阵
        aligned_pairs: 对齐的文本-表格对
        confidence: 对齐置信度
    """
    text_features: Tensor
    table_features: Tensor
    alignment_scores: Tensor
    aligned_pairs: Optional[List[Tuple[int, int]]] = None
    confidence: Optional[float] = None


class ProjectionHead(nn.Module):
    """
    投影头
    Projection Head

    将特征投影到对齐空间
    用于对比学习的特征变换

    Args:
        input_dim: 输入维度
        hidden_dim: 隐藏层维度
        output_dim: 输出维度
        dropout: Dropout概率
        num_layers: 层数

    Example:
        >>> proj = ProjectionHead(4096, 2048, 512)
        >>> projected = proj(features)
    """

    def __init__(
            self,
            input_dim: int,
            hidden_dim: int,
            output_dim: int,
            dropout: float = 0.1,
            num_layers: int = 2
    ):
        super().__init__()

        layers = []
        current_dim = input_dim

        for i in range(num_layers - 1):
            layers.extend([
                nn.Linear(current_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout)
            ])
            current_dim = hidden_dim

        # 最后一层不使用激活函数
        layers.append(nn.Linear(current_dim, output_dim))

        self.projection = nn.Sequential(*layers)

        logger.info(f"ProjectionHead: {input_dim} -> {output_dim}")

    def forward(self, x: Tensor) -> Tensor:
        """
        前向传播

        Args:
            x: 输入特征 (batch, input_dim)

        Returns:
            投影后的特征 (batch, output_dim)
        """
        return self.projection(x)


class ContrastiveLoss(nn.Module):
    """
    对比学习损失函数
    Contrastive Learning Loss

    实现InfoNCE损失，用于跨模态对齐

    Args:
        temperature: 温度参数
        label_smoothing: 标签平滑系数

    公式:
        L = -log(exp(sim(t_i, v_i)/τ) / Σ_j exp(sim(t_i, v_j)/τ))
    """

    def __init__(
            self,
            temperature: float = 0.07,
            label_smoothing: float = 0.0
    ):
        super().__init__()
        self.temperature = temperature
        self.label_smoothing = label_smoothing

        logger.info(f"ContrastiveLoss: temperature={temperature}")

    def forward(
            self,
            text_features: Tensor,
            table_features: Tensor
    ) -> Tuple[Tensor, Dict[str, float]]:
        """
        计算对比损失

        Args:
            text_features: 文本特征 (batch, dim)
            table_features: 表格特征 (batch, dim)

        Returns:
            loss: 对比损失
            metrics: 包含准确率等指标的字典
        """
        # L2归一化
        text_features = F.normalize(text_features, dim=-1)
        table_features = F.normalize(table_features, dim=-1)

        batch_size = text_features.size(0)

        # 计算相似度矩阵
        # (batch, dim) @ (dim, batch) -> (batch, batch)
        similarity_matrix = torch.matmul(text_features, table_features.T) / self.temperature

        # 创建标签（对角线为正样本）
        labels = torch.arange(batch_size, device=text_features.device)

        # 计算损失
        loss_i2t = F.cross_entropy(
            similarity_matrix,
            labels,
            label_smoothing=self.label_smoothing
        )
        loss_t2i = F.cross_entropy(
            similarity_matrix.T,
            labels,
            label_smoothing=self.label_smoothing
        )

        loss = (loss_i2t + loss_t2i) / 2

        # 计算准确率
        with torch.no_grad():
            pred_i2t = similarity_matrix.argmax(dim=1)
            pred_t2i = similarity_matrix.T.argmax(dim=1)
            acc_i2t = (pred_i2t == labels).float().mean()
            acc_t2i = (pred_t2i == labels).float().mean()

        metrics = {
            "loss_i2t": loss_i2t.item(),
            "loss_t2i": loss_t2i.item(),
            "acc_i2t": acc_i2t.item(),
            "acc_t2i": acc_t2i.item()
        }

        return loss, metrics


class CrossModalAttention(nn.Module):
    """
    跨模态注意力机制
    Cross-Modal Attention Mechanism

    实现文本和表格之间的交叉注意力

    Args:
        hidden_size: 隐藏层维度
        num_heads: 注意力头数
        dropout: Dropout概率

    Example:
        >>> cross_attn = CrossModalAttention(4096, 8)
        >>> attended = cross_attn(text_features, table_features)
    """

    def __init__(
            self,
            hidden_size: int,
            num_heads: int = 8,
            dropout: float = 0.1
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads

        assert hidden_size % num_heads == 0, "hidden_size必须能被num_heads整除"

        # Q, K, V 投影
        self.q_proj = nn.Linear(hidden_size, hidden_size)
        self.k_proj = nn.Linear(hidden_size, hidden_size)
        self.v_proj = nn.Linear(hidden_size, hidden_size)
        self.o_proj = nn.Linear(hidden_size, hidden_size)

        self.dropout = nn.Dropout(dropout)
        self.scale = self.head_dim ** -0.5

        logger.info(f"CrossModalAttention: hidden={hidden_size}, heads={num_heads}")

    def forward(
            self,
            query: Tensor,
            key_value: Tensor,
            attention_mask: Optional[Tensor] = None
    ) -> Tuple[Tensor, Tensor]:
        """
        前向传播

        Args:
            query: 查询特征 (batch, seq_q, hidden)
            key_value: 键值特征 (batch, seq_kv, hidden)
            attention_mask: 注意力掩码 (batch, seq_kv)

        Returns:
            output: 注意力输出 (batch, seq_q, hidden)
            attention_weights: 注意力权重 (batch, heads, seq_q, seq_kv)
        """
        batch_size = query.size(0)

        # 投影
        Q = self.q_proj(query)
        K = self.k_proj(key_value)
        V = self.v_proj(key_value)

        # 重塑为多头形式
        Q = Q.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)

        # 计算注意力分数
        scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale

        # 应用注意力掩码
        if attention_mask is not None:
            attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
            scores = scores.masked_fill(attention_mask == 0, float('-inf'))

        # Softmax和Dropout
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)

        # 加权求和
        output = torch.matmul(attention_weights, V)

        # 重塑并输出投影
        output = output.transpose(1, 2).contiguous().view(batch_size, -1, self.hidden_size)
        output = self.o_proj(output)

        return output, attention_weights


class CrossModalAligner(nn.Module):
    """
    跨模态对齐器
    Cross-Modal Aligner

    整合所有跨模态对齐组件，提供端到端的对齐能力

    Args:
        hidden_size: 隐藏层维度
        projection_dim: 投影空间维度
        num_heads: 注意力头数
        dropout: Dropout概率
        temperature: 对比学习温度

    Example:
        >>> aligner = CrossModalAligner(4096, 512)
        >>> result = aligner.align(text_features, table_features)
        >>> loss = aligner.compute_loss(result)
    """

    def __init__(
            self,
            hidden_size: int = 4096,
            projection_dim: int = 512,
            num_heads: int = 8,
            dropout: float = 0.1,
            temperature: float = 0.07
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.projection_dim = projection_dim

        # 投影头
        self.text_projection = ProjectionHead(
            hidden_size, hidden_size // 2, projection_dim, dropout
        )
        self.table_projection = ProjectionHead(
            hidden_size, hidden_size // 2, projection_dim, dropout
        )

        # 跨模态注意力
        self.cross_attention = CrossModalAttention(
            hidden_size, num_heads, dropout
        )

        # 融合层
        self.fusion = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.GELU(),
            nn.Dropout(dropout)
        )

        # 对比学习损失
        self.contrastive_loss = ContrastiveLoss(temperature)

        logger.info(f"CrossModalAligner: hidden={hidden_size}, proj_dim={projection_dim}")

    def align(
            self,
            text_features: Tensor,
            table_features: Tensor,
            text_mask: Optional[Tensor] = None,
            table_mask: Optional[Tensor] = None
    ) -> AlignmentResult:
        """
        对齐文本和表格特征

        Args:
            text_features: 文本特征 (batch, seq_t, hidden)
            table_features: 表格特征 (batch, seq_tab, hidden)
            text_mask: 文本注意力掩码
            table_mask: 表格注意力掩码

        Returns:
            AlignmentResult: 对齐结果
        """
        # 池化得到全局特征
        if text_mask is not None:
            text_global = self._masked_mean(text_features, text_mask)
        else:
            text_global = text_features.mean(dim=1)

        if table_mask is not None:
            table_global = self._masked_mean(table_features, table_mask)
        else:
            table_global = table_features.mean(dim=1)

        # 投影到对齐空间
        text_projected = self.text_projection(text_global)
        table_projected = self.table_projection(table_global)

        # 计算对齐分数
        alignment_scores = torch.matmul(
            F.normalize(text_projected, dim=-1),
            F.normalize(table_projected, dim=-1).T
        )

        # 跨模态注意力融合
        text_attended, _ = self.cross_attention(
            text_features, table_features, table_mask
        )
        table_attended, _ = self.cross_attention(
            table_features, text_features, text_mask
        )

        return AlignmentResult(
            text_features=text_projected,
            table_features=table_projected,
            alignment_scores=alignment_scores,
            confidence=alignment_scores.diag().mean().item()
        )

    def fuse(
            self,
            text_features: Tensor,
            table_features: Tensor
    ) -> Tensor:
        """
        融合文本和表格特征

        Args:
            text_features: 文本特征
            table_features: 表格特征

        Returns:
            融合后的特征
        """
        # 跨模态注意力
        text_attended, _ = self.cross_attention(text_features, table_features)

        # 拼接并融合
        combined = torch.cat([text_features, text_attended], dim=-1)
        fused = self.fusion(combined)

        return fused

    def compute_loss(
            self,
            alignment_result: AlignmentResult
    ) -> Tuple[Tensor, Dict[str, float]]:
        """
        计算对齐损失

        Args:
            alignment_result: 对齐结果

        Returns:
            loss: 对比学习损失
            metrics: 评估指标
        """
        return self.contrastive_loss(
            alignment_result.text_features,
            alignment_result.table_features
        )

    def _masked_mean(self, features: Tensor, mask: Tensor) -> Tensor:
        """带掩码的平均池化"""
        mask = mask.unsqueeze(-1).float()
        return (features * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)

    def forward(
            self,
            text_features: Tensor,
            table_features: Tensor,
            **kwargs
    ) -> AlignmentResult:
        """前向传播"""
        return self.align(text_features, table_features, **kwargs)


class AlignmentTrainer:
    """
    对齐训练器
    Alignment Trainer

    用于训练跨模态对齐模型

    Args:
        aligner: 跨模态对齐器
        lr: 学习率
        weight_decay: 权重衰减
        warmup_steps: 预热步数
    """

    def __init__(
            self,
            aligner: CrossModalAligner,
            lr: float = 1e-4,
            weight_decay: float = 0.01,
            warmup_steps: int = 100
    ):
        self.aligner = aligner

        self.optimizer = torch.optim.AdamW(
            aligner.parameters(),
            lr=lr,
            weight_decay=weight_decay
        )

        self.scheduler = torch.optim.lr_scheduler.LinearLR(
            self.optimizer,
            start_factor=0.1,
            end_factor=1.0,
            total_iters=warmup_steps
        )

        logger.info("AlignmentTrainer initialized")

    def train_step(
            self,
            text_features: Tensor,
            table_features: Tensor
    ) -> Dict[str, float]:
        """
        训练一步

        Args:
            text_features: 文本特征
            table_features: 表格特征

        Returns:
            训练指标字典
        """
        self.aligner.train()
        self.optimizer.zero_grad()

        # 对齐
        result = self.aligner.align(text_features, table_features)

        # 计算损失
        loss, metrics = self.aligner.compute_loss(result)

        # 反向传播
        loss.backward()
        self.optimizer.step()
        self.scheduler.step()

        return {**metrics, "loss": loss.item()}

    def evaluate(
            self,
            text_features: Tensor,
            table_features: Tensor
    ) -> Dict[str, float]:
        """
        评估

        Args:
            text_features: 文本特征
            table_features: 表格特征

        Returns:
            评估指标字典
        """
        self.aligner.eval()

        with torch.no_grad():
            result = self.aligner.align(text_features, table_features)
            _, metrics = self.aligner.compute_loss(result)

            # 计算检索指标
            retrieval_metrics = self._compute_retrieval_metrics(result.alignment_scores)
            metrics.update(retrieval_metrics)

        return metrics

    def _compute_retrieval_metrics(
            self,
            alignment_scores: Tensor
    ) -> Dict[str, float]:
        """计算检索指标"""
        batch_size = alignment_scores.size(0)

        # Text-to-Table检索
        t2t_preds = alignment_scores.argmax(dim=1)
        t2t_correct = (t2t_preds == torch.arange(batch_size, device=t2t_preds.device)).float()

        # Table-to-Text检索
        t2b_preds = alignment_scores.T.argmax(dim=1)
        t2b_correct = (t2b_preds == torch.arange(batch_size, device=t2b_preds.device)).float()

        return {
            "text2table_acc": t2t_correct.mean().item(),
            "table2text_acc": t2b_correct.mean().item(),
            "mean_reciprocal_rank": self._compute_mrr(alignment_scores)
        }

    def _compute_mrr(self, scores: Tensor) -> float:
        """计算Mean Reciprocal Rank"""
        batch_size = scores.size(0)

        # 排序
        sorted_indices = scores.argsort(dim=1, descending=True)

        # 计算每个查询的reciprocal rank
        rr_sum = 0.0
        for i in range(batch_size):
            rank = (sorted_indices[i] == i).nonzero(as_tuple=True)[0][0].item() + 1
            rr_sum += 1.0 / rank

        return rr_sum / batch_size
