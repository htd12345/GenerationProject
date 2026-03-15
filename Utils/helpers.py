"""
工具模块
Utility Module

提供常用的辅助函数和工具类
"""

import torch
import numpy as np
from typing import Dict, List, Optional, Any, Union, Callable
from pathlib import Path
import json
import logging
import time
from functools import wraps

logger = logging.getLogger(__name__)


def set_seed(seed: int = 42) -> None:
    """
    设置随机种子，确保可复现性
    Set random seed for reproducibility

    Args:
        seed: 随机种子值
    """
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    logger.info(f"Random seed set to {seed}")


def count_parameters(model: torch.nn.Module) -> Dict[str, int]:
    """
    统计模型参数数量
    Count model parameters

    Args:
        model: PyTorch模型

    Returns:
        参数统计字典
    """
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen = total - trainable

    return {
        "total": total,
        "trainable": trainable,
        "frozen": frozen,
        "trainable_ratio": trainable / total if total > 0 else 0
    }


def get_device() -> str:
    """
    获取可用设备
    Get available device

    Returns:
        设备名称 ('cuda' 或 'cpu')
    """
    if torch.cuda.is_available():
        device_name = torch.cuda.get_device_name(0)
        logger.info(f"Using GPU: {device_name}")
        return "cuda"
    else:
        logger.info("Using CPU")
        return "cpu"


def timing(func: Callable) -> Callable:
    """
    计时装饰器
    Timing decorator

    记录函数执行时间
    """

    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()

        elapsed = end_time - start_time
        logger.info(f"{func.__name__} executed in {elapsed:.2f}s")

        return result

    return wrapper


async def async_timing(func: Callable) -> Callable:
    """
    异步计时装饰器
    Async timing decorator
    """

    @wraps(func)
    async def wrapper(*args, **kwargs):
        start_time = time.time()
        result = await func(*args, **kwargs)
        end_time = time.time()

        elapsed = end_time - start_time
        logger.info(f"{func.__name__} executed in {elapsed:.2f}s")

        return result

    return wrapper


def load_json(path: Union[str, Path]) -> Any:
    """
    加载JSON文件
    Load JSON file

    Args:
        path: 文件路径

    Returns:
        解析后的数据
    """
    path = Path(path)
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def save_json(data: Any, path: Union[str, Path], indent: int = 2) -> None:
    """
    保存为JSON文件
    Save as JSON file

    Args:
        data: 要保存的数据
        path: 文件路径
        indent: 缩进空格数
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=indent)

    logger.info(f"Data saved to {path}")


def truncate_text(text: str, max_length: int = 100, suffix: str = "...") -> str:
    """
    截断文本
    Truncate text

    Args:
        text: 原始文本
        max_length: 最大长度
        suffix: 截断后缀

    Returns:
        截断后的文本
    """
    if len(text) <= max_length:
        return text
    return text[:max_length - len(suffix)] + suffix


def batch_iterate(data: List, batch_size: int):
    """
    批量迭代器
    Batch iterator

    Args:
        data: 数据列表
        batch_size: 批次大小

    Yields:
        批次数据
    """
    for i in range(0, len(data), batch_size):
        yield data[i:i + batch_size]


class EarlyStopping:
    """
    早停机制
    Early Stopping Mechanism

    监控指标，当连续多轮没有改善时停止训练

    Args:
        patience: 等待轮数
        min_delta: 最小改善阈值
        mode: 'min' 或 'max'
    """

    def __init__(
            self,
            patience: int = 5,
            min_delta: float = 0.0,
            mode: str = 'min'
    ):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False

        if mode == 'min':
            self.compare = lambda current, best: current < best - min_delta
        else:
            self.compare = lambda current, best: current > best + min_delta

    def __call__(self, score: float) -> bool:
        """
        检查是否应该停止

        Args:
            score: 当前分数

        Returns:
            是否应该停止
        """
        if self.best_score is None:
            self.best_score = score
            return False

        if self.compare(score, self.best_score):
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                return True

        return False

    def reset(self) -> None:
        """重置状态"""
        self.counter = 0
        self.best_score = None
        self.early_stop = False


class AverageMeter:
    """
    平均值计算器
    Average Meter

    计算滑动平均值

    Args:
        name: 指标名称
    """

    def __init__(self, name: str = ""):
        self.name = name
        self.reset()

    def reset(self) -> None:
        """重置"""
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val: float, n: int = 1) -> None:
        """
        更新值

        Args:
            val: 新值
            n: 数量
        """
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count if self.count > 0 else 0

    def __str__(self) -> str:
        return f"{self.name}: {self.avg:.4f}"


class ProgressLogger:
    """
    进度日志记录器
    Progress Logger

    记录训练/推理进度
    """

    def __init__(
            self,
            total: int,
            desc: str = "Progress",
            log_interval: int = 10
    ):
        self.total = total
        self.desc = desc
        self.log_interval = log_interval
        self.current = 0
        self.start_time = time.time()

    def update(self, n: int = 1) -> None:
        """更新进度"""
        self.current += n

        if self.current % self.log_interval == 0 or self.current >= self.total:
            elapsed = time.time() - self.start_time
            progress = self.current / self.total * 100
            eta = elapsed / self.current * (self.total - self.current) if self.current > 0 else 0

            logger.info(
                f"{self.desc}: {self.current}/{self.total} "
                f"({progress:.1f}%) - "
                f"Elapsed: {elapsed:.1f}s, ETA: {eta:.1f}s"
            )

    def close(self) -> None:
        """完成"""
        elapsed = time.time() - self.start_time
        logger.info(f"{self.desc} completed in {elapsed:.2f}s")
