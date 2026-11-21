"""
并行配置类
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class ParallelConfig:
    """并行配置"""
    tp_size: int = 1
    tp_rank: int = 0
    pipeline_size: int = 1
    pipeline_rank: int = 0
    dp_size: int = 1
    dp_rank: int = 0
    
    def __str__(self):
        return (f"ParallelConfig(TP={self.tp_size}, PP={self.pipeline_size}, "
                f"DP={self.dp_size})")