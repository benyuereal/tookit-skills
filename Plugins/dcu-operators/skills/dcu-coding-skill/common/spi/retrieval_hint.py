"""
RetrievalHint —— 意图分类器输出的结构化检索指令

在 IntentClassifier 输出 Intent 后，由 Dispatcher 填充本结构体，
作为下游检索引擎（ShardAggregator）的输入参数。

各字段指导：
- primary_domains: 只查哪些分片领域（如 ["migration", "cuda-hip-mapping"]）
- expand_domains: 窄域不足时扩到哪些兄弟分片
- channels: 用哪些检索通道（api/fts/vector/graph）
- mode: 关键词/hybrid/向量
- retrieval_policy: local-first（先本 skill 再兄弟）还是 aggregator（全量）
- self_docs_enabled: 自修复时可关闭用户补充知识
- rrf_k: RRF 公式的 k 值（越小 Top-1 越突出）
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Literal


# 默认通道权重（可在 config 中覆盖）
DEFAULT_CHANNEL_WEIGHTS = {
    "api": 1.5,     # 精确符号匹配，高置信
    "fts": 1.0,     # 关键词基线
    "vector": 0.8,  # 语义可能漂移
    "graph": 1.2,   # 关系补充，高结构性
}

# 各 intent 的默认检索配置（可被 RetrievalHint 覆盖）
INTENT_HINT_MAP = {
    "code-migrate": {
        "primary_domains": ["migration", "cuda-hip-mapping", "migration-perf", "migration-debug"],
        "channels": ["api", "fts", "graph"],
        "rrf_k": 30,
        "retrieval_policy": "local-first",
        "self_docs_enabled": True,
    },
    "api-lookup": {
        "primary_domains": ["hip-api", "hip-runtime", "hip-concept"],
        "channels": ["api", "fts"],
        "rrf_k": 30,
        "retrieval_policy": "local-first",
        "self_docs_enabled": True,
    },
    "ci-test": {
        "primary_domains": ["testing", "benchmark", "ci", "test-debug"],
        "channels": ["fts"],
        "rrf_k": 60,
        "retrieval_policy": "aggregator",
        "self_docs_enabled": True,
    },
    "triton": {
        "primary_domains": ["triton-api", "triton-patterns", "triton-perf", "triton-debug"],
        "channels": ["fts"],
        "rrf_k": 60,
        "retrieval_policy": "local-first",
        "self_docs_enabled": True,
    },
    "unknown": {
        "primary_domains": [],
        "channels": ["fts"],
        "rrf_k": 60,
        "retrieval_policy": "aggregator",
        "self_docs_enabled": False,
    },
}


@dataclass
class RetrievalHint:
    """
    意图分类器输出的结构化检索指令。

    Attributes:
        primary_domains: 主要检索领域列表（来自 marketplace.shard_domains）
        expand_domains: 窄域不足时的扩展领域（兄弟 skill 补充）
        channels: 检索通道子集 ["api", "fts", "vector", "graph"]
        mode: 检索模式 ("keyword" | "hybrid" | "vector")
        retrieval_policy: 检索策略 ("local-first" | "aggregator")
        local_order: 本 skill 分片查找顺序（默认 ["kb_shard", "kb_shard_ext"]）
        self_docs_enabled: 是否启用用户补充知识（self_docs）
        rrf_k: RRF 公式的 k 参数（默认 60，API 查询场景可降到 30）
        channel_weights: 各通道的权重覆盖（默认用 DEFAULT_CHANNEL_WEIGHTS）
    """
    primary_domains: List[str] = field(default_factory=list)
    expand_domains: List[str] = field(default_factory=list)
    channels: List[str] = field(default_factory=lambda: ["fts"])
    mode: Literal["keyword", "hybrid", "vector"] = "hybrid"
    retrieval_policy: Literal["local-first", "aggregator"] = "aggregator"
    local_order: List[str] = field(default_factory=lambda: ["kb_shard", "kb_shard_ext"])
    self_docs_enabled: bool = True
    rrf_k: int = 60
    channel_weights: dict[str, float] = field(default_factory=lambda: DEFAULT_CHANNEL_WEIGHTS.copy())

    @classmethod
    def from_intent(cls, intent_type: str) -> "RetrievalHint":
        """
        根据 intent_type 自动生成默认的 RetrievalHint。

        Args:
            intent_type: 意图类型（如 "code-migrate", "api-lookup" 等）

        Returns:
            配置好的 RetrievalHint 实例
        """
        config = INTENT_HINT_MAP.get(intent_type, INTENT_HINT_MAP["unknown"])
        return cls(
            primary_domains=config.get("primary_domains", []),
            channels=config.get("channels", ["fts"]),
            rrf_k=config.get("rrf_k", 60),
            retrieval_policy=config.get("retrieval_policy", "aggregator"),
            self_docs_enabled=config.get("self_docs_enabled", True),
        )

    def with_expanded_domains(self, extra_domains: List[str]) -> "RetrievalHint":
        """
        返回一个扩展了 expand_domains 的新 Hint（链式调用）。

        Args:
            extra_domains: 要添加的扩展领域

        Returns:
            新的 RetrievalHint 实例
        """
        new_hint = RetrievalHint(
            primary_domains=self.primary_domains,
            expand_domains=self.expand_domains + extra_domains,
            channels=self.channels,
            mode=self.mode,
            retrieval_policy=self.retrieval_policy,
            local_order=self.local_order,
            self_docs_enabled=self.self_docs_enabled,
            rrf_k=self.rrf_k,
            channel_weights=self.channel_weights.copy(),
        )
        return new_hint

    def disable_self_docs(self) -> "RetrievalHint":
        """
        返回一个禁用 self_docs 的新 Hint（用于自修复场景）。
        """
        new_hint = RetrievalHint(
            primary_domains=self.primary_domains,
            expand_domains=self.expand_domains,
            channels=self.channels,
            mode=self.mode,
            retrieval_policy=self.retrieval_policy,
            local_order=self.local_order,
            self_docs_enabled=False,
            rrf_k=self.rrf_k,
            channel_weights=self.channel_weights.copy(),
        )
        return new_hint


# ---- 便捷入口 ----


def get_hint_for_intent(intent_type: str) -> RetrievalHint:
    """
    根据意图类型获取默认检索配置的便捷函数。

    Args:
        intent_type: 意图类型

    Returns:
        RetrievalHint 实例
    """
    return RetrievalHint.from_intent(intent_type)