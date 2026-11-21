"""
Weighted Reciprocal Rank Fusion (带权 RRF) —— §6.5

公式：score(d) = Σ_i  w_i / (k + rank_i(d))
其中 k=60（平滑常数），w_i 为分片综合权重。

P4 增量：增加语义去重（semantic_deduplicate）
"""
from __future__ import annotations
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set

from .shard import SearchHit


@dataclass
class RankedResult:
    """融合后的单条结果"""
    chunk_id: str
    shard_id: str
    content: str
    rrf_score: float
    source_hits: List[SearchHit] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


def weighted_rrf(
    ranked_lists: List[List[SearchHit]],
    weights: Optional[Dict[str, float]] = None,
    k: int = 60,
    top_k: int = 10,
) -> List[RankedResult]:
    """
    对多个分片的检索结果做带权 RRF 融合并去重。

    Args:
        ranked_lists: 每个分片的 SearchHit 列表（已按 score 降序排列）
        weights:      {shard_id: weight}；None 则所有分片等权（w=1.0）
        k:            RRF 平滑常数，默认 60
        top_k:        返回 top_k 个结果

    Returns:
        融合后的 RankedResult 列表，按 rrf_score 降序
    """
    if weights is None:
        weights = {}

    # chunk_id → 累积 RRF 分数 + 来源 hit
    fused: Dict[str, dict] = {}

    for hits in ranked_lists:
        for hit in hits:
            w = weights.get(hit.shard_id, 1.0)
            contrib = w / (k + hit.rank)
            if hit.chunk_id not in fused:
                fused[hit.chunk_id] = {
                    "shard_id": hit.shard_id,
                    "content": hit.content,
                    "rrf_score": 0.0,
                    "source_hits": [],
                }
            fused[hit.chunk_id]["rrf_score"] += contrib
            fused[hit.chunk_id]["source_hits"].append(hit)

    results: List[RankedResult] = [
        RankedResult(
            chunk_id=cid,
            shard_id=data["shard_id"],
            content=data["content"],
            rrf_score=data["rrf_score"],
            source_hits=data["source_hits"],
        )
        for cid, data in fused.items()
    ]
    results.sort(key=lambda r: -r.rrf_score)
    return results[:top_k]


# ---- P4: 语义去重 ----


def _normalize_text(text: str) -> str:
    """文本规范化：转小写、去除标点、压缩空白"""
    # 转小写
    text = text.lower()
    # 去除代码片段（反引号内的内容）
    text = re.sub(r"`[^`]+`", " ", text)
    # 去除多余空白
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _get_ngrams(text: str, n: int = 3) -> Set[str]:
    """提取 n-gram 集合（用于 Jaccard 相似度）"""
    words = text.split()
    if len(words) < n:
        return {text} if words else set()
    return {" ".join(words[i:i+n]) for i in range(len(words) - n + 1)}


def jaccard_similarity(text1: str, text2: str) -> float:
    """
    计算两个文本的 Jaccard 相似度（基于 3-gram）。

    Returns:
        0.0 ~ 1.0 的相似度分数
    """
    ngrams1 = _get_ngrams(_normalize_text(text1), n=3)
    ngrams2 = _get_ngrams(_normalize_text(text2), n=3)

    if not ngrams1 or not ngrams2:
        return 0.0

    intersection = ngrams1 & ngrams2
    union = ngrams1 | ngrams2
    return len(intersection) / len(union) if union else 0.0


def semantic_deduplicate(
    results: List[RankedResult],
    similarity_threshold: float = 0.8,
) -> List[RankedResult]:
    """
    对 RRF 后的结果做语义级去重。

    解决跨分片同一文档段生成不同 chunk_id 的问题：
    - 两篇内容相似度 > threshold 时，保留得分高的
    - 较低分结果标记为 duplicate_of 指向高分红

    Args:
        results: RRF 后的结果列表（已按 score 降序）
        similarity_threshold: 相似度阈值，默认 0.8

    Returns:
        去重后的结果列表
    """
    if not results:
        return results

    keep: List[RankedResult] = []
    duplicate_of: Dict[str, str] = {}  # chunk_id -> 被去重的来源

    for result in results:
        # 检查是否与已保留结果相似；元数据需挂到实际命中的「保留项」上，非 keep 最后一项
        matched: Optional[RankedResult] = None
        for kept in keep:
            sim = jaccard_similarity(result.content, kept.content)
            if sim >= similarity_threshold:
                matched = kept
                dup_key = f"{result.chunk_id}_duplicate_of_{kept.chunk_id}"
                duplicate_of[dup_key] = f"similarity={sim:.2f}"
                break

        if matched is None:
            keep.append(result)
        else:
            if "deduplicated" not in matched.metadata:
                matched.metadata["deduplicated"] = []
            matched.metadata["deduplicated"].append({
                "chunk_id": result.chunk_id,
                "shard_id": result.shard_id,
                "score": result.rrf_score,
            })

    # 在返回结果中标记去重统计
    if duplicate_of:
        keep[0].metadata["deduplication_stats"] = {
            "total_input": len(results),
            "total_output": len(keep),
            "removed_count": len(results) - len(keep),
        }

    return keep
