"""DCUHipReferenceSkill —— SkillProvider 实现"""
from __future__ import annotations

from pathlib import Path
from typing import List, Optional

from common.spi import Capability, HealthStatus, SkillProvider
from common.kb import get_aggregator


class DCUHipReferenceSkill(SkillProvider):
    name    = "dcu-hip-reference"
    version = "1.0.0"

    def capabilities(self) -> List[Capability]:
        return [
            Capability(action="lookup_api",  cost="cheap",  description="精确查询 HIP API 签名"),
            Capability(action="search_docs", cost="medium", description="语义检索 HIP 文档"),
        ]

    def health_check(self) -> HealthStatus:
        shard_path = self.skill_dir() / "kb_shard" / "shard.sqlite"
        if not shard_path.exists():
            return HealthStatus(
                healthy=True,
                degraded=True,
                reason="hip-core 分片未构建，检索依赖 LLM 内置知识（建议运行 build_shard.py）",
                missing_shards=["hip-core"],
            )
        return HealthStatus(healthy=True)

    def skill_dir(self) -> Path:
        return Path(__file__).parent

    # ---- 业务能力 ----

    def lookup_api(self, api_name: str) -> dict:
        """
        精确查询 HIP API 文档（top-1 结果）。

        Returns:
            {"api_name": ..., "signature": ..., "description": ..., "example": ...}
        """
        kb = get_aggregator(skill_dir=self.skill_dir())
        results = kb.search(
            query=api_name,
            domains=["hip-api"],
            top_k=1,
            mode="keyword",    # 精确匹配优先
        )
        if results:
            hit = results[0]
            return {
                "api_name":   api_name,
                "content":    hit.content,
                "shard_id":   hit.shard_id,
                "score":      hit.rrf_score,
                "system_notices": kb.get_last_notices(),
            }
        return {
            "api_name": api_name,
            "content": None,
            "note": "分片中未找到，请参考官方文档",
            "system_notices": kb.get_last_notices(),
        }

    def search_docs(
        self,
        query: str,
        domains: Optional[List[str]] = None,
        top_k: int = 5,
    ) -> list:
        """
        语义 + 关键词混合检索 HIP 文档。

        Returns:
            RankedResult 列表
        """
        kb = get_aggregator(skill_dir=self.skill_dir())
        return kb.search(
            query=query,
            domains=domains or ["hip-runtime", "hip-api", "hip-concept", "hip-example"],
            top_k=top_k,
            mode="hybrid",
        )
