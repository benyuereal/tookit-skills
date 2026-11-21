"""
DomainRouter - DCU-first 前置门禁。

职责：
1) 识别 DCU 领域请求（命中阈值后强制走 dcu-hip-reference 起步）
2) 识别用户显式绕过指令（直接回答模式）
3) 输出统一 metadata，供 dispatcher/router 透传
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Set


@dataclass
class DomainDecision:
    force_dcu_first: bool
    bypass_skill: bool
    matched_keywords: List[str]
    reason: str


class DomainRouter:
    def __init__(self, marketplace: Dict[str, Any], installed_skills: Set[str]):
        self._installed_skills = installed_skills
        policy = marketplace.get("routing_policy", {})
        self._dcu_keywords = [k.lower() for k in policy.get("dcu_first_keywords", [])]
        self._bypass_keywords = [k.lower() for k in policy.get("bypass_keywords", [])]
        self._force_threshold = int(policy.get("force_threshold", 2))

    def pre_route(self, query: str) -> DomainDecision:
        query_lower = query.lower()
        bypass_hits = [k for k in self._bypass_keywords if k in query_lower]
        if bypass_hits:
            return DomainDecision(
                force_dcu_first=False,
                bypass_skill=True,
                matched_keywords=bypass_hits,
                reason="用户显式要求跳过 skill 工作流",
            )

        dcu_hits = [k for k in self._dcu_keywords if k in query_lower]
        if (
            len(dcu_hits) >= self._force_threshold
            and "dcu-hip-reference" in self._installed_skills
        ):
            return DomainDecision(
                force_dcu_first=True,
                bypass_skill=False,
                matched_keywords=dcu_hits,
                reason="命中 DCU 领域关键词阈值，触发 DCU-first",
            )

        return DomainDecision(
            force_dcu_first=False,
            bypass_skill=False,
            matched_keywords=dcu_hits,
            reason="未触发域级强制路由",
        )
