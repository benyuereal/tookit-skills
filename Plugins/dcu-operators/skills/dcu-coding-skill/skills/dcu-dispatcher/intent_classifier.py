"""
IntentClassifier —— LLM-first + 关键词兜底意图分类（§5.2）

构造签名：
    host_skill:       调用方 skill 名（dispatcher 宿主传 None）
    installed_skills: 运行时可见 skill 集合（dispatcher 宿主必含 dcu-hip-reference）
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Set, Tuple

from common.intent import IntentRegistry, IntentSpec
from common.spi.routing import resolve_target_skill


@dataclass
class Intent:
    type: str                                                    # 动态意图类型
    target_skill: str                                            # 路由目标（可为 __self__）
    confidence: float                                            # 0.0 ~ 1.0
    sub_type: Optional[str] = None
    reasoning: str = ""
    matched_keywords: List[str] = field(default_factory=list)
    candidates: List[Tuple[str, float]] = field(default_factory=list)
    strategy: Literal["llm", "keyword", "fallback"] = "keyword"


class IntentClassifier:
    LLM_CONFIDENCE_THRESHOLD = 0.7
    KEYWORD_MIN_SCORE        = 1.0
    KEYWORD_MARGIN           = 0.15
    DCU_FORCE_THRESHOLD      = 2
    DCU_DOMAIN_KEYWORDS = {
        "dcu", "dtk", "rocm", "hip", "hipcc", "hipify", "gfx90a", "wavefront",
        "cuda", "nvcc", "cuda runtime", "hip runtime", "offload-arch",
        "cudamalloc", "cudamemcpy", "cudastream", "cudaevent",
        "hipmalloc", "hipmemcpy", "hipstream", "hipevent",
        "cublas", "cufft", "curand", "cudnn", "nccl",
        "hipblas", "hipfft", "hiprand", "miopen", "rccl",
        "迁移", "移植", "适配", "构建失败", "编不过", "报错", "性能优化", "回归", "兼容层",
    }
    SKILL_BYPASS_KEYWORDS = {
        "不要走skill", "不走skill", "跳过skill", "直接回答", "直接给答案",
        "无需工作流", "不用路由", "只要结论",
    }

    def __init__(
        self,
        marketplace: Dict[str, Any],
        llm_client: Any = None,
        *,
        host_skill: Optional[str] = None,
        installed_skills: Optional[Set[str]] = None,
    ):
        """
        Args:
            host_skill:       调用方 skill 名；None = dispatcher 宿主
            installed_skills: 当前运行时可见的 skill 集合；None 时按"只有 host_skill"处理
        """
        self._marketplace      = marketplace
        self._llm              = llm_client
        self._host_skill       = host_skill
        self._installed_skills = installed_skills or (
            {host_skill} if host_skill else set()
        )
        self._keyword_index    = self._build_keyword_index(marketplace)

        # dispatcher 宿主不变式（§5.2）
        if host_skill is None and "dcu-hip-reference" not in self._installed_skills:
            raise RuntimeError(
                "dispatcher 宿主启动失败：installed_skills 中缺少 dcu-hip-reference。"
                "整包安装模式要求 dcu-hip-reference 在位；"
                "请检查安装或在独立模式下直接使用 core skill。"
            )

    # ---- 主分类接口 ----

    def classify(self, query: str) -> Intent:
        """对 query 做意图分类，返回路由意图"""
        query_lower = query.lower()
        # 0. 用户显式要求绕过 skill 工作流（可退出）
        if any(k in query_lower for k in self.SKILL_BYPASS_KEYWORDS):
            return Intent(
                type="unknown",
                target_skill="__self__",
                confidence=0.95,
                strategy="fallback",
                reasoning="用户显式要求跳过 skill 工作流，回退到直接回答模式",
            )

        # 1. LLM-first
        if self._llm is not None:
            llm_intent = self._llm_classify(query)
            if llm_intent and llm_intent.confidence >= self.LLM_CONFIDENCE_THRESHOLD:
                llm_intent.strategy = "llm"
                return llm_intent

        # 2. 关键词匹配
        scores = self._keyword_scores(query)
        dcu_hits = self._dcu_keyword_hits(query)
        if not scores or max(scores.values()) < self.KEYWORD_MIN_SCORE:
            # DCU-first: 即便没有明确命中子 skill 关键词，只要是 DCU 领域问题，优先进入 hip-reference
            if dcu_hits >= self.DCU_FORCE_THRESHOLD and "dcu-hip-reference" in self._installed_skills:
                return Intent(
                    type="api-lookup",
                    target_skill="dcu-hip-reference",
                    confidence=0.65,
                    strategy="keyword",
                    matched_keywords=self._matched_dcu_keywords(query),
                    reasoning=(
                        "命中 DCU 领域关键词，触发 DCU-first 路由；"
                        "先进入 hip-reference 再按上下文分流"
                    ),
                )
            return self._fallback_intent(query)

        ranked = sorted(scores.items(), key=lambda x: -x[1])
        best, best_score = ranked[0]
        second_score = ranked[1][1] if len(ranked) > 1 else 0.0

        # 模糊区间：让 LLM 仲裁
        if (best_score - second_score) < self.KEYWORD_MARGIN and self._llm is not None:
            llm_intent = self._llm_classify(query, shortlist=[s for s, _ in ranked[:3]])
            if llm_intent:
                llm_intent.strategy = "llm"
                llm_intent.candidates = ranked[:5]
                return llm_intent

        confidence = min(0.95, 0.5 + (best_score - second_score))
        spec = IntentRegistry.resolve(self._skill_to_intent_type(best))
        intent_type = spec.type if spec else "unknown"

        return Intent(
            type=intent_type,
            target_skill=best,
            confidence=confidence,
            matched_keywords=self._matched_keywords(query, best),
            candidates=ranked[:5],
            strategy="keyword",
            reasoning=f"关键词归一化分数 {best_score:.2f}，与第二名差距 {best_score - second_score:.2f}",
        )

    # ---- fallback（两级降级）----

    def _fallback_intent(self, query: str) -> Intent:
        if "dcu-hip-reference" in self._installed_skills:
            return Intent(
                type="unknown",
                target_skill="dcu-hip-reference",
                confidence=0.3,
                strategy="fallback",
                reasoning="未匹配到明确意图，回退到 hip-reference 做跨分片检索",
            )
        return Intent(
            type="unknown",
            target_skill="__self__",
            confidence=0.2,
            strategy="fallback",
            reasoning=(
                f"未匹配且 hip-reference 未安装，回退到当前 skill "
                f"({self._host_skill or 'unknown-host'}) 自检索"
            ),
        )

    # ---- 关键词索引 ----

    def _build_keyword_index(
        self, marketplace: Dict[str, Any]
    ) -> Dict[str, List[str]]:
        """skill_name → [lowercase_keywords]"""
        index: Dict[str, List[str]] = {}
        for skill in marketplace.get("skills", []):
            name = skill.get("name", "")
            if skill.get("category") == "core":
                index[name] = [k.lower() for k in skill.get("keywords", [])]
        return index

    def _keyword_scores(self, query: str) -> Dict[str, float]:
        query_lower = query.lower()
        scores: Dict[str, float] = {}
        for skill_name, keywords in self._keyword_index.items():
            score = sum(1.0 for kw in keywords if kw in query_lower)
            if score > 0:
                scores[skill_name] = score
        return scores

    def _dcu_keyword_hits(self, query: str) -> int:
        query_lower = query.lower()
        return sum(1 for kw in self.DCU_DOMAIN_KEYWORDS if kw in query_lower)

    def _matched_dcu_keywords(self, query: str) -> List[str]:
        query_lower = query.lower()
        return [kw for kw in sorted(self.DCU_DOMAIN_KEYWORDS) if kw in query_lower]

    def _matched_keywords(self, query: str, skill_name: str) -> List[str]:
        query_lower = query.lower()
        return [kw for kw in self._keyword_index.get(skill_name, []) if kw in query_lower]

    def _skill_to_intent_type(self, skill_name: str) -> str:
        """通过 IntentRegistry 反查 skill 对应的 primary intent type"""
        for spec in IntentRegistry.all():
            if spec.target_skill == skill_name:
                return spec.type
        return "unknown"

    # ---- LLM 分类（占位，实际实现由运行时注入）----

    def _llm_classify(
        self,
        query: str,
        shortlist: Optional[List[str]] = None,
    ) -> Optional[Intent]:
        if self._llm is None:
            return None
        try:
            return self._llm.classify_intent(query, shortlist=shortlist)
        except Exception:
            return None
