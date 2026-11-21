"""
ShardAggregator —— 分片自动发现与聚合检索（§6.4）

发现顺序（优先级从高到低）：
1. 调用方 skill 的 kb_shard/、kb_shard_ext/
2. 同级 skills/ 下其他 skill 的 kb_shard/、kb_shard_ext/（目录名排序）
3. 安装根目录 docs/self_docs/（用户补充；检索前按需增量建索引）

P6 增量：discover() 缓存 + self_docs 异步化
"""
from __future__ import annotations

import json
import threading
import time
import urllib.request
import warnings
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

from typing import TYPE_CHECKING

from .shard import Shard, SearchHit
from .rrf import weighted_rrf, RankedResult, semantic_deduplicate
from .self_docs import ensure_self_docs_shard
from common.spi.retrieval_hint import RetrievalHint

if TYPE_CHECKING:
    from common.kb.query_parser import ParsedQuery


SearchResults = List[RankedResult]

# 各来源基础权重（可被用户 config 覆盖）
_BASE_WEIGHTS: dict[str, float] = {
    "official":   1.0,
    "user_shard": 0.7,
    "self_docs":  0.5,
}

_DOMAIN_BOOST = 1.2
_VECTOR_WARNED = False

# P6: discover() 缓存配置
_DISCOVER_CACHE_TTL_SEC = 60  # 默认缓存 60 秒
_discover_cache: Dict[str, tuple[float, List[Shard]]] = {}  # skill_dir -> (timestamp, shards)
_cache_lock = threading.Lock()


class ShardAggregator:
    """
    自动发现并聚合所有可见的分片。

    线程安全：每次 search() 都是无状态的；shard 连接在 Shard 实例内部管理。
    """

    SHARD_DIR_CANDIDATES = ("kb_shard", "kb_shard_ext")

    def __init__(
        self,
        skill_dir: Optional[Path] = None,
        config: Optional[Dict[str, Any]] = None,
    ):
        self._skill_dir = Path(skill_dir) if skill_dir else None
        self._config = config or {}
        self._last_notices: list[str] = []

    # ---- P6: 分片发现（带缓存）----

    def discover(
        self,
        use_cache: bool = True,
        include_self_docs: bool = True,
        include_siblings: bool = True,
    ) -> List[Shard]:
        """
        按优先级顺序发现所有可见分片，去重（按 shard_id）。
        只返回 is_ready() == True 的分片。

        P6 优化：
        - 带缓存的 discover，避免每次都扫磁盘
        - include_self_docs 控制是否加载 self_docs（自修复场景可关闭）
        - include_siblings 控制是否扫描兄弟 skill 的 shards（整包模式开启，单 skill 模式关闭）

        Args:
            use_cache: 是否使用缓存（默认 True）
            include_self_docs: 是否包含 self_docs 分片（默认 True）
            include_siblings: 是否包含兄弟 skill 的 shards（默认 True）
        """
        # P6: 缓存 key（包含 include_self_docs 和 include_siblings 状态）
        cache_key = f"{self._skill_dir}:{include_self_docs}:{include_siblings}" if self._skill_dir else f"__none__:{include_self_docs}:{include_siblings}"

        if use_cache:
            with _cache_lock:
                if cache_key in _discover_cache:
                    ts, cached_shards = _discover_cache[cache_key]
                    if time.monotonic() - ts < _DISCOVER_CACHE_TTL_SEC:
                        # 缓存有效，直接返回
                        return cached_shards

        # 缓存不存在或已过期，执行真实 discover
        shards = self._discover_inner(include_self_docs=include_self_docs, include_siblings=include_siblings)

        # 写入缓存
        with _cache_lock:
            _discover_cache[cache_key] = (time.monotonic(), shards)

        return shards

    @staticmethod
    def invalidate_cache(skill_dir: Optional[Path] = None) -> None:
        """手动使缓存失效（供外部调用，如检测到文件变化时）"""
        with _cache_lock:
            if skill_dir:
                key_prefix = str(skill_dir)
                keys_to_remove = [k for k in _discover_cache if k.startswith(key_prefix)]
                for k in keys_to_remove:
                    del _discover_cache[k]
            else:
                _discover_cache.clear()

    def _discover_inner(
        self,
        include_self_docs: bool = True,
        include_siblings: bool = True,
    ) -> List[Shard]:
        """discover 的实际执行逻辑（P6 后由 discover 调用）"""
        seen: set[str] = set()
        shards: list[Shard] = []

        def _try_add(shard_dir: Path, source_type: str = "official") -> None:
            manifest = shard_dir / "shard.manifest.yaml"
            if not manifest.exists():
                return
            try:
                shard = Shard.from_dir(shard_dir)
                shard.source_type = source_type
                if shard.shard_id not in seen and shard.is_ready():
                    seen.add(shard.shard_id)
                    shards.append(shard)
            except Exception:
                pass

        def _candidate_dirs(skill_dir: Path) -> list[Path]:
            return [skill_dir / name for name in self.SHARD_DIR_CANDIDATES]

        # 检测是否为整包模式（skill_dir 是 skills/ 目录本身）
        is_bundle_mode = False
        if self._skill_dir:
            # 检查 skill_dir 是否是 skills/ 目录
            if self._skill_dir.name == "skills" and not any(
                (self._skill_dir / name).exists() for name in self.SHARD_DIR_CANDIDATES
            ):
                is_bundle_mode = True

        if is_bundle_mode:
            # 整包模式：扫描所有子 skill 的 shard
            for skill_subdir in sorted(self._skill_dir.iterdir()):
                if not skill_subdir.is_dir():
                    continue
                # 跳过非 skill 目录（如 __pycache__）
                if skill_subdir.name.startswith("_") or skill_subdir.name.startswith("."):
                    continue
                for shard_dir in _candidate_dirs(skill_subdir):
                    _try_add(shard_dir, "official")
        else:
            # 1. self
            if self._skill_dir:
                for shard_dir in _candidate_dirs(self._skill_dir):
                    _try_add(shard_dir, "official")

            # 2. siblings（同一 skills/ 目录下的其他 skill）- 仅当 include_siblings=True 时
            if include_siblings and self._skill_dir:
                skills_root = self._skill_dir.parent
                if skills_root.exists():
                    for sibling in sorted(skills_root.iterdir()):
                        if sibling == self._skill_dir or not sibling.is_dir():
                            continue
                        # 跳过 skills/ 目录本身
                        if sibling.name == "skills":
                            continue
                        for shard_dir in _candidate_dirs(sibling):
                            _try_add(shard_dir, "official")

        # 3. docs/self_docs（推荐路径）
        if include_self_docs and self._skill_dir:
            agent_root = self._skill_dir.parent.parent
            bundle_self_docs = agent_root / "docs" / "self_docs"
            if bundle_self_docs.exists():
                self._last_notices.extend(ensure_self_docs_shard(bundle_self_docs))
                _try_add(bundle_self_docs, "self_docs")

        return shards

    def get_last_notices(self) -> list[str]:
        return list(self._last_notices)

    # ---- 聚合检索 ----

    def search(
        self,
        query: str,
        top_k: int = 10,
        domains: Optional[List[str]] = None,
        mode: Literal["keyword", "hybrid", "vector"] = "hybrid",
    ) -> SearchResults:
        """
        聚合检索。

        Args:
            query:   检索查询字符串
            top_k:   返回结果数量
            domains: 按领域标签过滤分片（None 则查所有分片）
            mode:    keyword=FTS5 only, vector=embedding only, hybrid=两者 RRF 融合（默认）

        Returns:
            融合排序后的结果列表
        """
        self._last_notices = []
        shards = self.discover()
        if domains:
            shards = [s for s in shards if set(s.domains) & set(domains)]

        if not shards:
            return []

        weights = {s.shard_id: self._weight(s, query, domains) for s in shards}
        all_hits: List[List[SearchHit]] = []

        mode_effective = mode
        vector_enabled = self._is_vector_enabled()
        if mode in ("vector", "hybrid") and not vector_enabled:
            self._warn_vector_disabled()
            mode_effective = "keyword"

        query_embedding = None
        if mode_effective in ("vector", "hybrid"):
            query_embedding = self._embed(query)

        for shard in shards:
            hits: list[SearchHit] = []
            if mode_effective in ("keyword", "hybrid"):
                hits.extend(shard.search_fts(query, top_k=top_k * 2))
            if mode_effective in ("vector", "hybrid") and query_embedding:
                hits.extend(shard.search_vector(query_embedding, top_k=top_k * 2))
            if shard.graph_config.enabled:
                hits.extend(shard.search_graph(query, top_k=top_k * 2))
            if hits:
                hits.sort(key=lambda h: -h.score)
                for i, h in enumerate(hits):
                    h.rank = i + 1
                all_hits.append(hits)

        if not all_hits:
            return []
        results = weighted_rrf(all_hits, weights=weights, top_k=top_k)
        if self._last_notices:
            for item in results:
                notices = item.metadata.get("system_notices", [])
                item.metadata["system_notices"] = notices + self._last_notices
        return results

    # ---- P3: 分层检索引擎 ----

    def search_with_hint(
        self,
        query: str,
        top_k: int = 10,
        *,
        hint: Optional[RetrievalHint] = None,
        parsed_query: Optional["ParsedQuery"] = None,
        deduplicate: bool = True,
        similarity_threshold: float = 0.8,
        include_siblings: bool = True,
    ) -> SearchResults:
        """
        分层检索引擎（P3+P4）：基于 RetrievalHint 的三阶段检索 + 语义去重。

        阶段1 — 窄域：由 hint.primary_domains 驱动
        阶段2 — 扩域：扩展到 hint.expand_domains（兄弟 skill 补充）
        阶段3 — 全量兜底：所有分片参与（含 self_docs）

        Args:
            query: 原始查询字符串
            top_k: 返回结果数量
            hint: RetrievalHint（可选，不传则退化为 v4 行为）
            parsed_query: ParsedQuery（可选，用于分离符号/自然语言查询）
            deduplicate: 是否启用语义去重（默认 True）
            similarity_threshold: 语义去重相似度阈值（默认 0.8）
            include_siblings: 是否包含兄弟 skill 的 shards（默认 True，整包模式）

        Returns:
            融合排序后的结果列表
        """
        self._last_notices = []

        # 向后兼容：不传 hint 时退化为原 search()
        if hint is None:
            return self.search(query, top_k=top_k)

        # 1. 发现所有分片（考虑 hint.self_docs_enabled 和 include_siblings）
        include_self_docs = hint.self_docs_enabled if hint else True
        all_shards = self.discover(include_self_docs=include_self_docs, include_siblings=include_siblings)
        if not all_shards:
            return []

        # 2. 过滤出 primary_domains 对应的分片（阶段1）
        primary_shards = self._filter_by_domains(all_shards, hint.primary_domains)

        # 3. 阶段1：窄域检索
        results, stage = self._stage_search(
            primary_shards, query, top_k, hint, parsed_query, "stage1"
        )

        # 4. 如果阶段1结果不足，扩到阶段2
        if stage == "stage1" and len(results) < top_k and hint.expand_domains:
            expand_shards = self._filter_by_domains(all_shards, hint.expand_domains)
            # 避免重复搜索已在阶段1出现过的 shard
            existing_ids = {s.shard_id for s in primary_shards}
            expand_shards = [s for s in expand_shards if s.shard_id not in existing_ids]

            if expand_shards:
                stage1_results = results
                results2, _ = self._stage_search(
                    expand_shards, query, top_k, hint, parsed_query, "stage2"
                )
                # 合并阶段1和阶段2结果
                combined = stage1_results + results2
                results = self._rerank_combined(combined, top_k, hint)
                stage = "stage2"

        # 5. 如果阶段2仍不足，进入阶段3全量兜底
        if len(results) < top_k:
            # 计算已经检索过的 shard IDs
            searched_ids = {s.shard_id for s in primary_shards}
            if stage == "stage2":
                searched_ids.update(s.shard_id for s in expand_shards)
            
            # 剩余未检索的 shards（Stage3 全量兜底）
            remaining_shards = [s for s in all_shards if s.shard_id not in searched_ids]
            
            if remaining_shards:
                stage3_results, _ = self._stage_search(
                    remaining_shards, query, top_k, hint, parsed_query, "stage3"
                )
                # 合并之前所有阶段的结果
                combined = results + stage3_results
                results = self._rerank_combined(combined, top_k, hint)
                stage = "stage3"

        # P4: 语义去重
        if deduplicate and results:
            original_count = len(results)
            results = semantic_deduplicate(results, similarity_threshold)
            # 如果去掉了结果，可能需要补充一轮（但本版本暂不补）

        # 记录检索阶段信息
        for item in results:
            item.metadata["retrieval_stage"] = stage
            if self._last_notices:
                notices = item.metadata.get("system_notices", [])
                item.metadata["system_notices"] = notices + self._last_notices

        return results

    def _filter_by_domains(self, shards: List[Shard], domains: List[str]) -> List[Shard]:
        """按 domain 标签过滤分片"""
        if not domains:
            return shards
        return [s for s in shards if set(s.domains) & set(domains)]

    def _stage_search(
        self,
        shards: List[Shard],
        query: str,
        top_k: int,
        hint: RetrievalHint,
        parsed_query: Optional["ParsedQuery"],
        stage_name: str,
    ) -> tuple[SearchResults, str]:
        """
        执行单阶段检索（内部方法）。

        Returns:
            (results, stage_name)
        """
        if not shards:
            return [], stage_name

        # 计算分片权重
        weights = {s.shard_id: self._weight(s, query, hint.primary_domains) for s in shards}
        all_hits: List[List[SearchHit]] = []

        # 决定是否启用向量
        mode_effective = hint.mode
        vector_enabled = self._is_vector_enabled()
        if hint.mode in ("vector", "hybrid") and not vector_enabled:
            mode_effective = "keyword"

        query_embedding = None
        if mode_effective in ("vector", "hybrid"):
            query_embedding = self._embed(query)

        for shard in shards:
            hits: List[SearchHit] = []

            # 按通道检索
            for channel in hint.channels:
                if channel == "api" and parsed_query and parsed_query.symbols:
                    # API 精确通道：只对符号查询生效
                    for symbol in parsed_query.symbols:
                        channel_hits = shard.search_api(symbol, top_k=top_k * 2)
                        for h in channel_hits:
                            h.metadata["channel"] = "api"
                        hits.extend(channel_hits)

                elif channel == "fts" and mode_effective in ("keyword", "hybrid"):
                    channel_hits = shard.search_fts(query, top_k=top_k * 2)
                    for h in channel_hits:
                        h.metadata["channel"] = "fts"
                    hits.extend(channel_hits)

                elif channel == "vector" and mode_effective in ("vector", "hybrid") and query_embedding:
                    channel_hits = shard.search_vector(query_embedding, top_k=top_k * 2)
                    for h in channel_hits:
                        h.metadata["channel"] = "vector"
                    hits.extend(channel_hits)

                elif channel == "graph" and shard.graph_config.enabled:
                    channel_hits = shard.search_graph(query, top_k=top_k * 2)
                    for h in channel_hits:
                        h.metadata["channel"] = "graph"
                    hits.extend(channel_hits)

            if hits:
                # 应用通道权重
                for h in hits:
                    channel = h.metadata.get("channel", "fts")
                    w = hint.channel_weights.get(channel, 1.0)
                    h.score = h.score * w

                # 按分数排序并重编 rank
                hits.sort(key=lambda h: -h.score)
                for i, h in enumerate(hits):
                    h.rank = i + 1
                all_hits.append(hits)

        if not all_hits:
            return [], stage_name

        # 使用带 rrf_k 的 RRF 融合
        results = weighted_rrf(all_hits, weights=weights, k=hint.rrf_k, top_k=top_k)

        # v4.2 设计 §6.1：融合后再乘通道权重（跨通道再调优）
        for r in results:
            # 取 source_hits 中最高分的通道作为代表
            primary_channel = "fts"
            max_score = 0.0
            for h in r.source_hits:
                ch = h.metadata.get("channel", "fts")
                w = hint.channel_weights.get(ch, 1.0)
                if w * h.score > max_score:
                    max_score = w * h.score
                    primary_channel = ch
            channel_weight = hint.channel_weights.get(primary_channel, 1.0)
            r.rrf_score = r.rrf_score * channel_weight
            r.metadata["primary_channel"] = primary_channel

        # 重新排序（融合后权重可能改变顺序）
        results.sort(key=lambda r: -r.rrf_score)
        for i, r in enumerate(results):
            r.metadata["final_rank"] = i + 1

        return results, stage_name

    def _rerank_combined(
        self,
        combined: SearchResults,
        top_k: int,
        hint: RetrievalHint,
    ) -> SearchResults:
        """合并多阶段结果后重新排序"""
        # RankedResult 仅有 rrf_score，无 .score / .rank
        combined.sort(key=lambda r: -r.rrf_score)
        for i, r in enumerate(combined):
            r.metadata["merge_rank"] = i + 1
        return combined[:top_k]

    def search_api(
        self,
        symbol: str,
        top_k: int = 10,
        domains: Optional[List[str]] = None,
        *,
        domain: str = "",
        include_aliases: bool = False,
    ) -> SearchResults:
        """
        跨分片 API 符号检索：基于各分片 ``chunks.api_symbols``（JSON）与 ``api_ref`` 类型。

        Args:
            symbol:  API 名，如 ``hipMalloc``。
            top_k:   融合后返回条数。
            domains: 与 :meth:`search` 相同，按 manifest 的 ``domains`` 过滤分片。
            domain:  可选，在 SQL 层再按 ``chunks.domain`` 过滤。
            include_aliases: 无严格 JSON 片段命中时，是否用宽松 LIKE 回退（同 migrator）。
        """
        self._last_notices = []
        shards = self.discover()
        if domains:
            shards = [s for s in shards if set(s.domains) & set(domains)]

        if not shards:
            return []

        weights = {s.shard_id: self._weight(s, symbol, domains) for s in shards}
        all_hits: List[List[SearchHit]] = []

        for shard in shards:
            hits = shard.search_api(
                symbol,
                top_k=top_k * 2,
                domain=domain,
                include_aliases=include_aliases,
            )
            if hits:
                hits.sort(key=lambda h: -h.score)
                for i, h in enumerate(hits):
                    h.rank = i + 1
                all_hits.append(hits)

        if not all_hits:
            return []
        results = weighted_rrf(all_hits, weights=weights, top_k=top_k)
        if self._last_notices:
            for item in results:
                notices = item.metadata.get("system_notices", [])
                item.metadata["system_notices"] = notices + self._last_notices
        return results

    # ---- 权重计算 ----

    def _weight(
        self,
        shard: Shard,
        query: str = "",
        domains: Optional[List[str]] = None,
    ) -> float:
        """w = base_weight × priority_factor × domain_boost（§6.5）"""
        cfg_weights = self._config.get("shard_weights", {}).get("base", {})
        base = cfg_weights.get(shard.source_type, _BASE_WEIGHTS.get(shard.source_type, 1.0))
        priority_factor = shard.priority
        domain_boost_value = (
            self._config.get("shard_weights", {}).get("domain_boost", _DOMAIN_BOOST)
        )
        boost = domain_boost_value if (domains and set(shard.domains) & set(domains)) else 1.0
        return base * priority_factor * boost

    # ---- 嵌入生成（可替换） ----

    def _is_vector_enabled(self) -> bool:
        package_cap = bool(self._config.get("package_capabilities", {}).get("vectorization", False))
        runtime_enabled = bool(
            self._config.get("knowledge_base", {})
            .get("vectorization", {})
            .get("enabled", False)
        )
        return package_cap and runtime_enabled

    def _warn_vector_disabled(self) -> None:
        global _VECTOR_WARNED
        message = "vectorization is disabled; fallback to keyword mode."
        package_cap = bool(self._config.get("package_capabilities", {}).get("vectorization", False))
        runtime_enabled = bool(
            self._config.get("knowledge_base", {})
            .get("vectorization", {})
            .get("enabled", False)
        )
        if runtime_enabled and not package_cap:
            message = (
                "vectorization.enabled=true but current package capability is disabled; "
                "fallback to keyword mode."
            )
        self._last_notices.append(message)
        if _VECTOR_WARNED:
            return
        _VECTOR_WARNED = True
        warnings.warn(message, RuntimeWarning, stacklevel=3)

    def _embed(self, text: str) -> Optional[list[float]]:
        """
        生成查询文本的嵌入向量。
        优先使用 sentence-transformers；不可用时返回 None（降级到 keyword-only）。
        """
        vec_cfg = self._config.get("knowledge_base", {}).get("vectorization", {}) or {}
        provider = str(vec_cfg.get("provider", "openai_compat"))
        if provider == "openai_compat":
            api_url = str(vec_cfg.get("api_url", ""))
            api_key = str(vec_cfg.get("api_key", ""))
            model_name = str(vec_cfg.get("model", "qwen3-embedding"))
            if not api_url:
                return None
            payload = json.dumps({"input": [text], "model": model_name}).encode("utf-8")
            req = urllib.request.Request(
                api_url,
                data=payload,
                headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
                method="POST",
            )
            try:
                with urllib.request.urlopen(req, timeout=30) as resp:
                    data = json.loads(resp.read().decode("utf-8"))
                return data["data"][0]["embedding"]
            except Exception:
                return None
        if provider == "local_st":
            try:
                from sentence_transformers import SentenceTransformer

                model_name = str(vec_cfg.get("model", "all-MiniLM-L6-v2"))
                model = SentenceTransformer(model_name)
                return model.encode(text).tolist()
            except Exception:
                return None
        return None
