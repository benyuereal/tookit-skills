"""Shard 数据类 —— 单个知识分片的描述与检索接口"""
from __future__ import annotations
import hashlib
import json
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple


@dataclass
class SearchHit:
    """单条检索命中"""
    chunk_id: str
    shard_id: str
    content: str
    score: float
    rank: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class GraphConfig:
    """Graph RAG MVP 配置"""
    enabled: bool = False
    max_hops: int = 2
    rel_type_allowlist: List[str] = field(default_factory=lambda: ["maps_to", "depends_on", "see_also"])
    decay: float = 0.5


@dataclass
class Shard:
    """
    单个知识分片的运行时描述。

    对应 kb_shard/ 目录下的 shard.sqlite + shard.manifest.yaml + CHECKSUM。
    """
    shard_id: str
    shard_dir: Path
    domains: list[str] = field(default_factory=list)
    priority: float = 1.0
    source_type: str = "official"   # "official" | "user_shard" | "self_docs"
    graph_config: GraphConfig = field(default_factory=GraphConfig)

    # ---- 懒加载 ----
    _conn: Optional[sqlite3.Connection] = field(default=None, repr=False, compare=False)

    @classmethod
    def from_dir(cls, shard_dir: Path) -> "Shard":
        """从 kb_shard/ 目录加载 shard 元数据"""
        import yaml

        manifest_path = shard_dir / "shard.manifest.yaml"
        if not manifest_path.exists():
            raise FileNotFoundError(f"shard.manifest.yaml not found in {shard_dir}")

        with open(manifest_path, encoding="utf-8") as f:
            manifest = yaml.safe_load(f)

        graph_mvp = manifest.get("graph_mvp", {})
        graph_config = GraphConfig(
            enabled=bool(graph_mvp.get("enabled", False)),
            max_hops=int(graph_mvp.get("default", {}).get("max_hops", 2)),
            rel_type_allowlist=graph_mvp.get("default", {}).get("rel_type_allowlist", ["maps_to", "depends_on", "see_also"]),
            decay=float(graph_mvp.get("default", {}).get("decay", 0.5)),
        )

        return cls(
            shard_id=manifest.get("shard_id", shard_dir.name),
            shard_dir=shard_dir,
            domains=manifest.get("domains", []),
            priority=float(manifest.get("priority", 1.0)),
            source_type=manifest.get("source_type", "official"),
            graph_config=graph_config,
        )

    def is_ready(self) -> bool:
        """检查 shard.sqlite 是否存在且 checksum 通过"""
        sqlite_path = self.shard_dir / "shard.sqlite"
        if not sqlite_path.exists():
            return False
        # 已打开的连接：避免 SQLite WAL/旁路文件导致主库字节与 CHECKSUM 不一致而误判未就绪
        if self._conn is not None:
            return True
        checksum_path = self.shard_dir / "CHECKSUM"
        if checksum_path.exists():
            expected = checksum_path.read_text(encoding="utf-8").strip()
            actual = hashlib.sha256(sqlite_path.read_bytes()).hexdigest()
            if expected and expected != actual:
                import warnings
                warnings.warn(f"Shard '{self.shard_id}' checksum mismatch, skipping.")
                return False
        return True

    def _get_conn(self) -> sqlite3.Connection:
        if self._conn is None:
            sqlite_path = self.shard_dir / "shard.sqlite"
            self._conn = sqlite3.connect(str(sqlite_path), check_same_thread=False)
            self._conn.row_factory = sqlite3.Row
        return self._conn

    def search_fts(self, query: str, top_k: int = 20) -> list[SearchHit]:
        """FTS5 BM25 关键词检索"""
        if not self.is_ready():
            return []
        try:
            conn = self._get_conn()
            cur = conn.execute(
                """
                SELECT c.chunk_id, c.content, bm25(chunks_fts) AS score
                FROM chunks_fts
                JOIN chunks c ON chunks_fts.rowid = c.rowid
                WHERE chunks_fts MATCH ?
                ORDER BY score
                LIMIT ?
                """,
                (query, top_k),
            )
            rows = cur.fetchall()
            return [
                SearchHit(
                    chunk_id=row["chunk_id"],
                    shard_id=self.shard_id,
                    content=row["content"],
                    score=float(-row["score"]),   # bm25 返回负数，取反
                    rank=i + 1,
                )
                for i, row in enumerate(rows)
            ]
        except Exception:
            return []

    def search_api(
        self,
        symbol: str,
        top_k: int = 20,
        *,
        domain: str = "",
        include_aliases: bool = False,
    ) -> list[SearchHit]:
        """
        在 api_symbols（JSON 数组）中按 API 名检索，默认仅 chunk_type=api_ref。

        与 dcu-code-migrator 的 lookup / search --mode api 一致：
        先对 JSON 串做带双引号片段的匹配（等价于 ``LIKE '%\"symbol\"%'``），
        无命中且 ``include_aliases`` 时再对 ``symbol`` 做宽松子串 ``LIKE``。

        说明：在部分环境下将 ``%"symbol"%`` 整串作为单个 SQL 参数绑定会匹配失败；
        因此使用 ``LIKE ('%' || char(34) || ? || char(34) || '%')`` 只绑定纯符号，语义一致。
        """
        if not self.is_ready():
            return []
        sym = symbol.strip()
        if not sym:
            return []
        try:
            conn = self._get_conn()
            col_names = {row["name"] for row in conn.execute("PRAGMA table_info(chunks)").fetchall()}
            if "api_symbols" not in col_names:
                return []

            order_by = "position" if "position" in col_names else "chunk_id"
            has_type = "chunk_type" in col_names
            has_domain = "domain" in col_names

            def _run(*, use_json_quotes: bool) -> list[sqlite3.Row]:
                if use_json_quotes:
                    like_expr = "api_symbols LIKE ('%' || char(34) || ? || char(34) || '%')"
                    bind_params = [sym]
                else:
                    like_expr = "api_symbols LIKE ('%' || ? || '%')"
                    bind_params = [sym]
                sql = f"SELECT chunk_id, content, api_symbols FROM chunks WHERE {like_expr}"
                if has_type:
                    sql += " AND chunk_type = 'api_ref'"
                if domain and has_domain:
                    sql += " AND domain = ?"
                    bind_params.append(domain)
                sql += f" ORDER BY {order_by} LIMIT ?"
                bind_params.append(top_k)
                cur = conn.execute(sql, bind_params)
                return cur.fetchall()

            rows = _run(use_json_quotes=True)
            if not rows and include_aliases:
                rows = _run(use_json_quotes=False)

            hits: list[SearchHit] = []
            for i, row in enumerate(rows):
                raw = row["api_symbols"] or ""
                try:
                    syms = json.loads(raw) if raw else []
                except json.JSONDecodeError:
                    syms = []
                if isinstance(syms, list):
                    score = 1.0 if sym in syms else 0.5
                else:
                    score = 0.5
                hits.append(
                    SearchHit(
                        chunk_id=row["chunk_id"],
                        shard_id=self.shard_id,
                        content=row["content"],
                        score=score,
                        rank=i + 1,
                        metadata={"channel": "api"},
                    )
                )
            return hits
        except Exception:
            return []

    def search_vector(self, query_embedding: list[float], top_k: int = 20) -> list[SearchHit]:
        """向量余弦相似度检索（需要 faiss 或自定义实现）"""
        if not self.is_ready():
            return []
        try:
            import numpy as np
            conn = self._get_conn()
            cols = {row["name"] for row in conn.execute("PRAGMA table_info(embeddings)").fetchall()}
            vector_col = "embedding" if "embedding" in cols else ("vector" if "vector" in cols else None)
            if not vector_col:
                return []
            cur = conn.execute(f"SELECT chunk_id, content, {vector_col} AS emb FROM embeddings")
            rows = cur.fetchall()
            if not rows:
                return []

            q = np.array(query_embedding, dtype="float32")
            q_norm = q / (np.linalg.norm(q) + 1e-9)

            scored: List[Tuple[float, str, str]] = []
            for row in rows:
                emb = np.frombuffer(row["emb"], dtype="float32")
                emb_norm = emb / (np.linalg.norm(emb) + 1e-9)
                score = float(np.dot(q_norm, emb_norm))
                scored.append((score, row["chunk_id"], row["content"]))

            scored.sort(key=lambda x: -x[0])
            return [
                SearchHit(
                    chunk_id=chunk_id,
                    shard_id=self.shard_id,
                    content=content,
                    score=score,
                    rank=i + 1,
                )
                for i, (score, chunk_id, content) in enumerate(scored[:top_k])
            ]
        except Exception:
            return []

    def search_graph(self, query: str, top_k: int = 20) -> list[SearchHit]:
        """
        Graph RAG 检索：实体链接 → 子图扩展 → chunk 收集。

        返回的 SearchHit.metadata 包含 channel="graph" 和可选的 graph_trace。
        """
        if not self.graph_config.enabled or not self.is_ready():
            return []

        try:
            conn = self._get_conn()
            tables = {row[0] for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()}
            required_tables = {"kg_entities", "kg_relations", "kg_entity_chunk", "kg_symbol_index"}
            if not required_tables.issubset(tables):
                return []

            seed_entities = self._link_entities(conn, query)
            if not seed_entities:
                return []

            chunk_scores = self._expand_subgraph(conn, seed_entities, top_k)
            if not chunk_scores:
                return []

            hits: List[SearchHit] = []
            for i, (chunk_id, score, trace) in enumerate(chunk_scores[:top_k]):
                content = self._get_chunk_content(conn, chunk_id)
                if content is not None:
                    hits.append(SearchHit(
                        chunk_id=chunk_id,
                        shard_id=self.shard_id,
                        content=content,
                        score=score,
                        rank=i + 1,
                        metadata={"channel": "graph", "graph_trace": trace},
                    ))
            return hits
        except Exception:
            return []

    def _link_entities(self, conn: sqlite3.Connection, query: str, max_entities: int = 3) -> List[str]:
        """从 query 中提取 token 并链接到实体 ID"""
        import re
        tokens = set(re.findall(r"[a-zA-Z_][a-zA-Z0-9_]*", query.lower()))
        if not tokens:
            return []

        placeholders = ",".join("?" for _ in tokens)
        cur = conn.execute(
            f"SELECT DISTINCT entity_id FROM kg_symbol_index WHERE token IN ({placeholders})",
            list(tokens),
        )
        entity_ids = [row[0] for row in cur.fetchall()]
        return entity_ids[:max_entities]

    def _expand_subgraph(
        self,
        conn: sqlite3.Connection,
        seed_entities: List[str],
        top_k: int,
    ) -> List[Tuple[str, float, Dict[str, Any]]]:
        """BFS 扩展子图，收集 chunk_id 并计算 score"""
        max_hops = self.graph_config.max_hops
        decay = self.graph_config.decay
        allowlist = set(self.graph_config.rel_type_allowlist)

        visited_entities: set[str] = set()
        chunk_scores: Dict[str, Tuple[float, Dict[str, Any]]] = {}
        frontier = [(eid, 0) for eid in seed_entities]

        while frontier:
            entity_id, hop = frontier.pop(0)
            if entity_id in visited_entities:
                continue
            visited_entities.add(entity_id)

            base_score = decay ** hop

            cur = conn.execute(
                "SELECT chunk_id, link_type, weight FROM kg_entity_chunk WHERE entity_id = ?",
                (entity_id,),
            )
            for row in cur.fetchall():
                chunk_id = row[0]
                weight = row[2] if row[2] else 1.0
                score = base_score * weight
                trace = {"entity_id": entity_id, "link_type": row[1], "hop": hop}
                if chunk_id not in chunk_scores or chunk_scores[chunk_id][0] < score:
                    chunk_scores[chunk_id] = (score, trace)

            if hop < max_hops:
                cur = conn.execute(
                    "SELECT tail_id, rel_type, weight, evidence_chunk_id FROM kg_relations WHERE head_id = ?",
                    (entity_id,),
                )
                for row in cur.fetchall():
                    tail_id, rel_type, weight, evidence_chunk_id = row
                    if rel_type not in allowlist:
                        continue
                    if tail_id not in visited_entities:
                        frontier.append((tail_id, hop + 1))
                    if evidence_chunk_id:
                        ew = weight if weight else 1.0
                        score = base_score * ew
                        trace = {"entity_id": entity_id, "rel_type": rel_type, "hop": hop}
                        if evidence_chunk_id not in chunk_scores or chunk_scores[evidence_chunk_id][0] < score:
                            chunk_scores[evidence_chunk_id] = (score, trace)

        result = [(cid, score, trace) for cid, (score, trace) in chunk_scores.items()]
        result.sort(key=lambda x: -x[1])
        return result[:top_k]

    def _get_chunk_content(self, conn: sqlite3.Connection, chunk_id: str) -> Optional[str]:
        """从 chunks 表获取 content"""
        cur = conn.execute("SELECT content FROM chunks WHERE chunk_id = ?", (chunk_id,))
        row = cur.fetchone()
        return row[0] if row else None
