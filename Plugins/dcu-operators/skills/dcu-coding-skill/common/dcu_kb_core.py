"""
DCU Coding Skill Knowledge Base Core

知识库核心模块，提供 SQLite FTS5 索引和检索功能
"""

import sqlite3
import json
import hashlib
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass

from .utils import Logger, PathHandler, ConfigLoader


@dataclass
class SearchConfig:
    """搜索配置"""
    search_mode: str = "keyword"  # keyword, hybrid, api
    top_k: int = 5
    min_score: float = 0.0
    use_embeddings: bool = False


@dataclass
class SearchResult:
    """搜索结果"""
    chunk_id: str
    content: str
    score: float
    metadata: Dict[str, Any]
    citations: List[str]


class KnowledgeBase:
    """知识库核心类"""

    def __init__(self, path: Union[str, Path], config: Optional[Dict[str, Any]] = None):
        """
        初始化知识库

        Args:
            path: 数据库文件路径
            config: 配置字典
        """
        self.path = Path(path)
        self.config = config or {}
        self.logger = Logger(__name__)

        # 初始化数据库
        self._init_database()
        self.logger.info(f"知识库初始化完成: {self.path}")

    def _init_database(self) -> None:
        """初始化数据库和表结构"""
        PathHandler.ensure_parent(self.path)

        with sqlite3.connect(self.path) as conn:
            cursor = conn.cursor()

            # 创建 documents 表
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS documents (
                    doc_id TEXT PRIMARY KEY,
                    title TEXT NOT NULL,
                    role TEXT NOT NULL,
                    domain TEXT NOT NULL,
                    source_uri TEXT,
                    created_at INTEGER
                )
            """)

            # 创建 chunks 表
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS chunks (
                    chunk_id TEXT PRIMARY KEY,
                    doc_id TEXT NOT NULL,
                    content TEXT NOT NULL,
                    heading TEXT,
                    score REAL,
                    created_at INTEGER,
                    FOREIGN KEY (doc_id) REFERENCES documents(doc_id)
                )
            """)

            # 创建 chunks_fts 表（FTS5 全文搜索）
            cursor.execute("""
                CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
                    chunk_id,
                    content,
                    content=chunks,
                    content_rowid=rowid
                )
            """)

            # 创建 embeddings 表（可选，用于语义搜索）
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS embeddings (
                    chunk_id TEXT PRIMARY KEY,
                    vector BLOB,
                    created_at INTEGER
                )
            """)

            # 创建 source_state 表（用于增量更新）
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS source_state (
                    source_uri TEXT PRIMARY KEY,
                    content_hash TEXT,
                    updated_at INTEGER
                )
            """)

            conn.commit()

        self.logger.debug("数据库表结构初始化完成")

    def index_document(self, doc: Dict[str, Any]) -> bool:
        """
        索引文档

        Args:
            doc: CommonDoc 字典

        Returns:
            是否成功
        """
        try:
            doc_id = doc["doc_id"]

            # 插入文档元数据
            with sqlite3.connect(self.path) as conn:
                cursor = conn.cursor()

                # 插入 documents
                cursor.execute("""
                    INSERT OR REPLACE INTO documents
                    (doc_id, title, role, domain, source_uri, created_at)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, (
                    doc_id,
                    doc.get("title", ""),
                    doc.get("role", "guide"),
                    doc.get("domain", "general"),
                    doc.get("source_uri", ""),
                    int(doc.get("created_at", 0))
                ))

                # 插入 chunks（如果文档有内容）
                for i, chunk in enumerate(doc.get("chunks", [])):
                    chunk_id = f"{doc_id}_chunk_{i}"
                    heading = chunk.get("heading", "")
                    content = chunk.get("content", "")

                    # 插入 chunks 表
                    cursor.execute("""
                        INSERT OR REPLACE INTO chunks
                        (chunk_id, doc_id, content, heading, score, created_at)
                        VALUES (?, ?, ?, ?, ?, ?)
                    """, (
                        chunk_id,
                        doc_id,
                        content,
                        heading,
                        chunk.get("score", 0),
                        int(chunk.get("created_at", 0))
                    ))

                    # 插入 FTS 索引
                    cursor.execute("""
                        INSERT INTO chunks_fts (chunk_id, content)
                        VALUES (?, ?)
                    """, (chunk_id, content))

                # 插入 embeddings（如果有的话）
                for chunk in doc.get("chunks", []):
                    chunk_id = f"{doc_id}_chunk_{doc['chunks'].index(chunk)}"
                    embeddings = chunk.get("embeddings")
                    if embeddings:
                        import numpy as np
                        vector = np.array(embeddings).tobytes()
                        cursor.execute("""
                            INSERT OR REPLACE INTO embeddings
                            (chunk_id, vector, created_at)
                            VALUES (?, ?, ?)
                        """, (chunk_id, vector, int(chunk.get("created_at", 0))))

                conn.commit()

            self.logger.info(f"文档索引完成: {doc_id}")
            return True

        except Exception as e:
            self.logger.error(f"索引文档失败: {doc_id}, 错误: {e}")
            return False

    def search(self, query: str, config: Optional[SearchConfig] = None) -> List[SearchResult]:
        """
        搜索知识库

        Args:
            query: 搜索查询
            config: 搜索配置

        Returns:
            搜索结果列表
        """
        config = config or SearchConfig()
        search_mode = config.search_mode

        if search_mode == "keyword":
            return self._search_keyword(query, config)
        elif search_mode == "hybrid":
            return self._search_hybrid(query, config)
        elif search_mode == "api":
            return self._search_api(query, config)
        else:
            return self._search_keyword(query, config)

    def _search_keyword(self, query: str, config: SearchConfig) -> List[SearchResult]:
        """
        关键词搜索（FTS5）

        Args:
            query: 搜索查询
            config: 搜索配置

        Returns:
            搜索结果列表
        """
        results = []

        try:
            with sqlite3.connect(self.path) as conn:
                cursor = conn.cursor()

                # 使用 FTS5 搜索
                cursor.execute("""
                    SELECT c.chunk_id, c.content, c.heading, d.title, d.domain, d.role
                    FROM chunks_fts c
                    JOIN chunks ON c.chunk_id = chunks.chunk_id
                    JOIN documents d ON chunks.doc_id = d.doc_id
                    WHERE chunks_fts MATCH ?
                    ORDER BY rank
                    LIMIT ?
                """, (query, config.top_k))

                for row in cursor.fetchall():
                    chunk_id, content, heading, title, domain, role = row

                    results.append(SearchResult(
                        chunk_id=chunk_id,
                        content=content,
                        score=1.0,  # FTS5 的 rank 就是 score
                        metadata={
                            "heading": heading,
                            "title": title,
                            "domain": domain,
                            "role": role
                        },
                        citations=[]
                    ))

        except Exception as e:
            self.logger.error(f"关键词搜索失败: {e}")

        return results

    def _search_hybrid(self, query: str, config: SearchConfig) -> List[SearchResult]:
        """
        混合搜索（FTS5 + Embedding RRF）

        Args:
            query: 搜索查询
            config: 搜索配置

        Returns:
            搜索结果列表
        """
        # 暂时只使用关键词搜索
        # TODO: 实现完整的混合搜索
        return self._search_keyword(query, config)

    def _search_api(self, query: str, config: SearchConfig) -> List[SearchResult]:
        """
        API 符号查询

        Args:
            query: API 符号
            config: 搜索配置

        Returns:
            搜索结果列表
        """
        results = []

        try:
            with sqlite3.connect(self.path) as conn:
                cursor = conn.cursor()

                # 查询 API 函数
                cursor.execute("""
                    SELECT c.chunk_id, c.content, c.heading, d.title, d.domain, d.role
                    FROM chunks c
                    JOIN documents d ON c.doc_id = d.doc_id
                    WHERE c.content LIKE ?
                    LIMIT ?
                """, (f"%{query}%", config.top_k))

                for row in cursor.fetchall():
                    chunk_id, content, heading, title, domain, role = row

                    results.append(SearchResult(
                        chunk_id=chunk_id,
                        content=content,
                        score=1.0,
                        metadata={
                            "heading": heading,
                            "title": title,
                            "domain": domain,
                            "role": role
                        },
                        citations=[]
                    ))

        except Exception as e:
            self.logger.error(f"API 查询失败: {e}")

        return results

    def lookup_api(self, api_symbol: str) -> Optional[SearchResult]:
        """
        查询 API 符号

        Args:
            api_symbol: API 符号

        Returns:
            搜索结果或 None
        """
        results = self._search_api(api_symbol, SearchConfig(top_k=1))
        return results[0] if results else None

    def get_document_info(self, doc_id: str) -> Optional[Dict[str, Any]]:
        """
        获取文档信息

        Args:
            doc_id: 文档 ID

        Returns:
            文档信息或 None
        """
        try:
            with sqlite3.connect(self.path) as conn:
                cursor = conn.cursor()

                cursor.execute("""
                    SELECT title, role, domain, source_uri, created_at
                    FROM documents
                    WHERE doc_id = ?
                """, (doc_id,))

                row = cursor.fetchone()
                if row:
                    return {
                        "doc_id": doc_id,
                        "title": row[0],
                        "role": row[1],
                        "domain": row[2],
                        "source_uri": row[3],
                        "created_at": row[4]
                    }

        except Exception as e:
            self.logger.error(f"获取文档信息失败: {doc_id}, 错误: {e}")

        return None

    def get_all_documents(self) -> List[Dict[str, Any]]:
        """
        获取所有文档列表

        Returns:
            文档列表
        """
        documents = []

        try:
            with sqlite3.connect(self.path) as conn:
                cursor = conn.cursor()

                cursor.execute("""
                    SELECT doc_id, title, role, domain, source_uri, created_at
                    FROM documents
                    ORDER BY created_at DESC
                """)

                for row in cursor.fetchall():
                    documents.append({
                        "doc_id": row[0],
                        "title": row[1],
                        "role": row[2],
                        "domain": row[3],
                        "source_uri": row[4],
                        "created_at": row[5]
                    })

        except Exception as e:
            self.logger.error(f"获取文档列表失败: {e}")

        return documents

    def get_chunk_count(self) -> int:
        """
        获取总 chunk 数

        Returns:
            chunk 数量
        """
        try:
            with sqlite3.connect(self.path) as conn:
                cursor = conn.cursor()

                cursor.execute("SELECT COUNT(*) FROM chunks")
                return cursor.fetchone()[0]

        except Exception as e:
            self.logger.error(f"获取 chunk 数量失败: {e}")
            return 0

    def get_document_count(self) -> int:
        """
        获取总文档数

        Returns:
            文档数量
        """
        try:
            with sqlite3.connect(self.path) as conn:
                cursor = conn.cursor()

                cursor.execute("SELECT COUNT(*) FROM documents")
                return cursor.fetchone()[0]

        except Exception as e:
            self.logger.error(f"获取文档数量失败: {e}")
            return 0


def main():
    """命令行入口"""
    import argparse

    parser = argparse.ArgumentParser(description="DCU Coding Skill - 知识库核心")
    parser.add_argument("--init", action="store_true", help="初始化知识库")
    parser.add_argument("--search", type=str, help="搜索查询")
    parser.add_argument("--lookup-api", type=str, help="API 符号查询")
    parser.add_argument("--list-docs", action="store_true", help="列出所有文档")

    args = parser.parse_args()

    kb = KnowledgeBase("./references/kb.sqlite")

    if args.init:
        print("知识库初始化完成")
    elif args.search:
        results = kb.search(args.search)
        for result in results:
            print(f"[{result.chunk_id}] {result.score}: {result.content[:100]}...")
    elif args.lookup_api:
        result = kb.lookup_api(args.lookup_api)
        if result:
            print(f"找到: {result.content}")
        else:
            print("未找到")
    elif args.list_docs:
        docs = kb.get_all_documents()
        for doc in docs:
            print(f"{doc['doc_id']}: {doc['title']} ({doc['domain']})")


if __name__ == "__main__":
    main()
