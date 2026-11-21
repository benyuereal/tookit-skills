"""Self docs auto-indexer.

Users can drop raw docs into docs/self_docs, and this module builds/refreshes
the local shard files in place before retrieval.
"""
from __future__ import annotations

import hashlib
import json
import re
import sqlite3
import warnings
from pathlib import Path
from typing import Iterable

import yaml

SUPPORTED_EXTS = {
    ".md",
    ".markdown",
    ".txt",
    ".rst",
    ".adoc",
    ".pdf",
}

INDEX_META_FILE = ".index_meta.json"
AUTO_INDEX_MAX_FILES = 10
_AUTO_INDEX_WARNED_DIRS: set[str] = set()
_AUTO_INDEX_MODE_WARNED_DIRS: set[str] = set()


def _iter_doc_files(self_docs_dir: Path) -> list[Path]:
    files: list[Path] = []
    if not self_docs_dir.exists():
        return files
    for p in sorted(self_docs_dir.rglob("*")):
        if not p.is_file():
            continue
        if p.name in {"shard.sqlite", "shard.manifest.yaml", "CHECKSUM", INDEX_META_FILE}:
            continue
        if p.suffix.lower() in SUPPORTED_EXTS:
            files.append(p)
    return files


def _read_text(path: Path) -> str:
    if path.suffix.lower() == ".pdf":
        return _read_pdf_text(path)
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return path.read_text(encoding="utf-8", errors="ignore")


def _read_pdf_text(path: Path) -> str:
    try:
        from pypdf import PdfReader
    except ImportError:
        warnings.warn(
            (
                f"Skip PDF file {path.name}: pypdf is not installed. "
                "Install dependency and rerun build_self_docs_shard."
            ),
            RuntimeWarning,
            stacklevel=2,
        )
        return ""

    try:
        reader = PdfReader(str(path))
    except Exception as exc:
        warnings.warn(
            f"Skip PDF file {path.name}: failed to open PDF ({exc}).",
            RuntimeWarning,
            stacklevel=2,
        )
        return ""

    pages: list[str] = []
    for page in reader.pages:
        try:
            pages.append(page.extract_text() or "")
        except Exception:
            pages.append("")
    return "\n".join(pages).strip()


def _token_count(text: str) -> int:
    return len(re.findall(r"\S+", text))


def _split_chunks(text: str, max_chars: int = 1800, overlap_chars: int = 200) -> list[str]:
    if not text.strip():
        return []
    if len(text) <= max_chars:
        return [text.strip()]

    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = min(len(text), start + max_chars)
        chunks.append(text[start:end].strip())
        if end >= len(text):
            break
        start = max(0, end - overlap_chars)
    return [c for c in chunks if c]


def _build_keywords(text: str, filename: str) -> str:
    src = (filename + "\n" + text).lower()
    tokens = re.findall(r"[a-z0-9_]{4,}", src)
    counts: dict[str, int] = {}
    for t in tokens:
        counts[t] = counts.get(t, 0) + 1
    top = sorted(counts.items(), key=lambda x: (-x[1], x[0]))[:12]
    return ",".join(k for k, _ in top)


def _snapshot(files: Iterable[Path], base_dir: Path) -> dict:
    return {
        str(p.relative_to(base_dir)).replace("\\", "/"): p.stat().st_mtime_ns
        for p in files
    }


def _needs_rebuild(self_docs_dir: Path, files: list[Path]) -> bool:
    sqlite_path = self_docs_dir / "shard.sqlite"
    manifest_path = self_docs_dir / "shard.manifest.yaml"
    meta_path = self_docs_dir / INDEX_META_FILE
    if not sqlite_path.exists() or not manifest_path.exists() or not meta_path.exists():
        return True

    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except Exception:
        return True
    return meta.get("files") != _snapshot(files, self_docs_dir)


def _ensure_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
        DROP TABLE IF EXISTS chunks;
        DROP TABLE IF EXISTS chunks_fts;
        CREATE TABLE chunks (
            chunk_id TEXT PRIMARY KEY,
            doc_id TEXT NOT NULL,
            title TEXT,
            heading_path TEXT,
            content TEXT NOT NULL,
            token_count INTEGER DEFAULT 0,
            chunk_type TEXT DEFAULT 'guide',
            api_symbols TEXT DEFAULT '[]',
            keywords TEXT DEFAULT '',
            source_type TEXT DEFAULT 'self_docs',
            role TEXT DEFAULT 'guide',
            domain TEXT DEFAULT 'self-docs',
            version TEXT DEFAULT 'local',
            position INTEGER DEFAULT 0
        );
        CREATE VIRTUAL TABLE chunks_fts USING fts5(
            chunk_id,
            title,
            content,
            keywords
        );
        """
    )


def _build_shard(self_docs_dir: Path, files: list[Path]) -> dict[str, int]:
    sqlite_path = self_docs_dir / "shard.sqlite"
    manifest_path = self_docs_dir / "shard.manifest.yaml"
    checksum_path = self_docs_dir / "CHECKSUM"
    meta_path = self_docs_dir / INDEX_META_FILE

    conn = sqlite3.connect(str(sqlite_path))
    try:
        _ensure_schema(conn)
        rows: list[tuple] = []
        for position, path in enumerate(files):
            text = _read_text(path)
            chunks = _split_chunks(text)
            if not chunks:
                continue
            rel = str(path.relative_to(self_docs_dir)).replace("\\", "/")
            doc_hash = hashlib.sha1(rel.encode("utf-8")).hexdigest()[:10]
            doc_id = f"self-doc-{doc_hash}"
            title = path.stem
            keywords = _build_keywords(text, path.name)
            for local_idx, chunk_text in enumerate(chunks):
                chunk_id = f"{doc_id}_c{local_idx:04d}"
                rows.append(
                    (
                        chunk_id,
                        doc_id,
                        title,
                        json.dumps([], ensure_ascii=False),
                        chunk_text,
                        _token_count(chunk_text),
                        "guide",
                        "[]",
                        keywords,
                        "self_docs",
                        "guide",
                        "self-docs",
                        "local",
                        position,
                    )
                )

        if rows:
            conn.executemany(
                """
                INSERT INTO chunks (
                    chunk_id, doc_id, title, heading_path, content, token_count, chunk_type,
                    api_symbols, keywords, source_type, role, domain, version, position
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                rows,
            )
            conn.executemany(
                "INSERT INTO chunks_fts (chunk_id, title, content, keywords) VALUES (?, ?, ?, ?)",
                [(r[0], r[2], r[4], r[8]) for r in rows],
            )
        conn.commit()
    finally:
        conn.close()

    manifest = {
        "shard_id": "self-docs",
        "owner_skill": "self-docs",
        "source_type": "self_docs",
        "priority": 0.8,
        "domains": ["self-docs"],
        "embedding": {"model": "none", "dim": 0},
        "stats": {"chunk_count": len(rows), "doc_count": len(files)},
        "source_layer": "self_docs",
    }
    manifest_path.write_text(yaml.safe_dump(manifest, sort_keys=False, allow_unicode=True), encoding="utf-8")
    checksum_path.write_text(hashlib.sha256(sqlite_path.read_bytes()).hexdigest(), encoding="utf-8")
    meta_path.write_text(
        json.dumps({"files": _snapshot(files, self_docs_dir)}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return {"chunk_count": len(rows), "doc_count": len(files)}


def build_self_docs_shard(self_docs_dir: Path, force: bool = False) -> dict[str, int]:
    """Build or refresh docs/self_docs shard explicitly.

    This is a manual/offline entrypoint and does not apply AUTO_INDEX_MAX_FILES.
    """
    if not self_docs_dir.exists():
        return {"chunk_count": 0, "doc_count": 0}

    files = _iter_doc_files(self_docs_dir)
    if not force and not _needs_rebuild(self_docs_dir, files):
        return {"chunk_count": 0, "doc_count": len(files)}
    return _build_shard(self_docs_dir, files)


def ensure_self_docs_shard(self_docs_dir: Path) -> list[str]:
    """Build/refresh docs/self_docs shard from raw user docs.

    Auto-index is intentionally capped for latency predictability. When the
    number of source docs exceeds AUTO_INDEX_MAX_FILES, users should prebuild
    chunks offline and place shard artifacts in docs/self_docs directly.
    """
    notices: list[str] = []
    if not self_docs_dir.exists():
        return notices

    files = _iter_doc_files(self_docs_dir)
    if len(files) > AUTO_INDEX_MAX_FILES:
        key = str(self_docs_dir.resolve())
        mode_notice = (
            "docs/self_docs auto chunking is disabled when source documents exceed "
            f"{AUTO_INDEX_MAX_FILES}; manual shard mode is active."
        )
        if key not in _AUTO_INDEX_MODE_WARNED_DIRS:
            _AUTO_INDEX_MODE_WARNED_DIRS.add(key)
            warnings.warn(mode_notice, RuntimeWarning, stacklevel=2)
            notices.append(mode_notice)

        if _needs_rebuild(self_docs_dir, files):
            if key not in _AUTO_INDEX_WARNED_DIRS:
                _AUTO_INDEX_WARNED_DIRS.add(key)
            rebuild_notice = (
                "docs/self_docs contains more than "
                f"{AUTO_INDEX_MAX_FILES} source documents; "
                "auto chunking is disabled. Run "
                "'python scripts/build_self_docs_shard.py' "
                "to rebuild shard.sqlite before querying."
            )
            warnings.warn(rebuild_notice, RuntimeWarning, stacklevel=2)
            notices.append(rebuild_notice)
        return notices

    if not _needs_rebuild(self_docs_dir, files):
        return notices
    _build_shard(self_docs_dir, files)
    return notices
