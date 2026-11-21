"""Hybrid search script for the DCU knowledge base.

Usage:
    python search.py --query "hipMalloc" --top 5
    python search.py --query "how to do matrix multiply" --mode semantic --top 5
    python search.py --query "rocblas gemm" --mode hybrid --top 5
    python search.py --query "hipMalloc" --mode api
    python search.py --query "memory pool" --domain runtime --top 5
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sqlite3
import struct
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
SKILL_DIR = SCRIPT_DIR.parent

sys.path.insert(0, str(SCRIPT_DIR))
from db_paths import iter_search_targets, resolve_primary  # noqa: E402


def _load_embedding_config() -> dict:
    """Load embedding config with priority: env vars > user-config.json > config.json."""
    config: dict = {}

    base_cfg = SKILL_DIR / "references" / "config.json"
    if base_cfg.exists():
        config = json.loads(base_cfg.read_text(encoding="utf-8"))

    user_cfg = SKILL_DIR / "user-config.json"
    if user_cfg.exists():
        user = json.loads(user_cfg.read_text(encoding="utf-8"))
        for key in ("embedding_enabled", "embedding_api_url", "embedding_api_key", "embedding_model"):
            if key in user:
                config[key] = user[key]

    if os.environ.get("DCU_EMBEDDING_API_URL"):
        config["embedding_api_url"] = os.environ["DCU_EMBEDDING_API_URL"]
        config["embedding_enabled"] = True
    if os.environ.get("DCU_EMBEDDING_API_KEY"):
        config["embedding_api_key"] = os.environ["DCU_EMBEDDING_API_KEY"]
    if os.environ.get("DCU_EMBEDDING_MODEL"):
        config["embedding_model"] = os.environ["DCU_EMBEDDING_MODEL"]
    if os.environ.get("DCU_EMBEDDING_ENABLED", "").lower() in ("true", "1", "yes"):
        config["embedding_enabled"] = True
    elif os.environ.get("DCU_EMBEDDING_ENABLED", "").lower() in ("false", "0", "no"):
        config["embedding_enabled"] = False

    return config


def _search_single_db(
    db: Path,
    query: str,
    *,
    mode: str,
    top: int,
    domain: str,
    chunk_type: str,
) -> list[dict]:
    """Run the chosen search mode against a single DB file."""
    conn = sqlite3.connect(str(db))
    conn.row_factory = sqlite3.Row
    try:
        if mode == "api":
            return _search_api(conn, query, top, domain)
        if mode == "semantic":
            return _search_semantic(conn, query, top, domain, chunk_type)
        if mode == "hybrid":
            kw_results = _search_keyword(conn, query, top * 2, domain, chunk_type)
            sem_results = _search_semantic(conn, query, top * 2, domain, chunk_type)
            return _fuse_results(kw_results, sem_results, top)
        return _search_keyword(conn, query, top, domain, chunk_type)
    finally:
        conn.close()


def search(
    query: str,
    *,
    mode: str = "keyword",
    top: int = 5,
    domain: str = "",
    chunk_type: str = "",
    db_path: str = "",
    prefer: str = "",
) -> dict:
    """Run a search with layered fallback (native → ext → legacy).

    If ``db_path`` is provided explicitly, single-DB mode is used and
    the fallback pipeline is skipped.

    ``prefer`` applies a lightweight rerank bonus based on the query intent
    (``examples`` / ``reference`` / ``guide``) without touching the DB schema.
    """
    # Fetch more than ``top`` from each layer so the rerank has signal to work
    # with; we still return at most ``top`` rows.
    fetch_top = max(top * 3, top + 5) if prefer else top

    if db_path:
        db = Path(db_path)
        if not db.exists():
            return {
                "query": query,
                "mode": mode,
                "error": f"Database not found: {db}",
                "results": [],
                "total_found": 0,
                "searched_layers": [],
            }
        try:
            results = _search_single_db(
                db, query, mode=mode, top=fetch_top, domain=domain, chunk_type=chunk_type,
            )
        except sqlite3.OperationalError as exc:
            print(f"[SEARCH] DB error in {db}: {exc}", file=sys.stderr)
            results = []
        for r in results:
            r.setdefault("source_layer", "explicit")
        results = _apply_prefer_bonus(results, prefer)
        return {
            "query": query,
            "mode": mode,
            "results": results[:top],
            "total_found": len(results),
            "searched_layers": ["explicit"],
            "prefer": prefer,
        }

    targets = iter_search_targets(SKILL_DIR)
    if not targets:
        primary = resolve_primary(SKILL_DIR)
        return {
            "query": query,
            "mode": mode,
            "error": (
                f"No knowledge-base shard found. Expected one of: "
                f"{primary}, kb_shard_ext/shard.sqlite, references/kb.sqlite"
            ),
            "results": [],
            "total_found": 0,
            "searched_layers": [],
        }

    seen: set[str] = set()
    merged: list[dict] = []
    searched_layers: list[str] = []

    for layer_name, layer_path in targets:
        searched_layers.append(layer_name)
        try:
            layer_results = _search_single_db(
                layer_path, query,
                mode=mode, top=fetch_top, domain=domain, chunk_type=chunk_type,
            )
        except sqlite3.OperationalError as exc:
            print(f"[SEARCH] Layer {layer_name} failed ({layer_path}): {exc}", file=sys.stderr)
            continue

        for r in layer_results:
            cid = r.get("chunk_id")
            if not cid or cid in seen:
                continue
            r["source_layer"] = layer_name
            seen.add(cid)
            merged.append(r)

        if len(merged) >= fetch_top:
            break

    merged = _apply_prefer_bonus(merged, prefer)

    return {
        "query": query,
        "mode": mode,
        "results": merged[:top],
        "total_found": len(merged),
        "searched_layers": searched_layers,
        "prefer": prefer,
    }


def _check_fts5_available(conn: sqlite3.Connection) -> bool:
    """Check if FTS5 module is available in this SQLite build."""
    try:
        conn.execute("SELECT * FROM chunks_fts LIMIT 0")
        return True
    except sqlite3.OperationalError:
        return False


def _search_keyword(
    conn: sqlite3.Connection,
    query: str,
    top: int,
    domain: str,
    chunk_type: str,
) -> list[dict]:
    """FTS5 BM25 keyword search with automatic LIKE fallback."""
    if not _check_fts5_available(conn):
        print("[SEARCH] FTS5 not available, using LIKE fallback", file=sys.stderr)
        return _search_keyword_fallback(conn, query, top, domain, chunk_type)

    fts_query = _build_fts_query(query)
    if not fts_query:
        return _search_keyword_fallback(conn, query, top, domain, chunk_type)

    sql = """
        SELECT c.chunk_id, c.doc_id, c.title, c.heading_path, c.content,
               c.token_count, c.chunk_type, c.api_symbols, c.keywords,
               c.source_type, c.source_uri, c.role, c.domain, c.version,
               rank
        FROM chunks_fts f
        JOIN chunks c ON f.chunk_id = c.chunk_id
        WHERE chunks_fts MATCH ?
    """
    params: list = [fts_query]

    if domain:
        sql += " AND c.domain = ?"
        params.append(domain)
    if chunk_type:
        sql += " AND c.chunk_type = ?"
        params.append(chunk_type)

    sql += " ORDER BY rank LIMIT ?"
    params.append(top)

    try:
        rows = conn.execute(sql, params).fetchall()
    except sqlite3.OperationalError as exc:
        print(f"[SEARCH] FTS query failed: {exc} | query={fts_query}", file=sys.stderr)
        return _search_keyword_fallback(conn, query, top, domain, chunk_type)

    results = []
    for row in rows:
        results.append(_row_to_result(row, abs(row["rank"])))
    return results


def _search_keyword_fallback(
    conn: sqlite3.Connection,
    query: str,
    top: int,
    domain: str,
    chunk_type: str,
) -> list[dict]:
    """LIKE-based fallback when FTS5 is unavailable or fails."""
    import re
    tokens = re.findall(r"[\w\u3400-\u9fff]{2,}", query)
    if not tokens:
        tokens = re.findall(r"[\w\u3400-\u9fff]+", query)
    if not tokens:
        return []

    conditions = [f"(c.content LIKE ? OR c.title LIKE ? OR c.keywords LIKE ?)" for _ in tokens]
    params: list = []
    for t in tokens:
        like_pat = f"%{t}%"
        params.extend([like_pat, like_pat, like_pat])

    sql = f"""
        SELECT c.chunk_id, c.doc_id, c.title, c.heading_path, c.content,
               c.token_count, c.chunk_type, c.api_symbols, c.keywords,
               c.source_type, c.source_uri, c.role, c.domain, c.version
        FROM chunks c
        WHERE ({' OR '.join(conditions)})
    """

    if domain:
        sql += " AND c.domain = ?"
        params.append(domain)
    if chunk_type:
        sql += " AND c.chunk_type = ?"
        params.append(chunk_type)

    sql += " LIMIT ?"
    params.append(top)

    try:
        rows = conn.execute(sql, params).fetchall()
    except sqlite3.OperationalError as exc:
        print(f"[SEARCH] Fallback query also failed: {exc}", file=sys.stderr)
        return []

    results = []
    for row in rows:
        match_count = sum(1 for t in tokens if t.lower() in (row["content"] or "").lower())
        score = match_count / max(len(tokens), 1)
        results.append(_row_to_result(row, score))

    results.sort(key=lambda r: r["score"], reverse=True)
    return results


def _search_api(
    conn: sqlite3.Connection,
    query: str,
    top: int,
    domain: str,
) -> list[dict]:
    """Exact API symbol search in the api_symbols JSON field."""
    sql = """
        SELECT chunk_id, doc_id, title, heading_path, content,
               token_count, chunk_type, api_symbols, keywords,
               source_type, source_uri, role, domain, version
        FROM chunks
        WHERE api_symbols LIKE ?
    """
    params: list = [f"%{query}%"]

    if domain:
        sql += " AND domain = ?"
        params.append(domain)
    sql += " AND chunk_type = 'api_ref' ORDER BY position LIMIT ?"
    params.append(top)

    rows = conn.execute(sql, params).fetchall()
    results = []
    for row in rows:
        symbols = json.loads(row["api_symbols"]) if row["api_symbols"] else []
        score = 1.0 if query in symbols else 0.5
        results.append(_row_to_result(row, score))
    return results


def _search_semantic(
    conn: sqlite3.Connection,
    query: str,
    top: int,
    domain: str,
    chunk_type: str,
) -> list[dict]:
    """Vector similarity search using stored embeddings."""
    config = _load_embedding_config()
    if not config.get("embedding_enabled", False):
        return _search_keyword(conn, query, top, domain, chunk_type)

    query_vec = _get_query_embedding(query, config)
    if not query_vec:
        return _search_keyword(conn, query, top, domain, chunk_type)

    sql = """
        SELECT c.chunk_id, c.doc_id, c.title, c.heading_path, c.content,
               c.token_count, c.chunk_type, c.api_symbols, c.keywords,
               c.source_type, c.source_uri, c.role, c.domain, c.version,
               e.vector
        FROM embeddings e
        JOIN chunks c ON e.chunk_id = c.chunk_id
    """
    conditions = []
    params: list = []
    if domain:
        conditions.append("c.domain = ?")
        params.append(domain)
    if chunk_type:
        conditions.append("c.chunk_type = ?")
        params.append(chunk_type)
    if conditions:
        sql += " WHERE " + " AND ".join(conditions)

    rows = conn.execute(sql, params).fetchall()

    scored = []
    for row in rows:
        vec = _blob_to_vector(row["vector"])
        sim = _cosine_similarity(query_vec, vec)
        scored.append((sim, row))

    scored.sort(key=lambda x: x[0], reverse=True)
    results = []
    for sim, row in scored[:top]:
        results.append(_row_to_result(row, sim))
    return results


def _fuse_results(kw: list[dict], sem: list[dict], top: int) -> list[dict]:
    """Reciprocal rank fusion of keyword and semantic results."""
    k = 60
    scores: dict[str, float] = {}
    result_map: dict[str, dict] = {}

    for rank, r in enumerate(kw):
        cid = r["chunk_id"]
        scores[cid] = scores.get(cid, 0) + 1.0 / (k + rank + 1)
        result_map[cid] = r

    for rank, r in enumerate(sem):
        cid = r["chunk_id"]
        scores[cid] = scores.get(cid, 0) + 1.0 / (k + rank + 1)
        result_map[cid] = r

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    results = []
    for cid, score in ranked[:top]:
        entry = result_map[cid]
        entry["score"] = round(score, 4)
        results.append(entry)
    return results


_API_TOKEN_RE = None  # type: ignore[assignment]


def _is_api_token(tok: str) -> bool:
    """Detect CUDA/HIP/ROC/etc. API-style identifiers (camelCase with known prefix)."""
    import re
    global _API_TOKEN_RE
    if _API_TOKEN_RE is None:
        # Accept both camelCase (hipMemcpy) and snake_case (rocblas_sgemm) forms.
        _API_TOKEN_RE = re.compile(
            r"^(?:"
            r"(?:hip|miopen|rccl|hsa|cu|cuda|cublas|cufft|curand|cusparse|cusolver|du|dcu)[A-Z_]\w*"
            r"|(?:roc|rocblas|rocfft|rocsolver|rocsparse|rocrand|rocthrust|rocprim|hip|miopen)_[a-zA-Z]\w*"
            r"|(?:rocblas|rocfft|rocsolver|rocsparse|rocrand)[A-Z_]\w*"
            r")$"
        )
    return bool(_API_TOKEN_RE.match(tok))


def _build_fts_query(query: str) -> str:
    """Build an FTS5 query string from natural language query.

    API-like camelCase tokens (e.g. ``hipMemcpy``, ``cudaMalloc``) are treated
    as **mandatory** (AND), while natural-language / CJK tokens are combined
    with OR. When no API tokens are present, falls back to the original
    all-OR behaviour so plain natural-language queries still work.
    """
    import re
    tokens = re.findall(r"[\w\u3400-\u9fff]+", query)
    if not tokens:
        return ""

    api_tokens = [t for t in tokens if _is_api_token(t)]
    other_tokens = [t for t in tokens if not _is_api_token(t) and len(t) >= 2]
    if not other_tokens:
        other_tokens = [t for t in tokens if not _is_api_token(t)]

    if api_tokens:
        required = " AND ".join(f'"{t}"' for t in dict.fromkeys(api_tokens))
        if other_tokens:
            optional = " OR ".join(f'"{t}"' for t in dict.fromkeys(other_tokens))
            return f"{required} AND ({optional})"
        return required

    parts = [f'"{t}"' for t in dict.fromkeys(other_tokens)]
    return " OR ".join(parts) if parts else ""


_PREFER_PROFILES: dict[str, dict[str, float]] = {
    "": {},
    "examples": {
        "chunk_type:code_example": 0.25,
        "role:playbook": 0.1,
        "uri:/examples/": 0.2,
    },
    "reference": {
        "chunk_type:api_ref": 0.25,
        "role:authoritative": 0.15,
    },
    "guide": {
        "role:guide": 0.2,
        "role:authoritative": 0.05,
    },
}


def _apply_prefer_bonus(results: list[dict], prefer: str) -> list[dict]:
    """Add a small rerank bonus based on a preference profile, then re-sort."""
    profile = _PREFER_PROFILES.get(prefer or "", {})
    if not profile:
        return results
    for r in results:
        bonus = 0.0
        chunk_type = r.get("chunk_type", "")
        role = r.get("role", "")
        source_uri = str(r.get("source_uri") or "")
        for key, val in profile.items():
            kind, _, probe = key.partition(":")
            if kind == "chunk_type" and chunk_type == probe:
                bonus += val
            elif kind == "role" and role == probe:
                bonus += val
            elif kind == "uri" and probe in source_uri:
                bonus += val
        if bonus:
            r["score"] = round(float(r.get("score", 0.0)) + bonus, 4)
    results.sort(key=lambda x: x.get("score", 0.0), reverse=True)
    return results


def _row_to_result(row: sqlite3.Row, score: float) -> dict:
    try:
        source_uri = row["source_uri"]
    except (IndexError, KeyError):
        source_uri = ""
    return {
        "chunk_id": row["chunk_id"],
        "doc_id": row["doc_id"],
        "title": row["title"] or "",
        "heading_path": json.loads(row["heading_path"]) if row["heading_path"] else [],
        "content": row["content"][:3000],
        "token_count": row["token_count"],
        "chunk_type": row["chunk_type"],
        "role": row["role"],
        "domain": row["domain"],
        "version": row["version"] or "",
        "source_uri": source_uri or "",
        "score": round(score, 4),
    }


def _get_query_embedding(query: str, config: dict) -> list[float] | None:
    """Get embedding for query text via API."""
    import urllib.request

    api_url = config.get("embedding_api_url", "")
    api_key = (config.get("embedding_api_key", "")
               or os.environ.get("EMBEDDING_API_KEY", ""))
    model = config.get("embedding_model", "qwen3-embedding")

    if not api_url:
        return None

    payload = json.dumps({"input": [query], "model": model}).encode("utf-8")
    req = urllib.request.Request(api_url, data=payload, headers={
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }, method="POST")

    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            result = json.loads(resp.read().decode("utf-8"))
        return result["data"][0]["embedding"]
    except Exception:
        return None


def _blob_to_vector(blob: bytes) -> list[float]:
    count = len(blob) // 4
    return list(struct.unpack(f"{count}f", blob))


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    if len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def _ensure_utf8_stdout() -> None:
    import io
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    elif sys.stdout.encoding != "utf-8":
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")


def main() -> int:
    _ensure_utf8_stdout()
    parser = argparse.ArgumentParser(description="DCU KB Search")
    parser.add_argument("--query", required=True, help="Search query")
    parser.add_argument("--mode", default="keyword", choices=["keyword", "semantic", "hybrid", "api"])
    parser.add_argument("--top", type=int, default=5)
    parser.add_argument("--domain", default="")
    parser.add_argument("--chunk-type", default="")
    parser.add_argument("--db", default="")
    parser.add_argument(
        "--prefer",
        default="",
        choices=["", "examples", "reference", "guide"],
        help="Lightweight rerank: bias results toward examples, reference or guide docs.",
    )
    parser.add_argument("--pretty", action="store_true")
    args = parser.parse_args()

    result = search(
        args.query,
        mode=args.mode,
        top=args.top,
        domain=args.domain,
        chunk_type=args.chunk_type,
        db_path=args.db,
        prefer=args.prefer,
    )

    indent = 2 if args.pretty else None
    print(json.dumps(result, ensure_ascii=False, indent=indent))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
