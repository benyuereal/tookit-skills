"""Rebuild embedding vectors with a user-chosen model.

This script re-generates all vectors in kb.sqlite using your own embedding API,
so you can use semantic/hybrid search with any OpenAI-compatible embedding service.

Usage:
    python rebuild_embeddings.py --api-url http://your-server/v1/embeddings --model bge-large-zh-v1.5
    python rebuild_embeddings.py --api-url http://your-server/v1/embeddings --model bge-large-zh-v1.5 --api-key sk-xxx
    python rebuild_embeddings.py --help
"""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
import struct
import sys
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
SKILL_DIR = SCRIPT_DIR.parent
USER_CONFIG = SKILL_DIR / "user-config.json"

sys.path.insert(0, str(SCRIPT_DIR))
from db_paths import resolve_primary, resolve_legacy  # noqa: E402

DEFAULT_DB = resolve_primary(SKILL_DIR)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _vector_to_blob(vec: list[float]) -> bytes:
    return struct.pack(f"{len(vec)}f", *vec)


def _call_embedding_api(
    texts: list[str],
    api_url: str,
    api_key: str,
    model_name: str,
) -> list[list[float]]:
    """Call an OpenAI-compatible embedding API."""
    payload = json.dumps({"input": texts, "model": model_name}).encode("utf-8")
    req = urllib.request.Request(
        api_url,
        data=payload,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=120) as resp:
        result = json.loads(resp.read().decode("utf-8"))

    data = result.get("data", [])
    data.sort(key=lambda x: x.get("index", 0))
    return [item["embedding"] for item in data]


def _prepare_text(title: str, domain: str, content: str) -> str:
    """Prepare chunk text for embedding — prefix with title and domain."""
    parts = []
    if title:
        parts.append(title)
    if domain and domain != "general":
        parts.append(f"[{domain}]")
    parts.append(content[:2000])
    return " ".join(parts)


def _update_user_config(model_name: str, api_url: str, api_key: str) -> None:
    """Write back embedding settings to user-config.json."""
    cfg: dict = {}
    if USER_CONFIG.exists():
        cfg = json.loads(USER_CONFIG.read_text(encoding="utf-8"))

    cfg["embedding_enabled"] = True
    cfg["embedding_api_url"] = api_url
    cfg["embedding_model"] = model_name
    if api_key:
        cfg["embedding_api_key"] = api_key

    USER_CONFIG.write_text(
        json.dumps(cfg, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def rebuild(
    api_url: str,
    model_name: str,
    api_key: str = "",
    batch_size: int = 64,
    db_path: str = "",
) -> dict:
    """Re-generate all embeddings in the PRIMARY shard.

    By design this NEVER touches ``kb_shard_ext/`` — that layer is owned by
    an external framework and users must regenerate its vectors elsewhere.
    """
    if db_path:
        db = Path(db_path)
    else:
        db = DEFAULT_DB

    if not db.exists():
        legacy = resolve_legacy(SKILL_DIR)
        if legacy is not None and not db_path:
            return {
                "error": (
                    f"Primary shard not found at {db}. Legacy DB found at {legacy}. "
                    "Re-run your build to migrate it into kb_shard/shard.sqlite, "
                    "or pass --db to explicitly target the legacy file."
                )
            }
        return {"error": f"Database not found: {db}"}

    conn = sqlite3.connect(str(db))
    conn.row_factory = sqlite3.Row

    try:
        chunks = conn.execute(
            "SELECT chunk_id, title, domain, content FROM chunks ORDER BY position"
        ).fetchall()
        total = len(chunks)
        if total == 0:
            return {"error": "No chunks found in database."}

        print(f"[rebuild] {total} chunks to embed with model '{model_name}'")
        print(f"[rebuild] API: {api_url}")
        print(f"[rebuild] Batch size: {batch_size}")

        conn.execute("DELETE FROM embeddings")
        conn.commit()

        embedded = 0
        for i in range(0, total, batch_size):
            batch = chunks[i : i + batch_size]
            texts = [
                _prepare_text(r["title"] or "", r["domain"] or "", r["content"])
                for r in batch
            ]

            try:
                vectors = _call_embedding_api(texts, api_url, api_key, model_name)
            except Exception as exc:
                print(f"[rebuild] ERROR at batch {i // batch_size}: {exc}")
                conn.rollback()
                return {
                    "error": str(exc),
                    "embedded_before_error": embedded,
                    "total": total,
                }

            now = _utc_now()
            rows = [
                (r["chunk_id"], _vector_to_blob(vec), model_name, now)
                for r, vec in zip(batch, vectors)
            ]
            conn.executemany(
                "INSERT OR REPLACE INTO embeddings "
                "(chunk_id, vector, model_name, created_at) VALUES (?, ?, ?, ?)",
                rows,
            )
            conn.commit()
            embedded += len(batch)
            pct = embedded * 100 // total
            print(f"[rebuild] {embedded}/{total} ({pct}%)")

    finally:
        conn.close()

    _update_user_config(model_name, api_url, api_key)
    print(f"[rebuild] Done. user-config.json updated (embedding_enabled=true).")

    return {
        "status": "success",
        "model_name": model_name,
        "embedded_count": embedded,
        "built_at": _utc_now(),
    }


def _ensure_utf8_stdout() -> None:
    import io
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    elif sys.stdout.encoding != "utf-8":
        sys.stdout = io.TextIOWrapper(
            sys.stdout.buffer, encoding="utf-8", errors="replace"
        )


def main() -> int:
    _ensure_utf8_stdout()
    parser = argparse.ArgumentParser(
        description="Rebuild embedding vectors with your own model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Examples:
  python rebuild_embeddings.py \\
      --api-url http://your-server/v1/embeddings \\
      --model bge-large-zh-v1.5

  python rebuild_embeddings.py \\
      --api-url https://api.openai.com/v1/embeddings \\
      --model text-embedding-3-small \\
      --api-key sk-xxx

After running, user-config.json is automatically updated and
semantic/hybrid search will use your new model immediately.
""",
    )
    parser.add_argument("--api-url", required=True, help="OpenAI-compatible embedding API endpoint")
    parser.add_argument("--model", required=True, help="Embedding model name")
    parser.add_argument("--api-key", default="", help="API key (or set env EMBEDDING_API_KEY)")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size per API call (default: 64)")
    parser.add_argument("--db", default="", help="Path to shard.sqlite (default: kb_shard/shard.sqlite). Never targets kb_shard_ext/ unless you pass it explicitly.")
    args = parser.parse_args()

    api_key = args.api_key or os.environ.get("EMBEDDING_API_KEY", "")

    result = rebuild(
        api_url=args.api_url,
        model_name=args.model,
        api_key=api_key,
        batch_size=args.batch_size,
        db_path=args.db,
    )

    if "error" in result:
        print(f"\nFAILED: {result['error']}", file=sys.stderr)
        return 1

    print(f"\nSUCCESS: {result['embedded_count']} chunks embedded with '{result['model_name']}'")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
