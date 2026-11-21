"""Extract evidence snippets from the knowledge base for a given chunk.

Usage:
    python extract_snippet.py --chunk-id "dtk_26_04_hip_runtime_api_371342ae_api_0042"
    python extract_snippet.py --doc-id "dtk_26_04_hip_runtime_api_371342ae" --heading "hipMalloc"
    python extract_snippet.py --doc-id "dtk_26_04_hip_runtime_api_371342ae" --search "hipMalloc"
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Callable, Iterator, Optional

SCRIPT_DIR = Path(__file__).resolve().parent
SKILL_DIR = SCRIPT_DIR.parent

sys.path.insert(0, str(SCRIPT_DIR))
from db_paths import iter_search_targets, resolve_primary  # noqa: E402


@contextmanager
def _open_db(db: Path) -> Iterator[sqlite3.Connection]:
    conn = sqlite3.connect(str(db))
    conn.row_factory = sqlite3.Row
    try:
        yield conn
    finally:
        conn.close()


def _layered_targets(db_path: str) -> list[tuple[str, Path]]:
    """Return (layer, path) tuples for layered scan, honoring --db when set."""
    if db_path:
        p = Path(db_path)
        return [("explicit", p)] if p.exists() else []
    return iter_search_targets(SKILL_DIR)


def _no_shard_error() -> dict:
    primary = resolve_primary(SKILL_DIR)
    return {
        "error": (
            f"No knowledge-base shard found. Expected one of: "
            f"{primary}, kb_shard_ext/shard.sqlite, references/kb.sqlite"
        ),
        "searched_layers": [],
    }


def _run_layered(
    db_path: str,
    runner: Callable[[sqlite3.Connection], Optional[dict]],
) -> tuple[Optional[dict], list[str], Optional[str]]:
    """Scan layers until ``runner`` returns a non-None dict.

    Returns (result, searched_layers, hit_layer).
    """
    targets = _layered_targets(db_path)
    searched: list[str] = []
    if not targets:
        return None, searched, None

    for layer_name, layer_path in targets:
        searched.append(layer_name)
        try:
            with _open_db(layer_path) as conn:
                res = runner(conn)
        except sqlite3.OperationalError as exc:
            print(f"[EXTRACT] Layer {layer_name} failed ({layer_path}): {exc}", file=sys.stderr)
            continue
        if res is not None:
            return res, searched, layer_name

    return None, searched, None


def extract_by_chunk_id(chunk_id: str, db_path: str = "") -> dict:
    """Get full content of a specific chunk, scanning layered shards."""

    def runner(conn: sqlite3.Connection) -> Optional[dict]:
        row = conn.execute(
            "SELECT * FROM chunks WHERE chunk_id = ?", (chunk_id,)
        ).fetchone()
        if not row:
            return None
        return _row_to_snippet(row)

    result, searched, hit_layer = _run_layered(db_path, runner)
    if result is None:
        targets = _layered_targets(db_path)
        if not targets:
            return _no_shard_error()
        return {
            "error": f"Chunk not found: {chunk_id}",
            "searched_layers": searched,
        }
    result["source_layer"] = hit_layer
    result["searched_layers"] = searched
    return result


def extract_by_heading(doc_id: str, heading: str, db_path: str = "") -> dict:
    """Find chunks in a document matching a heading, layered fallback."""

    def runner(conn: sqlite3.Connection) -> Optional[dict]:
        rows = conn.execute(
            "SELECT * FROM chunks WHERE doc_id = ? AND title LIKE ? "
            "ORDER BY position LIMIT 5",
            (doc_id, f"%{heading}%"),
        ).fetchall()
        if not rows:
            return None
        return {"snippets": [_row_to_snippet(r) for r in rows]}

    result, searched, hit_layer = _run_layered(db_path, runner)
    if result is None:
        targets = _layered_targets(db_path)
        if not targets:
            return _no_shard_error()
        return {
            "doc_id": doc_id,
            "heading": heading,
            "snippets": [],
            "searched_layers": searched,
        }
    for snip in result["snippets"]:
        snip["source_layer"] = hit_layer
    return {
        "doc_id": doc_id,
        "heading": heading,
        "snippets": result["snippets"],
        "source_layer": hit_layer,
        "searched_layers": searched,
    }


def extract_by_search(doc_id: str, search_text: str, db_path: str = "") -> dict:
    """Search within a specific document's chunks across layers."""

    def runner(conn: sqlite3.Connection) -> Optional[dict]:
        rows = conn.execute(
            "SELECT * FROM chunks WHERE doc_id = ? AND content LIKE ? "
            "ORDER BY position LIMIT 5",
            (doc_id, f"%{search_text}%"),
        ).fetchall()
        if not rows:
            return None
        return {"snippets": [_row_to_snippet(r) for r in rows]}

    result, searched, hit_layer = _run_layered(db_path, runner)
    if result is None:
        targets = _layered_targets(db_path)
        if not targets:
            return _no_shard_error()
        return {
            "doc_id": doc_id,
            "search": search_text,
            "snippets": [],
            "searched_layers": searched,
        }
    for snip in result["snippets"]:
        snip["source_layer"] = hit_layer
    return {
        "doc_id": doc_id,
        "search": search_text,
        "snippets": result["snippets"],
        "source_layer": hit_layer,
        "searched_layers": searched,
    }


def get_document_outline(doc_id: str, db_path: str = "") -> dict:
    """Get the heading outline of a document from layered shards."""

    def runner(conn: sqlite3.Connection) -> Optional[dict]:
        rows = conn.execute(
            "SELECT chunk_id, title, heading_path, token_count, chunk_type, position "
            "FROM chunks WHERE doc_id = ? ORDER BY position",
            (doc_id,),
        ).fetchall()
        if not rows:
            return None
        sections = []
        for row in rows:
            sections.append({
                "chunk_id": row["chunk_id"],
                "title": row["title"] or "(untitled)",
                "heading_path": json.loads(row["heading_path"]) if row["heading_path"] else [],
                "token_count": row["token_count"],
                "chunk_type": row["chunk_type"],
                "position": row["position"],
            })
        return {"sections": sections}

    result, searched, hit_layer = _run_layered(db_path, runner)
    if result is None:
        targets = _layered_targets(db_path)
        if not targets:
            return _no_shard_error()
        return {
            "doc_id": doc_id,
            "chunk_count": 0,
            "sections": [],
            "searched_layers": searched,
        }
    sections = result["sections"]
    return {
        "doc_id": doc_id,
        "chunk_count": len(sections),
        "sections": sections,
        "source_layer": hit_layer,
        "searched_layers": searched,
    }


def get_context_window(chunk_id: str, window: int = 1, db_path: str = "") -> dict:
    """Get a chunk plus its neighbors; neighbors are fetched from the same layer."""

    def runner(conn: sqlite3.Connection) -> Optional[dict]:
        row = conn.execute(
            "SELECT * FROM chunks WHERE chunk_id = ?", (chunk_id,)
        ).fetchone()
        if not row:
            return None
        doc_id = row["doc_id"]
        position = row["position"]
        neighbors = conn.execute(
            "SELECT * FROM chunks WHERE doc_id = ? AND position BETWEEN ? AND ? "
            "ORDER BY position",
            (doc_id, position - window, position + window),
        ).fetchall()
        return {
            "center_chunk_id": chunk_id,
            "doc_id": doc_id,
            "snippets": [_row_to_snippet(r) for r in neighbors],
        }

    result, searched, hit_layer = _run_layered(db_path, runner)
    if result is None:
        targets = _layered_targets(db_path)
        if not targets:
            return _no_shard_error()
        return {
            "error": f"Chunk not found: {chunk_id}",
            "searched_layers": searched,
        }
    for snip in result["snippets"]:
        snip["source_layer"] = hit_layer
    result["source_layer"] = hit_layer
    result["searched_layers"] = searched
    return result


def _row_to_snippet(row: sqlite3.Row) -> dict:
    return {
        "chunk_id": row["chunk_id"],
        "doc_id": row["doc_id"],
        "title": row["title"] or "",
        "heading_path": json.loads(row["heading_path"]) if row["heading_path"] else [],
        "content": row["content"],
        "token_count": row["token_count"],
        "chunk_type": row["chunk_type"],
        "api_symbols": json.loads(row["api_symbols"]) if row["api_symbols"] else [],
        "role": row["role"],
        "domain": row["domain"],
        "position": row["position"],
    }


def _ensure_utf8_stdout() -> None:
    import io
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    elif sys.stdout.encoding != "utf-8":
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")


def main() -> int:
    _ensure_utf8_stdout()
    parser = argparse.ArgumentParser(description="DCU KB Snippet Extractor")
    parser.add_argument("--chunk-id", help="Extract a specific chunk")
    parser.add_argument("--doc-id", help="Target document for heading/search")
    parser.add_argument("--heading", help="Search by heading within a doc")
    parser.add_argument("--search", help="Search by text within a doc")
    parser.add_argument("--outline", action="store_true", help="Get document outline")
    parser.add_argument("--context", type=int, default=0, help="Number of surrounding chunks to include")
    parser.add_argument("--db", default="")
    args = parser.parse_args()

    if args.chunk_id:
        if args.context > 0:
            result = get_context_window(args.chunk_id, args.context, args.db)
        else:
            result = extract_by_chunk_id(args.chunk_id, args.db)
    elif args.doc_id and args.outline:
        result = get_document_outline(args.doc_id, args.db)
    elif args.doc_id and args.heading:
        result = extract_by_heading(args.doc_id, args.heading, args.db)
    elif args.doc_id and args.search:
        result = extract_by_search(args.doc_id, args.search, args.db)
    else:
        parser.print_help()
        return 1

    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
