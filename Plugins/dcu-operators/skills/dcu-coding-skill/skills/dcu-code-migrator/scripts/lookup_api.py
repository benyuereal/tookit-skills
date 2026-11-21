"""Fast API symbol lookup — finds exact matches in the knowledge base.

Usage:
    python lookup_api.py --symbol hipMalloc
    python lookup_api.py --symbol rocblas_sgemm --include-aliases
    python lookup_api.py --family hip --list-symbols
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
SKILL_DIR = SCRIPT_DIR.parent
API_SUMMARY = SKILL_DIR / "references" / "api-summary.json"

sys.path.insert(0, str(SCRIPT_DIR))
from db_paths import iter_search_targets, resolve_primary  # noqa: E402


def _lookup_single_db(
    db: Path,
    symbol: str,
    *,
    include_aliases: bool,
    limit: int,
) -> list[dict]:
    """Query one shard for the symbol and return serialized rows."""
    conn = sqlite3.connect(str(db))
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute(
            "SELECT chunk_id, doc_id, title, heading_path, content, "
            "token_count, chunk_type, api_symbols, role, domain, version "
            "FROM chunks WHERE api_symbols LIKE ? AND chunk_type = 'api_ref' "
            "ORDER BY position LIMIT ?",
            (f'%"{symbol}"%', limit),
        ).fetchall()

        if not rows and include_aliases:
            rows = conn.execute(
                "SELECT chunk_id, doc_id, title, heading_path, content, "
                "token_count, chunk_type, api_symbols, role, domain, version "
                "FROM chunks WHERE api_symbols LIKE ? "
                "ORDER BY position LIMIT ?",
                (f"%{symbol}%", limit),
            ).fetchall()
    finally:
        conn.close()

    results: list[dict] = []
    for row in rows:
        results.append({
            "chunk_id": row["chunk_id"],
            "doc_id": row["doc_id"],
            "title": row["title"] or "",
            "heading_path": json.loads(row["heading_path"]) if row["heading_path"] else [],
            "content": row["content"][:3000],
            "token_count": row["token_count"],
            "chunk_type": row["chunk_type"],
            "api_symbols": json.loads(row["api_symbols"]) if row["api_symbols"] else [],
            "role": row["role"],
            "domain": row["domain"],
        })
    return results


def lookup_symbol(
    symbol: str,
    *,
    include_aliases: bool = False,
    db_path: str = "",
    limit: int = 10,
) -> dict:
    """Look up an API symbol across layered shards (native → ext → legacy)."""
    if db_path:
        db = Path(db_path)
        if not db.exists():
            return {
                "symbol": symbol,
                "error": f"Database not found: {db}",
                "results": [],
                "count": 0,
                "searched_layers": [],
            }
        try:
            rows = _lookup_single_db(db, symbol, include_aliases=include_aliases, limit=limit)
        except sqlite3.OperationalError as exc:
            print(f"[LOOKUP] DB error in {db}: {exc}", file=sys.stderr)
            rows = []
        for r in rows:
            r.setdefault("source_layer", "explicit")
        return {
            "symbol": symbol,
            "results": rows,
            "count": len(rows),
            "searched_layers": ["explicit"],
        }

    targets = iter_search_targets(SKILL_DIR)
    if not targets:
        primary = resolve_primary(SKILL_DIR)
        return {
            "symbol": symbol,
            "error": (
                f"No knowledge-base shard found. Expected one of: "
                f"{primary}, kb_shard_ext/shard.sqlite, references/kb.sqlite"
            ),
            "results": [],
            "count": 0,
            "searched_layers": [],
        }

    seen: set[str] = set()
    merged: list[dict] = []
    searched_layers: list[str] = []
    for layer_name, layer_path in targets:
        searched_layers.append(layer_name)
        try:
            rows = _lookup_single_db(
                layer_path, symbol, include_aliases=include_aliases, limit=limit,
            )
        except sqlite3.OperationalError as exc:
            print(f"[LOOKUP] Layer {layer_name} failed ({layer_path}): {exc}", file=sys.stderr)
            continue
        for r in rows:
            cid = r.get("chunk_id")
            if not cid or cid in seen:
                continue
            r["source_layer"] = layer_name
            seen.add(cid)
            merged.append(r)

        if merged:
            break

    return {
        "symbol": symbol,
        "results": merged,
        "count": len(merged),
        "searched_layers": searched_layers,
    }


def list_symbols(family: str = "") -> dict:
    """List all known API symbols, optionally filtered by family."""
    if not API_SUMMARY.exists():
        return {"error": "api-summary.json not found", "symbols": []}

    data = json.loads(API_SUMMARY.read_text(encoding="utf-8"))
    if family:
        symbols = data.get("families", {}).get(family.lower(), [])
        return {"family": family, "symbols": symbols, "count": len(symbols)}
    else:
        return {
            "families": data.get("families", {}),
            "symbol_count": data.get("symbol_count", 0),
            "family_count": data.get("family_count", 0),
        }


def _ensure_utf8_stdout() -> None:
    import io, sys
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    elif sys.stdout.encoding != "utf-8":
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")


def main() -> int:
    _ensure_utf8_stdout()
    parser = argparse.ArgumentParser(description="DCU API Symbol Lookup")
    parser.add_argument("--symbol", help="API symbol to look up")
    parser.add_argument("--family", help="List symbols in a family")
    parser.add_argument("--list-symbols", action="store_true")
    parser.add_argument("--include-aliases", action="store_true")
    parser.add_argument("--db", default="")
    args = parser.parse_args()

    if args.list_symbols or args.family:
        result = list_symbols(args.family or "")
    elif args.symbol:
        result = lookup_symbol(args.symbol, include_aliases=args.include_aliases, db_path=args.db)
    else:
        parser.print_help()
        return 1

    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
