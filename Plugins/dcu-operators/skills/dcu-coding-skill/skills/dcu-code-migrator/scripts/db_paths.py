"""Centralized DB path resolution for the dcu-coding-fast skill.

Layered lookup order:
    1. native  — kb_shard/shard.sqlite      (shipped with the skill)
    2. ext     — kb_shard_ext/shard.sqlite  (optional, owned by external framework)
    3. legacy  — references/kb.sqlite       (backward-compat for older installs)
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Tuple

PRIMARY_REL = Path("kb_shard") / "shard.sqlite"
EXT_REL = Path("kb_shard_ext") / "shard.sqlite"
LEGACY_REL = Path("references") / "kb.sqlite"

# Chunk-id namespaces shipped / accepted by this skill.
#
# ``CHUNK_ID_PREFIX`` is the canonical default (the owning sub-skill, e.g.
# ``dcu-code-migrator:``). ``_KNOWN_PREFIXES`` additionally lists legacy /
# alternate prefixes we still need to recognise when a caller cites a chunk
# id built by an older pipeline or by another sub-skill in the same family.
#
# The actual prefix baked into freshly-built shards is controlled by the
# pipeline (``pipeline/config.yaml::target.chunk_id_namespace``) and is
# re-exported via ``dcu-coding-fast/pipeline/processors/chunker.py``.
CHUNK_ID_PREFIX = "dcu-code-migrator:"

_KNOWN_PREFIXES: tuple[str, ...] = (
    "dcu-code-migrator:",
    "dcu-coding-fast:",
)


def _strip_any_known_prefix(chunk_id: str) -> str:
    for prefix in _KNOWN_PREFIXES:
        if chunk_id.startswith(prefix):
            return chunk_id[len(prefix):]
    return chunk_id


def namespace_chunk_id(chunk_id: str, prefix: str = CHUNK_ID_PREFIX) -> str:
    """Return ``chunk_id`` with the canonical namespace prefix applied.

    If ``chunk_id`` already carries any of the known namespaces it is
    returned as-is. Otherwise the supplied ``prefix`` (default: the current
    canonical prefix) is prepended.
    """
    if not chunk_id:
        return chunk_id
    for known in _KNOWN_PREFIXES:
        if chunk_id.startswith(known):
            return chunk_id
    return f"{prefix}{chunk_id}"


def strip_chunk_id_namespace(chunk_id: str) -> str:
    """Return ``chunk_id`` with any known namespace prefix stripped."""
    return _strip_any_known_prefix(chunk_id)


def candidate_chunk_ids(chunk_id: str) -> list[str]:
    """Return every plausible form of ``chunk_id`` to try against the DB.

    Includes the raw (un-prefixed) form plus every known namespaced form so
    that shards built under either ``dcu-code-migrator:`` or the older
    ``dcu-coding-fast:`` namespace can both be queried with the same cite.
    """
    if not chunk_id:
        return []
    raw = _strip_any_known_prefix(chunk_id)
    out: list[str] = []
    for value in (chunk_id, raw, *(f"{p}{raw}" for p in _KNOWN_PREFIXES)):
        if value and value not in out:
            out.append(value)
    return out


def resolve_primary(skill_dir: str | Path) -> Path:
    """Return the path to the primary (native) shard (may not exist)."""
    return Path(skill_dir) / PRIMARY_REL


def resolve_ext(skill_dir: str | Path) -> Path | None:
    """Return the external shard path, or None when not present."""
    p = Path(skill_dir) / EXT_REL
    return p if p.exists() else None


def resolve_legacy(skill_dir: str | Path) -> Path | None:
    """Return the legacy `references/kb.sqlite` path, or None when absent."""
    p = Path(skill_dir) / LEGACY_REL
    return p if p.exists() else None


def iter_search_targets(skill_dir: str | Path) -> List[Tuple[str, Path]]:
    """Yield (layer_name, path) tuples in search priority order.

    Only existing files are returned. The order is always:
        native  → ext → legacy
    """
    skill_dir = Path(skill_dir)
    targets: List[Tuple[str, Path]] = []

    primary = resolve_primary(skill_dir)
    if primary.exists():
        targets.append(("native", primary))

    ext = resolve_ext(skill_dir)
    if ext is not None:
        targets.append(("ext", ext))

    legacy = resolve_legacy(skill_dir)
    if legacy is not None:
        targets.append(("legacy", legacy))

    return targets


def iter_existing_layers(
    skill_dir: str | Path,
    layers: Iterable[str] = ("native", "ext", "legacy"),
) -> List[Tuple[str, Path]]:
    """Return only the requested layers that actually exist, preserving order."""
    allowed = set(layers)
    return [(name, p) for name, p in iter_search_targets(skill_dir) if name in allowed]


__all__ = [
    "PRIMARY_REL",
    "EXT_REL",
    "LEGACY_REL",
    "CHUNK_ID_PREFIX",
    "resolve_primary",
    "resolve_ext",
    "resolve_legacy",
    "iter_search_targets",
    "iter_existing_layers",
    "namespace_chunk_id",
    "strip_chunk_id_namespace",
    "candidate_chunk_ids",
    "_KNOWN_PREFIXES",
]
