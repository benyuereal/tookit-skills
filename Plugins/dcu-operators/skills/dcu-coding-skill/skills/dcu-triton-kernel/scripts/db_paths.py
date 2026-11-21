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
    "resolve_primary",
    "resolve_ext",
    "resolve_legacy",
    "iter_search_targets",
    "iter_existing_layers",
]
