"""Shared paths for versioned seed directories (e.g. seeds/v1.2)."""
from __future__ import annotations

import re
from pathlib import Path

SEMVER_PATTERN = re.compile(r"^(\d+)\.(\d+)\.(\d+)$")


def parse_semver_tuple(version: str) -> tuple[int, int, int]:
    s = version.strip()
    m = SEMVER_PATTERN.fullmatch(s)
    if not m:
        raise ValueError(f"Invalid semantic version: {version!r} (expected x.y.z)")
    return int(m.group(1)), int(m.group(2)), int(m.group(3))


def minor_dir_name(version: str) -> str:
    """1.2.0 -> v1; 2.1.0 -> v2. 只取 major 版本号作为种子目录。

    约定：所有 v1.x 版本从 seeds/v1 取，v2.x 从 seeds/v2 取，以此类推。
    """
    major, _, _ = parse_semver_tuple(version)
    return f"v{major}"


def seeds_dir_for_version(project_root: Path, version: str) -> Path:
    return project_root / "seeds" / minor_dir_name(version)
