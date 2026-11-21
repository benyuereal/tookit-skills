"""路径工具（从 v3 common/utils.py 迁移，接口不变）"""
from pathlib import Path
from typing import Optional, Union


class PathHandler:
    @staticmethod
    def ensure_dir(path: Union[str, Path]) -> Path:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        return path

    @staticmethod
    def ensure_parent(path: Union[str, Path]) -> Path:
        path = Path(path)
        PathHandler.ensure_dir(path.parent)
        return path

    @staticmethod
    def resolve_path(path: Union[str, Path]) -> Path:
        path = Path(path)
        if not path.is_absolute():
            path = Path(__file__).parent.parent.parent / path
        return path.resolve()

    @staticmethod
    def get_relative_path(
        path: Union[str, Path],
        base: Optional[Union[str, Path]] = None,
    ) -> Path:
        path = Path(path)
        base = Path(base) if base else Path.cwd()
        return path.relative_to(base)
