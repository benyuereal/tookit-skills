"""配置加载器（从 v3 common/utils.py 迁移，接口不变）"""
import json
from pathlib import Path
from typing import Any, Dict, Union

from .path import PathHandler


class ConfigLoader:
    @staticmethod
    def load_yaml(config_path: Union[str, Path]) -> Dict[str, Any]:
        import yaml
        config_path = Path(config_path)
        if not config_path.exists():
            raise FileNotFoundError(f"配置文件不存在: {config_path}")
        with open(config_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)

    @staticmethod
    def load_json(config_path: Union[str, Path]) -> Dict[str, Any]:
        config_path = Path(config_path)
        if not config_path.exists():
            raise FileNotFoundError(f"配置文件不存在: {config_path}")
        with open(config_path, "r", encoding="utf-8") as f:
            return json.load(f)

    @staticmethod
    def save_yaml(config: Dict[str, Any], config_path: Union[str, Path]) -> None:
        import yaml
        config_path = PathHandler.ensure_parent(config_path)
        with open(config_path, "w", encoding="utf-8") as f:
            yaml.dump(config, f, default_flow_style=False, allow_unicode=True, sort_keys=False)

    @staticmethod
    def save_json(config: Dict[str, Any], config_path: Union[str, Path]) -> None:
        config_path = PathHandler.ensure_parent(config_path)
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(config, f, ensure_ascii=False, indent=2)

    @staticmethod
    def merge_config(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
        result = base.copy()
        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = ConfigLoader.merge_config(result[key], value)
            else:
                result[key] = value
        return result
