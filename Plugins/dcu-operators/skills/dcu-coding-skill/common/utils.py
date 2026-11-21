"""
DCU Coding Skill Utilities

提供共享的工具类和函数
"""

import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Union
from dataclasses import dataclass, asdict


@dataclass
class LogConfig:
    """日志配置"""
    level: str = "INFO"
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    log_file: Optional[str] = None


class Logger:
    """日志工具类"""

    def __init__(self, name: str, config: Optional[LogConfig] = None):
        """
        初始化日志记录器

        Args:
            name: 日志记录器名称
            config: 日志配置
        """
        self.name = name
        self.config = config or LogConfig()

        # 创建 logger
        self.logger = logging.getLogger(name)
        self.logger.setLevel(getattr(logging, self.config.level.upper()))

        # 避免重复添加 handler
        if not self.logger.handlers:
            # 控制台 handler
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setLevel(getattr(logging, self.config.level.upper()))
            console_format = logging.Formatter(self.config.format)
            console_handler.setFormatter(console_format)
            self.logger.addHandler(console_handler)

            # 文件 handler（如果配置了）
            if self.config.log_file:
                log_path = Path(self.config.log_file)
                log_path.parent.mkdir(parents=True, exist_ok=True)
                file_handler = logging.FileHandler(log_path, encoding="utf-8")
                file_handler.setLevel(getattr(logging, self.config.level.upper()))
                file_format = logging.Formatter(self.config.format)
                file_handler.setFormatter(file_format)
                self.logger.addHandler(file_handler)

    def debug(self, message: str, **kwargs):
        """Debug 级别日志"""
        self.logger.debug(message, extra=kwargs)

    def info(self, message: str, **kwargs):
        """Info 级别日志"""
        self.logger.info(message, extra=kwargs)

    def warning(self, message: str, **kwargs):
        """Warning 级别日志"""
        self.logger.warning(message, extra=kwargs)

    def error(self, message: str, **kwargs):
        """Error 级别日志"""
        self.logger.error(message, extra=kwargs)

    def critical(self, message: str, **kwargs):
        """Critical 级别日志"""
        self.logger.critical(message, extra=kwargs)


class PathHandler:
    """路径处理工具类"""

    @staticmethod
    def ensure_dir(path: Union[str, Path]) -> Path:
        """
        确保目录存在，不存在则创建

        Args:
            path: 目录路径

        Returns:
            Path 对象
        """
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        return path

    @staticmethod
    def ensure_parent(path: Union[str, Path]) -> Path:
        """
        确保父目录存在，不存在则创建

        Args:
            path: 文件路径

        Returns:
            Path 对象
        """
        path = Path(path)
        PathHandler.ensure_dir(path.parent)
        return path

    @staticmethod
    def resolve_path(path: Union[str, Path]) -> Path:
        """
        解析路径，转换为绝对路径

        Args:
            path: 路径字符串或 Path 对象

        Returns:
            绝对路径的 Path 对象
        """
        path = Path(path)
        if not path.is_absolute():
            path = Path(__file__).parent.parent.parent / path
        return path.resolve()

    @staticmethod
    def get_relative_path(path: Union[str, Path], base: Optional[Union[str, Path]] = None) -> Path:
        """
        获取相对于基路径的相对路径

        Args:
            path: 路径字符串或 Path 对象
            base: 基路径，默认为当前目录

        Returns:
            相对路径的 Path 对象
        """
        path = Path(path)
        base = Path(base) if base else Path.cwd()
        return path.relative_to(base)


class ConfigLoader:
    """配置加载器"""

    @staticmethod
    def load_yaml(config_path: Union[str, Path]) -> Dict[str, Any]:
        """
        加载 YAML 配置文件

        Args:
            config_path: 配置文件路径

        Returns:
            配置字典
        """
        import yaml

        config_path = Path(config_path)
        if not config_path.exists():
            raise FileNotFoundError(f"配置文件不存在: {config_path}")

        with open(config_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)

    @staticmethod
    def load_json(config_path: Union[str, Path]) -> Dict[str, Any]:
        """
        加载 JSON 配置文件

        Args:
            config_path: 配置文件路径

        Returns:
            配置字典
        """
        config_path = Path(config_path)
        if not config_path.exists():
            raise FileNotFoundError(f"配置文件不存在: {config_path}")

        with open(config_path, "r", encoding="utf-8") as f:
            return json.load(f)

    @staticmethod
    def save_yaml(config: Dict[str, Any], config_path: Union[str, Path]) -> None:
        """
        保存 YAML 配置文件

        Args:
            config: 配置字典
            config_path: 配置文件路径
        """
        import yaml

        config_path = PathHandler.ensure_parent(config_path)

        with open(config_path, "w", encoding="utf-8") as f:
            yaml.dump(config, f, default_flow_style=False, allow_unicode=True, sort_keys=False)

    @staticmethod
    def save_json(config: Dict[str, Any], config_path: Union[str, Path]) -> None:
        """
        保存 JSON 配置文件

        Args:
            config: 配置字典
            config_path: 配置文件路径
        """
        config_path = PathHandler.ensure_parent(config_path)

        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(config, f, ensure_ascii=False, indent=2)

    @staticmethod
    def merge_config(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
        """
        合并两个配置字典

        Args:
            base: 基础配置
            override: 覆盖配置

        Returns:
            合并后的配置
        """
        result = base.copy()
        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = ConfigLoader.merge_config(result[key], value)
            else:
                result[key] = value
        return result
