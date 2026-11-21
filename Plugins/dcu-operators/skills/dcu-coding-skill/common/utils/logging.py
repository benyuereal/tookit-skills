"""日志工具（从 v3 common/utils.py 迁移，接口不变）"""
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass
class LogConfig:
    level: str = "INFO"
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    log_file: Optional[str] = None


class Logger:
    def __init__(self, name: str, config: Optional[LogConfig] = None):
        self.name = name
        self.config = config or LogConfig()
        self.logger = logging.getLogger(name)
        self.logger.setLevel(getattr(logging, self.config.level.upper()))

        if not self.logger.handlers:
            handler = logging.StreamHandler(sys.stdout)
            handler.setFormatter(logging.Formatter(self.config.format))
            self.logger.addHandler(handler)

            if self.config.log_file:
                log_path = Path(self.config.log_file)
                log_path.parent.mkdir(parents=True, exist_ok=True)
                fh = logging.FileHandler(log_path, encoding="utf-8")
                fh.setFormatter(logging.Formatter(self.config.format))
                self.logger.addHandler(fh)

    def debug(self, msg: str, **kw): self.logger.debug(msg, extra=kw)
    def info(self, msg: str, **kw): self.logger.info(msg, extra=kw)
    def warning(self, msg: str, **kw): self.logger.warning(msg, extra=kw)
    def error(self, msg: str, **kw): self.logger.error(msg, extra=kw)
    def critical(self, msg: str, **kw): self.logger.critical(msg, extra=kw)
