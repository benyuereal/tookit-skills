"""SkillProvider ABC —— v4.1 所有 core skill 必须实现的契约（§3.6）"""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional


@dataclass
class Capability:
    """Skill 对外暴露的单个 action 描述"""
    action: str
    cost: str = "medium"      # "cheap" | "medium" | "expensive"
    description: str = ""


@dataclass
class HealthStatus:
    """Skill 健康状态（§8.3）"""
    healthy: bool
    degraded: bool = False
    reason: str = ""
    missing_shards: List[str] = field(default_factory=list)


class SkillProvider(ABC):
    """
    v4.1 core skill 契约。

    每个 core skill 的主类必须继承此 ABC，并实现 capabilities() 与 health_check()。
    dispatcher 在启动期通过 SPI 契约验证所有已装 core skill 的合规性。

    子类示例：
        class DCUCodeMigratorSkill(SkillProvider):
            name    = "dcu-code-migrator"
            version = "1.0.0"

            def capabilities(self) -> list[Capability]: ...
            def health_check(self) -> HealthStatus: ...
    """

    name: str
    version: str

    @abstractmethod
    def capabilities(self) -> List[Capability]:
        """返回该 skill 对外暴露的 action 清单"""
        ...

    @abstractmethod
    def health_check(self) -> HealthStatus:
        """
        检查 skill 运行时健康状态。
        - healthy=True, degraded=False  → 完全正常
        - healthy=True, degraded=True   → 可用但功能受限（如分片缺失）
        - healthy=False                 → 不可用，dispatcher 应跳过
        """
        ...

    def skill_dir(self) -> Path:
        """返回本 skill 所在目录（用于定位 kb_shard/ 等资源）"""
        return Path(__file__).parent

    def on_install(self) -> None:
        """安装后钩子（可选覆盖）"""

    def on_uninstall(self) -> None:
        """卸载前钩子（可选覆盖）"""
