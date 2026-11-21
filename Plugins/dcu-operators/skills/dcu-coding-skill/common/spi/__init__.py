"""
common.spi — Skill Provider Interface

所有 core skill 必须实现的契约，定义 v4.1 的 skill 标准接口。
"""
from .provider import SkillProvider, Capability, HealthStatus
from .workflow import RoutingContext, StepResult, WorkflowResult, StepStatus
from .routing import resolve_target_skill

__all__ = [
    "SkillProvider",
    "Capability",
    "HealthStatus",
    "RoutingContext",
    "StepResult",
    "WorkflowResult",
    "StepStatus",
    "resolve_target_skill",
]
