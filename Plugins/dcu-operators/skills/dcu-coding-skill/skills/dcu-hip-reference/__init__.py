"""dcu-hip-reference —— HIP 基础 API 查询 core skill"""
from common.intent import IntentRegistry, IntentSpec

# 代码式注册（frontmatter 声明式为主，此处作为运行时备份）
IntentRegistry.register(IntentSpec(
    type="api-lookup",
    target_skill="dcu-hip-reference",
    description="HIP API / runtime 查询与示例",
    examples=["hipMalloc 怎么用", "查 HIP stream API"],
    priority=100,
))

from .hip_reference import DCUHipReferenceSkill

__all__ = ["DCUHipReferenceSkill"]
