"""
IntentRegistry —— 动态意图注册表（§15.3）

注册时机（任选其一，推荐 frontmatter 声明式）：
1. SKILL.md frontmatter 的 intents: 段（由 validate_skills.py 扫描注册）
2. skill 的 __init__.py 中显式调用 IntentRegistry.register(...)
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class IntentSpec:
    """单个意图的完整描述"""
    type: str               # 全局唯一的意图类型字符串，如 "code-migrate"
    target_skill: str       # 处理该意图的 skill 名，如 "dcu-code-migrator"
    description: str        # 给 LLM 看的语义说明
    examples: List[str] = field(default_factory=list)
    priority: int = 100     # 冲突时比较优先级（数值越小越优先）


class IntentRegistry:
    """
    全局意图注册表（进程级单例，类变量实现）。

    不变式：
    - type 字段全局唯一；重复注册相同 type 时按 priority 较小者保留（priority 相同则后注册者覆盖并发 warning）
    - dispatcher 宿主在启动时调用 all() 构建路由表，无需硬编码任何 skill 名
    """
    _specs: Dict[str, IntentSpec] = {}

    @classmethod
    def register(cls, spec: IntentSpec) -> None:
        """注册或更新意图。priority 更小的规格会替换已有的。"""
        existing = cls._specs.get(spec.type)
        if existing is not None:
            if spec.priority < existing.priority:
                cls._specs[spec.type] = spec
            elif spec.priority == existing.priority and spec.target_skill != existing.target_skill:
                import warnings
                warnings.warn(
                    f"IntentRegistry: type '{spec.type}' 已被 '{existing.target_skill}' 注册，"
                    f"'{spec.target_skill}' 以相同 priority={spec.priority} 尝试覆盖。"
                    f"保留后注册者。",
                    stacklevel=2,
                )
                cls._specs[spec.type] = spec
        else:
            cls._specs[spec.type] = spec

    @classmethod
    def all(cls) -> List[IntentSpec]:
        """返回所有已注册意图的列表（按 priority 升序）"""
        return sorted(cls._specs.values(), key=lambda s: s.priority)

    @classmethod
    def resolve(cls, intent_type: str) -> Optional[IntentSpec]:
        """查找意图，不存在返回 None"""
        return cls._specs.get(intent_type)

    @classmethod
    def all_types(cls) -> List[str]:
        """返回所有已注册的意图 type 字符串"""
        return list(cls._specs.keys())

    @classmethod
    def clear(cls) -> None:
        """清空注册表（测试用）"""
        cls._specs.clear()
