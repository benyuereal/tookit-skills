"""
resolve_target_skill —— 执行期 __self__ 路由解析（§5.2）

IntentClassifier 在独立模式 + hip-reference 未装时产生 target_skill="__self__"，
router 在执行前调用此函数将其解析为真实 skill 名。
"""
from __future__ import annotations


__SELF_MARKER = "__self__"


from typing import Optional


def resolve_target_skill(
    target_skill: str,
    host_skill: Optional[str],
) -> str:
    """
    将 target_skill 中的 __self__ 标记解析为实际 skill 名。

    Args:
        target_skill:  IntentClassifier 返回的 target_skill 字段
        host_skill:    调用方 skill 名；dispatcher 宿主传 None

    Returns:
        解析后的实际 skill 名

    Raises:
        RuntimeError: dispatcher 宿主场景下产生了 __self__（不应发生）
    """
    if target_skill != __SELF_MARKER:
        return target_skill

    if host_skill is None:
        raise RuntimeError(
            "__self__ 路由要求调用方提供 host_skill。"
            "dispatcher 宿主不应产生 __self__——请检查 installed_skills 传入是否包含 dcu-hip-reference。"
        )
    return host_skill
