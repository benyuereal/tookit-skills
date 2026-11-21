"""
common.intent — Intent Registry（§15.3）

动态意图注册机制，取代 v4.0 的硬编码 Literal IntentType。
新增 core skill 只需在 SKILL.md frontmatter 声明 intents:，
validate_skills.py 扫描后自动注册，dispatcher 无需修改。
"""
from .registry import IntentRegistry, IntentSpec
from .keywords import find_conflicts, KeywordConflict

__all__ = ["IntentRegistry", "IntentSpec", "find_conflicts", "KeywordConflict"]
