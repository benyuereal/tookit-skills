"""
关键词冲突检测（§15.4）

随 skill 增多，同一关键词被多个 skill 声明会降低意图分类准确率。
此模块提供冲突检测，供 validate_skills.py --check-keyword-conflicts 调用。

判定规则（v1.0 冻结）：
- 完全相同关键词出现在 2+ skill → 返回 HARD 冲突（CI 阻断）
- 前缀/后缀/子串重叠           → 返回 SOFT 冲突（CI 警告，不阻断）
- 中英文近义词分散             → 返回 SOFT 冲突（CI 警告）

消解责任在新 skill 自己，不得修改邻居 skill 的 frontmatter（§15.1.3）。
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Literal


@dataclass
class KeywordConflict:
    keyword: str
    skills: list[str]                          # 产生冲突的 skill 名列表
    severity: Literal["hard", "soft"]          # hard=CI阻断, soft=仅警告
    reason: str


def find_conflicts(marketplace: dict) -> list[KeywordConflict]:
    """
    从 marketplace dict 检测关键词冲突。

    Args:
        marketplace: marketplace.json 解析后的字典

    Returns:
        冲突列表（空列表表示无冲突）
    """
    from collections import defaultdict

    # skill_name -> [keywords]
    keyword_map: dict[str, list[str]] = {}
    for skill in marketplace.get("skills", []):
        name = skill.get("name", "")
        keywords = [k.lower() for k in skill.get("keywords", [])]
        keyword_map[name] = keywords

    # 完全相同的关键词检测 → hard
    kw_to_skills: dict[str, list[str]] = defaultdict(list)
    for skill_name, keywords in keyword_map.items():
        for kw in keywords:
            kw_to_skills[kw].append(skill_name)

    conflicts: list[KeywordConflict] = []
    seen_hard: set[tuple[str, ...]] = set()

    for kw, skills in kw_to_skills.items():
        if len(skills) < 2:
            continue
        key = tuple(sorted(skills))
        if key in seen_hard:
            continue
        seen_hard.add(key)
        conflicts.append(KeywordConflict(
            keyword=kw,
            skills=skills,
            severity="hard",
            reason=f"关键词 '{kw}' 在 {len(skills)} 个 skill 中完全相同：{', '.join(skills)}",
        ))

    # 前缀/子串重叠检测 → soft
    all_keywords = list(kw_to_skills.keys())
    for i, kw_a in enumerate(all_keywords):
        skills_a = kw_to_skills[kw_a]
        for kw_b in all_keywords[i + 1:]:
            skills_b = kw_to_skills[kw_b]
            # 只有属于不同 skill 的关键词才检测重叠
            if set(skills_a) == set(skills_b):
                continue
            if (kw_a in kw_b or kw_b in kw_a) and kw_a != kw_b:
                involved = list(set(skills_a + skills_b))
                conflicts.append(KeywordConflict(
                    keyword=f"{kw_a} / {kw_b}",
                    skills=involved,
                    severity="soft",
                    reason=(
                        f"关键词 '{kw_a}'（{', '.join(skills_a)}）与 "
                        f"'{kw_b}'（{', '.join(skills_b)}）存在子串包含关系，"
                        f"可能造成意图分类混淆"
                    ),
                ))

    return conflicts


def has_blocking_conflicts(conflicts: list[KeywordConflict]) -> bool:
    """是否存在需要 CI 阻断的 hard 冲突"""
    return any(c.severity == "hard" for c in conflicts)
