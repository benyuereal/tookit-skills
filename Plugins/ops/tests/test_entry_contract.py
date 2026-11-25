#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""入口契约测试：文档里承诺的入口必须真实存在且可用。

为什么需要它：入口写在文档里，代码改了文档不动，或者反过来 ——
agent 按文档找过去发现是空的。这类错**不会让任何测试变红**，
只会让人在真用的时候撞墙。

本文件把「文档承诺」变成机检：
  · CLAUDE.md 指向的文件/脚本必须存在
  · 里面写的命令必须真能跑
  · marketplace.json 的 ops 描述必须提到 op-optimize（否则 Claude 匹配不到）
"""
import json
import re
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[3]
CLAUDE_MD = REPO / "CLAUDE.md"
MARKET = REPO / ".claude-plugin/marketplace.json"
ENTRY_SKILL = REPO / "Plugins/ops/skills/op-optimize/SKILL.md"


def test_claude_md_exists():
    """仓库根必须有 CLAUDE.md —— 否则 Claude 进来不知道从哪起手。"""
    assert CLAUDE_MD.is_file(), (
        "仓库根没有 CLAUDE.md。Claude Code 启动时读的是它，"
        "不是 AGENTS.md、不是 SKILL.md。")


def test_claude_md_points_to_real_entry():
    """CLAUDE.md 必须指向真实存在的 op-optimize，且说明它是唯一入口。"""
    text = CLAUDE_MD.read_text(encoding="utf-8")
    assert "Plugins/ops/skills/op-optimize" in text, "没指向 op-optimize"
    assert ENTRY_SKILL.is_file(), "指向了 op-optimize 但那个 skill 不存在"
    assert "唯一入口" in text, "没说明它是唯一入口"


def test_claude_md_declares_the_three_hard_rules():
    """三条硬规矩必须在 CLAUDE.md 里 —— 否则 agent 会绕过门禁。"""
    text = CLAUDE_MD.read_text(encoding="utf-8")
    for kw in ("loop.py", "event_isolated", "card_sync"):
        assert kw in text, "CLAUDE.md 缺硬规矩关键词 %r" % kw


def test_every_script_mentioned_in_claude_md_exists():
    """**核心反向**：CLAUDE.md 里 `scripts/xxx.py` 形式的引用必须真实存在。

    这是防「文档写了但文件没了」——实测踩过：文档里写 `scripts/ab.py`，
    实际它在各卡的 bench/ 下。
    """
    text = CLAUDE_MD.read_text(encoding="utf-8")
    missing = []
    for m in re.finditer(r"scripts/([a-z_]+\.py)", text):
        if not (REPO / "scripts" / m.group(1)).is_file():
            missing.append(m.group(1))
    assert missing == [], "CLAUDE.md 提到但不存在的脚本：%s" % sorted(set(missing))


def test_marketplace_ops_description_mentions_optimize():
    """**核心反向**：ops 插件的 description 必须提到「优化」。

    实测：原来只写「算子卡域 —— 查卡片」，Claude 看到不会想到用它做优化。
    description 是 Claude 匹配 skill 的唯一依据。
    """
    d = json.loads(MARKET.read_text(encoding="utf-8"))
    ops = [p for p in d["plugins"] if p["name"] == "ops"]
    assert ops, "marketplace.json 里没有 ops 插件"
    desc = ops[0].get("description") or ""
    assert "op-optimize" in desc, "description 没提 op-optimize，Claude 匹配不到"
    assert "优化" in desc, "description 没提「优化」"


def test_entry_skill_description_covers_new_operator():
    """op-optimize 的 description 必须覆盖「新算子」与「语言选型」两类场景。"""
    text = ENTRY_SKILL.read_text(encoding="utf-8")
    m = re.search(r'^description:\s*"(.+?)"$', text, re.M | re.S)
    assert m, "op-optimize 没有 description"
    desc = m.group(1)
    assert "op_bootstrap" in desc, "description 没提新算子建卡"
    assert "lang_select" in desc, "description 没提语言选型"


@pytest.mark.parametrize("cmd", [
    ["python3", "-I", "scripts/card_sync.py", "--check"],
    ["python3", "-I", "scripts/check_schema_fields.py", "--selftest"],
])
def test_documented_commands_actually_run(cmd):
    """文档里的命令必须真能跑 —— 不是摆设。"""
    p = subprocess.run(cmd, cwd=str(REPO), capture_output=True, text=True,
                       timeout=120)
    assert p.returncode == 0, "%s 失败：\n%s\n%s" % (cmd, p.stdout[-400:],
                                                     p.stderr[-400:])
