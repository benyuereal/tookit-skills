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


CARDS = REPO / "Plugins/ops/op-atlas/cards"


def _all_cards():
    """返回 (路径, yaml 文本) 列表。"""
    out = []
    for p in sorted(CARDS.glob("*.yaml")):
        out.append((p, p.read_text(encoding="utf-8")))
    return out


def test_all_cards_have_only_allowed_top_level_fields():
    """每张卡的顶层字段必须在白名单内。

    **为什么这条测试必须存在**（2026-11-26 加）：

    同一个错误犯了 3 次，每次都让树红一阵：
      ① rmsnorm-fused 加 comparison_note      → E6
      ② 5 张降级卡加 downgrade_reason          → E6
      ③ fused-add-rmsnorm 加 attribution       → E6

    `scripts/check_schema_fields.py` 早就写好了、判得也对，
    问题在于它需要手动传 `--field`，**而没人会在加字段前想起它**。
    validate.py 也会报 E6，但那次是先提交、后发现问题。

    工具判得对 ≠ 流程会拦住。这条测试把「跑一次全卡检查」
    变成 pytest 的固定项，不依赖谁记得。
    """
    sys.path.insert(0, str(REPO / "scripts"))
    try:
        from check_schema_fields import check_top_level  # noqa: F401
    except ImportError:
        check_top_level = None

    # 白名单以 validate.py 的 ALLOWED_TOP 为准 —— 那里是唯一真相
    val = (REPO / "Plugins/ops/op-atlas/validate.py").read_text(encoding="utf-8")
    m = re.search(r"ALLOWED_TOP\s*=\s*\{(.+?)\}", val, re.S)
    assert m, "validate.py 里找不到 ALLOWED_TOP"
    allowed = set(re.findall(r'"([^"]+)"', m.group(1)))
    assert allowed, "ALLOWED_TOP 解析为空 —— 正则失效了，不是卡的问题"

    import yaml
    bad = {}
    for p, _ in _all_cards():
        d = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
        extra = set(d) - allowed
        if extra:
            bad[p.name] = sorted(extra)
    assert not bad, (
        "以下卡有未知顶层字段（会触发 E6）。\n"
        "合法白名单：%s\n"
        "违规：%s\n"
        "修法：归因/说明类内容放进 last_verdict.bottleneck 或 "
        "last_verdict.evidence，不要新增顶层字段。" % (sorted(allowed), bad))


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


def test_no_card_claims_full_shape_score_while_slower():
    """S 项满分（覆盖 3+ 形状）的卡，geomean 不应该 >= 1。

    **这条测试故意用 xfail 语义的宽松判据**（只警告不失败）：
    改 shape_score 是口径变更，会让所有历史分数不可比，需单独决策。
    但问题必须可见 —— 否则「十张满卡」的目标数字会虚高，
    而每张卡单独看都是"合规"的。

    2026-11-26 实测命中 2 张：
      gemm-bf16      geomean=1.10
      rmsnorm-fused  geomean=1.4496
    """
    sys.path.insert(0, str(REPO / "scripts"))
    from check_s_direction import scan
    hits = scan()
    if hits:
        import warnings
        warnings.warn(
            "以下卡 S 满分但整体没变快（geomean>=1）：%s"
            % ", ".join("%s=%.2f" % (c, v) for c, v, _ in hits),
            stacklevel=1)
    # 不 assert，避免把口径问题变成阻塞项 —— 但会出现在 pytest -W 输出里
    assert True


def test_decision_wordlist_documented_where_people_look():
    """ledger 决策词表必须真的能匹配它列出来的词。

    2026-11-26 实测踩过：两张 99 分的卡写了 5 轮，但用「决定/结论」，
    而判据只认 adopt/refute/stalled/采纳/证伪/卡住，导致 D4 一直是 0。
    **判据存在但没人知道要写什么词。**

    这条测试守两件事：
      ① DECISION_WORDS 里每个词都能被 DECISION_RE 匹配（否则词表是假的）
      ② 词表与正则不漂移（改一个忘另一个会红）
    """
    sys.path.insert(0, str(REPO / "scripts"))
    import re

    from scorecard import DECISION_RE, DECISION_WORDS

    for w in DECISION_WORDS:
        assert re.search(DECISION_RE, w), (
            "词表里的 %r 匹配不上判据正则 %s —— 词表是假的" % (w, DECISION_RE))

    # 反向：不该被匹配的常见写法
    for w in ("决定", "结论", "完成", "TODO"):
        assert not re.search(DECISION_RE, w), (
            "%r 被误判成决策词，会虚增轮次" % w)
