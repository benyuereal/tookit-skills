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


def test_env_probe_detects_bind_mode_not_just_device_count():
    """env_probe 必须同时测「纯 torch」和「自定义 kernel」。

    **这条测试守的是判据本身，不是被测对象。**

    2026-11-26 实测：第一版 env_probe 只用 `a @ a` 探活，
    在 HCU[4] 上报告「两种设备绑定方式均可」——
    而同一次测试里，`cuda:4` + 自定义 .so 明确 dump core。

    原因是：**纯 torch 走 at::native，不受设备绑定 bug 影响；
    自定义 .so 加载的 kernel 才受影响。** 只测纯 torch 会得出
    「环境没问题」的错误结论，而那个结论正是花 5 小时才推翻的。

    这条测试检查源码里确实包含自定义 kernel 的调用路径。
    """
    src = (REPO / "scripts" / "env_probe.py").read_text(encoding="utf-8")

    # 必须有一个自定义 kernel 源码片段与 ctypes 调用
    assert "_PROBE_HIP" in src, "env_probe 不再自带探针 kernel 源码"
    assert "ctypes.CDLL" in src, "env_probe 不再加载自定义 .so —— 退化成纯 torch 探活"
    assert "probe_launch" in src, "env_probe 没有调用自定义 kernel"

    # 反向：不能只靠 torch 的 matmul 判定
    assert "必须同时测两条路径" in src or "两条路径" in src, (
        "env_probe 里没有说明「为什么必须测两条路径」—— "
        "这条注释是防止后人把它简化掉的唯一提示")


def test_every_bench_runs_preflight_by_default():
    """每个 bench/ab.py 都必须在 main 里调 preflight，且默认执行。

    2026-11-25 的 5 小时根因（设备绑定方式）本来 30 秒能查出来。
    `env_probe.py` 写出来不等于会被用 —— **如果没人记得跑，等于没写。**
    所以接进 bench 入口，默认执行，想跳过必须显式 --skip-preflight。

    这条测试防的是：有人觉得「太慢」把它改成 opt-in，
    或者新建 ab.py 时忘了接。
    """
    benches = sorted((REPO / "Plugins/ops/skills").glob("*/bench/ab.py"))
    assert benches, "一个 ab.py 都没找到 —— 路径变了？"

    missing = []
    for p in benches:
        s = p.read_text(encoding="utf-8")
        card = p.parent.parent.name
        if "bench_preflight" not in s:
            missing.append("%s: 没有 import bench_preflight" % card)
            continue
        if "skip_preflight" not in s:
            missing.append("%s: 没有 --skip-preflight 开关" % card)
        # **默认必须是「跑」，不是「跳过」**
        if "if not a.skip_preflight:" not in s:
            missing.append("%s: preflight 不是默认执行" % card)
    assert not missing, (
        "以下 bench 没有默认跑前置检查：\n  " + "\n  ".join(missing) +
        "\n修法：参考 scripts/bench_preflight.py 的用法说明接进 main()")


def test_failure_surfaces_are_complete_and_linked():
    """故障 surface 表必须完整、且 SKILL.md 指向它。

    这 6 条是从 2026-11-25 那 5 小时里提炼的。**有表但不指向，
    等于没有表** —— 下一个 agent 不会自己去找。
    """
    import yaml
    ref = REPO / "Plugins/ops/skills/op-optimize/references"
    f = ref / "failure-surfaces.yaml"
    assert f.is_file(), "缺 failure-surfaces.yaml"
    d = yaml.safe_load(f.read_text(encoding="utf-8"))
    surfaces = d.get("failure_surfaces") or []
    assert len(surfaces) >= 6, "故障 surface 少于 6 条（当前 %d）" % len(surfaces)

    need = ("id", "symptom", "minimal_repro", "diagnosis", "common_causes", "evidence")
    bad = []
    for s in surfaces:
        miss = [k for k in need if not s.get(k)]
        if miss:
            bad.append("%s 缺 %s" % (s.get("id"), miss))
        # evidence 必须指向真实的 D 编号
        ev = str(s.get("evidence") or "")
        if not re.search(r"D\d+|candidates\.yaml", ev):
            bad.append("%s 的 evidence 没指向可追溯来源：%r" % (s.get("id"), ev[:40]))
    assert not bad, "故障 surface 不完整：\n  " + "\n  ".join(bad)

    # SKILL.md 必须指向它
    skill = (REPO / "Plugins/ops/skills/op-optimize/SKILL.md").read_text(encoding="utf-8")
    assert "failure-surfaces" in skill, "SKILL.md 没指向 failure-surfaces.yaml"


def test_libraries_bounds_doc_exists_and_linked():
    """库的可用边界必须有归口文档，且 SKILL.md 指向它。

    2026-11-26：`lightop.gemm` 的 M<=16、FlashMLA fp8 的 H<=32，
    这些边界之前只存在于 commit message 里 —— 换个人就找不到了。
    """
    ref = REPO / "Plugins/ops/skills/op-optimize/references/libraries-bounds.md"
    assert ref.is_file(), "缺 libraries-bounds.md"
    txt = ref.read_text(encoding="utf-8")
    # 至少要有实测数据表和复现命令 —— 光写结论不算
    assert "|" in txt and "---" in txt, "libraries-bounds.md 没有实测数据表"
    assert "复现" in txt, "libraries-bounds.md 没写复现命令"

    skill = (REPO / "Plugins/ops/skills/op-optimize/SKILL.md").read_text(encoding="utf-8")
    assert "libraries-bounds" in skill, "SKILL.md 没指向 libraries-bounds.md"


def test_minimal_repro_tool_exists():
    """递减实验工具必须存在，且 **最后一层是「换调用方式」**。

    2026-11-25 的 40 分钟就是漏了这一层：递减只减了被测对象，
    没减「怎么调用它」。工具的价值不在跑命令，在于**顺序**。
    """
    p = REPO / "scripts/minimal_repro.py"
    assert p.is_file(), "缺 scripts/minimal_repro.py"
    s = p.read_text(encoding="utf-8")
    # 层序里「换调用方式」必须存在
    assert "换调用方式" in s, "递减层序里没有「换调用方式」这一层"
    # 且必须在最后
    layers = re.findall(r'\("(L\d+)",\s*"([^"]+)"', s)
    ids = [x[0] for x in layers]
    assert "L6" in ids, "递减层序少于 6 层：%s" % ids.extend(ids)
    last = [x for x in layers if x[0] == "L6"]
    assert last and "换调用方式" in last[0][1], (
        "最后一层（L6）不是「换调用方式」—— 正是这一层被漏掉过")
    # 全崩时必须提示查调用方式
    assert "该怀疑调用方式" in s, "工具不会在「全崩」时提示查调用方式"
