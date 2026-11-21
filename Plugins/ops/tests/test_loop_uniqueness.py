#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""闭环实现唯一性守卫（engine-check 的一条）。

背景：仓库里曾经有三份「算子优化闭环」——ops/op-optimize 的 loop.py、
dcu-core/dcu-orchestrator 的 op-optimize 工作流、dcu-perf/dcu-op-optimization
的 5 步闭环。三份互相矛盾，用户从任一入口进来看到不同的判定口径。

收敛后的约定：
  - 判定与状态只有一处实现：Plugins/ops/skills/op-optimize/loop.py
  - 方向预算常量（MAX_DIRECTION_REFUTES / MAX_IMPL_FAILS）只允许在那里定义
  - 其他位置只能「路由到它」，不能自己定义一套

本测试守的是「不再长出第二套」，不是「一次改完」——存量违规用 KNOWN_DUPLICATES
冻结并附迁移说明，新增则直接失败。
"""
import re
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
LOOP_DIR = REPO / "Plugins/ops/skills/op-optimize"

# 冻结的存量重复：{相对路径: 迁移说明}。
# 清掉一处就从这里删一行；不要往里加新条目。
KNOWN_DUPLICATES = {
    "Plugins/dcu-core/skills/dcu-orchestrator/workflows/op-optimize.yaml":
        "六节点闭环；应收敛为 profile→roofline→op-loop→distill 四节点",
    "Plugins/dcu-perf/skills/dcu-op-optimization/SKILL.md":
        "自带「标准 5 步闭环」；应降为纯路由 + references/casebook.md",
    "Plugins/dcu-perf/skills/dcu-op-optimization/README.md":
        "同一 skill 的 README 又定义一遍「算子优化标准闭环」；随 SKILL.md 一起收敛",
}

BUDGET_RE = re.compile(r"(?m)^\s*(MAX_DIRECTION_REFUTES|MAX_IMPL_FAILS)\s*=")
# 「自己定义一套**算子优化**闭环」的可疑信号。
# 必须同时出现「算子优化」这个限定——「性能分析闭环」「正确性闭环」
# 「验证闭环」是别的概念，不是本仓库唯一化的对象。
# （第一版只认「闭环」，误伤了 6 个无关 skill；这是收紧后的判据。）
LOOP_DEFINITION_RE = re.compile(
    r"(?m)^#{1,3}[^\n]*算子优化[^\n]*闭环"
    r"|(?m)^#{1,3}[^\n]*标准\s*\d+\s*步闭环"
    r"|(?m)^#{1,3}[^\n]*闭合优化循环")


def _declares_itself_the_only_loop(text):
    """该文档是否明确声明「闭环以 ops/op-optimize 为准」——若是，它只是路由。"""
    return bool(re.search(r"ops/op-optimize|op-optimize/loop\.py", text))


def _py_files():
    for p in REPO.rglob("*.py"):
        if ".git" in p.parts or "__pycache__" in p.parts:
            continue
        yield p


def _rel(p):
    return str(p.relative_to(REPO))


class TestLoopUniqueness(unittest.TestCase):
    def test_budget_constants_defined_in_exactly_one_place(self):
        """方向预算常量只允许在 op-optimize/loop.py 定义。"""
        hits = []
        for p in _py_files():
            try:
                text = p.read_text(encoding="utf-8", errors="ignore")
            except OSError:
                continue
            for m in BUDGET_RE.finditer(text):
                hits.append((_rel(p), m.group(1)))
        owners = sorted({h[0] for h in hits})
        self.assertEqual(
            owners, ["Plugins/ops/skills/op-optimize/loop.py"],
            "方向预算常量出现在了非引擎文件里：%s\n"
            "判定口径必须只有一处实现，否则两个入口会给出不同结论。" % owners)

    def test_no_new_loop_definitions(self):
        """不允许出现 KNOWN_DUPLICATES 之外的「第二套闭环」定义。"""
        found = []
        for p in REPO.rglob("*.md"):
            if ".git" in p.parts or "__pycache__" in p.parts:
                continue
            rel = _rel(p)
            if rel in KNOWN_DUPLICATES:
                continue
            try:
                text = p.read_text(encoding="utf-8", errors="ignore")
            except OSError:
                continue
            if LOOP_DEFINITION_RE.search(text) and not _declares_itself_the_only_loop(text):
                found.append(rel)
        self.assertEqual(
            found, [],
            "出现了新的闭环定义（应改为路由到 ops/op-optimize）：\n" + "\n".join(found))

    def test_frozen_duplicates_still_exist(self):
        """冻结清单必须指向真实文件；文件被删就该把条目也删掉。"""
        stale = [rel for rel in KNOWN_DUPLICATES if not (REPO / rel).is_file()]
        self.assertEqual(
            stale, [],
            "以下冻结条目指向的文件已不存在，请从 KNOWN_DUPLICATES 删除：\n"
            + "\n".join(stale))


if __name__ == "__main__":
    unittest.main()
