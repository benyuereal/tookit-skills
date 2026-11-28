#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""闭环实现唯一性守卫（engine-check 的一条）。

背景：仓库里曾经有三份「算子优化闭环」——ops/op-optimize 的 loop.py、
dcu-core/dcu-orchestrator 的 op-optimize 工作流、dcu-perf/dcu-op-optimization
的 5 步闭环。三份互相矛盾，用户从任一入口进来看到不同的判定口径。

收敛后的约定：
  - 判定与状态只有一处实现：Plugins/ops/engine/loop.py（M2/M3 迁移后）
  - 旧路径 Plugins/ops/skills/op-optimize/loop.py 只做 shim 转发，不含实现
  - 方向预算常量（MAX_DIRECTION_REFUTES / MAX_IMPL_FAILS）只允许在 engine 版定义
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
        # 运行时目录里可能有整仓副本（driver 工作区），不是仓库实现
        if (".git" in p.parts or "__pycache__" in p.parts
                or ".artifacts" in p.parts or "runs" in p.parts):
            continue
        yield p


def _rel(p):
    return str(p.relative_to(REPO))


# shim 允许的最大行数。判据 2 与它的反向测试共用这一处常量。
SHIM_MAX_LINES = 30


def _shim_impl_violations(text):
    """shim 文本里「有实现」的证据清单；空列表 = 它是纯转发壳。

    只有这一处判据：正向（真实 shim 必须干净）与反向（塞进实现必须被抓到）
    都调它，避免「校验器说通过、其实根本不会报错」。
    用 AST 找真的函数定义，不做字符串匹配 —— shim 的 docstring 里就写着
    「不许 def cmd_」，字符串匹配会自伤。
    """
    import ast
    probs = []
    n = len(text.splitlines())
    if n > SHIM_MAX_LINES:
        probs.append("行数 %d > %d" % (n, SHIM_MAX_LINES))
    tree = ast.parse(text)
    funcs = sorted({node.name for node in ast.walk(tree)
                    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))})
    if funcs:
        probs.append("定义了函数 %s" % funcs)
    return probs


class TestLoopUniqueness(unittest.TestCase):
    def test_budget_constants_defined_in_exactly_one_place(self):
        """方向预算常量只允许在 engine/loop.py 定义（原 op-optimize/loop.py 已迁走）。"""
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
            owners, ["Plugins/ops/engine/loop.py"],
            "方向预算常量出现在了非引擎文件里：%s\n"
            "判定口径必须只有一处实现，否则两个入口会给出不同结论。" % owners)

    def test_no_new_loop_definitions(self):
        """不允许出现 KNOWN_DUPLICATES 之外的「第二套闭环」定义。"""
        found = []
        for p in REPO.rglob("*.md"):
            if (".git" in p.parts or "__pycache__" in p.parts
                    or ".artifacts" in p.parts or "runs" in p.parts):
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


class TestEngineMigration(unittest.TestCase):
    """ENGINE 迁移（M2+M3）的守卫。

    背景：loop.py 有 1942 行、12 个子命令，被另一个插件
    （dcu-core/dcu-orchestrator/tasks/op-optimize.yaml）引用。
    它已经是「调度引擎」而不是 op-optimize 这个 skill 的实现细节，
    物理位置迁到 Plugins/ops/engine/，旧路径保留 shim。

    本测试守四条：
      A. shim 里不许有实现（行数 ≤30，且 AST 里零函数定义）
      B. 全仓只允许一份 `def cmd_run(` 的完整实现
      C. engine 版与 shim 的 `--help` stdout 必须逐字节相同（行为没变）
      D. 反向：往 shim 塞 `def cmd_fake()`，A 的判据必须报错
    """

    # parents[1] = Plugins/ops（parents[3] 是仓库根，会算错）
    _OPS = Path(__file__).resolve().parents[1]
    ENGINE = _OPS / "engine" / "loop.py"
    SHIM = _OPS / "skills" / "op-optimize" / "loop.py"

    def test_shim_has_no_implementation(self):
        txt = self.SHIM.read_text(encoding="utf-8")
        self.assertEqual(_shim_impl_violations(txt), [],
                         "shim 里出现了实现（行数上限 %d，且不许定义函数）——"
                         "实现必须留在 engine/" % SHIM_MAX_LINES)

    def test_engine_exists_and_has_implementation(self):
        self.assertTrue(self.ENGINE.is_file(),
                        "engine/loop.py 不存在 —— 迁移未完成")
        txt = self.ENGINE.read_text(encoding="utf-8")
        self.assertIn("def cmd_run(", txt, "engine/loop.py 里没有实现")
        self.assertGreater(len(txt.splitlines()), 500,
                           "engine/loop.py 太短，实现可能没搬全")

    def test_only_one_full_implementation(self):
        """全仓（不只 Plugins/ops）只许有一份 `def cmd_run(`。"""
        hits = []
        for p in REPO.rglob("*.py"):
            if (".git" in p.parts or "__pycache__" in p.parts
                    or ".artifacts" in p.parts or "runs" in p.parts):
                continue
            if "tests" in p.parts:          # 测试里的字符串是断言，不是实现
                continue
            try:
                if "def cmd_run(" in p.read_text(encoding="utf-8", errors="ignore"):
                    hits.append(_rel(p))
            except OSError:
                continue
        self.assertEqual(hits, ["Plugins/ops/engine/loop.py"],
                         "`def cmd_run(` 的完整实现只许有一份（engine/loop.py），"
                         "实际：%s" % hits)

    def test_shim_help_matches_engine(self):
        """shim 的 --help stdout 必须与 engine 版逐字节相同（行为没变）。

        shim 的 stderr 弃用告警不算——只比 stdout。
        """
        import subprocess, sys
        def help_of(path):
            r = subprocess.run([sys.executable, "-I", str(path), "--help"],
                               capture_output=True, text=True)
            self.assertEqual(r.returncode, 0,
                             "%s --help 退出码 %d：%s" % (path, r.returncode, r.stderr))
            return r.stdout
        shim_out, engine_out = help_of(self.SHIM), help_of(self.ENGINE)
        self.assertTrue(engine_out.strip(), "engine --help 没有输出")
        self.assertEqual(shim_out, engine_out,
                         "shim 与 engine 的 --help 不一致 —— 转发没生效")

    def test_shim_impl_check_catches_injected_function(self):
        """反向测试：判据 2 必须真的会报错，不能是空判据。

        往当前 shim 末尾塞一个 `def cmd_fake(): pass`，_shim_impl_violations
        必须非空。这条防的是「测试写成了永远通过」。
        """
        injected = self.SHIM.read_text(encoding="utf-8") + "\n\ndef cmd_fake():\n    pass\n"
        probs = _shim_impl_violations(injected)
        self.assertTrue(probs, "塞进 `def cmd_fake()` 后判据没报错 —— 它是空判据")
        self.assertTrue(any("cmd_fake" in p for p in probs),
                        "报错了但没点出 cmd_fake：%s" % probs)

