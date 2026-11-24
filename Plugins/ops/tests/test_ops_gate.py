#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""P0-4 唯一的 PreToolUse hook —— ops_gate.py 的测试。

三类证据：
  ① 触发面：只有 3 个动作会进闸，别的工具/命令一律 exit 0（不然会挡住正常干活）；
  ② 验收：改源码不重建 / 删 result.json 字段 / 直接 commit ⇒ 三次都被拦（exit 2）；
  ③ 失败关闭：门禁自己坏了（脚本缺失、manifest 读不出来、stdin 不是 JSON）必须 exit 2，
     不能写 `sys.exit(0)` 把「坏了」当「放行」。

每个用例都在临时目录里复制一份**真的** opverify.py / opmanifest.py / loop.py / ops_gate.py，
再从子进程喂 stdin JSON —— 走的就是 Claude Code 调 hook 的那条路，不是 mock。
"""
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
HOOK = REPO / "Plugins" / "ops" / "hooks" / "ops_gate.py"
sys.path.insert(0, str(HOOK.parent))
import ops_gate  # noqa: E402

CONTRACT = """\
op: demo
version: 1

shapes:
  q: [8, 8, 64]

dtypes:
  q: bfloat16

tolerance:
  vs_fp32_ref:
    max_abs: 1.0e-2
    cosine: 0.999
  variant_vs_variant: bit_exact

invariants:
  - name: determinism
    mode: bit_exact

gates:
  - name: correctness
    script: bench/gate.sh
    required: true

floor:
  metric: hbm_bandwidth
  peak: 1000
  unit: GB/s
  basis: "demo 地板"
  at_floor_threshold: 0.85

manifest:
  required_roles: [source, loaded]
  env_keys: [gfx]
"""

CANDIDATES_OK = """\
directions:
  - id: d1
    statement: "演示用攻击面"
    status: open
    refutes: 0

candidates:
  - id: c1
    direction: d1
    status: pending
    priority: 1
"""

CANDIDATES_NO_DIRECTION = """\
directions:
  - id: d1
    statement: "演示用攻击面"
    status: open
    refutes: 0

candidates:
  - id: c1
    status: pending
    priority: 1
"""


def _sha(p):
    h = hashlib.sha256()
    h.update(Path(p).read_bytes())
    return h.hexdigest()


def record_manifest(src, loaded, out, op="demo", shape="s", src_older=60.0):
    """按「源码先落地、产物后编译」的顺序冻一份 manifest。

    模块级而不是 Fixture 的方法：验收脚本要在**同一棵已存在的树**上重新 record，
    不该为此把整棵树重建一遍。
    """
    now = time.time()
    os.utime(src, (now - src_older, now - src_older))
    os.utime(loaded, (now - 10, now - 10))
    man = {
        "schema": "opmanifest/v1",
        "op": op,
        "shape": shape,
        # 相对时间：这份 manifest 是「刚 record 的」，不是历史记录。
        # 写成固定日期会让 `STALE_DAYS` 的陈旧豁免意外生效，测不到本来要测的拦截。
        "recorded_at": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(now)),
        "artifacts": [
            {"role": "source", "path": str(src), "sha256": _sha(src),
             "bytes": Path(src).stat().st_size, "mtime": Path(src).stat().st_mtime},
            {"role": "loaded", "path": str(loaded), "sha256": _sha(loaded),
             "bytes": Path(loaded).stat().st_size, "mtime": Path(loaded).stat().st_mtime},
        ],
        "env": {"gfx": "gfx938"},
    }
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(json.dumps(man))


class Fixture:
    """一棵可写的最小仓库树：一张卡 + 一个 kernel 源码 + 一份 manifest。"""

    def __init__(self, root):
        self.root = Path(root)
        (self.root / "scripts").mkdir(exist_ok=True)
        for n in ("opverify.py", "opmanifest.py", "opgate_checks.py"):
            shutil.copy(REPO / "scripts" / n, self.root / "scripts" / n)
        hookdir = self.root / "Plugins" / "ops" / "hooks"
        hookdir.mkdir(parents=True)
        self.hook = hookdir / "ops_gate.py"
        shutil.copy(HOOK, self.hook)
        opt = self.root / "Plugins" / "ops" / "skills" / "op-optimize"
        opt.mkdir(parents=True)
        # 这个测试模拟装好的插件布局：hook 期望 <skills>/op-optimize/loop.py 存在。
        # 实现现在在 engine/，但**装好后的布局仍需要那个路径**（shim 的作用），
        # 所以把 shim 拷过去，而不是 engine 实现。
        shutil.copy(REPO / "Plugins" / "ops" / "skills" / "op-optimize" / "loop.py",
                    opt / "loop.py")
        # shim 转发到 engine/loop.py，模拟布局里也要有它，否则 shim 会 sys.exit
        eng = self.root / "Plugins" / "ops" / "engine"
        eng.mkdir(parents=True, exist_ok=True)
        shutil.copy(REPO / "Plugins" / "ops" / "engine" / "loop.py",
                    eng / "loop.py")

        self.card = self.root / "Plugins" / "ops" / "skills" / "op-demo"
        (self.card / "bench").mkdir(parents=True)
        (self.card / "contract.yaml").write_text(CONTRACT, encoding="utf-8")
        (self.card / "bench" / "gate.sh").write_text("#!/bin/sh\nexit 0\n")
        (self.card / "bench" / "ab.py").write_text("print('ab')\n")

        self.src = self.root / "kernel" / "demo_kernel.hip"
        self.src.parent.mkdir(parents=True)
        self.src.write_text("// v1\n")
        self.other_src = self.root / "kernel" / "another.hip"
        self.other_src.write_text("// other\n")
        self.loaded = self.root / "kernel" / "build" / "executable.so"
        self.loaded.parent.mkdir(parents=True)
        self.loaded.write_text("so-v1")

        self.run = self.root / "runs" / "op-optimize" / "2026-10-28-demo" / "profile"
        self.run.mkdir(parents=True)
        self.result = self.run / "result.json"
        self.result.write_text(json.dumps(
            {"node": "profile", "op": "demo", "fingerprint": {"gfx": "gfx938"}}))
        self.record()

    # ---------------------------------------------------------------- 工具
    def record(self, src_older=60.0):
        record_manifest(self.src, self.loaded, self.run / "manifest.json", src_older=src_older)

    def gate(self):
        return ops_gate.Gate(self.root)

    def hook_rc(self, payload):
        """真的走一遍 Claude Code 那条路：子进程 + stdin JSON。"""
        cp = subprocess.run([sys.executable, "-I", str(self.hook)],
                            input=json.dumps(payload), capture_output=True,
                            text=True, cwd=str(self.root), timeout=180)
        return cp.returncode, cp.stdout + cp.stderr

    def write_payload(self, path, tool="Edit"):
        return {"tool_name": tool, "tool_input": {"file_path": str(path)},
                "cwd": str(self.root)}

    def bash_payload(self, cmd):
        return {"tool_name": "Bash", "tool_input": {"command": cmd}, "cwd": str(self.root)}

    def bench(self):
        return self.hook_rc(self.bash_payload(
            "%s %s" % (sys.executable, self.card / "bench" / "ab.py")))

    def commit(self):
        return self.hook_rc(self.bash_payload("git add -A && git commit -m 'x'"))


class Base(unittest.TestCase):
    def setUp(self):
        self.td = tempfile.mkdtemp(prefix="opsgate-")
        self.f = Fixture(self.td)

    def tearDown(self):
        shutil.rmtree(self.td, ignore_errors=True)


# ------------------------------------------------------------------ ① 触发面
class TestTriggers(Base):
    def test_ordinary_tools_are_not_gated(self):
        g = self.f.gate()
        self.assertIsNone(g.classify({"tool_name": "Read",
                                      "tool_input": {"file_path": str(self.f.src)}})[0])
        self.assertIsNone(g.classify({"tool_name": "Grep", "tool_input": {"pattern": "x"}})[0])
        self.assertIsNone(g.classify(self.f.bash_payload("ls -l"))[0])
        self.assertIsNone(g.classify(self.f.bash_payload("python3 -m pytest Plugins/ops -q"))[0])
        self.assertIsNone(g.classify(self.f.bash_payload("git status"))[0])
        self.assertIsNone(g.classify(self.f.bash_payload("cat /tmp/x.py"))[0])
        # 写普通文件不算写 kernel 源码
        self.assertIsNone(g.classify(self.f.write_payload(self.td + "/notes.txt", "Write"))[0])

    def test_write_kernel_source_is_gated(self):
        g = self.f.gate()
        self.assertEqual(g.classify(self.f.write_payload(self.f.src))[0], "write_source")
        # 没进过 manifest 的 .hip 也算 kernel 源码（新算子的第一天）
        self.assertEqual(g.classify(self.f.write_payload(self.f.other_src))[0], "write_source")
        perf = Path(self.td) / "tl" / "perf" / "k.py"
        perf.parent.mkdir(parents=True)
        perf.write_text("x = 1\n")
        self.assertEqual(g.classify(self.f.write_payload(perf))[0], "write_source")

    def test_fast_path_skips_ordinary_actions(self):
        """`_worth_checking` 是便宜的预筛：普通 Bash / 普通写文件根本不进判定。"""
        self.assertFalse(ops_gate._worth_checking(self.f.bash_payload("ls -l")))
        self.assertFalse(ops_gate._worth_checking(self.f.bash_payload("git status")))
        self.assertFalse(ops_gate._worth_checking(
            self.f.write_payload(self.td + "/notes.txt", "Write")))
        self.assertFalse(ops_gate._worth_checking(
            {"tool_name": "Read", "tool_input": {"file_path": str(self.f.src)}}))
        # 值得查的四类仍进判定
        self.assertTrue(ops_gate._worth_checking(self.f.write_payload(self.f.src)))
        self.assertTrue(ops_gate._worth_checking(self.f.bash_payload("git commit -m x")))
        self.assertTrue(ops_gate._worth_checking(self.f.bash_payload(
            "%s %s" % (sys.executable, self.f.card / "bench" / "ab.py"))))

    def test_settings_matcher_is_narrow(self):
        cfg = json.loads((REPO / ".claude" / "settings.json").read_text(encoding="utf-8"))
        matcher = cfg["hooks"]["PreToolUse"][0]["matcher"]
        self.assertEqual(matcher, "Bash|Write|Edit")
        self.assertNotIn("MultiEdit", matcher)
        self.assertNotIn("NotebookEdit", matcher)

    def test_bench_and_commit_are_gated(self):
        g = self.f.gate()
        self.assertEqual(g.classify(self.f.bash_payload(
            "%s %s" % (sys.executable, self.f.card / "bench" / "ab.py")))[0], "run_bench")
        self.assertEqual(g.classify(self.f.bash_payload(
            "python3 Plugins/ops/engine/loop.py -C x run c1"))[0], "run_bench")
        self.assertEqual(g.classify(self.f.bash_payload(
            "cd Plugins && git -C . commit -m 'x'"))[0], "git_commit")
        # loop.py 的只读子命令不是 bench
        self.assertIsNone(g.classify(self.f.bash_payload(
            "python3 Plugins/ops/engine/loop.py -C x status"))[0])


# ------------------------------------------------------------ ② 三次都被拦
class TestAcceptance(Base):
    def test_clean_state_passes(self):
        rc, out = self.f.bench()
        self.assertEqual(rc, 0, out)
        rc, out = self.f.commit()
        self.assertEqual(rc, 0, out)

    def test_edit_source_without_rebuild_blocks_bench(self):
        """验收①：改源码不重建 —— 跑 bench 就被拦。"""
        self.f.src.write_text("// v2 改了一行没重建\n")
        rc, out = self.f.bench()
        self.assertEqual(rc, 2, out)
        self.assertIn("sha256 与记录不符", out)
        self.assertIn("改完源码不重建就是这样被拦的", out)

    def test_edit_source_without_rebuild_blocks_commit(self):
        self.f.src.write_text("// v2\n")
        rc, out = self.f.commit()
        self.assertEqual(rc, 2, out)
        self.assertIn("sha256 与记录不符", out)

    def test_touch_source_without_rebuild_is_caught(self):
        """mtime 链同样算「没重建」：内容一样但源码比产物新。"""
        now = time.time()
        os.utime(self.f.src, (now, now))
        rc, out = self.f.bench()
        self.assertEqual(rc, 2, out)
        self.assertIn("疑似改完没重建", out)

    def test_rebuild_then_pass(self):
        self.f.src.write_text("// v2\n")
        self.assertEqual(self.f.bench()[0], 2)
        self.f.record()          # 重建 + 重新 record
        rc, out = self.f.bench()
        self.assertEqual(rc, 0, out)

    def test_deleting_a_result_field_blocks_commit(self):
        """验收②：删 result.json 的字段 —— 交付件不再能回溯，commit 被拦。"""
        d = json.loads(self.f.result.read_text())
        del d["fingerprint"]
        self.f.result.write_text(json.dumps(d))
        rc, out = self.f.commit()
        self.assertEqual(rc, 2, out)
        self.assertIn("缺字段 fingerprint", out)

    def test_direct_commit_is_checked(self):
        """验收③：直接 commit 也要过闸（不是只有 bench 才查）。"""
        self.f.result.write_text("not json")
        rc, out = self.f.commit()
        self.assertEqual(rc, 2, out)
        self.assertIn("读不出来", out)

    def test_a_node_without_manifest_is_still_checked(self):
        """删 manifest 不能顺手让交付件免检 —— 那是绕路，不是合规。"""
        (self.f.run / "manifest.json").unlink()
        self.f.result.write_text(json.dumps({"node": "profile"}))
        rc, out = self.f.commit()
        self.assertEqual(rc, 2, out)
        self.assertIn("缺字段 op, fingerprint", out)

    def test_legacy_run_level_result_is_out_of_scope(self):
        """`<run>/result.json` 是一次 run 的汇总件，不是节点交付件，不按节点规则查。"""
        legacy = self.f.root / "runs" / "op-optimize" / "2026-10-27-legacy" / "result.json"
        legacy.parent.mkdir(parents=True)
        legacy.write_text(json.dumps({"op": "demo", "metrics": {}}))
        probs = self.f.gate().check_result_shape()
        self.assertEqual(probs, [], "\n".join(m for _, m in probs))


class TestManifestOwnership(Base):
    """manifest 认领谁，看它自己的 `op` 字段，不看 run 目录名。

    这是沙箱验收时抓到的真 bug：跑 bench 全放行，因为 manifest 被
    `2026-10-28-p04` 这种人为起的目录名挡在了外面 —— 门禁挂到了邻居身上。
    """

    def _rename_run(self, new):
        old = self.f.run
        p = old.parent / new
        old.rename(p)
        self.f.run = p
        self.f.result = p / "result.json"

    def test_run_dir_name_without_the_op_still_binds(self):
        self._rename_run("2026-10-28-p04")
        self.f.src.write_text("// 改一行没重建\n")
        rc, out = self.f.bench()
        self.assertEqual(rc, 2, out)
        self.assertIn("sha256 与记录不符", out)

    def test_another_ops_manifest_is_not_claimed(self):
        """别人的 manifest 不能拿来给我这张卡背书。"""
        other = self.f.root / "runs" / "op-optimize" / "2026-10-28-x" / "profile"
        other.mkdir(parents=True)
        man = json.loads((self.f.run / "manifest.json").read_text())
        man["op"] = "somebody-else"
        (other / "manifest.json").write_text(json.dumps(man))
        os.utime(other / "manifest.json", (time.time() + 60, time.time() + 60))
        self.assertEqual(self.f.gate().newest_manifest(self.f.card),
                         self.f.run / "manifest.json")


# ------------------------------------------------------------------ 写源码
class TestWriteAction(Base):
    def test_writing_the_declared_source_is_allowed(self):
        """编辑源码本身就是要制造漂移，拿身份闸拦自己是自相矛盾。"""
        rc, out = self.f.hook_rc(self.f.write_payload(self.f.src))
        self.assertEqual(rc, 0, out)

    def test_writing_another_source_while_drifted_is_blocked(self):
        """身份链已经断了还去动另一条 kernel 源码 ⇒ 拦（别在流沙上继续盖）。"""
        self.f.src.write_text("// v2\n")
        rc, out = self.f.hook_rc(self.f.write_payload(self.f.other_src))
        self.assertEqual(rc, 2, out)
        self.assertIn("身份", out)

    def test_writing_a_compiled_artifact_is_blocked(self):
        rc, out = self.f.hook_rc(self.f.write_payload(self.f.loaded))
        self.assertEqual(rc, 2, out)
        self.assertIn("role=build/loaded", out)

    def test_stale_manifest_warns_instead_of_blocking_the_whole_repo(self):
        """陈旧 manifest 只提示、不拦。

        为什么：manifest 记的是某次 run 的产物指纹，换机器/清缓存必然失效。
        让它永久挡住所有 `git commit`，身份闸就成了仓库锁 —— 那是重约束，
        而且它挡的动作与那条过期结论无关。`OPS_GATE_STRICT=1` 才恢复全量拦截。
        """
        man_path = self.f.root / "runs" / "op-optimize" / "2026-10-28-demo" / "profile" / "manifest.json"
        man = json.loads(man_path.read_text())
        man["recorded_at"] = "2020-01-01T00:00:00"
        man_path.write_text(json.dumps(man))
        self.f.src.write_text("// drifted after the record\n")   # 制造 sha 漂移

        rc, out = self.f.hook_rc(self.f.write_payload(self.f.other_src))
        self.assertEqual(rc, 0, out)
        self.assertIn("陈旧 manifest", out)

        # strict 模式恢复拦截：要全量核验时仍然拦得住
        old = os.environ.get("OPS_GATE_STRICT")
        os.environ["OPS_GATE_STRICT"] = "1"
        try:
            rc, out = self.f.hook_rc(self.f.write_payload(self.f.other_src))
        finally:
            if old is None:
                os.environ.pop("OPS_GATE_STRICT", None)
            else:
                os.environ["OPS_GATE_STRICT"] = old
        self.assertEqual(rc, 2, out)


# --------------------------------------------------- verified 提案门（F）
class TestVerifiedProposal(Base):
    """机制二：写 `verified` 资产（SKILL.md / references/*.md）必须经过提案。

    参照物是 ProposeSkills —— 只出 review card，用户保存才落盘，且必须带 evidence[]。
    这里把「落盘」这一步的闸做进 PreToolUse：**新增或改写** verified 标记才拦，
    普通文档编辑一律放行（改标题 / 改错别字 / 改普通段落 / 保留原标记行的整篇写入）。
    """

    def setUp(self):
        super().setUp()
        self.docs = self.f.card / "references"
        self.docs.mkdir(parents=True)
        self.doc = self.docs / "cards.md"
        self.doc.write_text("# 知识卡\n\nstatus: verified\n\n普通段落：这是描述。\n",
                            encoding="utf-8")
        self.skill = self.f.card / "SKILL.md"
        self.skill.write_text("# demo\n\n普通说明。\n", encoding="utf-8")

    # ---------------------------------------------------------------- 工具
    def edit(self, path, old, new, tool="Edit"):
        return {"tool_name": tool,
                "tool_input": {"file_path": str(path), "old_string": old, "new_string": new},
                "cwd": str(self.f.root)}

    def write(self, path, content):
        return {"tool_name": "Write",
                "tool_input": {"file_path": str(path), "content": content},
                "cwd": str(self.f.root)}

    def proposal(self, path, data):
        p = path.with_name(path.name + ".proposal.json")
        p.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
        return p

    def ok_proposal(self, path):
        return self.proposal(path, {"target": path.name,
                                    "evidence": ["Plugins/ops/tests/test_ops_gate.py（本门禁的验收用例）"],
                                    "approved_by": "reviewer"})

    # ------------------------------------------------------------ 正例/反例
    def test_verified_write_is_a_fourth_gated_action(self):
        g = self.f.gate()
        self.assertEqual(g.classify(self.edit(self.doc, "x", "y"))[0], "write_verified")
        self.assertEqual(g.classify(self.edit(self.skill, "x", "y"))[0], "write_verified")
        # 不在 references/ 下、也不是 SKILL.md 的 .md 不管
        ledger = self.f.card / "ledger.md"
        ledger.write_text("x\n", encoding="utf-8")
        self.assertIsNone(g.classify(self.edit(ledger, "x", "y"))[0])

    def test_adding_a_verified_line_without_proposal_blocks(self):
        rc, out = self.f.hook_rc(self.edit(
            self.doc, "普通段落：这是描述。", "普通段落：这是描述。\nstatus: verified"))
        self.assertEqual(rc, 2, out)
        self.assertIn("写 verified 需要提案", out)
        self.assertIn("cards.md.proposal.json", out)
        self.assertIn("evidence", out)

    def test_adding_a_verified_line_with_proposal_passes(self):
        self.ok_proposal(self.doc)
        rc, out = self.f.hook_rc(self.edit(
            self.doc, "普通段落：这是描述。", "普通段落：这是描述。\nstatus: verified"))
        self.assertEqual(rc, 0, out)

    def test_modifying_an_existing_verified_line_blocks(self):
        """「修改」也拦：把标记行改写（哪怕只是加注释）等于重写一条断言。"""
        rc, out = self.f.hook_rc(self.edit(
            self.doc, "status: verified", "status: verified  # 2026-11-13 复验"))
        self.assertEqual(rc, 2, out)
        self.assertIn("写 verified 需要提案", out)

    def test_skill_md_is_in_scope(self):
        rc, out = self.f.hook_rc(self.edit(
            self.skill, "普通说明。", "普通说明。\n本机已验证：这条路径可用。"))
        self.assertEqual(rc, 2, out)
        self.assertIn("写 verified 需要提案", out)

    def test_multiedit_adding_verified_blocks(self):
        payload = {"tool_name": "MultiEdit",
                   "tool_input": {"file_path": str(self.doc), "edits": [
                       {"old_string": "普通段落：这是描述。", "new_string": "普通段落：这是描述。"},
                       {"old_string": "# 知识卡", "new_string": "# 知识卡\n\n已验证：测过。"}]},
                   "cwd": str(self.f.root)}
        rc, out = self.f.hook_rc(payload)
        self.assertEqual(rc, 2, out)
        self.assertIn("写 verified 需要提案", out)

    # -------------------------------------------------- 提案本身的字段校验
    def test_proposal_missing_evidence_blocks(self):
        self.proposal(self.doc, {"target": "cards.md", "evidence": [], "approved_by": "r"})
        rc, out = self.f.hook_rc(self.edit(
            self.doc, "普通段落：这是描述。", "普通段落：这是描述。\nstatus: verified"))
        self.assertEqual(rc, 2, out)
        self.assertIn("evidence", out)

    def test_proposal_wrong_target_blocks(self):
        self.proposal(self.doc, {"target": "other.md", "evidence": ["x"], "approved_by": "r"})
        rc, out = self.f.hook_rc(self.edit(
            self.doc, "普通段落：这是描述。", "普通段落：这是描述。\nstatus: verified"))
        self.assertEqual(rc, 2, out)
        self.assertIn("不符", out)

    def test_proposal_without_approver_blocks(self):
        self.proposal(self.doc, {"target": "cards.md", "evidence": ["x"], "approved_by": "  "})
        rc, out = self.f.hook_rc(self.edit(
            self.doc, "普通段落：这是描述。", "普通段落：这是描述。\nstatus: verified"))
        self.assertEqual(rc, 2, out)
        self.assertIn("approved_by", out)

    def test_unreadable_proposal_blocks(self):
        (self.doc.with_name(self.doc.name + ".proposal.json")).write_text("{ 不是 json")
        rc, out = self.f.hook_rc(self.edit(
            self.doc, "普通段落：这是描述。", "普通段落：这是描述。\nstatus: verified"))
        self.assertEqual(rc, 2, out)
        self.assertIn("提案读不出来", out)

    # ------------------------------------------------------------ 反例（不误伤）
    def test_heading_edit_passes(self):
        rc, out = self.f.hook_rc(self.edit(self.doc, "# 知识卡", "# 知识卡（2026-11-13 修订）"))
        self.assertEqual(rc, 0, out)

    def test_typo_edit_passes(self):
        rc, out = self.f.hook_rc(self.edit(self.doc, "这是描述。", "这是描述（改错别字）。"))
        self.assertEqual(rc, 0, out)

    def test_plain_paragraph_edit_passes(self):
        rc, out = self.f.hook_rc(self.edit(
            self.doc, "普通段落：这是描述。",
            "普通段落：这是描述，而且还补了一句解释（只是正文补充）。"))
        self.assertEqual(rc, 0, out)

    def test_rewriting_the_paragraph_next_to_a_marker_passes(self):
        """标记行原样保留、只改它旁边的正文 ⇒ 不是新增断言，放行。"""
        rc, out = self.f.hook_rc(self.edit(
            self.doc, "# 知识卡\n\nstatus: verified\n\n普通段落：这是描述。",
            "# 知识卡\n\nstatus: verified\n\n普通段落：换了说法。"))
        self.assertEqual(rc, 0, out)

    def test_write_that_keeps_all_markers_passes(self):
        rc, out = self.f.hook_rc(self.write(
            self.doc, "# 知识卡\n\nstatus: verified\n\n普通段落：整篇重写但保留原标记。\n"))
        self.assertEqual(rc, 0, out)

    def test_removing_a_verified_line_passes(self):
        """删断言不会凭空造出一条 verified 结论；删证据的风险不在本门范围内。"""
        rc, out = self.f.hook_rc(self.write(self.doc, "# 知识卡\n\n普通段落。\n"))
        self.assertEqual(rc, 0, out)

    def test_unverified_and_verified_on_do_not_count(self):
        rc, out = self.f.hook_rc(self.edit(
            self.doc, "普通段落：这是描述。",
            "普通段落：status: unverified；口径带 verified_on；这两者都不算断言。"))
        self.assertEqual(rc, 0, out)

    def test_non_scope_documents_pass(self):
        # 卡片目录下的 ledger.md 不是 verified 资产落点
        ledger = self.f.card / "ledger.md"
        ledger.write_text("# ledger\n\n正文。\n", encoding="utf-8")
        rc, out = self.f.hook_rc(self.edit(ledger, "正文。", "正文。\nstatus: verified"))
        self.assertEqual(rc, 0, out)

    def test_docs_outside_the_repo_are_not_ours(self):
        outside = Path(self.td).parent / ("outside-%s.md" % os.getpid())
        outside.write_text("# 别的项目\n\n正文。\n", encoding="utf-8")
        rc, out = self.f.hook_rc(self.edit(outside, "正文。", "正文。\nstatus: verified"))
        self.assertEqual(rc, 0, out)

    # ------------------------------------------------------------ 失败关闭
    def test_write_without_content_fails_closed(self):
        """读不出要写什么 ⇒ 判不了 ⇒ 不许放行（普通 Write 一定带 content）。"""
        rc, out = self.f.hook_rc({"tool_name": "Write",
                                  "tool_input": {"file_path": str(self.doc)},
                                  "cwd": str(self.f.root)})
        self.assertEqual(rc, 2, out)
        self.assertIn("失败关闭", out)


# ------------------------------------------------------------ 方向预算（C）
class TestBudgetGate(Base):
    def _set(self, text):
        (self.f.card / "candidates.yaml").write_text(text, encoding="utf-8")

    def test_clean_candidates_do_not_block(self):
        self._set(CANDIDATES_OK)
        rc, out = self.f.commit()
        self.assertEqual(rc, 0, out)

    def test_candidate_without_direction_blocks_commit(self):
        self._set(CANDIDATES_NO_DIRECTION)
        rc, out = self.f.commit()
        self.assertEqual(rc, 2, out)
        self.assertIn("方向预算", out)


# ------------------------------------------------------------ ③ 失败关闭
class TestFailClosed(Base):
    def test_broken_contract_blocks(self):
        (self.f.card / "contract.yaml").write_text("op: demo\n")  # 缺一堆必填
        rc, out = self.f.bench()
        self.assertEqual(rc, 2, out)
        self.assertIn("契约", out)

    def test_unreadable_manifest_blocks(self):
        (self.f.run / "manifest.json").write_text("{ 不是 json")
        rc, out = self.f.bench()
        self.assertEqual(rc, 2, out)

    def test_missing_verifier_module_blocks(self):
        """门禁自己坏了 ⇒ exit 2，绝不 exit 0（失败关闭）。"""
        (self.f.root / "scripts" / "opverify.py").unlink()
        rc, out = self.f.bench()
        self.assertEqual(rc, 2, out)
        self.assertIn("自己坏了", out)

    def test_stdin_not_json_blocks(self):
        cp = subprocess.run([sys.executable, "-I", str(self.f.hook)],
                            input="这不是 JSON", capture_output=True, text=True,
                            cwd=str(self.td), timeout=180)
        self.assertEqual(cp.returncode, 2, cp.stdout + cp.stderr)
        self.assertIn("自己坏了", cp.stdout + cp.stderr)

    def test_empty_stdin_is_not_an_action(self):
        cp = subprocess.run([sys.executable, "-I", str(self.f.hook)],
                            input="", capture_output=True, text=True,
                            cwd=str(self.td), timeout=180)
        self.assertEqual(cp.returncode, 0, cp.stderr)


# ------------------------------------------------------------ 真仓库回归
class TestRealRepo(unittest.TestCase):
    def test_every_real_contract_is_valid(self):
        g = ops_gate.Gate(REPO)
        cards = g.cards()
        self.assertTrue(cards, "仓库里一张卡都没找到？")
        probs = g.check_contracts(cards)
        self.assertEqual(probs, [], "\n".join(m for _, m in probs))

    def test_verified_gate_does_not_fire_on_an_ordinary_real_doc_edit(self):
        """真文档回归：一个已经满是 `status: verified` 的 cards.md，改正文不许被拦。"""
        g = ops_gate.Gate(REPO)
        doc = REPO / "Plugins" / "ops" / "skills" / "op-gate" / "references" / "cards.md"
        payload = {"tool_name": "Edit",
                   "tool_input": {"file_path": str(doc),
                                  "old_string": "### 知识点",
                                  "new_string": "### 知识点（正文口径补充）"},
                   "cwd": str(REPO)}
        rc, msg, action = g.decide(payload)
        self.assertEqual(rc, 0, msg)
        self.assertEqual(action, "write_verified")

    def test_gate_does_not_fire_on_ordinary_work(self):
        g = ops_gate.Gate(REPO)
        for cmd in ("python3 -m pytest Plugins/ops -q",
                    "git status --short",
                    "python3 -I scripts/opverify.py --selftest"):
            self.assertIsNone(g.classify({"tool_name": "Bash",
                                          "tool_input": {"command": cmd},
                                          "cwd": str(REPO)})[0], cmd)


if __name__ == "__main__":
    unittest.main(verbosity=2)
