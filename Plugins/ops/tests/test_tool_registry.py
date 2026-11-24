#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""工具注册表的两条硬规则必须被守住。

业主 2026-11-24：
  「没有 probe 证据不许写可用」
  「playbook 引用的工具必须在 registry 里」

这两条很容易在下一轮被绕过——只要有人手改 YAML 就能写个假的 available。
所以测试直接读生成出来的 `data/ops/tool-registry.yaml` 校验。
"""
import importlib.util
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
REGISTRY = REPO / "data/ops/tool-registry.yaml"
CATS = {"runtime", "crash", "static", "build", "numeric", "env"}


def _load_gen():
    spec = importlib.util.spec_from_file_location(
        "_gen_tr", str(REPO / "scripts/gen_tool_registry.py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


class TestToolRegistry(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import yaml
        if not REGISTRY.is_file():
            raise unittest.SkipTest("registry 尚未生成")
        cls.d = yaml.safe_load(REGISTRY.read_text(encoding="utf-8"))
        cls.tools = cls.d.get("tools") or []

    def test_registry_exists_and_has_six_categories(self):
        cats = {t["category"] for t in self.tools}
        self.assertEqual(cats, CATS,
                         "六类必须齐全；实得 %s" % sorted(cats))

    def test_available_requires_probe_evidence(self):
        """硬规则 1：status=available 必须有非空 probe。"""
        bad = [t["id"] for t in self.tools
               if t.get("status") == "available"
               and not str(t.get("probe") or "").strip()]
        self.assertEqual(bad, [], "以下工具标了 available 却没有 probe 证据：%s" % bad)

    def test_status_values_are_legal(self):
        legal = {"available", "degraded", "missing", "unverified"}
        bad = [(t["id"], t.get("status")) for t in self.tools
               if t.get("status") not in legal]
        self.assertEqual(bad, [], "非法 status：%s" % bad)

    def test_every_tool_has_required_fields(self):
        need = ("id", "category", "question", "command", "probe",
                "status", "fallback", "cost", "stage")
        bad = []
        for t in self.tools:
            miss = [k for k in need if k not in t]
            if miss:
                bad.append((t.get("id"), miss))
        self.assertEqual(bad, [], "缺字段：%s" % bad)

    def test_playbook_refs_all_exist(self):
        """硬规则 2：playbook 的 chain 里每个 id 都必须在 tools 里。"""
        ids = {t["id"] for t in self.tools}
        bad = []
        for pb in self.d.get("playbooks") or []:
            for c in pb.get("chain") or []:
                if c not in ids:
                    bad.append((pb.get("id"), c))
        self.assertEqual(bad, [], "playbook 引用了不存在的工具：%s" % bad)

    def test_four_required_playbooks(self):
        got = {pb["id"] for pb in self.d.get("playbooks") or []}
        want = {"correctness-failure", "perf-below-target",
                "crash-or-hang", "not-reproducible"}
        self.assertEqual(got, want, "playbook 应为这四条；实得 %s" % sorted(got))

    def test_playbook_chain_order_is_meaningful(self):
        """每条 playbook 至少两个工具——一个工具不叫"链"。"""
        bad = [pb["id"] for pb in self.d.get("playbooks") or []
               if len(pb.get("chain") or []) < 2]
        self.assertEqual(bad, [], "以下 playbook 的 chain 太短：%s" % bad)

    def test_known_missing_tools_are_not_available(self):
        """三个已实测不可用的，不许被写成 available。"""
        by_id = {t["id"]: t for t in self.tools}
        for tid, why in (("llvm_objdump", "本机不存在"),
                         ("hipcc_resource_usage", "本机 DTK 报 unsupported"),
                         ("bounds_check", "sanitizer 编不过")):
            if tid in by_id:
                self.assertNotEqual(
                    by_id[tid]["status"], "available",
                    "%s 被标成 available，但%s" % (tid, why))


class TestRegistryGenerator(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.m = _load_gen()

    def test_check_playbooks_catches_missing_tool(self):
        """反向测试：playbook 引用不存在的工具，必须被抓到。"""
        bad_doc = {"tools": [{"id": "a", "status": "available", "probe": "x"}],
                   "playbooks": [{"id": "p", "chain": ["a", "nonexistent"]}]}
        import yaml
        txt = yaml.dump(bad_doc, allow_unicode=True)
        probs = self.m.check_playbooks(txt, {"a"})
        self.assertTrue(any("nonexistent" in p for p in probs), probs)

    def test_check_playbooks_catches_available_without_probe(self):
        """反向测试：available 无 probe，必须被抓到。"""
        bad_doc = {"tools": [{"id": "a", "status": "available", "probe": ""}],
                   "playbooks": []}
        import yaml
        txt = yaml.dump(bad_doc, allow_unicode=True)
        probs = self.m.check_playbooks(txt, {"a"})
        self.assertTrue(any("没有 probe 证据" in p for p in probs), probs)


if __name__ == "__main__":
    unittest.main()
