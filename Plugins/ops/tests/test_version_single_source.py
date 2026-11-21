#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""SK010：版本号单一事实源。

`.claude-plugin/marketplace.json` 是唯一权威。其余出现版本号的地方必须与它一致，
否则会出现「marketplace 说 1.6.0、插件 SKILL.md 说 1.5.1」这种安装态与文档不符。

本测试只守**结构**，不改内容：
  - 8 个插件 SKILL.md 的 frontmatter version
  - 8 个 plugin.json 的 version
  - package.env / tools/*/package.env 的 VERSION
  - install.sh 的 TOOLKIT_VERSION
不检查正文里提到的历史版本（如某个驱动叫 V1.6.0，那是产品版本不是本包版本）。
"""
import json
import re
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
MARKETPLACE = REPO / ".claude-plugin/marketplace.json"


def _authoritative():
    return json.loads(MARKETPLACE.read_text(encoding="utf-8"))["version"]


def _frontmatter_version(path):
    text = path.read_text(encoding="utf-8")
    m = re.match(r"^---\n(.*?)\n---\n", text, re.S)
    if not m:
        return None
    v = re.search(r"(?m)^version:\s*(\S+)\s*$", m.group(1))
    return v.group(1) if v else None


class TestVersionSingleSource(unittest.TestCase):
    def setUp(self):
        self.version = _authoritative()

    def test_plugins_all_share_the_marketplace_version(self):
        bad = []
        for skill in sorted((REPO / "Plugins").glob("*/SKILL.md")):
            got = _frontmatter_version(skill)
            if got != self.version:
                bad.append("%s: %r ≠ marketplace %r"
                           % (skill.relative_to(REPO), got, self.version))
        self.assertEqual(bad, [], "插件 SKILL.md 版本与 marketplace 不一致：\n" + "\n".join(bad))

    def test_plugin_json_matches(self):
        bad = []
        for pj in sorted((REPO / "Plugins").glob("*/.claude-plugin/plugin.json")):
            got = json.loads(pj.read_text(encoding="utf-8")).get("version")
            if got != self.version:
                bad.append("%s: %r ≠ %r" % (pj.relative_to(REPO), got, self.version))
        self.assertEqual(bad, [], "\n".join(bad))

    def test_package_env_matches(self):
        bad = []
        paths = [REPO / "package.env"] + sorted((REPO / "tools").glob("*/package.env"))
        for p in paths:
            if not p.is_file():
                continue
            m = re.search(r'(?m)^VERSION="?([^"\n]+)"?', p.read_text(encoding="utf-8"))
            if m and m.group(1) != self.version:
                bad.append("%s: %r ≠ %r" % (p.relative_to(REPO), m.group(1), self.version))
        self.assertEqual(bad, [], "\n".join(bad))

    def test_install_sh_matches(self):
        p = REPO / "install.sh"
        if not p.is_file():
            self.skipTest("install.sh 不存在")
        m = re.search(r'(?m)^TOOLKIT_VERSION="?([^"\n]+)"?', p.read_text(encoding="utf-8"))
        if m:
            self.assertEqual(m.group(1), self.version,
                             "install.sh TOOLKIT_VERSION 与 marketplace 不一致")


if __name__ == "__main__":
    unittest.main()
