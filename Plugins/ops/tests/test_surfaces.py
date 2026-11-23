#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""攻击面表（surfaces.yaml）的一致性守卫。

surfaces.yaml 把「凭空发明 direction」变成「从表里挑」。它一旦漂移，
候选生成就会引用到不存在的方向，或者挂到不存在的定律上——判死时
就无法追溯到机制。

四条断言：
  1. 每条 surface 的 id 唯一且是 kebab-case
  2. law 必须是 pipeline-laws.md 里真实存在的 L 号
  3. 必填字段齐全，class 在枚举内，cost 在枚举内
  4. check 不像「看看是不是瓶颈」这类无法执行的废话（必须有可执行锚点）
"""
import re
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
SURFACES = REPO / "Plugins/ops/skills/op-optimize/references/surfaces.yaml"
LAWS = REPO / "Plugins/ops/skills/op-stack-design/references/pipeline-laws.md"

REQUIRED = {"id", "statement", "law", "class", "precondition", "check", "evidence", "cost"}
CLASSES = {"residency", "pipeline", "layout", "tier", "precision", "memory-path",
           # isa-policy：指令/编译器层面的策略（缓存提示、等待深度）。
           # 与上面六个的区别是「改的是编译选项或 ISA 提示，不是 kernel 结构」。
           "isa-policy"}
COSTS = {"S", "M", "L"}
# check 字段必须含一个可执行锚点：pmc 计数器名 / 具体工具 / 具体命令
EXECUTABLE = re.compile(
    r"pmc:|SQ_|GRBM|hipcc|rocminfo|hipprof|opcheck|opbench|probe_|--[a-z]"
    r"|\.py\b|[a-z_]+\(\)|[a-z]+_[a-z_]+",   # 脚本名 / 函数名 / 函数调用
    re.I)
# 空泛措辞：出现即视为 check 不可执行
VAGUE = re.compile(r"看看|确认一下瓶颈|观察是否|感觉|大概|应该是瓶颈")


def _load():
    try:
        import yaml
    except ImportError:                                   # pragma: no cover
        return None
    return yaml.safe_load(SURFACES.read_text(encoding="utf-8"))


def _law_anchor(text):
    """返回文件里所有可被引用的标题锚点（`## Q5 xxx` → `Q5`；`## 四点五、xx` → `四点五`）。

    中文小节号也要认：isa-census.md 用「四点五 / 四点六」编号，
    早期只认 [A-Za-z]\d+，导致引用中文小节永远被判「锚点不存在」。
    与 validate.py 的 E9 保持同一宽窄度——两处判据不一致本身就是隐患。
    """
    out = set(re.findall(r"(?m)^#{1,4}\s+([A-Za-z]\d+)\b", text))
    out |= set(re.findall(r"(?m)^#{1,4}\s+([一二三四五六七八九十]+点[一二三四五六七八九十]+)", text))
    return out


class TestSurfaces(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.surfaces = _load()
        if cls.surfaces is None:
            raise unittest.SkipTest("PyYAML 未安装")

    def test_ids_unique_and_kebab(self):
        ids = [s.get("id") for s in self.surfaces]
        self.assertEqual(len(ids), len(set(ids)), "surface id 有重复：%s" % ids)
        bad = [i for i in ids if not re.fullmatch(r"[a-z0-9]+(-[a-z0-9]+)*", str(i))]
        self.assertEqual(bad, [], "id 不是 kebab-case：%s" % bad)

    def test_laws_resolve_to_real_anchors(self):
        """law 必须写成 <file>#<anchor>，且锚点在那个文件里真实存在。

        早期版本用 `L1`..`L10` 编号，而定律本体文件根本不存在——
        定律悬空，`law` 变成装饰性编号。
        """
        bad = []
        for s in self.surfaces:
            law = str(s.get("law") or "")
            sid = s.get("id")
            if "#" not in law:
                bad.append("%s -> %r（缺锚点）" % (sid, law))
                continue
            fname, _, anchor = law.partition("#")
            target = REPO / fname
            if not target.is_file():
                bad.append("%s -> %r（文件不存在）" % (sid, fname))
                continue
            if anchor not in _law_anchor(target.read_text(encoding="utf-8", errors="ignore")):
                bad.append("%s -> %r（锚点 %s 不存在）" % (sid, law, anchor))
        self.assertEqual(bad, [], "surface 的 law 无法解析到实体：\n" + "\n".join(bad))

    def test_required_fields_and_enums(self):
        bad = []
        for s in self.surfaces:
            sid = s.get("id", "?")
            missing = REQUIRED - set(s)
            if missing:
                bad.append("%s 缺 %s" % (sid, sorted(missing)))
            if s.get("class") not in CLASSES:
                bad.append("%s class=%r 非法" % (sid, s.get("class")))
            if s.get("cost") not in COSTS:
                bad.append("%s cost=%r 非法" % (sid, s.get("cost")))
        self.assertEqual(bad, [], "\n".join(bad))

    def test_check_is_executable(self):
        bad = []
        for s in self.surfaces:
            chk = str(s.get("check") or "")
            if VAGUE.search(chk):
                bad.append("%s: check 含空泛措辞 -> %r" % (s.get("id"), chk))
            elif not EXECUTABLE.search(chk):
                bad.append("%s: check 无可执行锚点 -> %r" % (s.get("id"), chk))
        self.assertEqual(bad, [],
                         "check 必须能真的跑（pmc 计数器 / 工具名 / 命令）：\n" + "\n".join(bad))


if __name__ == "__main__":
    unittest.main()
