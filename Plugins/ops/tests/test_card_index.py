#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""知识卡（references/cards.md）的静态质量门。

每个 skill 的卡都收在同一张 `cards.md` 里，一卡一个 `## OP-XXX-NNN · <name>` 段。
（2026-10-28 合并：原先是 `references/OP-XXX-NNN-<slug>.md` 一卡一文件，12 个文件的
frontmatter + 六段模板逐份复制，改一处格式要改 12 遍。合并后格式只有一份定义，
本测试也从「逐个文件读 frontmatter」改成「按 `## OP-` 切块」。）

检查：
  * id 全局唯一，且与所在段标题一致；
  * 每卡有 fenced yaml 头，五字段齐全（id / name / category / applies_to / status）且取值合法；
  * 六段结构齐全且顺序固定：知识点 / 错误写法 / 正确写法 / 识别方法 / 证据要求 / 参考；
  * 正文（「### 参考」之外）不允许出现绝对耗时/带宽数字——绝对数归 ~/.config/dcu-toolkit/kb/op-opt/；
  * 无孤儿卡：每张卡都被同 skill 的 SKILL.md「## 知识卡索引」提到；索引里的 id 也必须真实存在；
  * 「### 参考」里的仓内路径必须真实存在。

无卡环境可跑：纯文本静态检查。
"""
import re
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
SKILLS_ROOT = REPO / "Plugins" / "ops" / "skills"
CARD_SKILLS = ("op-gate", "op-profiling", "op-roofline")

SECTIONS = ["知识点", "错误写法", "正确写法", "识别方法", "证据要求", "参考"]
REQUIRED_FM = ["id", "name", "category", "applies_to", "status"]
CATEGORIES = {"memory", "compute", "instruction", "gate", "caliber"}
STATUSES = {"verified", "unverified"}

# 绝对耗时/带宽：数字 + 单位。只允许出现在「### 参考」里（引用文件/命令），正文一律不许。
ABSOLUTE_TIME_RE = re.compile(r"\d+(?:\.\d+)?\s*(?:us|µs|μs|ms)\b", re.IGNORECASE)
ABSOLUTE_BW_RE = re.compile(r"\d+(?:\.\d+)?\s*(?:GB/s|GBps|TB/s)\b", re.IGNORECASE)
# 「### 参考」里的仓内路径（只校验 .py/.md/.yaml/.yml/.json）
REPO_PATH_RE = re.compile(
    r"(?:Plugins|scripts|data|docs|tools)/[\w./-]+\.(?:py|md|yaml|yml|json)")
INDEX_HEADING = "## 知识卡索引"
CARD_ID_RE = re.compile(r"\bOP-[A-Z]+-\d{3}\b")
CARD_HEADING_RE = re.compile(r"(?m)^## (OP-[A-Z]+-\d{3}) · (.+)$")
SECTION_RE = re.compile(r"(?m)^### (.+?)\s*$")
FM_BLOCK_RE = re.compile(r"(?s)```yaml\n(.*?)```")


class Card(object):
    """一张卡 = cards.md 里从 `## OP-…` 到下一个 `## OP-…` 之间的正文。"""

    def __init__(self, skill, where, cid, title, block):
        self.skill = skill
        self.where = "%s/references/cards.md#%s" % (skill, cid)
        self.id = cid
        self.title = title
        self.block = block
        self.frontmatter = _parse_fm(block)
        self.headings = SECTION_RE.findall(block)

    def section(self, name):
        """返回 `### <name>` 到下一个三级标题之间的正文。"""
        idx = self.block.find("### " + name)
        if idx < 0:
            return ""
        rest = self.block[idx + len("### " + name):]
        nxt = re.search(r"(?m)^###? ", rest)
        return rest[:nxt.start()] if nxt else rest

    def body_before_reference(self):
        return self.block.split("### 参考", 1)[0]


def _parse_fm(block):
    m = FM_BLOCK_RE.search(block)
    if not m:
        return {}
    fm = {}
    for raw in m.group(1).splitlines():
        if not raw.strip():
            continue
        key, sep, val = raw.partition(":")
        if sep:
            fm[key.strip()] = val.strip()
    return fm


def _load_cards():
    cards = []
    for skill in CARD_SKILLS:
        p = SKILLS_ROOT / skill / "references" / "cards.md"
        if not p.is_file():
            continue
        text = p.read_text(encoding="utf-8")
        marks = list(CARD_HEADING_RE.finditer(text))
        for i, m in enumerate(marks):
            end = marks[i + 1].start() if i + 1 < len(marks) else len(text)
            cards.append(Card(skill, p, m.group(1), m.group(2), text[m.start():end]))
    return cards


CARDS = _load_cards()


def _index_text(skill):
    p = SKILLS_ROOT / skill / "SKILL.md"
    text = p.read_text(encoding="utf-8")
    idx = text.find(INDEX_HEADING)
    return text[idx:] if idx >= 0 else ""


class TestCardInventory(unittest.TestCase):
    def test_enough_cards(self):
        self.assertGreaterEqual(len(CARDS), 10, "三个 skill 合计至少 10 张知识卡")
        per_skill = {}
        for c in CARDS:
            per_skill[c.skill] = per_skill.get(c.skill, 0) + 1
        for skill in CARD_SKILLS:
            self.assertGreaterEqual(per_skill.get(skill, 0), 1, "%s 至少一张知识卡" % skill)

    def test_ids_are_globally_unique(self):
        seen = {}
        for c in CARDS:
            self.assertTrue(c.id, "%s 缺 id" % c.where)
            seen.setdefault(c.id, []).append(c.where)
        dups = {k: v for k, v in seen.items() if len(v) > 1}
        self.assertEqual(dups, {}, "id 重复: %s" % dups)

    def test_id_prefix_matches_skill(self):
        """卡放错文件 = 索引也必定指错，直接判死。"""
        allowed = {"op-gate": {"OP-GATE", "OP-CALIBER"},
                   "op-profiling": {"OP-PROFILING", "OP-CALIBER"},
                   "op-roofline": {"OP-ROOFLINE", "OP-CALIBER"}}
        for c in CARDS:
            prefix = c.id.rsplit("-", 1)[0]
            self.assertIn(prefix, allowed[c.skill],
                          "%s 放在 %s 里但前缀是 %s" % (c.id, c.skill, prefix))

    def test_frontmatter_id_matches_heading(self):
        for c in CARDS:
            self.assertEqual(c.frontmatter.get("id", "").strip(), c.id,
                             "%s 的 yaml 头 id 与段标题不一致" % c.where)

    def test_every_card_starts_with_yaml_header(self):
        for c in CARDS:
            self.assertTrue(FM_BLOCK_RE.search(c.block), "%s 缺开头的 ```yaml 头" % c.where)


class TestFrontmatter(unittest.TestCase):
    def test_required_fields_present(self):
        for c in CARDS:
            for field in REQUIRED_FM:
                self.assertTrue(c.frontmatter.get(field, "").strip(),
                                "%s 缺字段 %s" % (c.where, field))

    def test_category_and_status_are_legal(self):
        for c in CARDS:
            self.assertIn(c.frontmatter.get("category", "").strip(), CATEGORIES,
                          "%s category 非法" % c.where)
            self.assertIn(c.frontmatter.get("status", "").strip(), STATUSES,
                          "%s status 非法" % c.where)

    def test_name_matches_heading_title(self):
        for c in CARDS:
            self.assertEqual(c.frontmatter.get("name", "").strip(), c.title.strip(),
                             "%s 的 yaml 头 name 与段标题不一致（两处会漂）" % c.where)


class TestStructure(unittest.TestCase):
    def test_six_sections_in_fixed_order(self):
        for c in CARDS:
            self.assertEqual(c.headings, SECTIONS,
                             "%s 的三级标题必须且只能是固定六段（顺序不变），实得 %s"
                             % (c.where, c.headings))

    def test_reference_section_is_not_empty(self):
        for c in CARDS:
            self.assertTrue(c.section("参考").strip(), "%s 的「参考」段为空" % c.where)


class TestNoAbsoluteNumbers(unittest.TestCase):
    def _scan(self, regex, label):
        bad = []
        for c in CARDS:
            # 只扫「### 参考」之前的部分（参考段允许引用带单位的命令/文件名）
            for lineno, line in enumerate(c.body_before_reference().splitlines(), 1):
                if regex.search(line):
                    bad.append("%s:%d: %s" % (c.where, lineno, line.strip()))
        self.assertEqual(bad, [], "%s 只允许出现在「### 参考」里：\n%s" % (label, "\n".join(bad)))

    def test_no_absolute_duration_in_body(self):
        self._scan(ABSOLUTE_TIME_RE, "绝对耗时（us/µs/ms）")

    def test_no_absolute_bandwidth_in_body(self):
        self._scan(ABSOLUTE_BW_RE, "绝对带宽（GB/s）")


class TestIndexCoverage(unittest.TestCase):
    def test_every_card_is_indexed(self):
        index_by_skill = {s: _index_text(s) for s in CARD_SKILLS}
        for skill in CARD_SKILLS:
            self.assertIn(INDEX_HEADING, (SKILLS_ROOT / skill / "SKILL.md").read_text(encoding="utf-8"),
                          "%s/SKILL.md 缺「%s」小节" % (skill, INDEX_HEADING))
        orphans = [c.where for c in CARDS if c.id not in index_by_skill[c.skill]]
        self.assertEqual(orphans, [], "孤儿卡（未在对应 SKILL.md 知识卡索引中登记）: %s" % orphans)

    def test_index_has_no_dangling_ids(self):
        known = {c.id for c in CARDS}
        for skill in CARD_SKILLS:
            mentioned = set(CARD_ID_RE.findall(_index_text(skill)))
            dangling = sorted(mentioned - known)
            self.assertEqual(dangling, [], "%s 的索引引用了不存在的卡: %s" % (skill, dangling))


class TestReferencePathsExist(unittest.TestCase):
    def test_repo_paths_in_reference_section_exist(self):
        missing = []
        for c in CARDS:
            for rel in sorted(set(REPO_PATH_RE.findall(c.section("参考")))):
                if not (REPO / rel).exists():
                    missing.append("%s -> %s" % (c.where, rel))
        self.assertEqual(missing, [], "「参考」指向了不存在的路径:\n%s" % "\n".join(missing))


if __name__ == "__main__":
    unittest.main()
