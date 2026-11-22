#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""S5：证伪必须能追溯到机制。

纪律原文：「判死时要能追溯到机制，不许写『试过，慢』」。
为什么这条重要：**机制能迁移到下一个算子，结论不能**。
一条「layout-pack 试过，慢了 3%」对后来者毫无价值；
「layout-pack 让 LDS 从 16KB 涨到 32KB，把 3 CTA/CU 打到 2」才是可复用的知识。

本测试守两件事：
  1. 新增的 refuted 条目必须带 mechanism（或 PMC 计数器引用）
  2. mechanism 不能是空话（长度与内容都要有实质）

存量条目用 LEGACY_FREEZE 冻结——它们写于本纪律之前，`why` 里其实
有机制描述但没进独立字段。**不批量补**：补不出真实机制就是编造。
"""
import pathlib
import re
import unittest

REPO = pathlib.Path(__file__).resolve().parents[3]
CARDS = REPO / "Plugins/ops/skills"

# 冻结：本纪律生效前已存在的条目数（按卡）。只许减，不许增。
LEGACY_FREEZE = {
    "op-sparse-mla": 43,      # 写于 S5 之前；why 里有机制但无独立字段
}

# 机制必须有点实质：太短、或全是泛词，等于没写
VAGUE = re.compile(r"^(试过|慢|不行|没效果|不work|无效)[，。 ]*$", re.I)
PMC_HINT = re.compile(r"SQ_[A-Z_]+|GRBM[A-Z_]*|PMC|LDS|VGPR|CTA/CU|L2", re.I)
MIN_LEN = 20


def _cards_with_candidates():
    return sorted(p.parent for p in CARDS.glob("*/candidates.yaml"))


def _load(p):
    try:
        import yaml
    except ImportError:                        # pragma: no cover
        return None
    return yaml.safe_load(p.read_text(encoding="utf-8")) or {}


class TestRefuteMechanism(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = {}
        for card in _cards_with_candidates():
            d = _load(card / "candidates.yaml")
            if d is not None:
                cls.data[card.name] = d
        if not cls.data:
            raise unittest.SkipTest("没有 candidates.yaml 或 PyYAML 未安装")

    def test_new_refutes_have_mechanism(self):
        bad = []
        for name, d in self.data.items():
            refs = d.get("refuted") or []
            frozen = LEGACY_FREEZE.get(name, 0)
            # 只检查「冻结额度之外」的新条目（从末尾数）
            new_items = refs[frozen:] if len(refs) > frozen else []
            for r in new_items:
                rid = r.get("id", "?")
                mech = str(r.get("mechanism") or "")
                text = str(r.get("why") or "") + mech
                if not mech and not PMC_HINT.search(text):
                    bad.append("%s/%s：无 mechanism 也无 PMC 引用" % (name, rid))
                    continue
                if mech and len(mech) < MIN_LEN:
                    bad.append("%s/%s：mechanism 过短（%d 字）" % (name, rid, len(mech)))
                if mech and VAGUE.match(mech.strip()):
                    bad.append("%s/%s：mechanism 是空话 %r" % (name, rid, mech))
        self.assertEqual(bad, [],
                         "新增证伪必须写机制（机制能迁移，结论不能）：\n" + "\n".join(bad))

    def test_freeze_counts_still_match(self):
        """冻结额度必须不小于实际存量——否则说明条目被删了，冻结数要跟着降。"""
        bad = []
        for name, frozen in LEGACY_FREEZE.items():
            d = self.data.get(name)
            if d is None:
                bad.append("%s 的 candidates.yaml 不见了" % name)
                continue
            n = len(d.get("refuted") or [])
            if n < frozen:
                bad.append("%s: 实际 %d 条 < 冻结 %d 条（条目被删了，请降冻结数）"
                           % (name, n, frozen))
        self.assertEqual(bad, [], "\n".join(bad))

    def test_at_least_three_cards_have_candidates(self):
        """S2：至少 3 个不同家族的卡有 candidates。"""
        self.assertGreaterEqual(len(self.data), 3,
                                "只有 %d 张卡有 candidates" % len(self.data))


if __name__ == "__main__":
    unittest.main()
