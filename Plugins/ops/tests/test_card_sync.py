#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""scripts/card_sync.py 的测试：一个入口控制 5 个登记位置。

背景（2026-11-25）：新建 `norm-act-swiglu` 卡时分 3 次才把 5 处补齐，
每次都是测试报错才想起来。这是「同一事实多处副本」的第 7 次复现。

本文件守的核心命题：**同步器不能误伤**。
它的第一版遍历了全部 54 张卡（包括没有 skill 目录的），
给每张都往 registry 塞一条 —— 914 行误伤。所以反向用例全部围绕「不该动的别动」。
"""
import importlib.util
import json
import sys
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[3]
SCRIPT = REPO / "scripts" / "card_sync.py"


def _load():
    spec = importlib.util.spec_from_file_location("_card_sync_t", str(SCRIPT))
    m = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = m
    spec.loader.exec_module(m)
    return m


C = _load()


def test_selftest_passes():
    assert C.selftest() == 0


def test_only_cards_with_skill_dir_are_handled():
    """**核心反向**：只处理有 skill 目录的卡，不是全部卡片。

    registry 登记的是「有可运行壳的算子」（7 个），
    而 cards/ 下有 54 张卡片。第一版搞混了，914 行误伤。
    """
    handled = {f.stem for f in C._card_files()}
    all_cards = {f.stem for f in C.CARDS.glob("*.yaml")}
    assert handled < all_cards, "应当只处理一部分卡"
    for name in handled:
        assert (C.SKILLS / ("op-%s" % name)).is_dir(), name
    # 反向：至少有一张卡没有 skill 目录，否则这条判据是空转
    assert (all_cards - handled), "所有卡都有 skill 目录 —— 这条判据没被验证过"


def test_real_repo_is_consistent():
    """真实仓库：5 个位置必须一致（这是回归守卫）。"""
    probs = C.sync(check=True)
    assert probs == [], "5 个位置不一致：\n" + "\n".join(probs)


def test_check_mode_does_not_write():
    """**反向**：--check 不许改任何文件。"""
    files = [C.REGISTRY, C.OPS_ENTRY]
    before = {f: f.read_text(encoding="utf-8") for f in files if f.is_file()}
    C.sync(check=True)
    for f, old in before.items():
        assert f.read_text(encoding="utf-8") == old, "%s 被 --check 改了" % f
