#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""课程化（curriculum）的测试。

背景（2026-11-25）：`surfaces.yaml` 的 17 条攻击面是**平铺的**，没有先后顺序。
实测代价：`norm-act-swiglu` 上先扫 BLOCK_ROWS 1/2/4 三轮才发现 2 行最优——
如果先定架构（1 行/CTA 归约延迟藏不住 → 必须多行），一轮就到。

ForgeKernel 的做法是把「最容易搞错、后来最难改」的决策放 M0 并冻结，
M1..Mk 只在这个结构内调参。它的 27 个算子证明**顺序比选项更重要**。

本文件守的核心命题：**17 条 surface 与 4 个层必须一一对应，且 M0 有 gate。**
这件事不会自动保持 —— 加了新 surface 忘了分层不会让任何测试变红，
只会在 agent 真按课程跑的时候发现「这条没层」。
"""
import importlib.util
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[3]
SCRIPT = REPO / "scripts" / "check_curriculum.py"


def _load():
    spec = importlib.util.spec_from_file_location("_curriculum_t", str(SCRIPT))
    m = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = m
    spec.loader.exec_module(m)
    return m


C = _load()


def test_selftest_passes():
    assert C.selftest() == 0


def test_real_repo_is_consistent():
    """真实仓库：17 条 surface 与 curriculum 必须一致（回归守卫）。"""
    surfaces = C._load_yaml(C.SURFACES)
    cur = C._load_yaml(C.CURRICULUM)
    probs = C.check(surfaces, cur)
    assert probs == [], "课程与 surface 不一致：\n" + "\n".join(probs)


def test_real_repo_covers_all_17_surfaces():
    """17 条一条都不能漏 —— 漏了那条 agent 就永远不知道什么时候做它。"""
    surfaces = C._load_yaml(C.SURFACES)
    cur = C._load_yaml(C.CURRICULUM)
    s5 = C.summary(surfaces, cur)
    assert s5["total_surfaces"] == 17, s5
    assert sum(s5["layers"].values()) == 17, s5


def test_m0_has_gate():
    """**核心**：M0 必须有 gate。

    没有验收判据的架构层等于没有冻结 —— agent 会在结构没验证的情况下
    就往下调参，然后在错的结构上浪费几轮（这正是我们踩过的）。
    """
    cur = C._load_yaml(C.CURRICULUM)
    gate = (cur.get("layers") or {}).get("M0", {}).get("gate") or ""
    assert gate.strip(), "M0 缺 gate"
    # gate 至少要说清「正确性」与「地板」两件事
    assert "正确性" in gate or "correctness" in gate.lower(), gate
    assert "地板" in gate or "floor" in gate.lower(), gate


def test_each_surface_in_exactly_one_layer():
    surfaces = C._load_yaml(C.SURFACES)
    cur = C._load_yaml(C.CURRICULUM)
    cmap, _ = C.curriculum_map(cur)
    for sid in C.surface_ids(surfaces):
        assert len(cmap.get(sid, [])) == 1, "%r 落在 %s 层" % (sid, cmap.get(sid))


# ---------------------------------------------------------------- 反向用例

def test_detects_unlayered_surface():
    """**反向**：新加的 surface 忘了分层，必须被报出来。"""
    surfaces = [{"id": "a"}, {"id": "b"}, {"id": "new"}]
    cur = {"layers": {
        "M0": {"gate": "g", "surfaces": ["a"]},
        "M1": {"surfaces": ["b"]},
        "M2": {"surfaces": []},
        "M3": {"surfaces": []},
    }}
    probs = C.check(surfaces, cur)
    assert any("'new' 没有分层" in p for p in probs), probs


def test_detects_duplicate_layer():
    """**反向**：同一条 surface 分到两层 —— 顺序就不唯一了。"""
    surfaces = [{"id": "a"}]
    cur = {"layers": {
        "M0": {"gate": "g", "surfaces": ["a"]},
        "M1": {"surfaces": ["a"]},
        "M2": {"surfaces": []},
        "M3": {"surfaces": []},
    }}
    probs = C.check(surfaces, cur)
    assert any("多个层" in p for p in probs), probs


def test_detects_phantom_surface():
    """**反向**：curriculum 里写了 surfaces.yaml 没有的 id。"""
    surfaces = [{"id": "a"}]
    cur = {"layers": {
        "M0": {"gate": "g", "surfaces": ["a"]},
        "M1": {"surfaces": ["ghost"]},
        "M2": {"surfaces": []},
        "M3": {"surfaces": []},
    }}
    probs = C.check(surfaces, cur)
    assert any("'ghost' 在 surfaces.yaml 里不存在" in p for p in probs), probs
