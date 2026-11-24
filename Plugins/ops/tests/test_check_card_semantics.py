#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""scripts/check_card_semantics.py 的独立测试（D8 的守卫）。

D8 是 mhc-fused.yaml 把 conv1d+hadamard+norm 的 math 挂在了 post+proj 的源文件上。
本测试守的是**判据本身**，不是那一张卡：
  · 该抓的时候真抓（反向，复现 D8 形态）
  · 不该抓的时候不误伤（正向）
  · 判据为空时不许假装通过（边界）
"""
import importlib.util
import sys
from pathlib import Path

import pytest
import yaml

REPO = Path(__file__).resolve().parents[3]
SCRIPT = REPO / "scripts" / "check_card_semantics.py"


def _load():
    spec = importlib.util.spec_from_file_location("_ccs_t", str(SCRIPT))
    m = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = m
    spec.loader.exec_module(m)
    return m


C = _load()


def _card(tmp_path, name, math, src_text, src_name="k.hip"):
    (tmp_path / src_name).write_text(src_text, encoding="utf-8")
    f = tmp_path / (name + ".yaml")
    f.write_text(yaml.safe_dump({
        "id": name, "math": math,
        "variants": [{"source_ref": {"path": str(tmp_path / src_name)}}]},
        allow_unicode=True), encoding="utf-8")
    return f


def test_d8_shape_is_caught(tmp_path):
    """**反向**：math 说 conv1d/hadamard/norm，源文件是 post+proj → 必须报。"""
    f = _card(tmp_path, "bad", "conv1d + hadamard + norm 的融合链",
              '// Post + next-layer-projection fusion\n'
              'extern "C" void hc_post_proj_launch(void);\n')
    probs = C.check_card(f)
    assert probs and "找不到" in probs[0], probs


def test_matching_semantics_not_flagged(tmp_path):
    """正向：语义相符的卡不该被报 —— 否则判据就是噪音。"""
    f = _card(tmp_path, "good", "conv1d + hadamard + norm 的融合链",
              '// conv1d then hadamard then norm\n'
              'extern "C" void conv1d_hadamard_norm_launch(void);\n',
              src_name="ok.hip")
    assert C.check_card(f) == []


def test_vague_math_is_skipped_not_passed(tmp_path):
    """边界：math 里没有可识别的算子族词 → 不检。

    返回空表**不等于通过**，只是「无可检项」。这一点必须在 docstring 里写清，
    否则空表会被当成绿灯。
    """
    f = _card(tmp_path, "vague", "把三件事串起来",
              'extern "C" void x(void);\n', src_name="v.hip")
    assert C.check_card(f) == []


def test_missing_source_file_is_not_a_pass(tmp_path):
    """边界：源文件不在本机 → 不检，但也不许声称通过。

    跳过的理由要能分辨：这里是因为跨机器路径，不是因为语义一致。
    """
    f = tmp_path / "m.yaml"
    f.write_text(yaml.safe_dump({
        "id": "m", "math": "sinkhorn 归一",
        "variants": [{"source_ref": {"path": "/nonexistent/x.hip"}}]}),
        encoding="utf-8")
    assert C.check_card(f) == []


def test_real_repo_is_clean_after_d8_fix():
    """真实仓库：D8 修好后必须全绿（这是回归守卫）。"""
    probs = C.check_all()
    assert probs == [], "真实卡仍有语义不符：%s" % probs


def test_selftest_passes():
    assert C.selftest() == 0
