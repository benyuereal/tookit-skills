#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""scripts/lang_select.py 的场景路由测试。

核心命题：**语言选型必须按场景判，不许默认从 Triton 起手。**

背景（2026-11-25，opbench 三臂对照实验）：A 臂裸 CC / C 臂本 skill / D 臂 SkillHub
**三臂都没写过一行 HIP kernel**，而本仓 op-rmsnorm-fused/runs/forged 下已有
4 个实测更快的现成 HIP 实现。根因不是能力，是这条链上没有任何一处说
「什么时候该考虑 HIP」——agent 按习惯从 Triton 起手。

本文件守的就是那条缺失的判据：它必须来自 roofline 的数字，而不是定性描述。
"""
import importlib.util
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
SCRIPT = REPO / "scripts" / "lang_select.py"


def _load():
    spec = importlib.util.spec_from_file_location("_lang_select_t", str(SCRIPT))
    m = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = m
    spec.loader.exec_module(m)
    return m


L = _load()




# ---------------------------------------------------------------- 场景路由

class TestScenarioRouting:
    """语言选型必须按场景判，不许默认 Triton。

    背景（2026-11-25，opbench 三臂对照）：三臂都没写过一行 HIP，
    但本仓有实测更快的 HIP 实现。根因不是能力，是**没有一条规则说
    「什么时候该考虑 HIP」**，agent 按习惯从 Triton 起手。
    """

    def test_memory_large_gap_prefers_hip(self):
        """**核心**：访存受限 + 差距大 ⇒ 必须建议先评估 HIP。"""
        roof = {"bound": "memory", "t_floor_us": 116.4, "t_actual_us": 140.0,
                "remaining": 0.203}
        r = L.route_scenario(roof, "memory")
        assert r["rule"] == "memory-bound-large-gap", r
        assert r["langs"][0] == "hip", r
        assert "HIP" in r["action"], r

    def test_near_ceiling_does_not_recommend_hip(self):
        """**反向**：已接近上限时不该建议 HIP —— 迭代成本不值。

        空间 <10% 还要付 HIP 的迭代成本，是把「能优化」当「值得优化」。
        """
        roof = {"bound": "memory", "t_floor_us": 200.0, "t_actual_us": 202.0,
                "remaining": 0.01}
        r = L.route_scenario(roof, "memory")
        assert r["rule"] == "memory-bound-near-ceiling", r
        assert "hip" not in r["langs"], r

    def test_launch_bound_beats_everything(self):
        """**反向**：launch 占比 >30% 时，换语言无效 —— 该做融合。"""
        roof = {"bound": "memory", "t_floor_us": 100.0, "t_actual_us": 200.0,
                "remaining": 1.0, "launch_us": 70.0}
        r = L.route_scenario(roof, "memory")
        assert r["rule"] == "launch-bound", r
        assert r["langs"] == [], "launch 受限时不该推荐任何语言"

    def test_compute_near_ceiling_uses_library(self):
        """反向：算力受限且接近上限 ⇒ 库实现已够，别手写。"""
        roof = {"bound": "compute", "t_floor_us": 200.0, "t_actual_us": 210.0,
                "remaining": 0.05}
        r = L.route_scenario(roof, "compute")
        assert r["rule"] == "compute-bound-near-ceiling", r
        assert r["langs"] == ["triton"], r

    def test_no_roofline_does_not_guess(self):
        """**反向**：没有 roofline 就明说不猜，不许默认给 Triton。"""
        assert L.route_scenario(None, "memory") is None

    def test_rules_are_loaded_from_yaml_not_hardcoded(self):
        """规则必须来自 yaml（单一真源），不是代码里的 if 链。

        反向意义：如果哪天有人在 _rule_hit 里加了一条 yaml 里没有的规则，
        这条会失败 —— 避免「文档说一套、代码做一套」。
        """
        rules = L._routing_rules()
        assert len(rules) >= 5, rules
        ids = {r["id"] for r in rules}
        for rid in ("launch-bound", "memory-bound-large-gap",
                    "memory-bound-near-ceiling", "compute-bound-near-ceiling"):
            assert rid in ids, "yaml 里缺规则 %s" % rid
