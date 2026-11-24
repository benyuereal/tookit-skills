#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""scripts/ablate.py 的独立测试。

为什么必须独立于脚本内的 `--selftest`：selftest 是**被实现者自己写的**
断言，改实现时很容易顺手把断言也改了。独立测试文件才是外部约束。

重点全在**反向**：消融归因的失败模式是「凑出一个数字」，而不是报错。
  · 全不显著却硬给 share        —— 把噪声写成结论
  · 各项重叠导致 share > 1.2    —— 重复计算同一笔收益
  · back_to_back 也计入合计     —— 用了不可 claim 的口径
  · 冠军本身不显著              —— 给一个不成立的前提做归因
  · 判据自成一套（不调 judge）  —— 3×MAD 长成第二份实现
"""
import importlib.util
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[3]
SCRIPT = REPO / "scripts" / "ablate.py"


def _load():
    spec = importlib.util.spec_from_file_location("_ablate_t", str(SCRIPT))
    m = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = m
    spec.loader.exec_module(m)
    return m


A = _load()


# ---------------------------------------------------------------- 正向

def test_two_equal_contributions():
    """两项各贡献一半 → sum_share == 1。"""
    r = A._case([A._r("base", 1.0), A._r("champ", .9), A._r("a", .95), A._r("b", .95)],
                [("a", "a"), ("b", "b")])
    assert r["status"] == "ok"
    assert abs(r["sum_share"] - 1) < 1e-9
    assert abs(r["items"][0]["attribution_share"] - 0.5) < 1e-9


# ---------------------------------------------------------------- 反向

def test_all_insignificant_refuses_to_invent_numbers():
    """**核心反向**：全部不显著时必须明说，不许凑一个 share。

    实测差距 0.1%（.9005 vs .8995），远在 3×MAD 内。硬给数字 = 把噪声当结论。
    """
    r = A._case([A._r("base", 1.0), A._r("champ", .9),
                 A._r("a", .9005), A._r("b", .8995)], [("a", "a"), ("b", "b")])
    assert r["status"] == "no_significant_attribution"
    assert r["sum_share"] is None


def test_share_below_0_8_is_unexplained():
    """**反向**：只解释了 <80% 的收益 → 还有没归因到的部分，不许当结论。"""
    r = A._case([A._r("base", 1.0), A._r("champ", .9), A._r("a", .92)],
                [("a", "a")])
    assert r["status"] == "unexplained"
    assert r["sum_share"] < 0.8


def test_share_above_1_2_is_unexplained():
    """**反向**：各项重叠（share>1.2）说明同一笔收益被重复计算。"""
    r = A._case([A._r("base", 1.0), A._r("champ", .9),
                 A._r("a", .98), A._r("c", .98)], [("a", "a"), ("c", "c")])
    assert r["status"] == "unexplained"
    assert r["sum_share"] > 1.2


def test_back_to_back_marked_not_claimable():
    """**反向**：back_to_back 口径不可 claim → 该项归因必须为空且不进合计。

    这是本项目最贵的一类错：偏差逐实现变化，无全局修正因子（k1/k2=1.29）。
    """
    r = A._case([A._r("base", 1.0), A._r("champ", .9),
                 A._r("a", .98, .001, "back_to_back")], [("a", "a")])
    it = r["items"][0]
    assert it["claimable"] is False
    assert it["attribution_share"] is None
    assert "measure_mode" in it["not_claimable_reason"]


def test_missing_tag_is_spec_error_not_verdict_failure():
    """**反向**：spec 引用不存在的 tag → SpecError（退出码 2），不是判据失败（1）。

    两者处置不同：前者是「你写错了 spec」，后者是「数据说明这个方向无效」。
    混在一起会让人以为实验做完了。

    注意 `_case` 把 SpecError **当返回值**而不是抛出（selftest 靠
    `isinstance(rep, SpecError)` 判断）。这是它自己的设计缺陷 ——
    返回值类型混合会让调用方必须每次 isinstance 检查，容易漏。
    这里按它的实际契约断言，同时在 DEBT 里登记。
    """
    r = A._case([A._r("base", 1.0), A._r("champ", .9), A._r("a", .95)],
                [("a", "a"), ("x", "nope")])
    assert isinstance(r, A.SpecError), "不存在的 tag 应当报 SpecError，实际 %r" % (r,)


def test_cli_exit_code_for_missing_tag_is_2(tmp_path):
    """**反向**（CLI 层）：spec 写错必须退 2，不许退 1。

    退 1 意味着「判据跑完了、结论是无效」——那会让人以为实验做过了。
    """
    import yaml
    card = tmp_path / "op-x"
    (card / "runs" / "loop").mkdir(parents=True)
    (card / "runs" / "loop" / "x.jsonl").write_text(
        "".join(__import__("json").dumps(A._r(t, ms)) + "\n"
                for t, ms in (("base", 1.0), ("champ", .9), ("a", .95))),
        encoding="utf-8")
    spec = tmp_path / "s.yaml"
    spec.write_text(yaml.safe_dump({
        "baseline": "base", "champion": "champ",
        "ablations": [{"id": "a", "tag": "nope"}]}), encoding="utf-8")
    assert A.main(["--card", str(card), "--spec", str(spec)]) == 2


def test_insignificant_champion_refuses_attribution():
    """**反向**：冠军本身不显著 → 不该给任何归因（前提不成立）。"""
    r = A._case([A._r("base", 1.0), A._r("champ", .999), A._r("a", .95)],
                [("a", "a")])
    assert r["status"] == "no_significant_attribution"
    assert r["champion_problems"]


# ---------------------------------------------------------------- 判据唯一性

def test_threshold_comes_from_judge_not_local_copy():
    """**核心**：3×MAD 边界必须与 loop._significant 完全一致 = 2.9× 不显著、3.5× 显著。

    反向意义：如果哪天有人在 ablate.py 里另写一套阈值，这条会立刻失败。
    """
    r1 = A._case([A._r("base", 1.0), A._r("champ", .9), A._r("a", .9029)],
                 [("a", "a")])
    r2 = A._case([A._r("base", 1.0), A._r("champ", .9), A._r("a", .9035)],
                 [("a", "a")])
    assert not r1["items"][0]["significant"], "2.9×MAD 不该判显著"
    assert r2["items"][0]["significant"], "3.5×MAD 该判显著"


def test_selftest_passes():
    assert A.selftest() == 0
