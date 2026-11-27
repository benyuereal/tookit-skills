#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""scripts/op_bootstrap.py 的测试：新算子冷启动。

缺口：已有卡的链路是通的（roofline → lang_select → advance），
但**新来的算子第一环就断**——derive.py 只认 sparse-mla，没有建卡入口，
卡片 schema 有 20+ 字段且 additionalProperties=false。

本文件守的核心命题：**bootstrap 算出的 bytes 必须是分母，算错就是结论反。**
所以反向用例全部围绕「什么情况下 bytes 会算错」。
"""
import importlib.util
import json
import re
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[3]
SCRIPT = REPO / "scripts" / "op_bootstrap.py"


def _load():
    spec = importlib.util.spec_from_file_location("_bootstrap_t", str(SCRIPT))
    m = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = m
    spec.loader.exec_module(m)
    return m


B = _load()


def test_selftest_passes():
    assert B.selftest() == 0


# ---------------------------------------------------------------- bytes 是分母

def test_two_reads_one_write():
    """正向：2 读 1 写 = 3 × M×H×2B。**这是分母**，算错方向就反。"""
    r = B.generate("t1", "t", "elementwise", "y = a + b", "M=1024,H=4096",
                   "a:bf16[M,H], b:bf16[M,H]", "y:bf16[M,H]",
                   "elementwise-2in", write=False)
    assert abs(r["bytes_mb"] - 25.166) < 0.01
    kinds = [d["kind"] for d in r["detail"]]
    assert kinds.count("read") == 2 and kinds.count("write") == 1


def test_missing_dim_is_rejected():
    """**反向**：维度名不在 shape 里必须拒。

    否则 `x:bf16[M,H]` 在 shape={M} 下只乘 M，bytes 少一个数量级，
    分母错 → 结论方向反（本仓已复现多次）。
    """
    with pytest.raises(ValueError, match="不在形状"):
        B.generate("t2", "t", "elementwise", "y=x", "M=1024",
                   "x:bf16[M,H]", "y:bf16[M,H]", "elementwise", write=False)


def test_gemm_pattern_produces_no_bytes():
    """正向：GEMM 不吃 bytes 分母（由 FLOPs 约束）。"""
    r = B.generate("t3", "t", "gemm", "C=A@B", "M=1024,N=1024,K=1024",
                   "A:bf16[M,K], B:bf16[K,N]", "C:bf16[M,N]", "gemm", write=False)
    assert r["bytes_mb"] is None


# ---------------------------------------------------------------- 复合算子

def test_composite_operator_rejected():
    """**核心**：含矩阵乘的 math 不许用 elementwise pattern。

    实测场景：「add + RMSNorm + SwiGLU」既有 elementwise（x+r）、又有归约
    （rmsnorm）、又有两个 GEMM。硬套 elementwise 只算到 201MB，
    漏掉 FLOPs 主导的部分 —— 分母只有一半。
    """
    with pytest.raises(ValueError, match="矩阵乘"):
        B.generate("t4", "t", "activation",
                   "h = rmsnorm(x+residual); y = silu(h @ Wg) * (h @ Wu)",
                   "M=8192,H=4096", "x:bf16[M,H]", "y:bf16[M,H]",
                   "elementwise-2in", write=False)


def test_multi_mechanism_math_rejected():
    """**反向**：math 里三个以上不同机制 ⇒ 多半是复合算子，该拆。"""
    with pytest.raises(ValueError, match="复合算子"):
        B.generate("t5", "t", "activation",
                   "rmsnorm then softmax then topk then reduce",
                   "M=8,H=8", "x:bf16[M,H]", "y:bf16[M,H]",
                   "elementwise", write=False)


def test_family_pattern_mismatch_rejected():
    """**反向**：family 与访问模式不匹配 ⇒ 分母大概率错，拒。"""
    with pytest.raises(ValueError, match="不匹配"):
        B.generate("t6", "t", "gemm", "y=A@B", "M=1024,N=1024,K=1024",
                   "A:bf16[M,K], B:bf16[K,N]", "C:bf16[M,N]",
                   "elementwise", write=False)


# ---------------------------------------------------------------- 不覆盖 / 不猜

def test_refuses_to_overwrite_existing_card():
    """**反向**：bootstrap 不覆盖既有卡 —— 那会毁掉已有实测证据。"""
    with pytest.raises(ValueError, match="已存在"):
        B.generate("rmsnorm-fused", "t", "norm", "y=x", "M=8,H=8",
                   "x:bf16[M,H]", "y:bf16[M,H]", "reduce", write=True)


def test_generated_common_does_not_fake_reference():
    """**核心**：生成的 reference 必须是显式 NotImplementedError。

    参考实现是正确性的唯一锚。猜一个能让整个门禁「通过」一个算错的实现
    —— 因为两边都错。宁可让它跑不了。
    """
    src = B.build_common("t7", [{"name": "x", "dtype": "bf16", "dims": ["M"]}],
                         "y = x")
    assert "NotImplementedError" in src
    assert "正确性的唯一锚" in src


def test_parse_io_handles_commas_inside_brackets():
    """**反向**：`x:bf16[M,H], y:bf16[M,H]` 不能按逗号切分（M,H 里也有逗号）。"""
    out = B.parse_io("x:bf16[M,H], y:bf16[M,H]")
    assert len(out) == 2, out
    assert out[0]["dims"] == ["M", "H"], out[0]
    assert out[1]["name"] == "y", out[1]


# ---------------------------------------------------------------- 建出来就得过仓库级校验
#
# 2026-11-27 实测：第一张 bootstrap 卡（op-gated-add）建出来之后，全仓测试
# 从 13 failed 变成 27 failed —— 多出来的 14 个红**全是仓库级一致性用例**，
# 因为「能进 loop.py status」不等于「仓库认这张卡」。
# 下面每一条都对着一个真实红过的用例，反向钉住「产物清单不许再缩水」。

def _card():
    return B.generate("tcard", "t", "elementwise", "y = x + gate * s",
                      "M=8,H=8", "x:bf16[M,H], gate:bf16[M,H]",
                      "y:bf16[M,H]", "elementwise-2in", write=False)["card"]


def test_card_semantics_are_schema_clean():
    """对着 validate.py 的 E36：semantics 只认五个键，traffic_passes 是 mapping。

    第一版生成的是 `{inputs, outputs, mutates, traffic_passes: 3, note}` ——
    四个错误一次报齐（未知字段两个、traffic_passes 类型错、formula 缺失）。
    """
    sem = _card()["semantics"]
    assert set(sem) <= {"formula", "inputs", "outputs", "traffic_passes",
                        "mutates_inputs"}, sem
    assert isinstance(sem["formula"], str) and sem["formula"].strip()
    tp = sem["traffic_passes"]
    assert isinstance(tp, dict), tp
    assert isinstance(tp["reads"], int) and tp["reads"] > 0
    assert isinstance(tp["writes"], int) and tp["writes"] > 0
    # E36 的核心判据：「产出几个」必须等于「写几份」（201MB vs 268MB 的来源）
    assert tp["writes"] == len(sem["outputs"]), sem
    assert tp["passes"] == tp["reads"] + tp["writes"], tp


def test_card_gemm_semantics_still_has_positive_passes():
    """**反向**：GEMM 的 PATTERNS 是 r=0/w=0，直接抄进 semantics 会撞 E36。

    「不吃 bytes 分母」是 DRAM 口径，不等于「不读不写」——
    语义上仍然读 A、读 B、写 C。
    """
    card = B.generate("tgemm", "t", "gemm", "C=A@B", "M=8,N=8,K=8",
                      "A:bf16[M,K], B:bf16[K,N]", "C:bf16[M,N]",
                      "gemm", write=False)["card"]
    tp = card["semantics"]["traffic_passes"]
    assert tp["reads"] >= 1 and tp["writes"] == 1, tp


def test_card_has_nonempty_variants_with_role():
    """对着 validate.py 的 E6（variants 必须非空）+ E28a（每条都要 role）。"""
    vs = _card()["variants"]
    assert isinstance(vs, list) and vs, vs
    for v in vs:
        assert v.get("role"), v
        for f in ("repo", "file_hint", "tier"):
            assert v.get(f), (f, v)


def test_card_has_baseline_ref():
    """对着 test_baseline_ref 三条 + validate E28b。

    新建卡只可能有 ref-heavy（role=correctness_only），而 baseline-policy.yaml
    既排除 correctness_only、也把 ref-heavy 列进 forbidden_impl_patterns ——
    所以**合法状态是 no_baseline_reason**，不是编一个假基准。
    """
    br = _card()["baseline_ref"]
    assert isinstance(br, dict) and br, br
    assert br.get("no_baseline_reason"), br
    assert "ref-heavy" in str(br["no_baseline_reason"]), br


def test_card_yaml_has_ratio_null_when_unmeasured():
    """对着 test_atlas_content::test_no_fabricated_ratio（按文本匹配，故意）。"""
    import yaml
    text = yaml.safe_dump(_card(), allow_unicode=True, sort_keys=False)
    assert re.search(r"(?m)^  status: none", text), text
    assert re.search(r"(?m)^  ratio: null", text), text


def test_generated_ab_runs_preflight_by_default():
    """对着 test_entry_contract::test_every_bench_runs_preflight_by_default。

    它匹配的是**字面量** `if not a.skip_preflight:`，所以这里也按字面量守。
    """
    src = B.build_ab_py()
    for need in ("bench_preflight", "skip_preflight", "if not a.skip_preflight:"):
        assert need in src, "ab.py 缺 %r" % need
    compile(src, "<ab.py>", "exec")          # 注入后必须还是合法 Python


def test_generated_candidates_mark_prose_expect_low_confidence():
    """对着 check_expect.py：expect 是散文时必须标 expect_confidence: low。"""
    cands = B.build_candidates_yaml("t", "elementwise-2in", 1.0)["candidates"]
    assert cands, cands
    for c in cands:
        assert c.get("expect"), c
        assert c.get("expect_confidence") == "low", c
        assert c.get("expect_basis"), c


def test_register_fills_registry_and_ops_entry(tmp_path):
    """对着 test_card_sync + test_doc_links::test_every_card_is_registered。

    只登记卡、不动别的位置：registry 里多一条、ops/SKILL.md 表格里多一行。
    """
    reg = tmp_path / "registry.json"
    reg.write_text(json.dumps({"schema_version": 1, "updated": "x",
                               "ops": [{"name": "other", "path": "p",
                                        "status": "draft"}]},
                              ensure_ascii=False, indent=2), encoding="utf-8")
    entry = tmp_path / "SKILL.md"
    entry.write_text("| `op-other` | `draft` | 别的卡 |\n\n## 六、别的段\n",
                     encoding="utf-8")
    atlas = tmp_path / "atlas"
    atlas.mkdir()
    marker = tmp_path / "index_called"
    (atlas / "sync_index.py").write_text(
        "from pathlib import Path\n"
        "Path(%r).write_text('1', encoding='utf-8')\n" % str(marker),
        encoding="utf-8")

    card = _card()
    card["id"] = "tcard"
    probs = B.register(card, skill_variants=[{"name": "ref-heavy"}],
                       registry=reg, entry=entry, atlas=atlas)
    assert probs == [], probs

    ops = json.loads(reg.read_text(encoding="utf-8"))["ops"]
    assert [o["name"] for o in ops] == ["other", "tcard"], ops
    assert ops[1]["path"] == "Plugins/ops/skills/op-tcard"
    assert ops[1]["status"] == "draft"
    assert ops[1]["variants"] == ["ref-heavy"]

    text = entry.read_text(encoding="utf-8")
    assert "| `op-tcard` | `draft` |" in text
    assert text.index("op-other") < text.index("op-tcard"), "新行必须插在表内"
    assert text.index("op-tcard") < text.index("## 六"), "新行不许掉到表格外"
    assert marker.is_file(), "没有调用 sync_index.py"


def test_register_is_idempotent(tmp_path):
    """跑两次不许写出第二份 —— 否则每建一次卡就是两行重复。"""
    reg = tmp_path / "registry.json"
    reg.write_text(json.dumps({"schema_version": 1, "updated": "x", "ops": []},
                              ensure_ascii=False, indent=2), encoding="utf-8")
    entry = tmp_path / "SKILL.md"
    entry.write_text("| `op-other` | `draft` | x |\n", encoding="utf-8")
    atlas = tmp_path / "atlas"
    atlas.mkdir()
    (atlas / "sync_index.py").write_text("pass\n", encoding="utf-8")
    card = _card()
    card["id"] = "tcard"
    for _ in range(2):
        B.register(card, skill_variants=[{"name": "ref-heavy"}],
                   registry=reg, entry=entry, atlas=atlas)
    ops = json.loads(reg.read_text(encoding="utf-8"))["ops"]
    assert [o["name"] for o in ops] == ["tcard"], ops
    assert entry.read_text(encoding="utf-8").count("op-tcard") == 1, entry.read_text()
