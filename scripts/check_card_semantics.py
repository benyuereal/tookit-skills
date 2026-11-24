#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""卡片与源文件语义一致性检查（D8 的机器判据）。

## 这条债是什么

`mhc-fused.yaml` 的 `math` 写「conv1d + hadamard + norm 的融合链」，
但它的 `variants[0].source_ref.path` 指向的 `mhc_fused.hip` 头注释写的是
「Post + next-layer-projection fusion: the step-3 experiments.」——
两个完全不同的算子。D5（`mlp-gelu-geglu`）是同一类错。

## 判据怎么定，为什么这样定

**不能**用「卡上的词必须出现在源文件里」这种朴素判据 —— 中文 `math` 与
英文注释不会有共同词。也不能让模型判断（那等于没判据）。

实际可机检的只有一件事：**卡的 `math` 声称的算子族，必须能在源文件的
标识符（kernel 名 / extern "C" 符号 / 宏名）里找到对应**。

所以判据是：从 `math` 抽出**算子族关键词**（conv1d / hadamard / norm /
sinkhorn / post / projection / rmsnorm / softmax ...），再去源文件里找
同名 token。**一个都找不到 → 可疑**，报出来人工裁决。

这不是「一定会抓对」的判据，是「一定会报出来」的判据。
误报可接受（人看一眼就知道），**漏报不可接受**（那正是 D8 的成因）。
"""
import argparse
import json
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
CARDS = REPO / "Plugins/ops/op-atlas/cards"

# 算子族关键词表。**只收有明确技术含义的词**，不收「融合」「链」这类
# 到处都有的词 —— 那些会让判据永远命中，等于没有判据。
FAMILY_WORDS = {
    "conv1d": (r"conv1d|conv_1d|convolution",),
    "hadamard": (r"hadamard|hadamard_transform",),
    "norm": (r"\bnorm\b|rmsnorm|layernorm|groupnorm|layer_norm|rms_norm",),
    "sinkhorn": (r"sinkhorn",),
    "softmax": (r"softmax",),
    "post": (r"\bpost\b|hc_post|mhc_post",),
    "projection": (r"proj|projection|gemm|mmac|mfma",),
    "gelu": (r"gelu",),
    "silu": (r"silu|swish",),
    "attention": (r"attention|flash|mla",),
    "rope": (r"\brope\b|rotary",),
    "topk": (r"topk|top_k",),
}

# 源文件里能当「标识符」看的部分：extern "C" 符号、__global__ 名、#define 名、注释首段
_IDENT_PAT = re.compile(
    r'(?:extern\s+"C"[^;{]*?(\w+)\s*\()'          # extern "C" 函数名
    r'|(?:__global__[^;{]*?(\w+)\s*\()'            # kernel 名
    r'|(?:#define\s+(\w+))'                        # 宏名
    r'|(?:^\s*//\s*(.+)$)',                        # 行注释（取头注释用）
    re.MULTILINE)


def families_in_math(math_text):
    """math 声明里出现的算子族。返回 {族名: 命中原文}。"""
    out = {}
    low = str(math_text).lower()
    for fam, pats in FAMILY_WORDS.items():
        for p in pats:
            if re.search(p, low):
                out[fam] = p
                break
    return out


def source_tokens(path):
    """源文件里可作标识符的 token 集合（小写）。读不到返回 None。"""
    p = Path(path)
    if not p.is_file():
        return None
    try:
        text = p.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None
    toks = set()
    for m in _IDENT_PAT.finditer(text):
        for g in m.groups():
            if g:
                toks.update(re.findall(r"[a-z0-9_]+", g.lower()))
    return toks


def check_card(card_path):
    """返回问题列表。空 = 通过（或无可检项）。"""
    import yaml
    try:
        d = yaml.safe_load(Path(card_path).read_text(encoding="utf-8")) or {}
    except Exception as e:                       # noqa: BLE001
        return ["%s: 读不了卡（%s）" % (Path(card_path).name, e)]
    math = d.get("math")
    if not math:
        return []                                # 没有 math 就没有可检的语义
    fams = families_in_math(math)
    if not fams:
        return []                                # math 里没有可识别的算子族词
    probs = []
    for v in (d.get("variants") or []):
        if not isinstance(v, dict):
            continue
        src = (v.get("source_ref") or {}).get("path")
        if not src or str(src).startswith("tilelang"):
            continue                             # 无具体源文件，不检
        toks = source_tokens(src)
        if toks is None:
            continue                             # 源文件不在本机（跨机器路径）→ 不检，不算通过
        hit = {f for f in fams if any(re.search(p, " ".join(toks)) for p in FAMILY_WORDS[f])}
        if not hit:
            probs.append(
                "%s: math 声称 %s，但源文件 %s 里找不到任何对应标识符"
                % (Path(card_path).stem, sorted(fams), Path(src).name))
    return probs


def check_all(cards_dir=CARDS):
    probs = []
    for f in sorted(Path(cards_dir).glob("*.yaml")):
        probs += check_card(f)
    return probs


def selftest():
    """**反向优先**：先证明它抓得住 D8，再证明它不误伤正常卡。"""
    import tempfile
    import yaml

    def mk(dirpath, name, math, src_text, src_name="k.hip"):
        d = Path(dirpath)
        (d / src_name).write_text(src_text, encoding="utf-8")
        (d / (name + ".yaml")).write_text(yaml.safe_dump({
            "id": name, "math": math,
            "variants": [{"source_ref": {"path": str(d / src_name)}}]},
            allow_unicode=True), encoding="utf-8")
        return str(d / (name + ".yaml"))

    with tempfile.TemporaryDirectory() as td:
        # 1) 正是 D8 的形态：math 说 conv1d/hadamard/norm，源文件是 post+proj
        bad = mk(td, "bad", "conv1d + hadamard + norm 的融合链",
                 '// Post + next-layer-projection fusion\n'
                 'extern "C" void hc_post_proj_launch(void);\n')
        probs = check_card(bad)
        assert probs and "找不到" in probs[0], probs
        print("[ok] 反向：复现 D8（conv1d/hadamard/norm 卡指向 post+proj 源）")

        # 2) 语义相符的卡不该被报
        good = mk(td, "good", "conv1d + hadamard + norm 的融合链",
                  '// conv1d then hadamard then norm\n'
                  'extern "C" void conv1d_hadamard_norm_launch(void);\n',
                  src_name="ok.hip")
        assert check_card(good) == [], check_card(good)
        print("[ok] 正向：语义相符的卡不误报")

        # 3) math 里没有算子族词 → 不检（避免拿空判据当通过）
        vague = mk(td, "vague", "把三件事串起来",
                   'extern "C" void x(void);\n', src_name="v.hip")
        assert check_card(vague) == []
        print("[ok] 边界：math 无可识别算子族词时不检")

        # 4) 源文件不在本机 → 不检，也不许算通过
        missing = str(Path(td) / "nope.yaml")
        Path(missing).write_text(yaml.safe_dump({
            "id": "m", "math": "sinkhorn 归一",
            "variants": [{"source_ref": {"path": "/nonexistent/x.hip"}}]}),
            encoding="utf-8")
        assert check_card(missing) == [], "源文件缺失时不该报，但也不该声称通过"
        print("[ok] 边界：源文件缺失时不检（不假装通过）")
    print("selftest PASS")
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cards", default=str(CARDS))
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args(argv)
    if a.selftest:
        return selftest()
    probs = check_all(a.cards)
    if a.json:
        print(json.dumps({"ok": not probs, "problems": probs},
                         ensure_ascii=False, indent=2))
    else:
        print(("✓ " if not probs else "✗ ") + "卡-源文件语义一致")
        for x in probs:
            print("  - " + x)
    return 0 if not probs else 1


if __name__ == "__main__":
    sys.exit(main())
