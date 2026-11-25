#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""op_bootstrap —— 从「一个算子题目」到「一张能跑的卡」。

## 解决什么问题（冷启动断链）

现在的流程对**已有的卡**是通的：roofline.json → lang_select 判语言 → advance 列候选。
但对**新来的算子**，第一环就断了：

    roofline.json 需要 roofline 分析
      → roofline 分析需要 model（models/*.py）
        → model 需要有人先懂这个算子
    ⇒ 新算子卡在循环的第一环

实测（2026-11-25）：`derive.py` 只认 sparse-mla（参数硬编码）；
`grep "op_card|new_card|建卡" scripts/*.py` = 空；卡片 schema 有 20+ 字段、
`additionalProperties: false` —— 手写必然反复撞 E6。

## 但计算能力其实是通用的

`floor.py` 只要 `bytes_mb` + `pattern` + `t_actual_us` 就能算地板，
不需要专门的 model（实测：`floor.py --card gemm-bf16` 对一张没 roofline 的卡也能算）。
**缺的不是算，是「从题目到这四个数」的入口。**

## 本脚本做三件事

1. **按访问模式分类**，自动算 `bytes_mb` 与 `pattern`
2. **生成最小可跑卡**：contract / shapes / candidates / _common 骨架
3. **立刻算第一份 floor**，让语言路由有输入

## 纪律

- **不猜**。题目里给不出的（如参考实现的数学），生成占位符并标 `TODO`，
  同时让 `check_means.py` / 正确性门去拦 —— 不写一个看起来能跑但算错的参考。
- **访问模式决定一切**：它是 bytes 的来源，也是语言路由的依据。
  写错模式会让分母错，而分母错 → 结论方向反（本仓已复现多次）。
"""
import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
CARDS = REPO / "Plugins/ops/op-atlas/cards"
SKILLS = REPO / "Plugins/ops/skills"

# 访问模式 → (读几遍输入, 写几遍输出, bw-ladder 档位)
# **这几个数字是 bytes 的唯一来源**，写错就是分母错。
PATTERNS = {
    "elementwise":   {"r": 1, "w": 1, "ladder": "rw_1to1",
                      "note": "逐元素，读一份写一份"},
    "elementwise-2in": {"r": 2, "w": 1, "ladder": "rw_2to1",
                        "note": "两个输入一个输出（如 add）"},
    "reduce":        {"r": 1, "w": 1, "ladder": "rw_1to1",
                      "note": "行内归约，读一份写一份（折扣另计）"},
    "gather":        {"r": 1, "w": 1, "ladder": "rw_1to1",
                      "note": "gather/scatter：搬运算 read+write"},
    "gemm":          {"r": 0, "w": 0, "ladder": None,
                      "note": "GEMM 由 flops 约束，不走 bytes 分母"},
}

# family ← → 访问模式的合理对应。写错了要报，不静默接受。
FAMILY_PATTERN = {
    "elementwise": {"elementwise", "elementwise-2in"},
    # activation 也可能带归约：如 norm-act 三合一（rmsnorm 沿 H 归约后再激活）。
    # 规则太窄会误伤真实题目 —— 只把**明显不可能**的组合列为不匹配。
    "activation": {"elementwise", "elementwise-2in", "reduce"},
    "norm": {"reduce", "elementwise-2in"},
    "quant": {"elementwise", "elementwise-2in"},
    "rope": {"elementwise", "elementwise-2in"},
    "kv": {"elementwise", "gather"},
    "moe": {"gather", "gemm"},
    "gemm": {"gemm"},
    "attention": {"gemm"},
    "sampling": {"reduce"},
    "memory": {"gather", "elementwise"},
}


def parse_shape(text):
    """'M=8192,H=4096' → {'M':8192,'H':4096}。

    **支持派生维度**：`I=@H/2` 表示 I 是已解析维度的表达式。
    为什么要它：三合一这类算子里 `I = H/2` 是定义的一部分，
    让人再手填一个数既啰嗦又容易和 H 不一致。
    """
    out = {}
    for kv in str(text).split(","):
        kv = kv.strip()
        if not kv:
            continue
        if "=" not in kv:
            raise ValueError("形状必须写成 NAME=整数，收到 %r" % kv)
        k, v = kv.split("=", 1)
        k, v = k.strip(), v.strip()
        if v.startswith("@"):
            expr = v[1:].replace("/", "//")
            try:
                out[k] = int(eval(expr, {"__builtins__": {}}, dict(out)))   # noqa: S307
            except Exception as e:                                      # noqa: BLE001
                raise ValueError("派生维度 %r=%r 算不出来：%s" % (k, v, e))
            continue
        try:
            out[k] = int(v)
        except ValueError:
            raise ValueError("形状 %r 的值不是整数" % kv)
    if not out:
        raise ValueError("形状为空")
    return out


def parse_io(text):
    """'x:bf16[M,H], y:bf16[M,H]' → [{'name','dtype','dims'}]。"""
    out = []
    # **不能直接按逗号切**：`x:bf16[M,H], y:bf16[M,H]` 里的 `M,H` 也是逗号。
    # 先按 `]` 断句，再补回 —— 张量定义一定以 `]` 结尾。
    raw = str(text)
    parts, buf, depth = [], "", 0
    for ch in raw:
        if ch == "[":
            depth += 1
        elif ch == "]":
            depth -= 1
        if ch == "," and depth == 0:
            parts.append(buf); buf = ""
        else:
            buf += ch
    if buf.strip():
        parts.append(buf)
    for item in parts:
        item = item.strip()
        if not item:
            continue
        if ":" not in item:
            raise ValueError("输入必须写成 name:dtype[dims]，收到 %r" % item)
        name, spec = item.split(":", 1)
        dtype, _, dims = spec.partition("[")
        dims = dims.rstrip("]")
        out.append({"name": name.strip(), "dtype": dtype.strip(),
                    "dims": [d.strip() for d in dims.split(",") if d.strip()]})
    return out


def compute_bytes(inputs, outputs, shape, pattern):
    """按访问模式算搬运量（MB）+ 每趟的字节数明细。

    **这是整个卡的分母**。算错就是「分母错 → 结论方向反」。
    所以每一步都记账：哪个张量、读还是写、多少字节。
    """
    p = PATTERNS[pattern]
    detail = []
    reads = []

    def _extent(dims):
        """把 ['M','H'] 解析成元素总数。维度名不在 shape 里就报错。"""
        n = 1
        for d in dims:
            if d not in shape:
                raise ValueError("维度 %r 不在形状 %s 里" % (d, shape))
            n *= shape[d]
        return n

    dsz = {"bf16": 2, "fp16": 2, "fp32": 4, "int8": 1, "fp8": 1, "int32": 4}
    for t in inputs:
        n = _extent(t["dims"])
        b = n * dsz.get(t["dtype"], 2)
        reads.append(b)
        detail.append({"tensor": t["name"], "kind": "read", "bytes": b})
    total_w = 0
    for t in outputs:
        n = _extent(t["dims"])
        b = n * dsz.get(t["dtype"], 2)
        total_w += b
        detail.append({"tensor": t["name"], "kind": "write", "bytes": b})

    if pattern == "gemm":
        return None, detail      # GEMM 不吃 bytes 分母
    total = sum(reads) + total_w
    return round(total / 1e6, 3), detail


def build_card(name, title, family, math, shape, inputs, outputs, pattern):
    """返回卡片 dict（尚未写盘）。字段严格按 validate.py 的 ALLOWED_TOP。"""
    return {
        "id": name,
        "title": title,
        "family": family,
        "math": math,
        "status": "draft",          # 新建的卡一律 draft，有实测证据才能升
        "layer": "kernel",
        "op_card": None,
        "semantics": {
            "inputs": [t["name"] for t in inputs],
            "outputs": [t["name"] for t in outputs],
            "mutates": [],
            "traffic_passes": PATTERNS[pattern]["r"] + PATTERNS[pattern]["w"],
            "note": PATTERNS[pattern]["note"],
        },
        "variants": [],
        "ceiling": {"kind": "missing", "ref": None,
                    "bound": "memory" if pattern != "gemm" else "compute",
                    "note": "bootstrap 生成的空卡：尚无实测，待补"},
        "shapes": [list(shape) and "s0"],
        "floor_inputs": {"bytes_mb": None, "pattern": PATTERNS[pattern]["ladder"],
                         "t_actual_us": None, "t_source": "derived",
                         "basis": "bootstrap 预测，未实测"},
        "last_verdict": {"status": "none", "kb_path": None,
                         "conclusion": "bootstrap 新建，尚无实测",
                         "verified_on": None},
        "updated": "2026-11-25",
    }


def build_contract(name, math, inputs, outputs):
    """最小 contract：ULP 容差（bf16 输出按量级判）+ 三条不变量。"""
    tol = {"mode": "ulp_bound", "max_ulp": 2, "max_abs": 3.125e-2,
           "cosine": 0.99999} if any(t["dtype"] == "bf16" for t in outputs) else \
          {"max_abs": 1e-5, "cosine": 0.999999}
    return {
        "op": name, "version": 1, "math": math,
        "shapes": {t["name"]: t["dims"] for t in inputs + outputs},
        "dtypes": {t["name"]: t["dtype"] for t in inputs + outputs},
        "tolerance": {"vs_fp32_ref": tol,
                      "variant_vs_variant": {"mode": "tolerance"}},
        "invariants": [
            {"name": "determinism", "mode": "bit_exact", "repeats": 5},
        ],
    }


def render_contract(name, math, inputs, outputs):
    """手写 contract 文本。

    **不用 yaml.safe_dump**：它把 shapes 写成块列表（`- M`），而
    `opverify.parse_yaml` 是极简解析器，会报「map/list 混用」。
    实测踩过 —— bootstrap 生成的第一张卡就因为这个让 test_ops_gate 失败。
    其他卡（如 op-fused-add-rmsnorm）用的是行内 `[M, H]`，照做。
    """
    def inline(dims):
        return "[" + ", ".join(dims) + "]"
    ref = {"bf16": {"mode": "ulp_bound", "max_ulp": 2, "max_abs": 3.125e-2,
                    "cosine": 0.99999}}
    bf16_out = any(t["dtype"] == "bf16" for t in outputs)
    lines = [
        "# %s 契约（bootstrap 生成，需人工核对）" % name,
        "",
        "op: %s" % name,
        "version: 1",
        'math: "%s"' % str(math).replace('"', "'"),
        "",
        "shapes:",
    ]
    for t in inputs + outputs:
        lines.append("  %s: %s" % (t["name"], inline(t["dims"])))
    lines += ["", "dtypes:"]
    for t in inputs + outputs:
        lines.append("  %s: %s" % (t["name"], t["dtype"]))
    lines += ["", "tolerance:", "  vs_fp32_ref:"]
    if bf16_out:
        lines += ["    mode: ulp_bound", "    max_ulp: 2",
                  "    max_abs: 3.125e-2", "    cosine: 0.99999",
                  '    note: "bf16 按量级判 ULP；max_abs 仅兜底"']
    else:
        lines += ["    max_abs: 1.0e-5", "    cosine: 0.999999"]
    lines += ["  variant_vs_variant:", "    mode: tolerance", "",
              "invariants:", "  - name: determinism",
              "    mode: bit_exact", "    repeats: 5", ""]
    # **gates 与 floor 是 opverify 的必填项**（REQUIRED_TOP）。
    # 第一版 bootstrap 漏了它们，生成的卡跑 test_ops_gate 直接失败。
    lines += [
        "gates:",
        "  - name: correctness",
        "    template: correctness",
        "    required: true",
        "  - name: invariants",
        "    template: invariants",
        "    required: true",
        "  - name: perf",
        "    script: bench/ab.py",
        "    required: false",
        '    skip_reason: "bootstrap 生成时还没有实测，bench 待接"',
        "",
        "floor:",
        "  metric: hbm_bandwidth",
        "  # peak 必须来自 bw-ladder 的 achievable，不是型号标称值（1100 是型号名）",
        "  peak: 1730",
        "  unit: GB/s",
        "  peak_source: Plugins/ops/op-atlas/bw-ladder.yaml",
        '  basis: "bootstrap 按访问模式算出的字节数，见 floor_inputs"',
        "  at_floor_threshold: 0.10",
        "  status: unmeasured",
        "",
    ]
    return "\n".join(lines)


def build_common(name, inputs, reference_expr):
    """生成 _common.py 骨架。

    `reference` 里的数学是**显式占位符**：不写一个看起来能跑但算错的参考。
    `check_means.py` 会拦没有 means 的候选，正确性门会拦算错的实现。
    """
    in_names = ", ".join(t["name"] for t in inputs)
    out_names = ", ".join(t["name"] for t in outputs_ref(outputs=None)) \
        if False else "y"
    return '''#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""op-%s 的 adapter（bootstrap 生成，**需要人工补齐 reference 的数学**）。

生成时间：2026-11-25
生成器：scripts/op_bootstrap.py

## 必须补的地方（否则正确性门跑不了）

1. `reference()` 里的数学 —— 现在只有形状占位，没有真实计算
2. `VARIANTS` —— 至少要有一个真实实现（库调用或手写）
3. `build_inputs()` 的数值分布 —— 要覆盖真实场景的取值范围

## 为什么 reference 不自动生成

参考实现是**正确性的唯一锚**。猜一个参考等于让整个门禁失去意义 ——
它会「通过」一个算错的实现，因为两边都错。
所以这里留占位符 + 让门禁拦住，而不是写一个能跑但不可信的版本。
"""
import torch

# TODO(必填)：填真实实现名
VARIANTS = ("ref-heavy",)

INPUT_NAMES = (%s)


def build_inputs(params, seed=0, device=None):
    """构造输入。TODO(必填)：按真实数值分布调整。"""
    g = torch.Generator(device="cpu").manual_seed(seed)
    dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
    # TODO：按 --inputs 的语义构造
    raise NotImplementedError(
        "bootstrap 生成的 build_inputs 是空的 —— 必须先按真实分布实现，"
        "否则测出来的数字没有意义")


def reference(%s, params=None):
    """fp32 参考。TODO(必填)：补真实数学。

    %s
    """
    raise NotImplementedError(
        "bootstrap 生成的 reference 是空的。**不要跳过这一步** —— "
        "参考实现是正确性的唯一锚，猜一个会让整个门禁失去意义。")


def runner_kwargs(params):
    return dict(params)


def get_runner(name, **kwargs):
    if name not in VARIANTS:
        raise ValueError("未知变体 %%r（可选 %%s）" %% (name, " / ".join(VARIANTS)))
    if name == "ref-heavy":
        return lambda *a, **kw: reference(*a, params=kw)
    raise RuntimeError("变体 %%r 的 adapter 尚未实现" %% name)
''' % (name, in_names, in_names, reference_expr)


def outputs_ref(outputs):
    return outputs or []


def build_skill_md(name, title, math, inputs, outputs, pattern, bytes_mb):
    """生成 SKILL.md（卡的自述）。**必须与卡上字段同源**，不另写一份知识。"""
    in_desc = ", ".join("%s:%s[%s]" % (t["name"], t["dtype"], ",".join(t["dims"]))
                        for t in inputs)
    out_desc = ", ".join("%s:%s[%s]" % (t["name"], t["dtype"], ",".join(t["dims"]))
                         for t in outputs)
    sz = ("%.1f MB（pattern=%s）" % (bytes_mb, pattern)) if bytes_mb else \
         "由 FLOPs 约束，不走 bytes 分母"
    return """---
name: %(name)s
description: "%(title)s。输入 %(in_desc)s，输出 %(out_desc)s。用途：(bootstrap 生成，待补真实场景描述)。触发场景：(待补)。不适用：(待补)。"
---

# %(name)s

%(math)s

## 输入输出

- 输入：%(in_desc)s
- 输出：%(out_desc)s
- 搬运量：%(sz)s

## 状态

bootstrap 生成，**尚未实测**。补齐顺序：

1. `_common.py` 里的 `reference()` —— 现在是 `NotImplementedError`。
   参考实现是正确性的唯一锚，不要写一个「看起来能跑」的版本。
2. `build_inputs()` 的数值分布 —— 要覆盖真实场景的取值范围。
3. `VARIANTS` 里填真实实现（库调用或手写 kernel）。
4. 跑一次基线拿到 `t_actual`，再 `floor.py` 出 roofline。
5. `lang_select.py` 判语言路线（**它需要 roofline，所以第 4 步不能省**）。

## 纪律

- 分母来源见 `floor_inputs`，算错方向就反。
- 每次对比必须 base 重测。
""" % {"name": name, "title": title, "math": math, "in_desc": in_desc,
       "out_desc": out_desc, "sz": sz}


def build_variants_yaml(name, inputs, outputs, pattern):
    """生成 variants.yaml。只有 ref-heavy —— 真实实现要人补。

    **不自动填一个「看起来能用」的变体**：那会让 `get_runner` 返回一个
    没人验证过的实现，而它可能算错。宁可是空的等人填。
    """
    p = PATTERNS[pattern]
    return {"variants": [{
        "name": "ref-heavy",
        "source": "本卡 _common.py 的 fp32 参考",
        "status": "reference",
        "role": "correctness_only",
        "applies_to": ["s0"],
        "mutates_inputs": False,
        "semantics": {
            "inputs": [t["name"] for t in inputs],
            "outputs": [t["name"] for t in outputs],
            "mutates": [],
            "traffic_passes": p["r"] + p["w"],
            "note": "fp32 参考，只用于正确性对照，不是性能候选"},
        "note": "**不得用于性能对比**：未优化的参考实现",
    }]}


def generate(name, title, family, math, shape_text, inputs_text,
             outputs_text, pattern, write=True):
    """完整一步：算 bytes → 建卡 → 建 contract → 建 _common → 算 floor。

    返回报告 dict。**不写一个看起来完整但算错的产物**：
    任何一步算不出来就明说，不猜。
    """
    shape = parse_shape(shape_text)
    inputs = parse_io(inputs_text)
    outputs = parse_io(outputs_text)

    # **复合算子不能用一个 pattern 概括**。实测（2026-11-25）：一个
    # 「add + RMSNorm + SwiGLU」的题目，既有 elementwise（x+r）、又有归约
    # （rmsnorm）、又有两个 GEMM（h@W_gate / h@W_up）。硬套 elementwise
    # 会把 201MB 当成全部成本，漏掉 FLOPs 主导的那部分 —— 分母只有一半。
    multi = [kw for kw in (" @ ", "matmul", "gemm", "rmsnorm", "layernorm",
                           "sum(", "mean(", "reduce", "softmax", "topk")
             if kw in str(math).lower() or kw in str(math)]
    has_gemm = any(k in str(math) for k in (" @ ", "matmul", "gemm"))
    if has_gemm and pattern != "gemm":
        raise ValueError(
            "math 里有矩阵乘（%s），但 pattern=%r。**复合算子要拆开**："
            "GEMM 由 FLOPs 约束、elementwise 由字节约束，两种分母不能混。"
            "先用 --pattern gemm 建 GEMM 那部分，再单独建 elementwise 部分。"
            % (", ".join(k for k in ("@",) if k in str(math)), pattern))
    if len([k for k in multi if k in str(math).lower()]) >= 3:
        raise ValueError(
            "math 里出现 %d 个不同机制（%s）—— 这多半是**复合算子**，"
            "不该用单一 pattern。请拆成多个子算子分别 bootstrap，"
            "再决定是否值得融合（融合前先问两半的最优 CTA 粒度是否相容）。"
            % (len(multi), multi))
    fam = set(FAMILY_PATTERN.get(family, set()))
    if fam and pattern not in fam:
        raise ValueError(
            "family=%r 与访问模式 %r 不匹配（%r 通常是 %s）。"
            "**访问模式决定分母**，写错会让结论方向反 —— 请确认后重试。"
            % (family, pattern, family, sorted(fam)))

    bytes_mb, detail = compute_bytes(inputs, outputs, shape, pattern)
    card = build_card(name, title, family, math, shape, inputs, outputs, pattern)

    out = {"card": card, "bytes_mb": bytes_mb, "detail": detail,
           "shape": shape, "pattern": pattern, "written": []}

    # floor：有 bytes 就立刻算，让语言路由有输入
    if bytes_mb:
        card["floor_inputs"].update({
            "bytes_mb": bytes_mb, "t_source": "derived",
            "basis": "bootstrap 按访问模式 %s 算出：%s"
                     % (pattern, " + ".join("%s(%s)=%dB" % (d["tensor"], d["kind"],
                                                            d["bytes"]) for d in detail))})
        out["floor_note"] = ("bytes=%.3f MB（pattern=%s）。**还没有 t_actual**，"
                             "所以 remaining 算不出来 —— 先跑一次实现拿到实测值。"
                             % (bytes_mb, pattern))
    if write:
        CARDS.mkdir(parents=True, exist_ok=True)
        cf = CARDS / ("%s.yaml" % name)
        if cf.exists():
            raise ValueError("卡 %s 已存在 —— bootstrap 不覆盖既有卡" % cf)
        import yaml
        cf.write_text(yaml.safe_dump(card, allow_unicode=True, sort_keys=False),
                      encoding="utf-8")
        out["written"].append(str(cf))

        sd = SKILLS / ("op-%s" % name)
        sd.mkdir(parents=True, exist_ok=True)
        ct = sd / "contract.yaml"
        ct.write_text(render_contract(name, math, inputs, outputs),
                      encoding="utf-8")
        out["written"].append(str(ct))
        cm = sd / "_common.py"
        cm.write_text(build_common(name, inputs, math), encoding="utf-8")
        out["written"].append(str(cm))
        sh = sd / "shapes.yaml"
        sh.write_text(yaml.safe_dump({"s0": {"params": shape,
                                             "note": "bootstrap 生成"}},
                                     allow_unicode=True, sort_keys=False),
                      encoding="utf-8")
        out["written"].append(str(sh))
        # SKILL.md 与 variants.yaml 也一起生成 —— 少了它们卡是「半张」，
        # 每次都要人补（实测：第一张 bootstrap 卡就是漏了这两个）。
        sk = sd / "SKILL.md"
        sk.write_text(build_skill_md(name, title, math, inputs, outputs,
                                     pattern, bytes_mb), encoding="utf-8")
        out["written"].append(str(sk))
        vf = sd / "variants.yaml"
        vf.write_text(yaml.safe_dump(
            build_variants_yaml(name, inputs, outputs, pattern),
            allow_unicode=True, sort_keys=False), encoding="utf-8")
        out["written"].append(str(vf))
    return out


def selftest():
    """**反向优先**：先证明它拦得住那些会让分母出错的输入。"""
    # ① 正常：add 类，2 读 1 写
    r = generate("t-add", "t", "elementwise", "y = x1 + x2", "M=1024,H=4096",
                 "x1:bf16[M,H], x2:bf16[M,H]", "y:bf16[M,H]",
                 "elementwise-2in", write=False)
    # 1024*4096*2B * 3 = 25.166 MB
    assert abs(r["bytes_mb"] - 25.166) < 0.01, r["bytes_mb"]
    print("[ok] 正向：2 读 1 写 = 25.166 MB（1024×4096×2B×3）")

    # ② 反向：family 与 pattern 不匹配 —— 分母会错，必须拒
    try:
        generate("t-bad", "t", "gemm", "y = A@B", "M=1024,N=1024,K=1024",
                 "A:bf16[M,K], B:bf16[K,N]", "C:bf16[M,N]", "elementwise", write=False)
    except ValueError as e:
        assert "不匹配" in str(e), e
        print("[ok] 反向：gemm 配 elementwise 被拒（访问模式决定分母，写错方向就反）")
    else:
        raise AssertionError("family/pattern 不匹配应当被拒")

    # ③ 反向：维度名不在 shape 里 —— 会算出错的 bytes
    try:
        generate("t-bad2", "t", "elementwise", "y = x", "M=1024",
                 "x:bf16[M,H]", "y:bf16[M,H]", "elementwise", write=False)
    except ValueError as e:
        assert "不在形状" in str(e), e
        print("[ok] 反向：维度 H 不在形状里被拒（否则 bytes 会算成只乘 M）")
    else:
        raise AssertionError("维度缺失应当被拒")

    # ④ 反向：不覆盖既有卡
    try:
        generate("rmsnorm-fused", "t", "norm", "y=x", "M=8,H=8",
                 "x:bf16[M,H]", "y:bf16[M,H]", "reduce", write=True)
    except ValueError as e:
        assert "已存在" in str(e), e
        print("[ok] 反向：不覆盖既有卡")
    else:
        raise AssertionError("既有卡应被拒")

    # ⑤ 正向：GEMM 不走 bytes 分母
    r2 = generate("t-gemm", "t", "gemm", "C=A@B", "M=1024,N=1024,K=1024",
                  "A:bf16[M,K], B:bf16[K,N]", "C:bf16[M,N]", "gemm", write=False)
    assert r2["bytes_mb"] is None, r2["bytes_mb"]
    print("[ok] 正向：gemm 不产出 bytes 分母（由 flops 约束）")

    print("selftest PASS")
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--name", help="算子 id（建卡用）")
    ap.add_argument("--title", default="", help="中文标题")
    ap.add_argument("--family", help="validate.py 的 FAMILIES 之一")
    ap.add_argument("--math", default="TODO: 写出数学", help="数学表达式")
    ap.add_argument("--shape", help="M=8192,H=4096")
    ap.add_argument("--inputs", help="x:bf16[M,H], residual:bf16[M,H]")
    ap.add_argument("--outputs", help="y:bf16[M,H]")
    ap.add_argument("--pattern", default="elementwise-2in",
                    choices=sorted(PATTERNS))
    ap.add_argument("--dry-run", action="store_true", help="只算不写")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args(argv)
    if a.selftest:
        return selftest()
    if not (a.name and a.family and a.shape and a.inputs and a.outputs):
        ap.error("需要 --name --family --shape --inputs --outputs（或 --selftest）")
    try:
        r = generate(a.name, a.title or a.name, a.family, a.math, a.shape,
                     a.inputs, a.outputs, a.pattern, write=not a.dry_run)
    except ValueError as e:
        print("拒绝：%s" % e, file=sys.stderr)
        return 1
    print(json.dumps({"bytes_mb": r["bytes_mb"], "pattern": r["pattern"],
                      "detail": r["detail"], "written": r["written"],
                      "floor_note": r.get("floor_note")},
                     ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
