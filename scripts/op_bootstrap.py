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

## 本脚本做四件事

1. **按访问模式分类**，自动算 `bytes_mb` 与 `pattern`
2. **生成最小可跑卡**：atlas 卡 / contract / shapes / candidates / _common / SKILL / variants / bench
3. **把卡登记进仓库**：`data/ops/registry.json`、`Plugins/ops/SKILL.md` 索引表、
   `op-atlas/INDEX.md`（调用 `sync_index.py`）
4. **立刻算第一份 floor**，让语言路由有输入

## 为什么第 3 步必须在建卡时就做（2026-11-27 实测）

第一张 bootstrap 卡（`op-gated-add`）建出来之后，全仓测试从 13 failed 变成
27 failed —— 多出来的 14 个红**全是仓库级一致性用例**：没注册、没
`baseline_ref`、`semantics` 撞 E36、`last_verdict` 缺 `ratio: null`、
`bench/ab.py` 没接 `bench_preflight`……「能进 `loop.py status`」和
「仓库认这张卡」是两件事，只做到前者等于把 14 个红留给下一个人。

**产物清单以「仓库级校验要求什么」为准**，不是以「跑得起来需要什么」为准。

`gaps.md` 例外：它是全局生成物，重算会改写**所有**卡的「连续缺证据批数」
（可能把别人的卡推到「连续 ≥2 批 → 降级 draft」），所以 bootstrap 不碰它，
只在报告里提示。

## 纪律

- **不猜**。题目里给不出的（如参考实现的数学），生成占位符并标 `TODO`，
  同时让 `check_means.py` / 正确性门去拦 —— 不写一个看起来能跑但算错的参考。
- **访问模式决定一切**：它是 bytes 的来源，也是语言路由的依据。
  写错模式会让分母错，而分母错 → 结论方向反（本仓已复现多次）。
"""
import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
ATLAS = REPO / "Plugins/ops/op-atlas"
CARDS = ATLAS / "cards"
SKILLS = REPO / "Plugins/ops/skills"
REGISTRY = REPO / "data/ops/registry.json"
OPS_ENTRY = REPO / "Plugins/ops/SKILL.md"
SYNC_INDEX = ATLAS / "sync_index.py"

# 文本骨架放 scripts/templates/contract.md；本文件只负责拼动态块。
# `-I` 不把脚本目录放进 sys.path（测试还按路径 importlib 加载），这里补一次。
_SCRIPTS = Path(__file__).resolve().parent
if str(_SCRIPTS) not in sys.path:
    sys.path.append(str(_SCRIPTS))

from template_render import render as _render_text  # noqa: E402


def _today():
    import datetime
    return datetime.date.today().isoformat()

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


def build_semantics(inputs, outputs, pattern, math):
    """atlas 卡的 semantics 块（validate.py 的 E36 只认这五个键）。

    为什么不能沿用第一版的 `{inputs, outputs, mutates, traffic_passes, note}`：
    E36 的白名单是 `{formula, inputs, outputs, traffic_passes, mutates_inputs}`，
    多一个键就是 error —— 第一张 bootstrap 卡就是这么红的。
    另外 E36 要求 `traffic_passes.writes == len(outputs)`：
    「产出几个」与「写几份」不一致正是 201MB / 268MB 结论反转的来源。
    """
    p = PATTERNS[pattern]
    # gemm 的 PATTERNS 里 r/w 都是 0（不吃 bytes 分母），但 semantics 的
    # traffic_passes 记的是**语义张量**读几遍写几份，不是 DRAM 遍数，
    # 所以 GEMM 也得写成「读 A、读 B、写 C」。写 0 会直接撞 E36（要求正整数）。
    reads = p["r"] or len(inputs)
    writes = len(outputs)
    return {
        "formula": str(math or "").strip() or "TODO: 写出数学",
        "inputs": ["%s (%s)" % (t["name"], ",".join(t["dims"])) for t in inputs],
        "outputs": ["%s (%s)" % (t["name"], ",".join(t["dims"])) for t in outputs],
        "traffic_passes": {
            "reads": reads,
            "writes": writes,
            "passes": reads + writes,
            "source": "bootstrap 按访问模式 %s 推定（**未实测，须用 --actual 校正**）"
                      % pattern,
        },
        "mutates_inputs": [],
    }


def build_baseline_ref(inputs, outputs, pattern):
    """atlas 卡的 baseline_ref（validate.py E28b + test_baseline_ref 要它）。

    **新建卡一律先写 no_baseline_reason，不写一个假基准。**
    理由：bootstrap 建出来的卡只可能有一条 ref-heavy（本卡 _common.py 的 fp32
    参考，role=correctness_only）。`baseline-policy.yaml` 既把 correctness_only
    排除在性能基准之外，又把 `ref-heavy` 列进 `forbidden_impl_patterns` ——
    把它写成基准，等于把「3.65×」那个假数再犯一次。

    E28b 与 test_baseline_ref 都把 `no_baseline_reason` 当合法状态
    （「确实没有基准时用 no_baseline_reason 显式声明，比编一个假基准好」），
    所以这里同时给出按 semantics 推定的 traffic_passes：等真有基准时，
    人只要填 impl + role，不用再算一遍读几遍写几份。
    """
    p = PATTERNS[pattern]
    reads = p["r"] or len(inputs)
    writes = len(outputs)
    return {
        "impl": None,
        "role": "unverified",
        "traffic_passes": {
            "reads": reads,
            "writes": writes,
            "passes": reads + writes,
            "source": "按 semantics 推定（**未实测，须用 --actual 校正**）",
        },
        "no_baseline_reason": (
            "本卡目前只有一条实现 ref-heavy（本卡 _common.py 的 fp32 参考，"
            "role=correctness_only）。baseline-policy.yaml 既把 correctness_only "
            "排除在性能基准之外，也把 ref-heavy 列进 forbidden_impl_patterns —— "
            "拿它当基准算出的倍数没有工程意义。等有厂商/第三方实现（或本卡自研"
            "优化版）时回填 impl + role=competitor|best_known，并按实际测量校正 "
            "traffic_passes。"),
        "note": "在补上真基准之前，本卡不许写任何「快多少」",
    }


def build_atlas_variants(name, inputs, outputs, pattern):
    """atlas 卡的 variants（E6 要求非空数组；E28a 要求每条都带 role）。

    只放一条 in-house 的 ref-heavy，`role: correctness_only`。

    **刻意不写 import_path**：`test_import_path_callable` 会 import 每一个
    import_path 并要求它指向可调用对象，而 `_common.py:reference` 这种卡内路径
    既 import 不了也不该冒充库函数。E13/E15（必须落到 库.函数、必须带
    source_ref）只对 `tier: primary` 生效，这里是 reference，所以不适用；
    仍然写 `l1_exempt_reason` 说明为什么没有库函数。
    """
    in_names = [t["name"] for t in inputs]
    return [{
        "repo": "in-house",
        "library_profile_ref": None,
        "file_hint": "本卡 _common.py 的 fp32 参考实现（变体名 ref-heavy）",
        "tier": "reference",
        "role": "correctness_only",
        "best_for": [
            "建立本算子的正确性锚与首个可运行基线（现成库没有同名入口）",
        ],
        "not_for": [
            "算收益：未优化的参考实现打赢了没有工程意义（baseline-policy 明令禁止）",
        ],
        "pros": [
            "与本卡 _common.py 同源，输入构造与语义口径不会漂",
        ],
        "cons": [
            "现成库不支持这个算子：要跑只能自研，参考实现的绝对值不代表库的上限",
        ],
        "source": "Plugins/ops/skills/op-%s/_common.py" % name,
        "l1_exempt_reason": (
            "自研参考实现，不是可 import 的库函数；真实入口是 "
            "op-%s/_common.py 的 reference(%s) 与 get_runner(\"ref-heavy\")"
            % (name, ", ".join(in_names) or "...")),
    }]


def build_card(name, title, family, math, shape, inputs, outputs, pattern):
    """返回卡片 dict（尚未写盘）。字段严格按 validate.py 的 ALLOWED_TOP。

    **产物清单是「仓库级校验要求什么」，不是「loop.py 跑得起来需要什么」。**
    第一版按后者写，于是：`semantics` 撞 E36、`variants` 空数组撞 E6、
    `last_verdict` 缺 `ratio: null` 撞 test_atlas_content（status=none 却有
    ratio 的口子）、没有 `baseline_ref` 撞 E28b。
    """
    p = PATTERNS[pattern]
    return {
        "id": name,
        "title": title,
        "family": family,
        "math": math,
        "status": "draft",          # 新建的卡一律 draft，有实测证据才能升
        "layer": "kernel",
        # 已登记进 registry，op_card 才能非空（E4：registry 无此算子时必须为 null）
        "op_card": "Plugins/ops/skills/op-%s" % name,
        "semantics": build_semantics(inputs, outputs, pattern, math),
        "variants": build_atlas_variants(name, inputs, outputs, pattern),
        "ceiling": {"kind": "missing", "ref": None,
                    "bound": "memory" if pattern != "gemm" else "compute",
                    "note": "bootstrap 生成的空卡：尚无实测天花板，待补"},
        "shapes": ["s0"],
        "floor_inputs": {"bytes_mb": None, "pattern": p["ladder"],
                         "t_actual_us": None, "t_source": "derived",
                         "basis": "bootstrap 预测，未实测"},
        "last_verdict": {
            "status": "none",
            "kb_path": None,
            # status=none 时 ratio 必须是 null：有 status 有 ratio 等于
            # 「没测却有倍数」，test_atlas_content::test_no_fabricated_ratio 守着。
            "ratio": None,
            "conclusion": "bootstrap 新建，尚无实测",
            "verified_on": None,
        },
        "baseline_ref": build_baseline_ref(inputs, outputs, pattern),
        "updated": _today(),
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
    """contract 文本走 scripts/templates/contract.md，本函数只拼动态块。

    **不用 yaml.safe_dump**：它把 shapes 写成块列表（`- M`），而
    `opverify.parse_yaml` 是极简解析器，会报「map/list 混用」。
    实测踩过 —— bootstrap 生成的第一张卡就因为这个让 test_ops_gate 失败。
    其他卡（如 op-fused-add-rmsnorm）用的是行内 `[M, H]`，照做。

    动态块（shapes/dtypes/tolerance）每条自带换行；为空时为空串，
    由模板里的单个换行保持原来的空行/无空行布局。
    """
    def inline(dims):
        return "[" + ", ".join(dims) + "]"

    def block(items):
        return "\n".join(items) + "\n" if items else ""

    bf16_out = any(t["dtype"] == "bf16" for t in outputs)
    shapes_body = block(["  %s: %s" % (t["name"], inline(t["dims"]))
                         for t in inputs + outputs])
    dtypes_body = block(["  %s: %s" % (t["name"], t["dtype"])
                         for t in inputs + outputs])
    if bf16_out:
        tol_lines = ["    mode: ulp_bound", "    max_ulp: 2",
                     "    max_abs: 3.125e-2", "    cosine: 0.99999",
                     '    note: "bf16 按量级判 ULP；max_abs 仅兜底"']
    else:
        tol_lines = ["    max_abs: 1.0e-5", "    cosine: 0.999999"]

    # **gates 与 floor 是 opverify 的必填项**（REQUIRED_TOP），
    # 模板里已固定；第一版 bootstrap 漏了它们，新卡跑 test_ops_gate 直接失败。
    return _render_text(
        "contract",
        op_name=name,
        math=str(math).replace('"', "'"),
        shapes_body=shapes_body,
        dtypes_body=dtypes_body,
        # tolerance 块后面没有空行：模板占位符行自带换行，这里不再补。
        tolerance_body="\n".join(tol_lines),
    )


def build_common(name, inputs, reference_expr):
    """生成 _common.py 骨架。

    `reference` 里的数学是**显式占位符**：不写一个看起来能跑但算错的参考。
    `check_means.py` 会拦没有 means 的候选，正确性门会拦算错的实现。
    """
    in_names = ", ".join(t["name"] for t in inputs)
    # 生成的是**字符串元组**，必须带引号；原模板写成 INPUT_NAMES = (x, r)
    # 会在 import 时 NameError（实测 2026-11-27 新卡第一次运行就炸）。
    in_names_lit = ", ".join('"%s"' % t["name"] for t in inputs)
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

INPUT_NAMES = (%s,)


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


def load_shape(arg=None):
    """读本卡 shapes.yaml。ab.py / loop.py 都靠它拿形状。

    bootstrap 以前不生成这个函数，新卡第一次 `loop.py run` 就报
    `module '_common' has no attribute 'load_shape'`（实测 2026-11-27）。
    """
    import sys as _sys
    from pathlib import Path as _P
    _root = next((q for q in _P(__file__).resolve().parents
                  if (q / "scripts" / "opverify.py").is_file()),
                 _P(__file__).resolve().parents[4])
    _sys.path.insert(0, str(_root / "scripts"))
    from opverify import parse_yaml

    p = _P(__file__).resolve().parent
    target = (p / str(arg)) if (arg and (p / str(arg)).is_file()) else (p / "shapes.yaml")
    data = parse_yaml(target.read_text(encoding="utf-8"))
    key = arg if arg in data else next(iter(data))
    node = dict(data[key])
    node.setdefault("_name", key)
    return node


def runner_kwargs(params):
    return dict(params)


def get_runner(name, **kwargs):
    if name not in VARIANTS:
        raise ValueError("未知变体 %%r（可选 %%s）" %% (name, " / ".join(VARIANTS)))
    if name == "ref-heavy":
        return lambda *a, **kw: reference(*a, params=kw)
    raise RuntimeError("变体 %%r 的 adapter 尚未实现" %% name)
''' % (name, in_names_lit, in_names, reference_expr)


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
- 本卡已登记进 `data/ops/registry.json`、`Plugins/ops/SKILL.md` 索引表、
  `Plugins/ops/op-atlas/INDEX.md`；状态变化要三处一起改，
  `python3 -I scripts/card_sync.py` 会核对。
- `baseline_ref` 现在是 `no_baseline_reason`：只有 ref-heavy 时算不出有意义的倍数。
  补上真基准（competitor / best_known）之前，本卡不许写任何「快多少」。
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


def build_candidates_yaml(name, pattern, bytes_mb):
    """建卡时生成一个**可跑的**候选队列。

    只有一条 `baseline-ref`：它的作用是先把参考实现的实测值记下来
    （`purpose: calibration`），不声称任何收益。之后的真实候选由
    op-optimize 按方向补进来。

    为什么必须有：`loop.py run` 会检查 tag 是否登记在 candidates.yaml，
    空队列直接拒绝，建完卡也跑不起来。
    """
    bound = {"elementwise": "memory", "elementwise-2in": "memory",
             "gather": "memory", "reduce": "memory",
             "gemm": "compute"}.get(pattern, "mixed")
    action = {"memory": "reduce-bytes", "compute": "reduce-flops",
              "latency": "raise-concurrency", "mixed": "reduce-bytes"}[bound]
    return {
        "defaults": {
            "base": "ref-heavy",
            "rounds": 3,
            "bench_args": ["--base", "ref-heavy", "--measure-mode",
                           "event_isolated", "--rounds", "3"],
            # 精度探针的参数**必须显式声明**。不声明时 loop.py 会退回
            # sparse-mla 的 `--qscale 1 10 --skv 32768`（那是 logit 尺度，
            # 逐元素/归约类算子没有这个概念），日志里还会打一句
            # "未声明 defaults.acc_args，退回 sparse-mla 默认" —— 读的人会
            # 误以为这张卡跟 sparse-mla 有关系。
            "acc_args": ["--shape", "s0"],
        },
        "bottleneck": {
            "bound": bound,
            "derive_evidence": ("bootstrap 按访问模式 %s 算出的分母：%s"
                                % (pattern, ("%.3f MB" % bytes_mb) if bytes_mb else "未算出")),
        },
        "directions": [{
            "id": "baseline",
            "statement": "先量参考实现/现成实现的基线，作为后续改动的比较口径",
            "status": "open",
            "refutes": 0,
        }],
        "candidates": [{
            "id": "baseline-ref",
            "direction": "baseline",
            "status": "pending",
            "priority": 1,
            "action_class": action,
            # direction 不在 surfaces.yaml 时，loop.py 要求候选自带 class，
            # 且 class 必须落在 bottleneck.bound 允许的集合里（见 bound-to-class.yaml）。
            "class": {"memory": "residency", "compute": "pipeline",
                      "mixed": "residency"}.get(bound, "residency"),
            "purpose": "calibration",
            "variant": "ref-heavy",
            "expect": "仅取基线；不声称收益",
            # `expect` 是散文，`check_expect.py` 要求它要么是数值、
            # 要么显式 expect_confidence: low + expect_basis。
            # 少了这一行，新卡一登记就多一个红（test_check_expect）。
            "expect_confidence": "low",
            "expect_basis": "calibration-only：baseline 是 correctness_only 的参考实现，"
                            "loop.py 只接受它作校准（CALIBRATION_ONLY），不计收益；"
                            "未实测，给不出数值",
        }],
        "refuted": [],
        "note": ("由 op_bootstrap 生成。真实候选请按 op-optimize 的方向预算补进 "
                 "candidates；baseline-ref 只用于校准。"),
    }


_PREFLIGHT_BLOCK = '''# **默认跑前置检查**（scripts/bench_preflight.py）。
# 2026-11-25 的 5 小时根因是「设备绑定方式」问题 —— 纯 torch 探活查不出，
# 必须用自定义 kernel 试。默认检查是为了不依赖「谁记得跑」。
# 想跳过必须显式 --skip-preflight，且那时测出的数字不可信。
if not %(a)s.skip_preflight:
    try:
        from bench_preflight import preflight
    except ImportError:                                      # 没装检查器不阻塞
        preflight = None
    if preflight is not None and not preflight(
            device=getattr(%(a)s, "device", None)):
        print("✗ 前置检查未全过 —— 加 --skip-preflight 可强制继续，"
              "但测出的数字不可信", file=sys.stderr)
        return 2'''


def _reindent(block, prefix):
    return "\n".join(prefix + ln if ln.strip() else ln
                     for ln in block.splitlines())


def _preflight_arg(parser, ind):
    cont = " " * (len(ind) + len(parser) + len(".add_argument("))
    return (ind + parser + '.add_argument("--skip-preflight", action="store_true",\n'
            + cont + 'dest="skip_preflight",\n'
            + cont + 'help="跳过环境前置检查（**只在已手工验过环境时用**）")\n')


def _inject_preflight(text):
    """往通用 A/B 适配器模板里注入 bench_preflight（默认执行）。

    为什么是「注入」而不是改模板：`Plugins/ops/engine/templates/ab_generic.py`
    属于 engine 层（另一拨人在改），建卡脚本不该依赖它的某个版本；
    而「新建的 ab.py 必须默认跑前置检查」是 test_entry_contract 的硬要求
    （连 `if not a.skip_preflight:` 这个字面量都要能匹配到），模板里没有就得在这里补。

    模板结构变了就**大声失败**：静默跳过等于让下一个人的新卡再红一次。
    """
    if "bench_preflight" in text:
        return text
    # 只在 main() 里那一行 `a = p.parse_args(argv)` 上动手，且变量名从文本里取
    # （不写死 a/p）—— 模板换变量名时仍然对，换结构时上面那条 RuntimeError 兜住。
    m = re.search(r"(?m)^(?P<ind>[ \t]*)(?P<args>\w+)[ \t]*=[ \t]*"
                  r"(?P<parser>\w+)\.parse_args\([^)\n]*\)[ \t]*$", text)
    if not m:
        raise RuntimeError(
            "ab 适配器模板里找不到 `<args> = <parser>.parse_args(...)` 这一行 —— "
            "模板结构变了，请同步更新 scripts/op_bootstrap._inject_preflight")
    ind, args, parser = m.group("ind"), m.group("args"), m.group("parser")
    arg_line = _preflight_arg(parser, ind)
    block = _reindent(_PREFLIGHT_BLOCK % {"a": args}, ind)
    line = text[m.start():m.end()]
    rest = text[m.end():]
    if rest.startswith("\n"):
        rest = rest[1:]
    rest = rest.lstrip("\n")
    return (text[:m.start()] + arg_line + line + "\n\n" + block + "\n\n" + rest)


def build_ab_py(template=None):
    """生成卡内 bench/ab.py：通用 A/B 适配器 + 默认执行的前置检查。"""
    tpl = Path(template) if template else (
        REPO / "Plugins/ops/engine/templates/ab_generic.py")
    if not tpl.is_file():
        raise RuntimeError(
            "找不到 A/B 适配器模板 %s —— loop.py 会以「本卡没有 A/B 适配器」拒绝该卡；"
            "模板缺失时无法生成 bench/ab.py" % tpl)
    return _inject_preflight(tpl.read_text(encoding="utf-8"))


def build_acc_probe(template=None):
    """生成卡内 `bench/acc_probe.py`：通用精度适配器。

    没有这个文件时 `loop.py` 会**静默跳过**精度检查（
    `if not a.no_acc and (card / acc_adapter).exists():`），
    于是新卡的 `acc_ok` 恒 False、永远给不出可 claim 的判定。
    实测 2026-11-28：op-gated-add / op-silu-mul 两张新卡都卡在这。
    """
    tpl = Path(template) if template else (
        REPO / "Plugins/ops/engine/templates/acc_probe_generic.py")
    if not tpl.is_file():
        raise RuntimeError(
            "找不到精度适配器模板 %s —— 没有它生成的卡可以 `run` 但永远 "
            "`acc_ok=False`，拿不出可 claim 的结论" % tpl)
    return tpl.read_text(encoding="utf-8")


# ---------------------------------------------------------------- 登记（5 处一致）

def _load_json(p):
    return json.loads(Path(p).read_text(encoding="utf-8"))


def _save_json(p, d):
    """原样写回：**不加尾随换行**（data/ops/registry.json 本来就没有），
    缩进 2 与仓库现状一致 —— 否则每建一张卡就把整个 registry 重排一遍，
    diff 里再也看不出来「这一张卡加了什么」。"""
    Path(p).write_text(json.dumps(d, ensure_ascii=False, indent=2), encoding="utf-8")


def sync_registry(card, variants=None, registry=None, skills_dir=None):
    """① data/ops/registry.json：卡必须有一条，状态与卡一致。

    `variants` 是**技能层** variants.yaml 里的变体名（ref-heavy / lightop …），
    registry 的 variants 字段记的就是这个，不是 atlas 卡的 repo 名。

    只补缺失、只在状态漂移时改状态。**不覆盖人写字段**
    （best_variant / floor_status / open_questions / evidence）。
    """
    registry = Path(registry or REGISTRY)
    skills_dir = Path(skills_dir or SKILLS)
    if not registry.is_file():
        return ["%s 不存在，跳过 registry 登记" % registry]
    name = card["id"]
    reg = _load_json(registry)
    ops = reg.setdefault("ops", [])
    skill_dir = skills_dir / ("op-%s" % name)
    want_path = ("Plugins/ops/skills/op-%s" % name)
    try:
        want_path = str(skill_dir.relative_to(REPO)).replace("\\", "/")
    except ValueError:
        pass
    for op in ops:
        if op.get("name") != name:
            continue
        if op.get("status") != card.get("status"):
            op["status"] = card.get("status")
            _save_json(registry, reg)
            return []
        return []
    ops.append({
        "name": name,
        "path": want_path,
        "status": card.get("status") or "draft",
        "shapes": list(card.get("shapes") or []),
        "variants": list(variants or []),
        "best_variant": None,
        "floor_status": None,
        "last_verified": card.get("updated"),
        "evidence": None,
        "open_questions": [],
    })
    reg["updated"] = _today()
    _save_json(registry, reg)
    return []


def sync_ops_entry(card, entry=None):
    """② Plugins/ops/SKILL.md 的索引表：`op-<name>` 必须有一行，状态列与卡一致。

    插入位置 = 最后一条 `| \\`op-...\\` |` 行之后（与 card_sync.py 一致），
    保证表格连续、不动别人的行。
    """
    entry = Path(entry or OPS_ENTRY)
    if not entry.is_file():
        return ["%s 不存在，跳过入口索引" % entry]
    name = "op-%s" % card["id"]
    status = card.get("status") or "draft"
    text = entry.read_text(encoding="utf-8")
    row_re = re.compile(r"(?m)^\|\s*`%s`\s*\|\s*([^|]*?)\s*\|" % re.escape(name))
    m = row_re.search(text)
    if m is not None:
        shown = m.group(1).strip().strip("`")
        if shown == status:
            return []
        text = text[:m.start(1)] + " `%s` " % status + text[m.end(1):]
        entry.write_text(text, encoding="utf-8")
        return []
    new_row = "| `%s` | `%s` | %s |" % (name, status,
                                        card.get("title") or name)
    rows = list(re.finditer(r"(?m)^\|\s*`op-[\w-]+`\s*\|.*$", text))
    if rows:
        at = rows[-1].end()
        text = text[:at] + "\n" + new_row + text[at:]
    else:
        text = text.rstrip() + "\n\n" + new_row + "\n"
    entry.write_text(text, encoding="utf-8")
    return []


def sync_index(atlas=None):
    """③ op-atlas/INDEX.md 是生成物（sync_index.py），不在这里重实现格式。"""
    atlas = Path(atlas or ATLAS)
    script = atlas / "sync_index.py"
    if not script.is_file():
        return ["%s 不存在，跳过 INDEX.md 同步" % script]
    p = subprocess.run([sys.executable, "-I", str(script), "--atlas", str(atlas)],
                       capture_output=True, text=True, cwd=str(REPO))
    if p.returncode != 0:
        return ["sync_index.py 失败：%s" % (p.stderr or p.stdout)[-300:]]
    return []


def register(card, skill_variants=None, registry=None, entry=None, atlas=None):
    """把新卡登记进仓库的三个索引位置，返回问题列表（空 = 一致）。

    **这是「建出来就过仓库级校验」和「建出来只能进 loop.py status」的分界线。**
    漏了它，新卡会让 test_card_sync / test_doc_links / test_entry_contract /
    validate 的 E1/E4 一起变红（实测：+14 个红）。
    """
    names = [v.get("name") or v.get("repo")
             for v in (skill_variants or []) if isinstance(v, dict)]
    probs = []
    probs += sync_registry(card, variants=names, registry=registry)
    probs += sync_ops_entry(card, entry=entry)
    probs += sync_index(atlas=atlas)
    return probs


def generate(name, title, family, math, shape_text, inputs_text,
             outputs_text, pattern, write=True, do_register=True):
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

        # ---- 候选队列 + A/B 适配器（2026-11-27 补）----
        # 之前只生成到 variants.yaml 就结束了。后果：`loop.py run` 既没有
        # 队列（拒绝"不在 candidates.yaml 里"）也没有 bench/ab.py（拒绝
        # "本卡没有 A/B 适配器"）——"建卡 → 闭环"必然断在最后一环，
        # 实测 agent 建完卡就退回手搓测量。
        cf2 = sd / "candidates.yaml"
        cf2.write_text(yaml.safe_dump(
            build_candidates_yaml(name, pattern, bytes_mb),
            allow_unicode=True, sort_keys=False), encoding="utf-8")
        out["written"].append(str(cf2))

        bd = sd / "bench"
        bd.mkdir(parents=True, exist_ok=True)
        ab = bd / "ab.py"
        ab.write_text(build_ab_py(), encoding="utf-8")
        out["written"].append(str(ab))
        # 精度适配器。少了它 loop.py 会静默跳过精度门（文件存在性判断），
        # 新卡永远 acc_ok=False —— 能跑但拿不出结论。
        ap = bd / "acc_probe.py"
        ap.write_text(build_acc_probe().replace("__OP__", name), encoding="utf-8")
        out["written"].append(str(ap))

        # ---- 登记进仓库（2026-11-27 补）----
        # 之前到这里就结束了：卡能进 `loop.py status`，但**仓库不认它** ——
        # registry / ops/SKILL.md / op-atlas/INDEX.md 三处全缺，
        # 全仓测试因此多出 14 个红（实测 op-gated-add）。
        if do_register:
            skv = yaml.safe_load(vf.read_text(encoding="utf-8")).get("variants") or []
            probs = register(card, skill_variants=skv)
            out["registered"] = not probs
            out["register_problems"] = probs
        out["gaps_note"] = (
            "gaps.md 是全局生成物：重算会改写**所有**卡的「连续缺证据批数」"
            "（可能把别人的卡推到「连续 ≥2 批 → 降级 draft」），所以 bootstrap "
            "不碰它。需要时手动跑 python3 -I scripts/gen_gaps.py 并单独提交。")
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

    # ⑦ 反向：必须生成 bench/acc_probe.py。
    # 没有它 loop.py 的 `(card / acc_adapter).exists()` 为假，整块精度检查
    # **静默跳过**，新卡 acc_ok 恒 False —— 能跑但永远拿不出可 claim 的结论。
    # 实测 2026-11-28：op-gated-add / op-silu-mul 两张卡都栽在这。
    tpl = REPO / "Plugins/ops/engine/templates/acc_probe_generic.py"
    assert tpl.is_file(), "缺少精度适配器模板 %s" % tpl
    probe = build_acc_probe()
    for must in ("qscale", "ref_absmax", "max_ulp", "nonfinite"):
        assert must in probe, "模板缺字段 %r —— loop.py 的 ADAPTER_ACC 契约不满足" % must
    compile(probe.replace("__OP__", "t"), "<acc_probe>", "exec")
    print("[ok] 反向：生成的卡带 bench/acc_probe.py，且满足 ADAPTER_ACC 契约（可编译）")

    # ⑥ 反向：生成的卡必须过仓库级 schema —— 不是「能跑」就算数
    card = build_card("t-schema", "t", "elementwise", "y = x + gate * s",
                      {"M": 8, "H": 8},
                      parse_io("x:bf16[M,H], gate:bf16[M,H]"),
                      parse_io("y:bf16[M,H]"), "elementwise-2in")
    sem = card["semantics"]
    assert set(sem) <= {"formula", "inputs", "outputs", "traffic_passes",
                        "mutates_inputs"}, sem
    tp = sem["traffic_passes"]
    assert tp["reads"] > 0 and tp["writes"] == len(sem["outputs"]), tp
    assert tp["passes"] == tp["reads"] + tp["writes"], tp
    assert card["variants"] and all(v.get("role") for v in card["variants"])
    assert card["baseline_ref"].get("no_baseline_reason"), card["baseline_ref"]
    assert card["last_verdict"]["ratio"] is None, card["last_verdict"]
    print("[ok] 反向：生成的卡 semantics/variants/baseline_ref/ratio 全部合 schema")

    # ⑦ 反向：ab.py 必须默认跑 bench_preflight（否则新卡又红一条）
    ab = build_ab_py()
    for need in ("bench_preflight", "skip_preflight", "if not a.skip_preflight:"):
        assert need in ab, "ab.py 缺 %r" % need
    compile(ab, "<ab.py>", "exec")     # 注入后必须还是合法 Python
    print("[ok] 反向：bench/ab.py 默认执行 bench_preflight 且可编译")

    # ⑧ 反向：散文 expect 必须带 expect_confidence，否则 check_expect 红
    cand = build_candidates_yaml("t-c", "elementwise", 1.0)["candidates"][0]
    assert cand.get("expect_confidence") == "low" and cand.get("expect_basis"), cand
    print("[ok] 反向：散文 expect 自带 expect_confidence: low + basis")

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
    ap.add_argument("--no-register", action="store_true",
                    help="只建卡，不登记进 registry / ops/SKILL.md / op-atlas/INDEX.md"
                         "（会导致全仓一致性测试变红；只在调试时用）")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args(argv)
    if a.selftest:
        return selftest()
    if not (a.name and a.family and a.shape and a.inputs and a.outputs):
        ap.error("需要 --name --family --shape --inputs --outputs（或 --selftest）")
    try:
        r = generate(a.name, a.title or a.name, a.family, a.math, a.shape,
                     a.inputs, a.outputs, a.pattern, write=not a.dry_run,
                     do_register=not a.no_register)
    except (ValueError, RuntimeError) as e:
        print("拒绝：%s" % e, file=sys.stderr)
        return 1
    print(json.dumps({"bytes_mb": r["bytes_mb"], "pattern": r["pattern"],
                      "detail": r["detail"], "written": r["written"],
                      "floor_note": r.get("floor_note"),
                      "registered": r.get("registered"),
                      "register_problems": r.get("register_problems"),
                      "gaps_note": r.get("gaps_note")},
                     ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
