#!/usr/bin/env python3
"""opcheck — 算子卡的共用检查模板（一个文件三套流程，别再给每张卡抄一份测试）。

    python3 opcheck.py correctness --contract <card>/contract.yaml --shape decode-s128 --variants tilelang,flash_mla
    python3 opcheck.py invariants  --contract <card>/contract.yaml --variant tilelang --tilelang-root <tl>
    python3 opcheck.py padding     --contract <card>/contract.yaml --shape decode-s128

每个子命令对应 `contract.yaml:gates` 里一条 `template: <子命令>`。**判据全部来自卡片的 YAML**
（`tolerance` / `invariants[].mode` / `variants.yaml` 的 `padding_support`+`mutates_inputs`），
流程、报告字段、退出码在本文件里只有一份 —— 改口径只改这里。

── adapter 契约（卡目录下的模块，默认 `_common.py`，用 `--adapter` 改名）──
必需（缺一个即退出码 2）：
  build_inputs(params, seed=, device=) -> (inp..., meta)   造输入；inp... 原样传给 run/run_isolated
  reference(*inp, params, chunk=) -> out                   高精度 oracle（fp32 或更高，不得用另一个 GPU 实现）
  runner_kwargs(params) -> dict                            传给 get_runner 的形状参数
  get_runner(name, tilelang_root=None, **kwargs) -> fn    变体注册表；fn(*inp) -> out
  run_isolated(fn, *inp) -> (out, [被原地改写的输入名])      可信调用：输入副本 + 输出毒化，必须可只给 inp 调用
可选：
  load_shape(arg) -> {params:…}                            缺省时本文件直接读 shapes.yaml
  compare(a, b) -> {max_abs, mean_abs, cosine}              缺省时本文件用 torch 现算
  split_runner(tilelang_root, params, num_split) -> fn      split_invariance 探针
  permute_valid(indices, seed=) -> indices                  index_permutation 探针（要懂本项目哨兵语义）
  head_layout(indices) -> indices                           把「无效槽位」从队尾挪到队首，供 padding 门

退出码：0 通过；1 失败；2 adapter 契约不满足。
"""

import argparse
import importlib
import re
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from opverify import parse_yaml  # noqa: E402

REQUIRED = ("build_inputs", "reference", "runner_kwargs", "get_runner", "run_isolated")
OPTIONAL = ("load_shape", "compare", "split_runner", "permute_valid", "head_layout")


class AdapterError(RuntimeError):
    pass


# 「参考实现不支持这个形状 / 造不出输入」时 adapter 通常会抛这些。
# 只吞这一组：真正的代码 bug（TypeError/AttributeError/NameError…）应当炸出来，
# 而不是被记成 SKIP 掩盖掉 —— 那正是门禁静默通过的老路。
ADAPTER_INPUT_ERRORS = (RuntimeError, ValueError, KeyError, OSError, ImportError)


# ------------------------------------------------------------------ 卡片与 adapter

def load_card(contract_path, adapter_name=None):
    path = Path(contract_path).resolve()
    if not path.exists():
        raise AdapterError("契约文件不存在: %s" % path)
    card = path.parent
    contract = parse_yaml(path.read_text(encoding="utf-8"))
    if not isinstance(contract, dict) or not contract.get("op"):
        raise AdapterError("契约缺 op 字段: %s" % path)
    name = adapter_name or "_common"
    sys.path.insert(0, str(card))
    try:
        mod = importlib.import_module(name)
    except Exception as e:  # noqa: BLE001
        # 必须是宽捕获：adapter 是第三方代码，模块顶层任意语句都可能失败
        # （ImportError / RuntimeError / 依赖初始化错误…）。这里不吞，统一
        # 转成 AdapterError（退出码 2）并保留异常链，调用方看不出细节就是 bug。
        raise AdapterError("导入 adapter %s 失败: %s: %s" % (name, type(e).__name__, e)) from e
    missing = [s for s in REQUIRED if not callable(getattr(mod, s, None))]
    if missing:
        raise AdapterError("adapter %s 缺必需符号: %s（契约见本文件 docstring）" % (name, ", ".join(missing)))
    return card, contract, mod


def resolve_shape(card, mod, arg):
    fn = getattr(mod, "load_shape", None)
    if callable(fn):
        return fn(arg)
    data = parse_yaml(Path(arg).read_text(encoding="utf-8")) if str(arg).endswith((".yaml", ".yml")) \
        else parse_yaml((card / "shapes.yaml").read_text(encoding="utf-8"))
    if str(arg).endswith((".yaml", ".yml")):
        return data
    if arg not in data:
        raise AdapterError("shapes.yaml 里没有档位 %r" % arg)
    out = dict(data[arg])
    out.setdefault("name", arg)
    return out


def tolerance(contract):
    ref = (contract.get("tolerance") or {}).get("vs_fp32_ref") or {}
    return float(ref.get("max_abs", 2.0e-3)), float(ref.get("cosine", 0.9999))


def invariant_modes(contract):
    out = {}
    for it in contract.get("invariants") or []:
        if isinstance(it, dict) and it.get("name"):
            out[it["name"]] = it.get("mode", "bit_exact")
    return out


def declared_variants(card):
    """读 variants.yaml（op-contract 规定的字段名，跨算子同一套 schema）。"""
    p = card / "variants.yaml"
    if not p.exists():
        return {}
    data = parse_yaml(p.read_text(encoding="utf-8"))
    out = {}
    for item in data.get("variants") or []:
        if isinstance(item, dict) and item.get("id"):
            out[item["id"]] = {
                "padding": str(item.get("padding_support", "unknown")),
                "mutates": str(item.get("mutates_inputs", "unknown")),
                "status": str(item.get("status", "unknown")),
            }
    return out


def _cmp(mod, a, b):
    fn = getattr(mod, "compare", None)
    if callable(fn):
        return fn(a, b)
    import torch
    d = (a.float() - b.float()).abs()
    cos = torch.nn.functional.cosine_similarity(a.float().flatten(), b.float().flatten(), dim=0)
    return {"max_abs": float(d.max()), "mean_abs": float(d.mean()), "cosine": float(cos)}


def _judge(mod, mode, a, b, max_abs, cosine):
    """返回 (通过?, 说明)。无论哪种 mode 都给误差量级，便于分清数值问题与结构问题。"""
    import torch
    m = _cmp(mod, a, b)
    bit = bool(torch.equal(a, b))
    desc = "max_abs=%.3e mean_abs=%.3e cosine=%.7f%s" % (
        m["max_abs"], m["mean_abs"], m["cosine"], "（逐位一致）" if bit else "")
    if mode == "bit_exact":
        return bit, desc
    return (m["max_abs"] <= max_abs and m["cosine"] >= cosine), \
        desc + " [容差 %.1e / %.4f]" % (max_abs, cosine)


def _write_json(path, payload):
    if not path:
        return
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print("# wrote %s" % path)


def _split_arg(s):
    return [x.strip() for x in (s or "").split(",") if x.strip()]


# ------------------------------------------------------------------ correctness

def cmd_correctness(a):
    card, contract, mod = load_card(a.contract, a.adapter)
    max_abs, cosine = tolerance(contract)
    shape = resolve_shape(card, mod, a.shape)
    params = dict(shape.get("params") or {})
    if a.pad_ratio < 1.0:
        params["effective_ratio"] = a.pad_ratio
    params["pad_position"] = a.pad_position

    inputs = mod.build_inputs(params, seed=a.seed, device=a.device)
    meta = inputs[-1]
    print("# shape=%s params=%s" % (shape.get("shape"), meta))
    if a.pad_ratio >= 1.0:
        print("# 注意：--pad-ratio>=1 时输入不含无效槽位，覆盖不到「部分填充块」这条路径")
    if a.pad_position == "head":
        print("# 注意：--pad-position head 是越界探测（厂商口径只垫队尾），声明只支持队尾的变体预期失败")

    ref = mod.reference(*inputs[:-1], params=params, chunk=a.ref_chunk)
    print("# 参考实现就绪（chunk=%d）" % a.ref_chunk)

    out = {"op": contract.get("op"), "shape": shape.get("shape"), "params": meta,
           "tolerance": {"max_abs": max_abs, "cosine": cosine}, "variants": {}}
    all_pass = True
    for v in _split_arg(a.variants):
        try:
            run = mod.get_runner(v, tilelang_root=a.tilelang_root, **mod.runner_kwargs(params))
            got, mutated = mod.run_isolated(run, *inputs[:-1])
        except Exception as e:  # noqa: BLE001
            # **故意宽捕获**：runner 是第三方 GPU kernel（torch / triton / ctypes），
            # 失败类型不可枚举（RuntimeError、AssertionError、HIP 运行时、SystemExit…）。
            # 但**不吞**：记进 out["variants"] 并置 all_pass=False，最后返回非 0，
            # 报告里能看到是哪个变体崩的。若这里改窄，未预料的崩溃会逃出报告，
            # 反而让一次「一个变体都没跑成」看起来像 checker 自己坏了。
            print("FAIL %-12s 无法运行: %s: %s" % (v, type(e).__name__, e))
            out["variants"][v] = {"status": "error", "reason": str(e)}
            all_pass = False
            continue
        m = _cmp(mod, got, ref)
        ok = m["max_abs"] <= max_abs and m["cosine"] >= cosine
        all_pass &= ok
        out["variants"][v] = dict(m, status="pass" if ok else "fail", mutates_inputs=mutated)
        print("%s %-12s max_abs=%.3e (限 %.1e)  mean_abs=%.3e  cosine=%.7f (限 %.4f)%s"
              % ("PASS" if ok else "FAIL", v, m["max_abs"], max_abs, m["mean_abs"], m["cosine"], cosine,
                 ("  [原地改输入: %s]" % ",".join(mutated)) if mutated else ""))

    out["verdict"] = "pass" if all_pass else "fail"
    _write_json(a.json, out)
    return 0 if all_pass else 1


# ------------------------------------------------------------------ robustness

# 未见过的 shape：与卡片 shapes.yaml 里声明的**全部形状都不同**。
# 为什么必须测：只在提交的形状上通过，等于「考试题和练习题一样」。
# CompoundingKernel 的 L2/079 就是通过了所有 shape，但 FP16 stress 输入溢出，
# 他们宁可从 1.62× 退到 0.48× 也不交——**检查的意义在于能拦住东西**。
UNSEEN_SHAPES = [
    {"S": 1, "H": 4, "D": 64, "topk": 8},        # 极小：边界/取整
    {"S": 7, "H": 3, "D": 96, "topk": 33},       # 非 2 的幂、奇数
    {"S": 129, "H": 5, "D": 160, "topk": 65},    # 跨 tile 边界（128+1）
    {"S": 2, "H": 64, "D": 128, "topk": 1},      # topk=1 退化
]

# stress 数值：不是「更大」，是**更坏**。溢出/下溢/非有限值。
STRESS_PATTERNS = ["huge", "tiny", "mixed_sign", "denormal", "zeros"]


def _stress_fill(t, torch, pattern, seed):
    """按 pattern 往张量里灌 stress 数值。返回 (张量, 描述)。"""
    g = torch.Generator(device=t.device).manual_seed(seed)
    finfo_max = torch.finfo(t.dtype).max if t.is_floating_point() else 1e4
    if pattern == "huge":
        v = finfo_max / 8.0
        t.fill_(v)
        return t, "填 %.3e（接近 dtype 上限的 1/8，压 fp16/bf16 的累加）" % v
    if pattern == "tiny":
        v = float(torch.finfo(t.dtype).tiny) if t.is_floating_point() else 1e-30
        t.fill_(v)
        return t, "填 %.3e（最小正规数，压下溢）" % v
    if pattern == "denormal":
        v = float(torch.finfo(t.dtype).tiny) / 2.0 if t.is_floating_point() else 1e-38
        t.fill_(v)
        return t, "填 %.3e（次正规数）" % v
    if pattern == "mixed_sign":
        # 一半最大正、一半最大负：压对消误差
        t.zero_()
        half = t.numel() // 2
        flat = t.view(-1)
        flat[:half] = finfo_max / 8.0
        flat[half:] = -finfo_max / 8.0
        return t, "一半 +max/8、一半 -max/8（压对消误差）"
    if pattern == "zeros":
        t.zero_()
        return t, "全 0（压除零/归一化）"
    return t, "未知 pattern"


FLAW_VARIANT = "__selftest_flaw__"


def _first_runnable(mod, variants, a, params):
    """取第一个能正常构造的变体，作为注入缺陷的底座。"""
    for v in variants:
        if v == FLAW_VARIANT:
            continue
        try:
            return mod.get_runner(v, tilelang_root=a.tilelang_root,
                                 **mod.runner_kwargs(params))
        except (SystemExit,) + ADAPTER_INPUT_ERRORS:
            # 探测「哪个变体能构造」：未接入 / 缺后端的变体由 adapter 用
            # ValueError / RuntimeError / SystemExit 报出。只吞这一组，
            # 其他异常（TypeError 等）继续冒泡 —— 那多半是 checker 的 bug。
            continue
    raise RuntimeError("没有可用的底座变体，无法注入缺陷")


def _flawed_runner(real_run, kind):
    """构造一个**已知有缺陷**的 runner，用于证明鲁棒性门生效。

    缺陷刻意做成「平时完全正确，只在极端处错」——这正是真实世界里漏检的那类：
      kind=stress：输出首元素注入 inf（模拟溢出）
      kind=unseen：输出整体加 1（模拟「边界形状下索引算错」）
    如果这套检查连它都抓不到，说明检查没生效（业主的验收要求）。
    """
    def _poison(out, kind):
        import torch
        if isinstance(out, torch.Tensor):
            t = out.clone()
            if kind == "stress":
                if t.is_floating_point() and t.numel():
                    t.view(-1)[0] = float("inf")
            else:
                if t.is_floating_point():
                    t = t + 1.0
            return t
        if isinstance(out, (list, tuple)):
            return type(out)(_poison(x, kind) for x in out)
        return out

    def _run(*inp):
        return _poison(real_run(*inp), kind)
    return _run


def _all_finite(torch, x):
    """参考/被测输出可能是 tensor / tuple / dict。全部为有限才返回 True；
    拿不到任何 tensor 返回 None（判不了，调用方应 SKIP 而不是判错）。
    """
    items = []
    if isinstance(x, torch.Tensor):
        items = [x]
    elif isinstance(x, (list, tuple)):
        items = [t for t in x if isinstance(t, torch.Tensor)]
    elif isinstance(x, dict):
        items = [t for t in x.values() if isinstance(t, torch.Tensor)]
    if not items:
        return None
    for t in items:
        if t.is_floating_point() and not bool(torch.isfinite(t).all().item()):
            return False
    return True


def _known_params(card, mod, shape_arg):
    """取一个已知形状的 params 作为「键名原型」。"""
    try:
        shape = resolve_shape(card, mod, shape_arg)
        return dict(shape.get("params") or {})
    except (SystemExit, OSError, ValueError):
        # 取不到「键名原型」只是一条启发式失败，调用方会跳过 A 类检查。
        # resolve_shape 的已知失败面：文件不存在(OSError)、YAML 解析(ValueError)、
        # adapter 的 load_shape 用 SystemExit 报「找不到形状文件」。
        return {}


def _derive_unseen(proto, sp):
    """把通用极端值映射到本卡实际的参数键名上。

    规则：按键名做语义匹配（大小写不敏感），匹配不到就保持原型值不动。
    匹配不到时返回 None —— 说明这张卡的形状参数与 attention 风格差太远，
    硬造一个「没见过的形状」没有意义，跳过比瞎造好。
    """
    if not proto:
        return None
    want = {"s": sp["S"], "seq": sp["S"], "seqlen": sp["S"], "l": sp["S"],
            "m": sp["S"], "n": sp["S"],
            "h": sp["H"], "heads": sp["H"], "hidden": sp["H"],
            "d": sp["D"], "dim": sp["D"], "head_dim": sp["D"],
            "topk": sp["topk"], "k": sp["topk"]}
    out = dict(proto)
    matched = False
    for k in list(out):
        low = str(k).lower()
        if low in want and isinstance(out[k], (int, float)):
            out[k] = want[low]
            matched = True
    return out if matched else None


def cmd_robustness(a):
    """未见过的 shape + stress 数值：检查能不能拦住东西。

    两类检查，任一构造出「参考实现正常但被测变体错/崩」就算抓到。
    退出码：0 = 两类都过；1 = 至少一例不通过（**这是正常且想要的结果**）；2 = adapter 契约问题。

    注意语义：本命令报 FAIL **不是坏事**——它证明检查有效。
    真正的坏结果是一例都没抓到（见 --require-catch）。
    """
    card, contract, mod = load_card(a.contract, a.adapter)
    max_abs, cosine = tolerance(contract)
    try:
        import torch
    except ImportError:
        print("ERROR: robustness 检查需要 torch", file=sys.stderr)
        return 2

    known = set()
    try:
        known = set((parse_yaml((card / "shapes.yaml").read_text(encoding="utf-8")) or {}).keys())
    except (OSError, ValueError):
        # known 只用于打印「卡片声明 N 个形状」；读不到/解析不了不该中断检查。
        pass

    out = {"op": contract.get("op"), "kind": "robustness",
           "tolerance": {"max_abs": max_abs, "cosine": cosine},
           "unseen_shape": {}, "stress": {}, "caught": 0}
    variants = _split_arg(a.variants)
    # --selftest-flaw：故意插一个**已知有缺陷**的变体，用来证明这套检查有效。
    # 每张卡都可以跑它：如果连这个构造的缺陷都抓不到，说明检查没生效。
    if a.selftest_flaw:
        variants = variants + ["__selftest_flaw__"]

    # ---- 一类：未见过的 shape
    # 参数键名必须**取自本卡某个已知形状**（不同卡的 params 键不一样：
    # attention 卡是 S/H/D，moE 卡是 M/H/E/K）。硬塞一套键名只会全 SKIP。
    print("== A 类：未见过的 shape（卡片声明 %d 个形状，以下均不在其中）==" % len(known))
    proto = _known_params(card, mod, a.shape)
    for sp in UNSEEN_SHAPES:
        params = _derive_unseen(proto, sp)
        if params is None:
            continue
        key = ",".join("%s=%s" % (k, params[k]) for k in sorted(params)
                       if isinstance(params[k], (int, float)))
        params["effective_ratio"] = 1.0
        try:
            inputs = mod.build_inputs(params, seed=a.seed, device=a.device)
            ref = mod.reference(*inputs[:-1], params=params, chunk=4)
        except (SystemExit,) + ADAPTER_INPUT_ERRORS as e:
            # 参考实现都造不出来 ⇒ 这个 shape 对本卡无意义，跳过（不算抓到）。
            # 只吞 adapter 报「不支持」的那组；TypeError/AttributeError 等
            # checker 侧写错要继续冒泡，否则检查会静默少测一类。
            print("  SKIP %-28s 参考实现不支持：%s" % (key, str(e)[:60]))
            out["unseen_shape"][key] = {"status": "skip", "reason": str(e)[:120]}
            continue
        for v in variants:
            try:
                if v == FLAW_VARIANT:
                    base = _first_runnable(mod, variants, a, params)
                    run = _flawed_runner(base, "unseen")
                else:
                    run = mod.get_runner(v, tilelang_root=a.tilelang_root,
                                         **mod.runner_kwargs(params))
                got, _ = mod.run_isolated(run, *inputs[:-1])
                m = _cmp(mod, got, ref)
                ok = m["max_abs"] <= max_abs and m["cosine"] >= cosine
                if not ok:
                    out["caught"] += 1
                out["unseen_shape"][key] = {"status": "pass" if ok else "fail",
                                            "variant": v, "max_abs": m["max_abs"]}
                print("  %s   %-28s %-12s max_abs=%.3e（限 %.1e）"
                      % ("PASS" if ok else "FAIL", key, v, m["max_abs"], max_abs))
            except Exception as e:  # noqa: BLE001
                # **故意宽捕获**：被测变体是第三方 GPU kernel，失败类型不可枚举
                # （RuntimeError、AssertionError、HIP 运行时、SystemExit…）。
                # 「跑不起来」不等于「鲁棒性有问题」。
                # 变体未接入 / 缺后端时**不算抓到**——否则任何缺失都会被记成
                # 「检查有效」，把真信号淹掉（本卡第一版就是这样抓到 12 例假的）。
                # 但仍逐条记进报告，不静默。
                out["unseen_shape"][key] = {"status": "error", "variant": v,
                                            "reason": "%s: %s" % (type(e).__name__, e)}
                print("  ERROR  %-28s %-12s 跑不起来（不计入抓到）：%s"
                      % (key, v, str(e)[:60]))

    # ---- 二类：stress 数值（在**已声明**的形状上灌极端值）
    print("== B 类：stress 数值（同一 shape，灌极端值）==")
    shape = resolve_shape(card, mod, a.shape)
    base_params = dict(shape.get("params") or {})
    base_params["effective_ratio"] = 1.0
    for pattern in STRESS_PATTERNS:
        try:
            inputs = mod.build_inputs(base_params, seed=a.seed, device=a.device)
        except (SystemExit,) + ADAPTER_INPUT_ERRORS as e:
            # 同 A 类：adapter 报「造不出」预期内；checker 侧 bug 应冒泡。
            print("  SKIP %-12s 造不出输入：%s" % (pattern, str(e)[:60]))
            continue
        filled, desc = [], []
        for t in inputs[:-1]:
            if isinstance(t, torch.Tensor) and t.is_floating_point():
                t2, d = _stress_fill(t.clone(), torch, pattern, a.seed)
                filled.append(t2)
                desc.append(d)
            else:
                filled.append(t)
        if not desc:
            print("  SKIP %-12s 该卡没有浮点输入" % pattern)
            continue
        try:
            ref = mod.reference(*filled, params=base_params, chunk=4)
            ref_finite = _all_finite(torch, ref)
        except (SystemExit,) + ADAPTER_INPUT_ERRORS as e:
            # 参考实现对极端值崩掉 ⇒ 该 stress 对本卡无意义。只吞 adapter 的
            # 「不支持」组；若 TypeError 之类冒泡，说明 checker 调用写错了。
            print("  SKIP %-12s 参考实现自己就崩：%s" % (pattern, str(e)[:50]))
            continue
        if ref_finite is None:
            print("  SKIP %-12s 参考输出不是 tensor，无法判有限性" % pattern)
            continue
        if ref_finite is False:
            # 参考实现自己也溢出 ⇒ 这个 stress 对本卡无意义。
            # **不能算「抓到」**——否则默认实现会被记成「检查有效」，
            # 而我们其实什么都没证明（CompoundingKernel 的检查是在
            # 「参考正常、被测异常」时才算抓到）。
            print("  SKIP %-12s 参考实现自己也溢出（该 stress 对本卡无意义）" % pattern)
            continue
        for v in variants:
            try:
                if v == FLAW_VARIANT:
                    base = _first_runnable(mod, variants, a, base_params)
                    run = _flawed_runner(base, "stress")
                else:
                    run = mod.get_runner(v, tilelang_root=a.tilelang_root,
                                         **mod.runner_kwargs(base_params))
                got, _ = mod.run_isolated(run, *filled)
                got_ok = _all_finite(torch, got)
                if ref_finite and not got_ok:
                    # **这是最有价值的一类**：参考正常，被测变体溢出
                    out["caught"] += 1
                    out["stress"]["%s/%s" % (pattern, v)] = {
                        "status": "fail", "reason": "输出含 inf/nan，参考实现却全有限"}
                    print("  FAIL   %-12s %-12s 输出含 inf/nan（参考全有限）—— 正是这类要抓" % (pattern, v))
                    continue
                m = _cmp(mod, got, ref)
                ok = m["max_abs"] <= max_abs and m["cosine"] >= cosine
                if not ok:
                    out["caught"] += 1
                out["stress"]["%s/%s" % (pattern, v)] = {
                    "status": "pass" if ok else "fail", "max_abs": m["max_abs"]}
                print("  %s   %-12s %-12s max_abs=%.3e" % ("PASS" if ok else "FAIL", pattern, v, m["max_abs"]))
            except Exception as e:  # noqa: BLE001
                # **故意宽捕获**：同 A 类内层。第三方 kernel 的失败类型不可枚举，
                # 记进报告且不计入抓到；不静默丢弃。
                out["stress"]["%s/%s" % (pattern, v)] = {
                    "status": "error", "reason": "%s: %s" % (type(e).__name__, e)}
                print("  ERROR  %-12s %-12s 跑不起来（不计入抓到）：%s"
                      % (pattern, v, str(e)[:60]))

    print()
    print("抓到 %d 例不通过。" % out["caught"])
    if a.require_catch and out["caught"] == 0:
        print("caveat --require-catch：一例都没抓到 —— 说明**检查没生效**，不是代码没问题。",
              file=sys.stderr)
        out["verdict"] = "check_ineffective"
        _write_json(a.json, out)
        return 1
    out["verdict"] = "ok"
    _write_json(a.json, out)
    return 0


# ------------------------------------------------------------------ invariants

def cmd_invariants(a):
    card, contract, mod = load_card(a.contract, a.adapter)
    modes = invariant_modes(contract)
    max_abs, cosine = tolerance(contract)
    shape = resolve_shape(card, mod, a.shape)
    params = dict(shape.get("params") or {})
    inputs = mod.build_inputs(params, seed=a.seed, device=a.device)
    meta = inputs[-1]
    run = mod.get_runner(a.variant, tilelang_root=a.tilelang_root, **mod.runner_kwargs(params))

    res, detail = {}, {}

    base, mutated = mod.run_isolated(run, *inputs[:-1])
    if mutated:
        print("# 注意：%s 原地改写了输入 %s；下面每次调用都喂独立副本" % (a.variant, ",".join(mutated)))
        # **必须重建输入**：run_isolated 内部虽然克隆了，但这里再往下还会多次
        # 直接引用 inputs（determinism / 不变量 / 精度对比），而 inputs 里
        # 已经被原地写过一次。原实现只在第一行注释里说"每次都喂副本"，
        # 代码却没重建——实测导致 lightop 被判 max_abs=3.1e-2（假失败）。
        inputs = mod.build_inputs(params, seed=a.seed, device=a.device)
        base, mutated = mod.run_isolated(run, *inputs[:-1])

    # 1) determinism
    same, bad_i = True, None
    for i in range(1, max(a.repeats, 1)):
        if not __import__("torch").equal(base, mod.run_isolated(run, *inputs[:-1])[0]):
            same, bad_i = False, i
            break
    res["determinism"] = bool(same)
    detail["determinism"] = ("重复 %d 次逐位一致" % a.repeats) if same else ("第 %d 次出现差异" % bad_i)
    print("%s determinism                   %s" % ("PASS" if same else "FAIL", detail["determinism"]))

    # 2) index_permutation_invariance（乱序有效索引，改变累加顺序）
    if "index_permutation_invariance" in modes:
        if callable(getattr(mod, "permute_valid", None)):
            permuted = list(inputs[:-1])
            permuted[-1] = mod.permute_valid(permuted[-1])
            ok, desc = _judge(mod, modes["index_permutation_invariance"],
                              mod.run_isolated(run, *permuted)[0], base, max_abs, cosine)
            res["index_permutation_invariance"] = bool(ok)
            detail["index_permutation_invariance"] = "乱序有效索引：" + desc
        else:
            res["index_permutation_invariance"] = None
            detail["index_permutation_invariance"] = "未测（adapter 未提供 permute_valid）"
        st = res["index_permutation_invariance"]
        print("%s index_permutation_invariance  %s"
              % ("PASS" if st else ("SKIP" if st is None else "FAIL"), detail["index_permutation_invariance"]))

    # 3) split_invariance（只有能强制 num_split 的实现才测）
    if "split_invariance" in modes:
        if callable(getattr(mod, "split_runner", None)) and a.tilelang_root:
            try:
                o1 = mod.run_isolated(mod.split_runner(a.tilelang_root, params, 1), *inputs[:-1])[0]
                o2 = mod.run_isolated(mod.split_runner(a.tilelang_root, params, 2), *inputs[:-1])[0]
                ok, desc = _judge(mod, modes["split_invariance"], o1, o2, max_abs, cosine)
                res["split_invariance"] = bool(ok)
                detail["split_invariance"] = "num_split 1 vs 2：" + desc
            except (SystemExit, RuntimeError, ValueError, OSError, ImportError) as e:
                # float kernel 在 num_split 边界上的已知失败面：缺后端/未编译
                # (ImportError/OSError)、内核构造/运行报错(RuntimeError)、
                # 参数不合法(ValueError)、adapter 报「无 runner」(SystemExit)。
                # 不在此列的异常（TypeError 等）继续冒泡 —— 那是 checker 的 bug。
                # 结果记为 None（verdict=incomplete），报告里有原因，不判通过。
                res["split_invariance"] = None
                detail["split_invariance"] = "未测（%s: %s）" % (type(e).__name__, e)
        else:
            res["split_invariance"] = None
            detail["split_invariance"] = "未测（需 adapter.split_runner + --tilelang-root）"
        st = res["split_invariance"]
        print("%s split_invariance              %s"
              % ("PASS" if st else ("SKIP" if st is None else "FAIL"), detail["split_invariance"]))

    # 合同里声明了但模板不知道的说不变量：显式报缺，不默许通过
    for name in modes:
        if name in res:
            continue
        res[name] = None
        detail[name] = "未测（模板无对应探针，需在 adapter 增加并在 opcheck 里接入）"
        print("SKIP %-28s %s" % (name, detail[name]))

    verdict = ("pass" if all(v is True for v in res.values())
               else ("incomplete" if any(v is None for v in res.values()) else "fail"))
    print("# verdict=%s" % verdict)
    _write_json(a.json, {"op": contract.get("op"), "shape": shape.get("shape"), "variant": a.variant,
                         "params": meta, "invariants": res, "detail": detail, "verdict": verdict})
    return 0 if verdict == "pass" else 1


# ------------------------------------------------------------------ padding

def cmd_padding(a):
    card, contract, mod = load_card(a.contract, a.adapter)
    declared = declared_variants(card)
    shape = resolve_shape(card, mod, a.shape)
    params = dict(shape.get("params") or {})
    params.update(effective_ratio=a.pad_ratio, pad_position="tail")
    inputs = mod.build_inputs(params, seed=a.seed, device=a.device)
    meta = inputs[-1]
    if not callable(getattr(mod, "head_layout", None)):
        raise AdapterError("adapter 未提供 head_layout(indices)，无法构造前置布局")
    head = list(inputs[:-1])
    head[-1] = mod.head_layout(head[-1])

    baseline = a.baseline or contract.get("capability_baseline")
    if not baseline:
        names = [k for k, v in declared.items()
                 if v["status"] in ("active", "usable") and k not in ("ref-fp32", "prod-vllm")]
        if not names:
            raise AdapterError("无法确定 padding 门的基准变体，用 --baseline 指定")
        baseline = names[0]

    # 默认只跑「现役 + 可用」；reference / broken 跳过，否则 get_runner 的 SystemExit 会打断整条检查
    names = _split_arg(a.variants) or [k for k, v in declared.items()
                                       if v["status"] in ("active", "usable")
                                       and k not in ("ref-fp32", "prod-vllm")]
    print("# shape=%s eff=%.3f 布局=队尾/前置 两种 基准=%s"
          % (shape.get("shape"), meta.get("effective_ratio", float("nan")), baseline))

    base_run = mod.get_runner(baseline, tilelang_root=a.tilelang_root, **mod.runner_kwargs(params))
    base_tail, mut_base = mod.run_isolated(base_run, *inputs[:-1])
    base_head, _ = mod.run_isolated(base_run, *head)
    if mut_base:
        print("# 基准 %s 原地改写了输入: %s（已用副本隔离，不影响判定）" % (baseline, ",".join(mut_base)))

    result, all_ok = {}, True
    for v in names:
        sup = declared.get(v, {}).get("padding", "unknown")
        decl_mut = declared.get(v, {}).get("mutates", "unknown")
        try:
            run = mod.get_runner(v, tilelang_root=a.tilelang_root, **mod.runner_kwargs(params))
            ot, mut_t = mod.run_isolated(run, *inputs[:-1])
            oh, mut_h = mod.run_isolated(run, *head)
        except SystemExit as e:  # 该变体没有 runner（示例代码 / 生产路径），跳过不算失败
            result[v] = {"declared": sup, "status": "no_runner", "reason": str(e)}
            print("SKIP  %-16s 无 runner: %s" % (v, e))
            continue
        except Exception as e:  # noqa: BLE001
            # **故意宽捕获**：同 correctness 内层 —— 第三方 kernel 失败类型不可枚举。
            # 记进报告并置 all_ok=False（最后返回非 0），不静默。
            result[v] = {"declared": sup, "status": "error", "reason": "%s: %s" % (type(e).__name__, e)}
            print("ERROR %-16s 无法运行: %s: %s" % (v, type(e).__name__, e))
            all_ok = False
            continue
        mutated = sorted(set(mut_t) | set(mut_h))

        mt = _cmp(mod, ot, base_tail)
        mh = _cmp(mod, oh, base_head)
        ok_tail = mt["max_abs"] <= a.tolerance
        ok_head = mh["max_abs"] <= a.tolerance
        conforms = {"both": ok_tail and ok_head,
                    "suffix_only": ok_tail and not ok_head}.get(sup)

        mut_ok = True
        if decl_mut not in ("unknown", "None"):
            want = [] if decl_mut in ("none", "no") else _split_arg(decl_mut)
            mut_ok = sorted(want) == sorted(mutated)
            if not mut_ok:
                print("%-16s 声明 mutates_inputs=%s 与实测 %s 不符" % (v, decl_mut, mutated or ["none"]))
        all_ok &= mut_ok

        result[v] = {"declared": sup, "declared_mutates": decl_mut,
                     "tail_vs_base": mt, "head_vs_base": mh, "conforms": conforms,
                     "mutates_inputs": mutated,
                     "mutates_conforms": mut_ok if decl_mut not in ("unknown", "None") else None}
        nt = "一致" if ok_tail else "**不一致**"
        nh = "一致" if ok_head else "**不一致**"
        print("%-16s 声明=%-11s 队尾=%-10s 前置=%-10s %s"
              % (v, sup, nt, nh, {True: "OK", False: "**不符声明**", None: "仅报告"}[conforms]))
        print("                    队尾 max_abs=%.3e cos=%.7f | 前置 max_abs=%.3e cos=%.7f%s"
              % (mt["max_abs"], mt["cosine"], mh["max_abs"], mh["cosine"],
                 ("  [原地改输入: %s]" % ",".join(mutated)) if mutated else ""))
        if conforms is False:
            all_ok = False
            print("                    ↑ " + ("声明「仅队尾」但队尾不一致：这个变体连支持的布局都不对"
                                              if sup == "suffix_only"
                                              else "声明「两种都行」但前置不一致：需改声明或修 kernel"))

    print("# verdict=%s" % ("pass" if all_ok else "fail"))
    _write_json(a.json, {"op": contract.get("op"), "shape": shape.get("shape"), "params": meta,
                         "baseline": baseline, "variants": result,
                         "verdict": "pass" if all_ok else "fail"})
    return 0 if all_ok else 1


# ------------------------------------------------------------------ CLI

def build_parser():
    p = argparse.ArgumentParser(description="算子卡共用检查模板（子命令 = contract.yaml 的 gate template）")
    sub = p.add_subparsers(dest="cmd", required=True)

    def common(sp):
        sp.add_argument("--contract", required=True, help="<card>/contract.yaml")
        sp.add_argument("--adapter", default=None, help="卡目录下的 adapter 模块名（默认 _common）")
        sp.add_argument("--shape", default="decode-s128", help="shapes.yaml 的档位键或 yaml 路径")
        sp.add_argument("--tilelang-root", default=None, help="需要强制 num_split / 加载 tilelang 变体时给")
        sp.add_argument("--device", default=None, help="如 cuda:0；默认由 adapter 决定")
        sp.add_argument("--seed", type=int, default=0)
        sp.add_argument("--json", default=None, help="把结果写成 JSON（字段对齐 opverify 的 result.json）")

    c = sub.add_parser("correctness", help="gate1：各实现 vs 高精度参考")
    common(c)
    c.add_argument("--variants", required=True, help="逗号分隔，如 tilelang,flash_mla")
    c.add_argument("--pad-ratio", type=float, default=0.5, help="有效索引占比；默认 0.5 才覆盖部分填充路径")
    c.add_argument("--pad-position", choices=["tail", "head"], default="tail",
                   help="tail=厂商口径；head 只用于探测「只支持队尾」的实现")
    c.add_argument("--ref-chunk", type=int, default=4, help="参考实现按 query 分块")

    rb = sub.add_parser("robustness",
                        help="未见过的 shape + stress 数值（能拦住才说明检查有效）")
    common(rb)
    rb.add_argument("--variants", required=True, help="逗号分隔")
    rb.add_argument("--require-catch", action="store_true",
                    help="一例都没抓到就判失败（证明检查有效，CI 用）")
    rb.add_argument("--selftest-flaw", action="store_true", dest="selftest_flaw",
                    help="注入一个已知有缺陷的变体，证明检查能抓到东西")

    i = sub.add_parser("invariants", help="gate2：参考对比抓不到的结构性错误")
    common(i)
    i.add_argument("--variant", required=True)
    i.add_argument("--repeats", type=int, default=5)

    d = sub.add_parser("padding", help="无效槽位布局能力：声明 vs 实测")
    common(d)
    d.add_argument("--variants", default=None, help="默认取 variants.yaml 里 status=active/usable 的全部")
    d.add_argument("--baseline", default=None, help="布局比较的基准变体（默认取 contract.capability_baseline）")
    d.add_argument("--pad-ratio", type=float, default=0.5)
    d.add_argument("--tolerance", type=float, default=2.0e-3, help="与基准比较的 max_abs")
    return p


def main(argv=None):
    a = build_parser().parse_args(argv)
    fn = {"correctness": cmd_correctness, "invariants": cmd_invariants,
          "padding": cmd_padding, "robustness": cmd_robustness}[a.cmd]
    try:
        return fn(a)
    except AdapterError as e:
        print("ERROR: %s" % e, file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
