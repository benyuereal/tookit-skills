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
        raise AdapterError("导入 adapter %s 失败: %s: %s" % (name, type(e).__name__, e))
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
            except Exception as e:  # noqa: BLE001
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
    fn = {"correctness": cmd_correctness, "invariants": cmd_invariants, "padding": cmd_padding}[a.cmd]
    try:
        return fn(a)
    except AdapterError as e:
        print("ERROR: %s" % e, file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
