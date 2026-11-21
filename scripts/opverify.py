#!/usr/bin/env python3
"""opverify — 按算子契约（contract.yaml）判定一次 run 是否通过。

三闸：正确性 / 不变量 / 性能；四档结论：ACCEPTED / REJECTED / INCONCLUSIVE / DIAGNOSTIC-ONLY。
纪律（硬性）：任何 required 项缺数据一律判 REJECTED，不许默认放行。

用法：
    python3 scripts/opverify.py --contract <card>/contract.yaml --check-contract
    python3 scripts/opverify.py --contract <card>/contract.yaml --run runs/op-optimize/<id> [--json] [--allow-missing perf]
    python3 -I scripts/opverify.py --selftest

退出码：0=ACCEPTED，2=INCONCLUSIVE，1=REJECTED/契约错误。

run 目录里的 result.json 期望字段：
{
  "op": "sparse-mla", "shape": "decode-s128", "variant": "tilelang",
  "env": {"gfx": "gfx938", "dtk": "...", "torch": "..."},
  "correctness": {"max_abs": 6.8e-4, "cosine": 0.999998, "n_compared": 1024},
  "invariants": {"determinism": true, "padding_invariance": true},
  "metrics":  {"median_ms": 1.00, "mad_ms": 0.01},
  "baseline": {"median_ms": 1.10, "mad_ms": 0.02}
}
"""

import argparse
import json
import os
import sys

DEFAULT_K = 3.0

# scripts/opcheck.py 提供的共享检查模板：gate 可用 `template: <name>` 代替 `script: <卡内路径>`
GATE_TEMPLATES = ("correctness", "invariants", "padding")


# ---------------------------------------------------------------- 迷你 YAML
# 只支持契约用到的最小子集：缩进 map、`- key: value` 列表项 + 续行、inline 列表、标量。

def _strip_comment(line):
    out = []
    quote = None
    for i, ch in enumerate(line):
        if quote:
            out.append(ch)
            if ch == quote:
                quote = None
        elif ch in "\"'":
            quote = ch
            out.append(ch)
        elif ch == "#" and (i == 0 or line[i - 1] in " \t"):
            break
        else:
            out.append(ch)
    return "".join(out)


def _indent(line):
    return len(line) - len(line.lstrip())


def _val(s):
    s = s.strip()
    if s == "":
        return None
    if len(s) >= 2 and s[0] == s[-1] and s[0] in "\"'":
        return s[1:-1]
    if s.startswith("[") and s.endswith("]"):
        inner = s[1:-1].strip()
        return [] if not inner else [_val(x) for x in inner.split(",")]
    low = s.lower()
    if low in ("true", "yes", "on"):
        return True
    if low in ("false", "no", "off"):
        return False
    if low in ("null", "none", "~"):
        return None
    try:
        return int(s)
    except ValueError:
        pass
    try:
        return float(s)
    except ValueError:
        pass
    return s


def _split_kv(s):
    """按第一个 ASCII 冒号拆 key/value；引号开头视为纯标量。"""
    if s[:1] in ("\"", "'"):
        return None, s
    k, sep, v = s.partition(":")
    if not sep:
        return None, s
    return k.strip(), v.strip()


def parse_yaml(text):
    lines = []
    for raw in text.splitlines():
        line = _strip_comment(raw.rstrip())
        if line.strip():
            lines.append(line)
    if not lines:
        return {}
    obj, _ = _parse_block(lines, 0, _indent(lines[0]))
    return obj


def _parse_block(lines, i, indent):
    container = None
    while i < len(lines):
        line = lines[i]
        cur = _indent(line)
        if cur < indent:
            break
        if cur > indent:
            raise ValueError("缩进异常: %r" % line)
        s = line.strip()
        if s.startswith("- "):
            if container is None:
                container = []
            elif not isinstance(container, list):
                raise ValueError("map/list 混用: %r" % line)
            item_s = s[2:].strip()
            k, v = _split_kv(item_s)
            if k is not None and v != "":
                item = {k: _val(v)}
                i += 1
                if i < len(lines) and _indent(lines[i]) == cur + 2 and not lines[i].strip().startswith("- "):
                    sub, i = _parse_block(lines, i, cur + 2)
                    if isinstance(sub, dict):
                        item.update(sub)
                container.append(item)
            elif k is not None:
                i += 1
                sub, i = _parse_block(lines, i, cur + 2)
                container.append({k: sub})
            else:
                container.append(_val(item_s))
                i += 1
        else:
            k, v = _split_kv(s)
            if k is None:
                raise ValueError("无法解析行: %r" % line)
            if container is None:
                container = {}
            elif not isinstance(container, dict):
                raise ValueError("list/map 混用: %r" % line)
            if v == "":
                i += 1
                if i < len(lines) and _indent(lines[i]) > cur:
                    sub, i = _parse_block(lines, i, _indent(lines[i]))
                    container[k] = sub
                else:
                    container[k] = {}
            else:
                container[k] = _val(v)
                i += 1
    return container, i


# ---------------------------------------------------------------- 契约校验

REQUIRED_TOP = ["op", "shapes", "dtypes", "tolerance", "invariants", "gates", "floor"]


def check_contract(c, base_dir=None):
    """返回问题列表（空 = 契约可用）。"""
    problems = []
    for k in REQUIRED_TOP:
        if k not in c or c[k] in (None, {}, []):
            problems.append("缺少必填字段: %s" % k)
    tol = c.get("tolerance") or {}
    ref = tol.get("vs_fp32_ref") or {}
    if "max_abs" not in ref:
        problems.append("tolerance.vs_fp32_ref.max_abs 缺失")
    if "cosine" not in ref:
        problems.append("tolerance.vs_fp32_ref.cosine 缺失（没有就显式写 0，但不许省略）")
    if not tol.get("variant_vs_variant"):
        problems.append("tolerance.variant_vs_variant 缺失（默认应为 bit_exact）")

    invs = c.get("invariants") or []
    if not isinstance(invs, list) or not invs:
        problems.append("invariants 为空（至少 determinism）")
    else:
        names = []
        for it in invs:
            if isinstance(it, dict):
                if not it.get("name"):
                    problems.append("invariants 条目缺少 name")
                    continue
                names.append(it["name"])
                if it.get("mode") not in ("bit_exact", "tolerance"):
                    problems.append("invariant %s 的 mode 必须是 bit_exact 或 tolerance" % it["name"])
            else:
                names.append(it)
        for must in ("determinism",):
            if must not in names:
                problems.append("缺少必备不变量: %s" % must)

    gates = c.get("gates") or []
    if not isinstance(gates, list) or not gates:
        problems.append("gates 为空")
    else:
        for g in gates:
            if not isinstance(g, dict) or "name" not in g:
                problems.append("gates 条目缺少 name: %r" % (g,))
                continue
            if g.get("required", True) in (True, "true", 1):
                # 两种给法：script=<卡内脚本路径>，或 template=<scripts/opcheck.py 的子命令>。
                # template 由共享实现提供流程（opcheck.py），卡只声明用哪一套。
                if g.get("template"):
                    if g["template"] not in GATE_TEMPLATES:
                        problems.append("gate %s 的 template 未知: %s（可选 %s）"
                                        % (g["name"], g["template"], "/".join(GATE_TEMPLATES)))
                    continue
                p = g.get("script")
                if not p:
                    problems.append("gate %s required=true 但没有 script 或 template" % g["name"])
                elif base_dir and not os.path.exists(os.path.join(base_dir, p)):
                    problems.append("gate %s 的脚本不存在: %s" % (g["name"], p))

    fl = c.get("floor") or {}
    for k in ("metric", "peak", "unit", "basis", "at_floor_threshold"):
        if k not in fl:
            problems.append("floor.%s 缺失" % k)
    if fl and fl.get("basis") in (None, ""):
        problems.append("floor.basis 为空：地板必须可复算")

    # manifest 段（T10/P0-1）：声明了就要自洽；整段不写 = 不做执行身份校验
    mf = c.get("manifest")
    if mf is not None:
        if not isinstance(mf, dict):
            problems.append("manifest 必须是对象（required_roles / env_keys）")
        else:
            roles = mf.get("required_roles")
            if not isinstance(roles, list) or not roles:
                problems.append(
                    "manifest.required_roles 为空：要么整段删掉（不做身份校验），要么写清要哪些 role")
            else:
                for r in roles:
                    if r not in ("source", "build", "loaded"):
                        problems.append("manifest.required_roles 含未知 role: %r（只认 source/build/loaded）" % (r,))
                if "build" not in roles and "loaded" not in roles:
                    problems.append("manifest.required_roles 至少要含 build 或 loaded 之一：只钉源码钉不住产物")
            keys = mf.get("env_keys")
            if not isinstance(keys, list) or not keys:
                problems.append("manifest.env_keys 为空：环境指纹缺失时结论不可回溯")
    return problems


# ---------------------------------------------------------------- 判定

def _num(d, *path):
    cur = d
    for p in path:
        if not isinstance(cur, dict) or p not in cur:
            return None
        cur = cur[p]
    return cur if isinstance(cur, (int, float)) else None


def evaluate(c, run, k=DEFAULT_K, allow_missing=None, env_check=True, identity_problems=None):
    allow_missing = set(allow_missing or [])
    missing = []
    reasons = []
    gates = {}

    # 指纹
    if env_check:
        env = run.get("env") or {}
        for f in ("gfx", "dtk"):
            if not env.get(f):
                missing.append("env.%s（环境指纹缺失，结论不可回溯）" % f)

    # 执行身份（T10/P0-1）：契约声明了 manifest.required_roles 就必须通过 opmanifest 校验。
    # identity_problems=None 表示「根本没提供 manifest」——不是「没有问题」。
    mf = c.get("manifest") or {}
    req_roles = (mf.get("required_roles") or []) if isinstance(mf, dict) else []
    if not req_roles:
        # 契约没声明就不出现这个闸：老契约的输出结构不变（兼容既有台账与测试）
        reasons.append("契约未声明 manifest.required_roles，未做执行身份校验")
    elif "manifest" in allow_missing:
        gates["identity"] = "skipped"
        reasons.append("执行身份校验被 --allow-missing manifest 显式跳过")
    elif identity_problems is None:
        gates["identity"] = "fail"
        reasons.append("执行身份未冻结：没有提供 manifest（源码/构建产物/加载产物三者必须钉住 sha256）")
    elif identity_problems:
        gates["identity"] = "fail"
        for p in identity_problems:
            reasons.append("身份不一致: " + p)
    else:
        gates["identity"] = "pass"
        reasons.append("执行身份一致：%s 的 sha256 与 mtime 链未断裂" % "/".join(req_roles))

    if run.get("op") and c.get("op") and run["op"] != c["op"]:
        missing.append("run.op=%s 与契约 op=%s 不一致" % (run["op"], c["op"]))

    # gate1 正确性
    tol = ((c.get("tolerance") or {}).get("vs_fp32_ref") or {})
    corr = run.get("correctness") or {}
    if "correctness" in allow_missing:
        gates["correctness"] = "skipped"
        reasons.append("correctness 被允许缺失（--allow-missing）")
    elif not corr:
        gates["correctness"] = "fail"
        missing.append("correctness（未提供）")
    else:
        ma = corr.get("max_abs")
        cs = corr.get("cosine")
        if ma is None:
            gates["correctness"] = "fail"
            missing.append("correctness.max_abs")
        elif cs is None:
            gates["correctness"] = "fail"
            missing.append("correctness.cosine")
        elif ma <= float(tol.get("max_abs", 0)) and cs >= float(tol.get("cosine", 1)):
            gates["correctness"] = "pass"
            reasons.append("max_abs=%.3g <= %.3g，cosine=%.7f >= %.7f" %
                           (ma, float(tol.get("max_abs")), cs, float(tol.get("cosine"))))
        else:
            gates["correctness"] = "fail"
            reasons.append("超出容差：max_abs=%.3g（限 %.3g），cosine=%.7f（限 %.7f）" %
                           (ma, float(tol.get("max_abs", 0)), cs, float(tol.get("cosine", 1))))

    # gate2 不变量
    inv_run = run.get("invariants") or {}
    if "invariants" in allow_missing:
        gates["invariants"] = "skipped"
        reasons.append("invariants 被允许缺失（--allow-missing）")
    elif not isinstance(c.get("invariants"), list):
        gates["invariants"] = "fail"
        missing.append("contract.invariants 格式错误")
    else:
        fails = []
        for it in c["invariants"]:
            name = it["name"] if isinstance(it, dict) else it
            if name not in inv_run:
                missing.append("invariant.%s（未测）" % name)
            elif not inv_run[name]:
                fails.append(name)
        if fails:
            gates["invariants"] = "fail"
            reasons.append("不变量失败: %s" % ", ".join(fails))
        elif any(x.startswith("invariant.") for x in missing):
            gates["invariants"] = "fail"
        else:
            gates["invariants"] = "pass"
            reasons.append("不变量全部通过（%d 项）" % len(c["invariants"]))

    # gate3 性能
    perf_gate = None
    for g in (c.get("gates") or []):
        if isinstance(g, dict) and g.get("name") == "perf":
            perf_gate = g
    has_perf_gate = perf_gate is not None or "perf" not in allow_missing
    med = _num(run, "metrics", "median_ms")
    base = _num(run, "baseline", "median_ms")
    if "perf" in allow_missing:
        gates["perf"] = "skipped"
        reasons.append("perf 被允许缺失（--allow-missing）")
    elif med is None or base is None:
        gates["perf"] = "fail"
        missing.append("metrics.median_ms / baseline.median_ms（无可比基线）")
    else:
        mad_m = _num(run, "metrics", "mad_ms") or 0.0
        mad_b = _num(run, "baseline", "mad_ms") or 0.0
        ratio = med / base if base else float("inf")
        delta = abs(med - base)
        noise = k * max(mad_m, mad_b)
        limit = 1.02
        if perf_gate and isinstance(perf_gate.get("require"), str) and "*" in perf_gate["require"]:
            try:
                limit = float(perf_gate["require"].split("*")[-1].strip())
            except ValueError:
                limit = 1.02
        if delta <= noise:
            gates["perf"] = "inconclusive"
            reasons.append("性能在噪声内：|Δ|=%.4g <= %.1f*MAD=%.4g（ratio=%.4f）——不许宣称收益" %
                           (delta, k, noise, ratio))
        elif ratio <= limit:
            gates["perf"] = "pass"
            reasons.append("性能显著改善：ratio=%.4f <= %.3f，|Δ|=%.4g > %.1f*MAD=%.4g" %
                           (ratio, limit, delta, k, noise))
        else:
            gates["perf"] = "fail"
            reasons.append("性能回退：ratio=%.4f > %.3f" % (ratio, limit))

    if missing:
        verdict = "REJECTED"
        reasons.insert(0, "缺项强制拒绝（%d 项）：%s" % (len(missing), ", ".join(missing)))
    elif any(v == "fail" for v in gates.values()):
        verdict = "REJECTED"
    elif gates.get("perf") == "inconclusive":
        verdict = "INCONCLUSIVE"
    elif gates.get("perf") == "skipped":
        verdict = "DIAGNOSTIC-ONLY"
    else:
        verdict = "ACCEPTED"

    return {"verdict": verdict, "gates": gates, "reasons": reasons, "missing": missing,
            "op": c.get("op"), "shape": run.get("shape"), "variant": run.get("variant")}


# ---------------------------------------------------------------- 入口

def _load_run(path):
    if os.path.isdir(path):
        path = os.path.join(path, "result.json")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f), path


def _resolve_identity(c, run_path, explicit):
    """按契约的 manifest 段解析并复核执行身份。

    返回 (identity_problems, manifest_path)：
      (None, None)     契约没要求，不做校验
      (None, path)     要求了但找不到 manifest —— evaluate 会判「未冻结」
      ([...], path)    复核结果（空列表 = 一致）
    """
    mf = c.get("manifest") or {}
    req = (mf.get("required_roles") or []) if isinstance(mf, dict) else []
    if not req:
        return None, None
    run_dir = run_path if os.path.isdir(run_path) else os.path.dirname(os.path.abspath(run_path))
    mpath = explicit or os.path.join(run_dir, "manifest.json")
    if not os.path.isfile(mpath):
        if explicit:
            return ["--manifest 指定的文件不存在：%s" % mpath], mpath
        return None, mpath
    try:
        here = os.path.dirname(os.path.abspath(__file__))
        if here not in sys.path:
            sys.path.insert(0, here)
        import opmanifest
        with open(mpath, "r", encoding="utf-8") as f:
            man = json.load(f)
        return opmanifest.validate_manifest(man, req, mf.get("env_keys") or []), mpath
    except Exception as e:   # 门禁自己坏了也必须失败关闭
        return ["加载/校验 manifest 失败（按不一致处理，不 fail-open）：%s: %s"
                % (type(e).__name__, e)], mpath


def _selftest():
    here = os.path.dirname(os.path.abspath(__file__))
    sample = os.path.join(here, "..", "Plugins", "ops", "skills", "op-sparse-mla", "contract.yaml")
    if not os.path.exists(sample):
        print("[skip] 找不到样例契约: %s" % sample)
        return 0
    with open(sample, "r", encoding="utf-8") as f:
        c = parse_yaml(f.read())
    probs = check_contract(c, os.path.dirname(sample))
    assert not probs, probs
    assert c["op"] == "sparse-mla", c.get("op")
    assert c["shapes"]["q"] == ["S", "H", 576], c["shapes"]["q"]
    assert abs(c["tolerance"]["vs_fp32_ref"]["max_abs"] - 2.0e-3) < 1e-12
    assert c["invariants"][0]["name"] == "determinism"
    # 按名字找，不按位置：卡片的门数会随卡变化（这张卡是 4 道门，perf 在第 4 位）。
    assert "perf" in [g["name"] for g in c["gates"]], c["gates"]
    print("[ok] 契约解析与校验通过（%d 项不变量 / %d 道门）" % (len(c["invariants"]), len(c["gates"])))

    run = {"op": "sparse-mla", "shape": "decode-s128", "variant": "tilelang",
           "env": {"gfx": "gfx938", "dtk": "dtk2604"},
           "correctness": {"max_abs": 6.8e-4, "cosine": 0.999998},
           "invariants": {it["name"]: True for it in c["invariants"]},
           "metrics": {"median_ms": 1.00, "mad_ms": 0.01},
           "baseline": {"median_ms": 1.10, "mad_ms": 0.02}}
    v = evaluate(c, run, identity_problems=[])
    assert v["verdict"] == "ACCEPTED", v
    assert v["gates"]["identity"] == "pass", v["gates"]
    print("[ok] ACCEPTED 路径: %s" % v["reasons"][0])

    # P0-1：没有 manifest 不等于「身份没问题」，必须拒绝
    assert (c.get("manifest") or {}).get("required_roles"), "样例契约应声明 manifest.required_roles"
    v0 = evaluate(c, run)
    assert v0["verdict"] == "REJECTED" and v0["gates"]["identity"] == "fail", v0
    print("[ok] 未冻结执行身份 -> REJECTED")

    bad = json.loads(json.dumps(run))
    bad["invariants"]["determinism"] = False
    v2 = evaluate(c, bad, identity_problems=[])
    assert v2["verdict"] == "REJECTED" and v2["gates"]["invariants"] == "fail", v2
    print("[ok] 不变量失败 -> REJECTED")

    noisy = json.loads(json.dumps(run))
    noisy["metrics"] = {"median_ms": 1.09, "mad_ms": 0.02}   # Δ=0.01 < 3*MAD(0.06) -> 噪声内
    v3 = evaluate(c, noisy, identity_problems=[])
    assert v3["verdict"] == "INCONCLUSIVE", v3
    print("[ok] 噪声内 -> INCONCLUSIVE")

    miss = json.loads(json.dumps(run))
    del miss["correctness"]
    v4 = evaluate(c, miss, identity_problems=[])
    assert v4["verdict"] == "REJECTED" and v4["missing"], v4
    print("[ok] 缺项 -> REJECTED（缺项不许放行）")

    stale = evaluate(c, run, identity_problems=["source 的 sha256 与记录不符"])
    assert stale["verdict"] == "REJECTED" and stale["gates"]["identity"] == "fail", stale
    print("[ok] 源码/产物不一致 -> REJECTED")
    return 0


def main(argv=None):
    p = argparse.ArgumentParser(description="opverify — 按契约判定算子改动")
    p.add_argument("--contract", help="contract.yaml 路径")
    p.add_argument("--run", help="run 目录（含 result.json）或 result.json 路径")
    p.add_argument("--json", action="store_true", help="输出机读 json")
    p.add_argument("--check-contract", action="store_true", help="只校验契约本身")
    p.add_argument("--allow-missing", default="", help="允许缺失的闸，逗号分隔（perf/invariants/correctness）")
    p.add_argument("--no-env-check", action="store_true", help="跳过环境指纹必填检查")
    p.add_argument("--manifest", help="执行身份 manifest.json（默认取 run 目录下的 manifest.json）")
    p.add_argument("--k", type=float, default=DEFAULT_K, help="显著性系数，默认 3")
    p.add_argument("--selftest", action="store_true")
    a = p.parse_args(argv)

    if a.selftest:
        return _selftest()
    if not a.contract:
        p.print_help()
        return 0

    with open(a.contract, "r", encoding="utf-8") as f:
        c = parse_yaml(f.read())
    base_dir = os.path.dirname(os.path.abspath(a.contract))
    probs = check_contract(c, base_dir)
    if a.check_contract or not probs:
        pass
    if probs:
        payload = {"verdict": "REJECTED", "reasons": ["契约不合格"] + probs, "missing": probs}
        if a.json:
            print(json.dumps(payload, ensure_ascii=False, indent=2))
        else:
            print("契约不合格（%d 项）：" % len(probs))
            for x in probs:
                print("  - " + x)
        return 1 if not a.check_contract else 0
    if a.check_contract:
        print("契约合格：%s（%d 不变量 / %d 道门）" % (c.get("op"), len(c.get("invariants") or []), len(c.get("gates") or [])))
        return 0

    if not a.run:
        p.error("给了 --contract 但没有 --run（或加 --check-contract）")
    run, path = _load_run(a.run)
    identity_problems, mpath = _resolve_identity(c, a.run, a.manifest)
    v = evaluate(c, run, k=a.k, allow_missing=[x for x in a.allow_missing.split(",") if x],
                 env_check=not a.no_env_check, identity_problems=identity_problems)
    v["run_path"] = path
    v["manifest_path"] = mpath
    if a.json:
        print(json.dumps(v, ensure_ascii=False, indent=2))
    else:
        print("verdict: %s  (op=%s shape=%s variant=%s)" % (v["verdict"], v["op"], v["shape"], v["variant"]))
        for g, s in v["gates"].items():
            print("  gate %-12s %s" % (g, s))
        for r in v["reasons"]:
            print("  - " + r)
    return {"ACCEPTED": 0, "INCONCLUSIVE": 2, "DIAGNOSTIC-ONLY": 2}.get(v["verdict"], 1)


if __name__ == "__main__":
    sys.exit(main())
