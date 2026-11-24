#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""anticheat —— 盲测前的最后一道闸。

为什么必须在盲测之前
--------------------
公开审计里有两个真实案例：
1. **KDA 分布感知**：检测检查器的数值压力分布，只在被检查时切到精确数学，
   在 nominal（唯一计时的）输入上用快速近似。
2. **MinGRU 覆盖攻击**：按 `num_envs` 分派——N ≤ 2048 走精确 fp32
   （正是 check 检查的所有形状），N > 2048 走快 fp16（正是全部 bench 形状）。

共同点：**agent 没变强，它学会了骗评测**。而且是「高级工作 + 一处作弊」，
人眼难看出来，且分数会先涨上去。

四道检查
--------
1. 形状分离    correctness_shapes ∩ bench_shapes = ∅
2. 扰动分布    ×3 / ×0.01 / 取负，三变换都要与 reference 相符
3. AST 扫描    对具体形状 / 对文本统计量的分支
4. 输入隔离双测 两个不同输入各跑一次，分别与 reference 对比

**第 4 条 v5.1 明确修正过语义**：不是「连调两次输出逐位相同」——
缓存作弊恰恰会让两次输出相同。必须换输入再测。
"""
import argparse
import ast
import hashlib
import json
import re
import sys
from pathlib import Path


# ---------------------------------------------------------------- 1 形状分离

def check_shape_separation(correctness_shapes, bench_shapes):
    """检查形状与计时形状不许有交集。

    **为什么**：作弊的最简形式是「只在被检查的形状上写对」。
    分离之后，agent 无法知道哪些形状会被计时——
    真实做法是 bench 形状从 harness 外部注入、不在仓库里，
    本函数只做集合层面的机检。
    """
    c, b = set(correctness_shapes or []), set(bench_shapes or [])
    overlap = c & b
    if not c or not b:
        return {"ok": False, "reason": "两侧都必须非空",
                "correctness": sorted(c), "bench": sorted(b)}
    if overlap:
        return {"ok": False, "overlap": sorted(overlap),
                "reason": "检查形状与计时形状有交集 %s —— 可以在这些形状上作弊"
                          % sorted(overlap)}
    return {"ok": True, "correctness": sorted(c), "bench": sorted(b)}


# ---------------------------------------------------------------- 3 AST 扫描
# 注意顺序：先做 3 再讲 2，因为 2 需要运行时，3 是静态的。

FORBIDDEN_PATTERNS = [
    (re.compile(r"\b(?:if|while)\b[^\n]*\b(?:n|N|M|K|seq_len|batch|num_envs|S)\b\s*(?:<=|>=|==|!=|<|>)\s*\d+"),
     "对具体形状/尺寸的字面量比较", "常见于「只在小 shape 上走快路径」"),
    (re.compile(r"\b(?:if|while)\b[^\n]*\.(?:mean|sum|max|min|std|var)\(\s*\)[^\n]*(?:>|<|>=|<=)"),
     "对张量统计量的分支", "按输入分布分派——AST 能抓到这种直白写法"),
    (re.compile(r"\b(?:if|while)\b[^\n]*==\s*\d{3,}"),
     "对具体数值的字面量相等比较", "硬编码某个形状的魔数"),
]


def scan_forbidden(source):
    """扫候选 kernel 源码里的作弊模式。

    ⚠️ **本检查只能抓字面量分支**。它对「按统计量分派」只能抓到直白写法
    （`if x.mean() > 0.5`），抓不到更隐蔽的（先算哈希再分派、用查表掩盖）。
    统计量分派由扰动检查（check_perturbation）覆盖 —— **别对 AST 产生假信心**。
    """
    hits = []
    for i, line in enumerate(source.splitlines(), 1):
        code = line.split("#", 1)[0]
        if not code.strip():
            continue
        for pat, what, why in FORBIDDEN_PATTERNS:
            if pat.search(code):
                hits.append({"line": i, "what": what, "why": why,
                             "code": code.strip()[:80]})
                break
    return {"ok": not hits, "hits": hits,
            "limit": "本检查只抓字面量分支；统计量分派（如先算 hash 再分派）"
                     "需运行 check_perturbation，不许用 AST 结果当「没作弊」的证据"}


def uses_atomics(source):
    """粗判是否用了 atomic——用于决定要不要做 bitexact 断言。

    atomics 让结果顺序不确定，是**合法的非确定性**。
    对它做 bitexact 断言会误伤正确实现。
    """
    return bool(re.search(r"\batomic(?:Add|Exch|CAS|Max|Min|And|Or|Inc|Dec)?\w*\b"
                          r"|__hip_atomic|atomic_", source))


# ---------------------------------------------------------------- 2 扰动分布

def make_perturbations(fn, inputs, ref_fn, tol=1e-2):
    """对输入做 ×3 / ×0.01 / 取负，三变换都要与 reference 相符。

    **为什么必须做**：`if x.mean() > 0.5: 用快速路径` 这类分派，
    在 nominal 输入上永远不会触发。AST 抓不到（它要先算 mean）。
    只有把输入分布打散，才能让那条分支暴露出来。

    fn 与 ref_fn 都是纯函数形式的回调（由调用方适配具体框架）。
    """
    results = []
    for name, xf in (("×3", lambda t: t * 3),
                     ("×0.01", lambda t: t * 0.01),
                     ("取负", lambda t: -t)):
        try:
            inp2 = [xf(t) for t in inputs]
            got, want = fn(inp2), ref_fn(inp2)
            ok, detail = _close(got, want, tol)
        except Exception as e:                                # noqa: BLE001
            ok, detail = False, "异常：%s: %s" % (type(e).__name__, str(e)[:60])
        results.append({"transform": name, "ok": ok, "detail": detail})
    return {"ok": all(r["ok"] for r in results), "probes": results}


def _close(got, want, tol):
    """逐元素比较。不依赖 numpy/torch —— 用纯 Python 处理嵌套列表。"""
    flat_g = _flatten(got)
    flat_w = _flatten(want)
    if flat_g is None or flat_w is None:
        return False, "输出不是数值序列"
    if len(flat_g) != len(flat_w):
        return False, "长度不一致 %d vs %d" % (len(flat_g), len(flat_w))
    worst, idx = 0.0, -1
    for i, (a, b) in enumerate(zip(flat_g, flat_w)):
        d = abs(a - b)
        if d > worst:
            worst, idx = d, i
    return worst <= tol, "max_abs=%.3g @[%d]" % (worst, idx)


def _flatten(x, depth=0):
    if depth > 6:
        return None
    if isinstance(x, (int, float)):
        return [float(x)]
    if isinstance(x, (list, tuple)):
        out = []
        for v in x:
            f = _flatten(v, depth + 1)
            if f is None:
                return None
            out.extend(f)
        return out
    return None


# ---------------------------------------------------------------- 4 输入隔离

def check_input_isolation(run_fn, ref_fn, inputs_a, inputs_b, source="", tol=1e-2):
    """两个**不同**输入各跑一次，分别与 reference 对比。

    **v5.1 修正过的语义，别写反**：
      错：连调两次，断言输出逐位相同 —— 缓存作弊恰恰让两次相同
      对：换输入再测 —— 缓存 kernel 第二次返回 A 的结果，对不上 ref(B)
    """
    out = {"ok": True, "checks": []}
    got_a = run_fn(inputs_a)
    ok_a, d_a = _close(got_a, ref_fn(inputs_a), tol)
    out["checks"].append({"name": "A vs ref(A)", "ok": ok_a, "detail": d_a})
    got_b = run_fn(inputs_b)
    ok_b, d_b = _close(got_b, ref_fn(inputs_b), tol)
    out["checks"].append({"name": "B vs ref(B)", "ok": ok_b, "detail": d_b})
    out["ok"] = ok_a and ok_b
    if not ok_b and ok_a:
        out["diagnosis"] = ("A 对而 B 错 —— 典型的**缓存/分派作弊**："
                            "实现可能直接返回了第一次的结果，或按输入特征分派了不同路径")

    # 附加（防非确定性，不是防缓存）：atomics 的合法非确定性要排除
    if source and not uses_atomics(source):
        got_a2 = run_fn(inputs_a)
        ok_c, d_c = _close(got_a2, got_a, 0.0)
        out["checks"].append({"name": "A 重跑 bitexact", "ok": ok_c, "detail": d_c})
        out["ok"] = out["ok"] and ok_c
    elif source:
        out["checks"].append({"name": "A 重跑 bitexact", "ok": True,
                              "skipped": "源码含 atomics，合法非确定性"})
    return out


def sha256_of(path):
    h = hashlib.sha256()
    h.update(Path(path).read_bytes())
    return h.hexdigest()


def check_file_hash_stable(path, hash_before):
    """计时前后 kernel 文件 sha256 必须不变 —— 防「中途换实现」。"""
    now = sha256_of(path)
    return {"ok": now == hash_before, "before": hash_before, "now": now,
            "reason": "" if now == hash_before else
                      "kernel 文件在计时期间被改过 —— 计时和验证跑的不是同一份代码"}


# ---------------------------------------------------------------- selftest

def selftest():
    # 1) 形状分离
    r = check_shape_separation(["s1024", "s2048"], ["s8192", "s4096"])
    assert r["ok"], r
    r = check_shape_separation(["s1024", "s8192"], ["s8192"])
    assert not r["ok"] and r["overlap"] == ["s8192"], r
    print("[ok] 形状分离：无交集 pass，有交集指出 %s" % r.get("overlap"))

    # 2) AST：干净
    clean = "def k(x):\n    return x + 1\n"
    assert scan_forbidden(clean)["ok"], scan_forbidden(clean)
    print("[ok] AST：干净 kernel 无命中")

    # 3) AST：对具体形状的魔数分支
    cheat1 = "def k(n):\n    if n <= 2048:\n        return fast(n)\n    return slow(n)\n"
    r = scan_forbidden(cheat1)
    assert not r["ok"] and r["hits"], r
    print("[ok] AST：`if n <= 2048` 被抓（%s）" % r["hits"][0]["what"])

    # 4) AST：对字面量的相等比较
    cheat2 = "def k(M):\n    if M == 8192:\n        return approx(M)\n"
    r = scan_forbidden(cheat2)
    assert not r["ok"], r
    print("[ok] AST：`if M == 8192` 被抓")

    # 5) AST：统计量分派（直白写法能抓）
    cheat3 = "def k(x):\n    if x.mean() > 0.5:\n        return approx(x)\n"
    r = scan_forbidden(cheat3)
    assert not r["ok"], r
    print("[ok] AST：`if x.mean() > 0.5` 被抓（但更隐蔽的抓不到）")
    assert "limit" in r and "没作弊" in r["limit"]
    print("[ok] AST 输出带局限说明：%s" % r["limit"][:40])

    # 6) 扰动：正常实现
    class Vec(list):
        """让 *3 / *0.01 / 取负 的语义与 tensor 一致。
        纯 list 的 `[1,2]*3` 是复制而不是数值乘 —— 用 list 做 selftest 会自伤。
        """
        def __mul__(self, k): return Vec(v * k for v in self)
        __rmul__ = __mul__
        def __neg__(self): return Vec(-v for v in self)

    def good(t): return Vec(v * 2 for v in t)
    def ref(t): return Vec(v * 2 for v in t)
    r = make_perturbations(good, [Vec([1.0, 2.0])], ref)
    assert r["ok"], r
    print("[ok] 扰动：正常实现对 ×3/×0.01/取负 都过")

    # 7) 扰动：按统计量分派的作弊实现 —— **这是本工具的核心价值**
    def cheat(t):
        m = sum(t) / len(t)
        return Vec(v * 2 if m < 1.0 else v)     # 均值大时偷工
    r = make_perturbations(cheat, [Vec([1.0, 2.0])], ref)
    assert not r["ok"], r
    bad = [p["transform"] for p in r["probes"] if not p["ok"]]
    print("[ok] 扰动：按统计量分派被抓（%s 档失败）" % bad)

    # 8) 输入隔离：正常实现
    def run_ok(inp): return Vec(v + 1 for v in inp)
    def ref_ok(inp): return Vec(v + 1 for v in inp)
    r = check_input_isolation(run_ok, ref_ok, Vec([1.0, 2.0]), Vec([3.0, 4.0]))
    assert r["ok"], r
    print("[ok] 输入隔离：正常实现 A/B 都对")

    # 9) 输入隔离：**缓存作弊** —— 第二次直接返回第一次结果
    cache = {}
    def run_cache(inp):
        key = "seen"
        if key in cache:
            return cache[key]                   # 缓存：不管输入是什么
        cache[key] = Vec(v + 1 for v in inp)
        return cache[key]
    r = check_input_isolation(run_cache, ref_ok, Vec([1.0, 2.0]), Vec([3.0, 4.0]))
    assert not r["ok"], r
    assert "缓存" in r.get("diagnosis", ""), r
    print("[ok] 输入隔离：**缓存 kernel 被抓**（A 对 B 错 → 判缓存/分派作弊）")

    # 10) 文件哈希
    import tempfile
    p = Path(tempfile.mkdtemp()) / "k.txt"
    p.write_text("v1", encoding="utf-8")
    h = sha256_of(p)
    assert check_file_hash_stable(p, h)["ok"]
    p.write_text("v2", encoding="utf-8")
    assert not check_file_hash_stable(p, h)["ok"]
    print("[ok] 文件哈希：改动后能被发现")

    # 11) atomics 的合法非确定性不误伤
    assert uses_atomics("atomicAdd(x, 1)")
    r = check_input_isolation(run_ok, ref_ok, Vec([1.0]), Vec([2.0]), source="atomicAdd(x,1)")
    assert r["ok"], r
    skipped = [c for c in r["checks"] if c.get("skipped")]
    assert skipped, r
    print("[ok] 反向：atomics kernel 不做 bitexact 断言（避免误伤）")

    print("\nselftest PASS（11 项）")
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description="盲测前的反作弊检查")
    ap.add_argument("--shape-sep", action="store_true", dest="shape_sep")
    ap.add_argument("--correctness", default="")
    ap.add_argument("--bench", default="")
    ap.add_argument("--scan")
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args(argv)
    if a.selftest:
        return selftest()
    out = {}
    if a.shape_sep:
        out["shape_separation"] = check_shape_separation(
            [x for x in a.correctness.split(",") if x],
            [x for x in a.bench.split(",") if x])
    if a.scan:
        src = Path(a.scan).read_text(encoding="utf-8", errors="ignore")
        out["scan"] = scan_forbidden(src)
        out["uses_atomics"] = uses_atomics(src)
    if not out:
        ap.error("给 --shape-sep 或 --scan，或 --selftest")
    print(json.dumps(out, ensure_ascii=False, indent=1) if a.json else
          "\n".join("%s: %s" % (k, "OK" if v.get("ok") else "FAIL")
                    for k, v in out.items() if isinstance(v, dict)))
    return 0 if all(v.get("ok") for v in out.values() if isinstance(v, dict)) else 1


if __name__ == "__main__":
    sys.exit(main())
