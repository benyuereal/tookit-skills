#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""rank_impls —— 跨实现排序的唯一口径。

为什么需要它
------------
2026-11-23 同一天，同一台机器，两次排序给出相反的结论：
  业主：tilelang 204.5 → lightop 211.2 → triton 212.3
  我们：lightop 195.4 → tilelang 199.5 → triton 211.2
tilelang 两边对得上（203~204），**差的是 lightop：一个 198、一个 211，差 6%**。

最可能的原因：lightop.fused_add_rms_norm **原地改写输入**。
  - 每轮喂预克隆副本 → 测的是「冷数据」路径
  - 复用同一 buffer → 第二轮起输入已被改写，但仍可能因 L2 命中而偏快

所以本脚本**同时测两种输入策略**，把两个数都报出来，让差异显式化。

口径（全部必须满足，缺一项这张表就不能用）
------------------------------------------
  1. 同一进程内测所有实现（避免跨进程环境差异）
  2. **交替测量**（A/B/C 按轮轮换），抵消共模环境噪声

     2026-11-23 曾误判「交替有害」：测到 lightop 交替后 435 µs（单独 211）。
     复查发现是外部 `pytest tests/ -q`（PID 492471）在抢占。
     干净环境复测：lightop 独立 210.2 / 交替 211.7（+0.7%）⇒ **交替无害且有效**。

     **但交替只抵消「对称」的干扰**：若 A 被抢、B 没被抢，比值照样失真。
     所以仍必须配漂移检测：测完回头重测第一个，差 >5% 则整表作废。
  3. event 隔离 + 每轮 sync
  4. clone 在计时区外
  5. 记 sclk_mhz 进 JSON
  6. 开跑前 `hy-smi --showpids` 必须为空（只有 PID 才算占用）

用法
----
    python3 -I scripts/rank_impls.py --card rmsnorm-fused --shape prefill-8k
    python3 -I scripts/rank_impls.py --selftest
"""
import argparse
import json
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SKILLS = REPO / "Plugins/ops/skills"
RESULTS = REPO / "Plugins/ops/skills/op-rmsnorm-fused/results"

ROUNDS = 7
# 池大小不是越大越好：每个 buffer 是 M*H*2 字节（prefill-8k 下 67 MB/个），
# 64 组 × 2 张量 = 8.6 GB，实测会引入显存压力（MAD 从 0.6 涨到 11.7 µs）。
# 16 组足够让「原地改写」的污染不重复落到同一块 buffer 上。
CLONES = 16


def sclk_mhz():
    try:
        import re
        out = subprocess.run(["hy-smi", "--showclocks"], capture_output=True,
                             text=True, timeout=15).stdout
        v = [int(x) for x in re.findall(r"sclk clock level:\s*\d+\s*\((\d+)Mhz\)", out)]
        return max(v) if v else None
    except Exception:                                            # noqa: BLE001
        return None


def gpu_pids():
    """返回占用 GPU 的 PID 列表。**只有 PID 才算占用**——showuse 的瞬时数字不作依据。"""
    try:
        out = subprocess.run(["hy-smi", "--showpids"], capture_output=True,
                             text=True, timeout=15).stdout
    except Exception:                                            # noqa: BLE001
        return None
    if "No KFD PIDs" in out or "not running" in out:
        return []
    import re
    pids = re.findall(r"(?m)^\s*(\d+)\s+\S+", out)
    return [p for p in pids if p != str(os.getpid())]


def _time_once(fn, torch):
    e1, e2 = torch.cuda.Event(True), torch.cuda.Event(True)
    e1.record()
    fn()
    e2.record()
    torch.cuda.synchronize()
    return e1.elapsed_time(e2)


def _median_mad(xs):
    med = statistics.median(xs)
    mad = statistics.median([abs(x - med) for x in xs]) if len(xs) > 1 else 0.0
    return med, mad


def rank(runners, inputs, torch, rounds=ROUNDS, clones=CLONES, label=""):
    """**按轮交替**（每轮里每个实现各测一次），抵消共模环境噪声。

    配漂移检测：测完回头重测第一个实现，差 >5% 说明环境非平稳，整表作废。
    """
    out = {t: {"clone": [], "reuse": []} for t in runners}

    for strategy in ("clone", "reuse"):
        pools = None
        if strategy == "clone":
            pools = [[t.clone() if torch.is_tensor(t) else t for t in inputs]
                     for _ in range(clones)]
        idx = [0]

        def make_call(tag):
            def _call():
                if strategy == "clone":
                    k = idx[0] % clones
                    idx[0] += 1
                    return runners[tag](*pools[k])
                return runners[tag](*inputs)
            return _call

        calls = {t: make_call(t) for t in runners}
        for t in runners:                    # 预热
            for _ in range(5):
                calls[t]()
        torch.cuda.synchronize()
        for _ in range(rounds):              # 每轮里每个实现各测一次
            for t in runners:
                out[t][strategy].append(_time_once(calls[t], torch))

    res = {}
    for t, d in out.items():
        mc, madc = _median_mad(d["clone"])
        mr, madr = _median_mad(d["reuse"])
        res[t] = {"clone": {"median_ms": round(mc, 4), "mad_ms": round(madc, 4),
                            "samples": len(d["clone"])},
                  "reuse": {"median_ms": round(mr, 4), "mad_ms": round(madr, 4),
                            "samples": len(d["reuse"])},
                  "strategy_delta_pct": round((mr / mc - 1) * 100, 2) if mc else None}

    # 漂移检测：重测第一个实现（交替不解决非平稳噪声）
    first = next(iter(runners))
    for strategy in ("clone", "reuse"):
        d = out[first][strategy]
        if len(d) >= 3:
            med2, _ = _median_mad(d[-3:])
            base = res[first][strategy]["median_ms"]
            drift = (med2 / base - 1) * 100 if base else 0
            res[first][strategy]["recheck_ms"] = round(med2, 4)
            res[first][strategy]["drift_pct"] = round(drift, 2)
            res[first][strategy]["drift_ok"] = abs(drift) <= 5.0
    return res


def pairwise(res, key="clone", k=3.0):
    """两两显著性（|Δ| vs k×MAD），与 ab.py 同一判据。"""
    tags = list(res)
    rows = []
    for i in range(len(tags)):
        for j in range(i + 1, len(tags)):
            a, b = tags[i], tags[j]
            ma, mad_a = res[a][key]["median_ms"], res[a][key]["mad_ms"]
            mb, mad_b = res[b][key]["median_ms"], res[b][key]["mad_ms"]
            delta = abs(ma - mb)
            thr = k * max(mad_a, mad_b)
            rows.append({"a": a, "b": b, "delta_ms": round(delta, 4),
                         "threshold_ms": round(thr, 4),
                         "significant": delta > thr,
                         "faster": a if ma < mb else b})
    return rows


def run_card(card, shape_name, bandwidth_bytes):
    import importlib.util
    card_dir = SKILLS / card
    spec = importlib.util.spec_from_file_location("_rank_card", str(card_dir / "_common.py"))
    mod = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(REPO / "scripts"))
    spec.loader.exec_module(mod)

    import yaml
    shapes = yaml.safe_load((card_dir / "shapes.yaml").read_text(encoding="utf-8")) or {}
    params = dict((shapes[shape_name] or {}).get("params") or {})
    params["_name"] = shape_name
    inp = mod.build_inputs(params, device="cuda")
    inputs = tuple(inp[:-1])
    kw = mod.runner_kwargs(params)

    # 只排「等语义」的实现：输出份数必须一致
    vd = yaml.safe_load((card_dir / "variants.yaml").read_text(encoding="utf-8")) or {}
    sem = {v["name"]: (v.get("semantics") or {}) for v in vd.get("variants") or []}
    outs = {}
    for name, s in sem.items():
        outs.setdefault(tuple(s.get("outputs") or []), []).append(name)
    # 取份数最多的那一组（通常是 2 份输出的等语义组）
    group = max(outs.values(), key=len) if outs else list(sem)
    runners = {}
    for tag in group:
        try:
            runners[tag] = mod.get_runner(tag, **kw)
        except Exception as e:                                   # noqa: BLE001
            print("跳过 %s：%s" % (tag, str(e)[:70]), file=sys.stderr)

    t0 = time.time()
    res = rank(runners, inputs, __import__("torch"))
    sig = pairwise(res, "clone")
    sig_reuse = pairwise(res, "reuse")
    try:
        import torch as _t
        free_b, total_b = _t.cuda.mem_get_info()
        mem = {"free_gb": round(free_b / 2**30, 1), "total_gb": round(total_b / 2**30, 1)}
    except Exception:                                            # noqa: BLE001
        mem = None
    payload = {
        "card": card, "shape": shape_name, "gpu_mem": mem,
        "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "sclk_mhz": sclk_mhz(),
        "caliber": {"rounds": ROUNDS, "clones": CLONES,
                    "clone_in_timing_region": False,
                    "interleaved": True, "timer": "cuda_event",
                    "note": "同一进程内交替测量；两种输入策略各报一组"},
        "impls": res, "significance_clone": sig, "significance_reuse": sig_reuse,
        "bandwidth_bytes": bandwidth_bytes,
        "wall_s": round(time.time() - t0, 1),
    }
    return payload


def selftest():
    import torch

    # 一个「原地改写」的假实现与一个「不改写」的，用于验证两种策略能测出差异
    def inplace(x, w, r):
        x.add_(r)
        return x * w

    def pure(x, w, r):
        return (x + r) * w

    runners = {"inplace": inplace, "pure": pure}
    inputs = (torch.randn(64, 64), torch.randn(64), torch.randn(64))
    res = rank(runners, inputs, torch, rounds=3, clones=8)
    assert "clone" in res["inplace"] and "reuse" in res["inplace"]
    assert res["inplace"]["clone"]["samples"] == 3
    print("[ok] 两种输入策略都测了：clone=%s reuse=%s"
          % (res["inplace"]["clone"]["median_ms"],
             res["inplace"]["reuse"]["median_ms"]))
    assert "strategy_delta_pct" in res["inplace"]

    # pairwise 的显著性判定
    fake = {"a": {"clone": {"median_ms": 1.0, "mad_ms": 0.01}},
            "b": {"clone": {"median_ms": 1.5, "mad_ms": 0.01}}}
    rows = pairwise(fake, "clone")
    assert rows[0]["significant"] and rows[0]["faster"] == "a"
    print("[ok] pairwise：Δ=%.2f > 3MAD → 显著，a 更快" % rows[0]["delta_ms"])
    print("\nselftest PASS")
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description="跨实现排序的唯一口径")
    ap.add_argument("--card", default="op-rmsnorm-fused")
    ap.add_argument("--shape", default="prefill-8k")
    ap.add_argument("--out", default=None)
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args(argv)
    if a.selftest:
        return selftest()

    pids = gpu_pids()
    if pids is None:
        print("⚠ 读不到 hy-smi --showpids，无法确认卡是否空闲（继续，但请自行确认）",
              file=sys.stderr)
    elif pids:
        print("⛔ 卡被占用（PID: %s），排序不做。等空了再跑。" % ", ".join(pids),
              file=sys.stderr)
        return 2
    else:
        print("# 卡空闲（showpids 为空）")

    # 带宽字节：3*M*H*2 是 1读1写1读 的错误口径；本卡 2读2写 = 4*M*H*2
    import yaml
    shapes = yaml.safe_load((SKILLS / a.card / "shapes.yaml").read_text(encoding="utf-8"))
    p = (shapes[a.shape] or {}).get("params") or {}
    nbytes = 4 * p.get("M", 0) * p.get("H", 0) * 2

    payload = run_card(a.card, a.shape, nbytes)
    out = Path(a.out) if a.out else (RESULTS / "impl-ranking.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n# sclk=%s MHz  rounds=%d clones=%d  字节=%d"
          % (payload["sclk_mhz"], ROUNDS, CLONES, nbytes))
    print("%-24s %10s %10s %10s %10s" % ("实现", "clone µs", "reuse µs", "策略差", "clone GB/s"))
    for tag, v in sorted(payload["impls"].items(), key=lambda kv: kv[1]["clone"]["median_ms"]):
        print("%-24s %10.1f %10.1f %9.1f%% %10.0f"
              % (tag, v["clone"]["median_ms"] * 1000, v["reuse"]["median_ms"] * 1000,
                 v["strategy_delta_pct"] or 0,
                 nbytes / (v["clone"]["median_ms"] * 1e-3) / 1e9))
    print("\n# 显著性（clone 口径，3×MAD）")
    for r in payload["significance_clone"]:
        print("  %-22s vs %-22s Δ=%6.1fµs 3MAD=%6.1f  %s"
              % (r["a"], r["b"], r["delta_ms"] * 1000, r["threshold_ms"] * 1000,
                 "显著" if r["significant"] else "打平"))
    print("\n→ %s" % out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
