#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""op-rmsnorm-fused A/B 适配器：把本卡测量喂给 loop.py run。

契约见 loop.py 的 ADAPTER_AB。测量一律走 opbench.bench_fn（median + MAD），
不在这里另写一套计时——口径不一致是本项目最贵的一类错。

**base 每轮重测**：只报一次 base 会把频率漂移算成收益。

    python3 bench/ab.py --json out.json --variants lightop-4stage --rounds 3
"""
import argparse
import json
import os
import re
import sys
from pathlib import Path

CARD = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CARD))
sys.path.insert(0, str(CARD.parents[3] / "scripts"))

import torch  # noqa: E402
from opbench import bench_fn, significant  # noqa: E402
import _common as C  # noqa: E402


def _measure(fn, inp, rounds, warmup_ms, rep_ms, mode="back_to_back"):
    """每轮喂**副本**——有些实现原地改写输入（lightop.fused_add_rms_norm 就是）。

    2026-11-23 实测：lightop 复用 buffer 196.8 µs / 每轮副本 199.9 µs（差 1.6%），
    说明它一次性完成读改写，复用与否几乎不影响。但**不隔离的实现在别的卡上会失真**
    （本项目 op-sparse-mla 就因 flash_mla 改 indices 把两个变体的结果看反过）。
    所以统一隔离，代价只有 1.6%。

    用 --reps 轮换预克隆的 buffer，避免把 clone 开销算进计时。
    """
    # 预克隆一批 buffer 轮换。数量要够 bench_fn 单轮内的 iters（rep_ms/单次耗时），
    # 否则原地改写实现（lightop.fused_add_rms_norm）会在同一轮内把 pool 反复改掉，
    # 后续迭代读到已变的输入 → 测出偏小的活（实测差 40%）。
    # 预克隆池：给**原地改写输入**的实现用的隔离手段。
    #
    # 显存约束（实测）：prefill-8k 单 buffer 67 MB，256 组 × 2 = 34.4 GB → OOM。
    # 而 rep_ms=100 / 单次 0.16 ms ≈ 625 次迭代，任何能装下的 REP 都不够
    # 「每 buffer 只用一次」。
    #
    # 所以按 shape 大小分两条路：
    #   小 shape（单 buffer ≤ 8 MB）→ 预克隆池，隔离原地改写，开销可忽略
    #   大 shape                    → 直连，不做隔离；改为**依赖实现自身不改写输入**
    #                                 并在记录里标明 isolation=none
    # **原地改写的实现必须隔离**，否则测出来的是被污染后的"快"。
    # 实测教训：lightop.fused_add_rms_norm 原地改写 x 与 residual，
    # 关闭隔离后它测出 202.2 µs（污染后），而 ledger.md 的隔离值是 227.0 µs——
    # 差 11%，且方向会让 lightop 显得更快，从而低估我们相对它的优势。
    #
    # 所以隔离不是可选项，是正确性前提。显存装不下时**降 REP 而不是关隔离**：
    # REP=2 也能保证每轮换 buffer，只是重建更频繁。
    _bytes = sum(t.numel() * t.element_size()
                 for t in inp[:-1] if torch.is_tensor(t))
    # 预算 16 GB：ledger.md 用的是「64 buffer 轮换」，
    # 而 2 GB 预算在 prefill-8k（67MB/buffer）只够 15 组，隔离不充分。
    # 16 GB 能让 REP 到 64，与 ledger 口径一致（实测 8.6 GB 可容纳）。
    _budget = int(16 * 1024 ** 3)
    REP = max(2, min(64, _budget // max(_bytes, 1)))
    _use_pool = True
    isolation = "preclone_pool(REP=%d)" % REP
    if _use_pool:
        # **必须克隆原始输入的快照**，不能直接克隆 inp——
        # inp 会被第一次调用改写（lightop 是原地写的），之后所有 clone
        # 都是克隆"已被污染"的数据，隔离形同虚设。
        # 实测证据：不存快照时 lightop 测出 202.9 µs，而 ledger 的隔离值是 227.0。
        _pristine = [t.clone() if torch.is_tensor(t) else t for t in inp[:-1]]

        def _clone_all():
            return [[t.clone() if torch.is_tensor(t) else t for t in _pristine]
                    for _ in range(REP)]

        pools = _clone_all()
        idx = [0]

        def _call():
            k = idx[0]
            idx[0] += 1
            if idx[0] >= REP:
                idx[0] = 0
                # 重建也只能从 _pristine 克隆（它从来没被传给 fn）
                for i in range(REP):
                    pools[i] = [t.clone() if torch.is_tensor(t) else t
                                for t in _pristine]
            return fn(*pools[k])
    else:
        # 大 shape：不克隆。原地改写的实现在这里会失真——
        # 这个代价必须显式记录（isolation=none），不能假装隔离了。
        _call = lambda: fn(*inp[:-1])                   # noqa: E731

    meds = []
    if mode == "event_isolated":
        # 每次调用前后插 event + synchronize，**杜绝 kernel 间重叠**。
        # 这才是可 claim 的口径。开销是每次多一次 sync（约 10µs 量级），
        # 对 160µs 的 kernel 影响 <7%，但能保证不重叠。
        def _timed():
            e1, e2 = torch.cuda.Event(True), torch.cuda.Event(True)
            e1.record()
            _call()
            e2.record()
            torch.cuda.synchronize()
            return e1.elapsed_time(e2)
        for _ in range(5):
            _timed()                                   # warmup
        for _ in range(max(rounds * 7, 21)):           # 21 次起，够算 MAD
            meds.append(_timed())
    else:
        for _ in range(rounds):
            st = bench_fn(_call, warmup_ms=warmup_ms, rep_ms=rep_ms, repeats=3)
            meds.append(st["median_ms"])
    meds.sort()
    n = len(meds)
    med = meds[n // 2] if n % 2 else 0.5 * (meds[n // 2 - 1] + meds[n // 2])
    mad = sorted(abs(x - med) for x in meds)[n // 2]
    return med, mad


def main(argv=None):
    p = argparse.ArgumentParser(description="op-rmsnorm-fused A/B 适配器")
    p.add_argument("--json", default=None)
    p.add_argument("--rounds", type=int, default=3)
    # measure_mode 必须能选：v5.1 P0-1 要求「同模式才能比」，
    # 而本文件原先写死 back_to_back（不可 claim）。S0 重测需要 event_isolated。
    p.add_argument("--measure-mode", dest="measure_mode",
                   choices=("back_to_back", "event_isolated"), default="back_to_back",
                   help="back_to_back=连续调用（筛方向用）；"
                        "event_isolated=每次 event 隔离（可 claim）")
    p.add_argument("--shape", default="decode-m32")
    p.add_argument("--variants", required=True, help="逗号分隔的变体名")
    p.add_argument("--base", default="ref-heavy")
    p.add_argument("--device", default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--warmup-ms", type=float, default=25.0)
    p.add_argument("--rep-ms", type=float, default=50.0)
    p.add_argument("--skip-preflight", action="store_true",
                   help="跳过环境前置检查（**只在已手工验过环境时用**）")
    a = p.parse_args(argv)

    # **默认跑前置检查**（scripts/bench_preflight.py）。
    # 2026-11-25 的 5 小时根因是「设备绑定方式」问题 —— 纯 torch 探活查不出，
    # 必须用自定义 kernel 试。默认检查是为了不依赖「谁记得跑」。
    if not a.skip_preflight:
        import sys as _sys
        from pathlib import Path as _P
        _sys.path.insert(0, str(_P(__file__).resolve().parents[4] / "scripts"))
        try:
            from bench_preflight import preflight
            if not preflight(device=getattr(a, "device", None)):
                print("✗ 前置检查未全过 —— 加 --skip-preflight 可强制继续，"
                      "但测出的数字不可信", file=sys.stderr)
                return 2
        except ImportError:
            pass   # 没装检查器时不阻塞（但会在 stderr 提一句）


    # **D24**：不要把 `cuda:N`（N>=4）直接传给 torch。
    # 实测同一张卡、同一个 .so：
    #     device='cuda:4'                          -> dumped core
    #     HIP_VISIBLE_DEVICES=4 + device='cuda:0'  -> OK
    # 纯 torch 路径不受影响，所以很容易误判成"平台坏了"。
    # 这里把 N 转成环境变量重映射，然后一律用 cuda:0。
    if a.device:
        _m = re.match(r"^(?:cuda:)?(\d+)$", str(a.device))
        if _m and int(_m.group(1)) > 0:
            os.environ["HIP_VISIBLE_DEVICES"] = _m.group(1)
            os.environ["CUDA_VISIBLE_DEVICES"] = _m.group(1)
            a.device = "cuda:0"

    shape = C.load_shape(a.shape)
    params = dict(shape.get("params") or {})
    kw = C.runner_kwargs(params)
    inp = C.build_inputs(params, seed=a.seed, device=a.device)

    row = {"skv": 0, "variants": []}
    try:
        bmed, bmad = _measure(C.get_runner(a.base, **kw), inp, a.rounds,
                              a.warmup_ms, a.rep_ms, mode=a.measure_mode)
    except Exception as e:                       # noqa: BLE001
        print("base=%s 失败: %s: %s" % (a.base, type(e).__name__, e), file=sys.stderr)
        return 1
    row["base_ms"], row["base_mad"] = bmed, bmad

    for tag in [x.strip() for x in a.variants.split(",") if x.strip()]:
        try:
            med, mad = _measure(C.get_runner(tag, **kw), inp, a.rounds,
                                a.warmup_ms, a.rep_ms, mode=a.measure_mode)
        except Exception as e:                   # noqa: BLE001
            print("variant=%s 跳过: %s: %s" % (tag, type(e).__name__, e), file=sys.stderr)
            continue
        sig, ratio, why = significant({"median_ms": bmed, "mad_ms": bmad},
                                      {"median_ms": med, "mad_ms": mad})
        row["variants"].append({"tag": tag, "ms": med, "mad": mad,
                                "delta_pct": (ratio - 1.0) * 100.0 if ratio else None,
                                "significant": sig, "why": why,
                                "maxdiff": None, "rel": None})
        print("# %s: %.4fms (%.4fx) %s" % (tag, med, ratio or 0, why))
        print("# 口径=%s（%s）" % (a.measure_mode,
              "可 claim" if a.measure_mode == "event_isolated"
              else "不可 claim，仅可筛方向"))

    # caveat 测量口径必须写进 JSON，否则读的人会把「背靠背无 sync」的数当成绩。
    # 2026-11-23 三方对账（业主亲自跑，同机 prefill-8k）：
    #   ab.py 背靠背无 per-iter sync   140 µs  偏乐观 23%（kernel 间重叠）
    #   events 隔离 + 预分配 y         172.5 µs 纯 kernel
    #   C.get_runner 端到端（含分配）   183.0 µs ← **卡片采用这个口径**
    # 本文件的 _measure 走 opbench.bench_fn，是连续调用（背靠背），
    # 所以这里报的 ms 天生偏乐观；卡片/周报一律用端到端口径。
    out = {"caliber": {"what": "op_call", "alloc": False, "state": "hot",
                       "rounds": a.rounds, "timer": "hip_event"},
           "measure_mode": a.measure_mode,
           "measure_note": ("back_to_back：偏乐观（kernel 间可重叠），不可 claim。"
                            "event_isolated：每次 sync，可 claim。"
                            "卡片/周报口径必须用端到端（每次 empty_like）："
                            "2026-11-23 实测 183.0 µs / 1081 GB/s / util 0.63。"
                            "纯 kernel（event 隔离）172.5 µs。"),
           "shape": dict(shape, shape=a.shape),
           "runs": [row]}
    if a.json:
        Path(a.json).parent.mkdir(parents=True, exist_ok=True)
        Path(a.json).write_text(json.dumps(out, ensure_ascii=False, indent=2),
                                encoding="utf-8")
        print("→ %s" % a.json)
    return 0 if row["variants"] else 1


if __name__ == "__main__":
    sys.exit(main())
