#!/usr/bin/env python3
"""derive.py —— 给定形状和设计，先算出地板和绑定约束，再和实测对账。

用法：
  python3 derive.py sparse-mla --s 128 --h 64 --topk 2048 --bm 32 --thr 256 --measured-ms 0.7796
  python3 derive.py sparse-mla --s 128 --h 64 --topk 2048 --bm 64 --thr 256      # 只出预测
  python3 derive.py --list

纪律：没有 --measured 也能用——它给的是**预测**，必须写下来再去实现。
有 --measured 时输出残差；残差 > 30% 先怀疑模型，不要急着换变体。

这个脚本只读 constants.yaml 里的常数，不内置任何魔数。
"""
import argparse
import importlib.util
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
MODELS = os.path.join(HERE, "models")


def _load_constants():
    """极简 YAML 子集读取（只认 key: value / 二级缩进），避免依赖 pyyaml。"""
    path = os.path.join(HERE, "constants.yaml")
    out, stack = {}, []
    for raw in open(path, encoding="utf-8"):
        line = raw.split("#", 1)[0].rstrip()
        if not line.strip():
            continue
        indent = len(line) - len(line.lstrip())
        key, _, val = line.strip().partition(":")
        val = val.strip()
        if not val:
            stack = stack[: indent // 2]
            stack.append(key)
            d = out
            for k in stack:
                d = d.setdefault(k, {})
            continue
        d = out
        for k in stack[: max(0, indent // 2 - 0)]:
            d = d.setdefault(k, {})
        try:
            d[key] = float(val) if val.replace(".", "", 1).replace("-", "", 1).isdigit() else val
        except Exception:
            d[key] = val
    return out


def _load_model(name):
    p = os.path.join(MODELS, name.replace("-", "_") + ".py")
    if not os.path.exists(p):
        raise SystemExit("没有模型：%s（现有：%s）" % (p, ", ".join(
            f[:-3] for f in os.listdir(MODELS) if f.endswith(".py") and not f.startswith("_"))))
    spec = importlib.util.spec_from_file_location("m_" + name.replace("-", "_"), p)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def _num(node, default):
    """常数既可能是裸值，也可能是 {value: ...} 结构。"""
    if isinstance(node, dict):
        node = node.get("value", default)
    try:
        return float(node)
    except (TypeError, ValueError):
        return float(default)


def emit_json(a, r, model, clock, probed, clock_note, bw):
    """机器可读输出。roofline 节点的 result.json 必须由这里生成，
    不得手写 —— 手写才出现过 pmc 算 38.9 / derive 算 77.9 并存的事（D1）。"""
    import json
    out = {
        "tool": "op-roofline/derive.py",
        "model": "sparse-mla" if a.op == "sparse-mla" else a.op,
        "design": {"S": a.s, "H": a.h, "topk": a.topk, "d": a.d,
                   "bm": a.bm, "bi": a.bi, "thr": a.thr, "split": a.split},
        "floors_us": {
            "memory": r["mem_us"],
            "compute": r["mma_us"],
            "overlap": max(r["mem_us"], r["mma_us"]),
            "serial": r["mem_us"] + r["mma_us"],
        },
        "cost_bound": "compute" if r["mma_us"] > r["mem_us"] else "memory",
        # 唯一可信的 ceiling：单来源（derive），带自己的字节分母
        "ceiling_us": max(r["mem_us"], r["mma_us"]),
        "ceiling_basis": "derive 重叠地板 = max(内存地板, 算力地板)",
        "bytes_denominator_mb": r["kv_mb"],
        "bytes_denominator_note": "derive 模型的块重读量；pooled touched 量是另一个分母，不可混用",
        "bandwidth_gbs": bw,
        "clock_ghz": clock,
        "clock_source": ("--clock-ghz 显式指定" if a.clock_ghz
                         else "constants.yaml"),
        "probed_clock_ghz": probed,
        "clock_notes": clock_note,
        "geometry": {k: r[k] for k in
                     ("blocks", "ctas", "mmas", "kv_mb", "cta_per_cu", "rounds",
                      "lds_kb", "lds_ratio", "lds_ok", "serial_chain", "h_per_block",
                      "split", "combine_us", "combine_mb")},
    }
    if a.measured_ms:
        m = a.measured_ms * 1000
        out["measured_us"] = m
        out["measured_source"] = "--measured-ms %.6g" % a.measured_ms
        out["floor_attainment"] = out["ceiling_us"] / m
        out["slowdown_vs_floor"] = m / out["ceiling_us"]
        out["level2_limit_us"] = out["ceiling_us"] * 1.5
        out["level2_within_1_5x_floor"] = m <= out["ceiling_us"] * 1.5
        if probed and abs(probed - clock) / clock > 0.10:
            out["clock_caliber_gap"] = {
                "model_clock_ghz": clock, "probed_clock_ghz": probed,
                "floors_at_probed_us": {
                    "memory": r["mem_us"],
                    "compute": r["mma_us"] * clock / probed},
                "binding_may_flip": True,
            }
    print(json.dumps(out, ensure_ascii=False, indent=2))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("op", nargs="?", help="稀疏 MLA 传 sparse-mla")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--s", type=int, default=128)
    ap.add_argument("--h", type=int, default=64)
    ap.add_argument("--topk", type=int, default=2048)
    ap.add_argument("--nvalid", type=int, default=None, help="有效索引数（-1 填充前）")
    ap.add_argument("--d", type=int, default=512)
    ap.add_argument("--bm", type=int, default=32)
    ap.add_argument("--bi", type=int, default=32)
    ap.add_argument("--thr", type=int, default=256)
    ap.add_argument("--split", type=int, default=1)
    ap.add_argument("--skv", type=int, default=333012, help="mqa: KV 总长（320k 上下文实测 333012）")
    ap.add_argument("--kv-bytes", type=int, default=1, help="mqa: KV/Q 每元素字节（fp8=1, bf16=2）")
    ap.add_argument("--fuse-topk", action="store_true", help="mqa: 假设 logits 不物化")
    ap.add_argument("--next-n", type=int, default=1,
                    help="mqa: 投机深度（生产 next_n=4）。未 fold 时不影响，fold 时 K 少读 next_n 倍")
    ap.add_argument("--q-per-ktile", type=int, default=None,
                    help="mqa: 多少条 query 共享一份 K（现成 decode 内核=1，prefill tile=BM，fold 路径=next_n）。"
                         "缺省按档位推断：给了 --fold-nextn 用 next_n，否则 1")
    ap.add_argument("--fold-nextn", action="store_true",
                    help="mqa: 假设 next_n 已并入 GEMM 的 N 维（生产 fold 路径）。"
                         "不加这个开关 = 现成内核，K 每条 query 读一遍 ⇒ 判访存受限")
    ap.add_argument("--replicate-h", type=int, default=None)
    ap.add_argument("--lds-bytes", type=int, default=None)
    ap.add_argument("--measured-ms", type=float, default=None)
    ap.add_argument("--clock-ghz", type=float, default=None,
                    help="实测耗时当时的 sclk 档位。缺省用 constants.yaml 的值，"
                         "但与当场探测值不一致时会警告——算力地板 ∝ 1/sclk，"
                         "0.6 vs 1.15 GHz 差 1.9×，足以让 cost_bound 从 memory 翻到 compute")
    ap.add_argument("--no-probe-clock", action="store_true", help="跳过当场 sclk 探测")
    ap.add_argument("--json", action="store_true",
                    help="输出机器可读 JSON（roofline 节点的 result.json 必须由此生成，不得手写）")
    a = ap.parse_args()

    if a.list:
        print("可用模型：", ", ".join(f[:-3] for f in sorted(os.listdir(MODELS))
                                    if f.endswith(".py") and not f.startswith("_")))
        return

    if not a.op:
        ap.error("要么给算子名，要么 --list")

    const = _load_constants()
    clock_cfg = _num(const.get("compute", {}).get("clock_ghz"), 1.15)
    bw = _num(const.get("memory", {}).get("stream_read_peak"), 1724)
    mma_peak = _num(const.get("compute", {}).get("mma_issue_peak"), 31.6)
    model = _load_model(a.op)

    # --- 时钟口径（D12）----------------------------------------------------
    # 算力地板 = MMAs / (31.6 条/cyc × sclk)。sclk 是唯一一个会在两次测量之间
    # 自己变的常数（DPM 档位，未锁频），不显式钉住就可能把「算力受限」判成「内存受限」。
    clock = a.clock_ghz if a.clock_ghz else clock_cfg
    clock_note = []
    probed = None
    if a.clock_ghz:
        clock_note.append("时钟来自 --clock-ghz=%.3f GHz" % a.clock_ghz)
    else:
        clock_note.append("时钟来自 constants.yaml clock_ghz=%.3f GHz" % clock_cfg)
        if not a.no_probe_clock:
            try:
                import re as _re
                import subprocess
                txt = subprocess.run(["hy-smi", "-c"], capture_output=True,
                                     text=True, timeout=5).stdout or ""
                s = [int(m) for m in _re.findall(
                    r"sclk clock level:[^(]*\((\d+)\s*Mhz", txt, _re.I)]
                if s:
                    probed = s[0] / 1000.0
            except Exception:  # noqa: BLE001
                probed = None
        if probed:
            clock_note.append("当场探测卡0 sclk=%.3f GHz" % probed)
        else:
            clock_note.append("未能探测当前 sclk")

    if a.op not in ("sparse-mla", "mqa-logits"):
        raise SystemExit("暂时只有 sparse-mla / mqa-logits 的模型")

    if a.op == "mqa-logits":
        peak = _num(const.get("compute", {}).get("fp8_gemm_peak_tf"), 540)
        r = model.analyze(a.s, a.h, a.skv, a.d, BM=a.bm, BN=a.bi,
                          kv_bytes=a.kv_bytes, q_bytes=a.kv_bytes,
                          peak_tf=peak, bw_gbs=bw, fuse_topk=a.fuse_topk,
                          next_n=a.next_n, fold_nextn=a.fold_nextn,
                          **({"q_per_ktile": a.q_per_ktile} if a.q_per_ktile else {}))
        print("mqa_logits S=%d H=%d SKV=%d D=%d | kv=%dB/元素 topk融合=%s "
              "q_per_ktile=%d p=%d"
              % (a.s, a.h, a.skv, a.d, a.kv_bytes, a.fuse_topk,
                 r["q_per_ktile"], r["k_passes"]))
        print()
        print(model.verdict(r, a.measured_ms * 1000 if a.measured_ms else None))
        return

    r = model.analyze(a.s, a.h, a.topk, a.d, a.bm, a.thr, a.bi, a.split,
                      replicate_h=a.replicate_h, nvalid=a.nvalid, lds_bytes=a.lds_bytes,
                      clock_ghz=clock, bw_gbs=bw)

    if a.json:
        emit_json(a, r, model, clock, probed, clock_note, bw)
        return 0

    print("形状 S=%d H=%d topk=%d(有效 %d) d=%d | 设计 bm=%d bi=%d thr=%d REPLICATE_H=%s(每 CTA %d 头)"
          % (a.s, a.h, a.topk, a.nvalid or a.topk, a.d, a.bm, a.bi, a.thr,
             a.replicate_h if a.replicate_h else "auto", r["h_per_block"]))
    print("  CTA=%d  每 CTA %d 块  MMA=%.2f M 条  读 KV=%.1f MB"
          % (r["ctas"], r["blocks"], r["mmas"] / 1e6, r["kv_mb"]))
    print("  每 CU 驻 %.1f 个 CTA ⇒ 跑 %d 轮；每 CTA 串行 %d 步"
          % (r["cta_per_cu"], r["rounds"], r["serial_chain"]))
    print("  LDS %.0f KB / 64 KB = %.0f%%  %s"
          % (r["lds_kb"], r["lds_ratio"] * 100, "" if r["lds_ok"] else "blocked 超 85% 阈值（R6）"))
    print()
    print("  内存地板 %8.1f µs  ( %.1f MB / 峰值 %.0f GB/s )" % (r["mem_us"], r["kv_mb"], bw))
    print("  算力地板 %8.1f µs  ( %.2f M 条 / %.1f MFMA/cyc @ %.2f GHz )"
          % (r["mma_us"], r["mmas"] / 1e6, mma_peak, clock))
    print()
    # 时钟口径：写在结论前面，因为它能改变结论本身
    if probed and abs(probed - clock) / clock > 0.10:
        mma_at_probed = r["mma_us"] * clock / probed
        flip = (r["mem_us"] < mma_at_probed) != (r["mem_us"] < r["mma_us"])
        print("  caveat 时钟口径不一致：本表按 %.3f GHz 算，当场探测卡0 是 %.3f GHz。"
              % (clock, probed))
        print("     若实测时也是 %.3f GHz，算力地板 = %.1f µs（本表 %.1f µs）。%s"
              % (probed, mma_at_probed, r["mma_us"],
                 "**绑定约束会翻转：%s → %s**" %
                 ("memory" if flip else "compute", "compute" if flip else "memory")
                 if flip else "绑定约束不变。"))
        print("     用 --clock-ghz <实测时的档位> 重算。")
        print()
    for line in clock_note:
        print("  时钟：%s" % line)
    if clock_note:
        print()
    print(model.verdict(r, a.measured_ms * 1000 if a.measured_ms else None))

    if a.measured_ms:
        pred = max(r["mem_us"], r["mma_us"])
        res = (pred - a.measured_ms * 1000) / (a.measured_ms * 1000) * 100
        print()
        print("  ── 对账 ──")
        print("  下界预测 %.1f µs vs 实测 %.1f µs ⇒ 残差 %+.0f%%（负 = 实测高于任何吞吐地板）"
              % (pred, a.measured_ms * 1000, res))
        if res < -30:
            print("  结论：数据一致 —— 没有单元被打满，耗时由依赖链/并发决定。")


if __name__ == "__main__":
    sys.exit(main())
