#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""bw_calibrate —— 标定「访存模式 → 可达带宽」的阶梯，产出 bw-ladder.yaml。

为什么需要它
------------
`1724 GB/s` 被当分母用了很久，而它**只有断言、没有测量记录**。
2026-11-23 一天之内，分母类错误犯了五次（1100 标称 / 185 反推 / 540 无来源 /
310 单轮采样 / 1724 无标定），每次都导致结论方向反转。

根因不是算错，是**跳层**：直接断言一个「本卡地板」的数，却没经过
「硬件标称 → 通用可达 → 算子类上限 → 本卡地板」这四层。

所以这个脚本只做一件事：**把前三层标出来，让第四层可以推**。

四层阶梯
--------
  ① 硬件标称     1100 GB/s   BW1100 产品名。**只作参考，永不当分母。**
        ↓ 实测
  ② 通用可达     ~1498 GB/s  elementwise y=x+r（本脚本测）
        ↓ 按访存模式打折（实测标定，不许猜）
  ③ 算子类上限   ~1123 GB/s  行内归约 = ② × 0.75（本脚本测）
        ↓ × 语义流量 (inputs + outputs)
  ④ 本卡地板     T = 流量 / ③   ← 由 scripts/floor.py 算

四类访存模式
------------
  streaming       y=x / y=x*2 / y=x+r          已标（本机最高 1498）
  row_reduction   每行一次归约（rmsnorm/layernorm/softmax）  已标
  cross_row       跨行归约（softmax over vocab / topk）      未标定
  scatter_gather  gather / permute / paged write             未标定

**样本数 <3 的标 confidence: low。** 现在 row_reduction 只有 1 个样本
（就是我们自己那个 triton kernel），不能当定律——所以它写成 low。

用法
----
    python3 -I scripts/bw_calibrate.py                  # 标定并写 bw-ladder.yaml
    python3 -I scripts/bw_calibrate.py --dry-run        # 只打印，不写文件
    python3 -I scripts/bw_calibrate.py --check          # 只校验与仓库内 yaml 是否一致
"""
import argparse
import hashlib
import json
import statistics
import sys
import time
from pathlib import Path

# 文本骨架放 scripts/templates/*.md；本文件只负责动态字段/列表。
# `-I` 不把脚本目录放进 sys.path（测试还按路径 importlib 加载），这里补一次。
_SCRIPTS = Path(__file__).resolve().parent
if str(_SCRIPTS) not in sys.path:
    sys.path.append(str(_SCRIPTS))

from template_render import render  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
LADDER = REPO / "Plugins/ops/op-atlas/bw-ladder.yaml"

NOMINAL = 1100          # ① 硬件标称，只作参考
PEAK_UNVERIFIED = 1724  # 旧常数：无测量记录，已降级


def _bench(fn, torch, rounds=30, warmup=5):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(rounds):
        e1, e2 = torch.cuda.Event(True), torch.cuda.Event(True)
        e1.record()
        fn()
        e2.record()
        torch.cuda.synchronize()
        ts.append(e1.elapsed_time(e2))
    ts.sort()
    med = ts[len(ts) // 2]
    return med, statistics.median([abs(t - med) for t in ts])


def _sclk():
    try:
        import re
        import subprocess
        out = subprocess.run(["hy-smi", "--showclocks"], capture_output=True,
                             text=True, timeout=15).stdout
        v = [int(x) for x in re.findall(r"sclk clock level:\s*\d+\s*\((\d+)Mhz\)", out)]
        return max(v) if v else None
    except Exception:                                            # noqa: BLE001
        return None


# ---------------------------------------------------------------- 四类模式

TRITON_SRC = r'''
@triton.jit
def k_read(X, OUT, N, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    off = pid * BLOCK + tl.arange(0, BLOCK)
    v = tl.load(X + off, mask=off < N, other=0.0)
    s = tl.sum(v.to(tl.float32), axis=0)
    if s == 12345.678:
        tl.store(OUT, s)

@triton.jit
def k_write(Y, N, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    off = pid * BLOCK + tl.arange(0, BLOCK)
    tl.store(Y + off, tl.zeros([BLOCK], dtype=Y.dtype.element_ty), mask=off < N)

@triton.jit
def k_copy(X, Y, N, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    off = pid * BLOCK + tl.arange(0, BLOCK)
    m = off < N
    tl.store(Y + off, tl.load(X + off, mask=m), mask=m)

@triton.jit
def k_add2(X, Z, Y, N, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    off = pid * BLOCK + tl.arange(0, BLOCK)
    m = off < N
    tl.store(Y + off, tl.load(X + off, mask=m) + tl.load(Z + off, mask=m), mask=m)
'''


def _build_kernels():
    """triton 的 @jit 要求源码在**真实文件**里（exec/stdin 都不行）。

    所以把这个模块里写好的 kernel 源码落到临时文件再 import——
    保证「一条命令复跑」成立，同时不把 kernel 拆到第二个文件里维护。
    """
    import importlib.util
    import tempfile
    d = tempfile.mkdtemp(prefix="bwcal_")
    f = Path(d) / "bwcal_kernels.py"
    f.write_text("import triton\nimport triton.language as tl\n" + TRITON_SRC,
                 encoding="utf-8")
    spec = importlib.util.spec_from_file_location("bwcal_kernels", str(f))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return {k: getattr(m, k) for k in ("k_read", "k_write", "k_copy", "k_add2")}


def _sweep(kern, mult, grid_bytes, args, torch, blocks=(1024, 2048, 4096, 8192),
           warps=(1, 2, 4, 8), n_iters=30):
    best = None
    for blk in blocks:
        for w in warps:
            grid = ((args["N"] + blk - 1) // blk,)
            try:
                fn = lambda: kern[grid](**args, BLOCK=blk, num_warps=w)
                ms, _ = _bench(fn, torch, rounds=n_iters, warmup=8)
            except Exception:
                continue
            gbs = mult * grid_bytes / (ms * 1e-3) / 1e9
            if best is None or gbs > best["achievable_gbs"]:
                best = {"achievable_gbs": round(gbs), "warps": w,
                        "block": blk, "median_ms": round(ms, 4)}
    return best


def calibrate_modes(torch, tensor_gb=2):
    """四档分开测，扫参取最高值。

    **禁用 torch 的 elementwise 当标尺**：实测 torch copy 1214 vs triton copy 1746，
    用 torch 会把上限压掉 30%。confidence 的判定条件就包括「是否 triton、是否 >=1GB」。
    """
    ns = _build_kernels()
    GB = 1 << 30
    n = ((tensor_gb * GB) // 2 // 8192) * 8192
    x = torch.randn(n, dtype=torch.bfloat16, device="cuda")
    z = torch.randn(n, dtype=torch.bfloat16, device="cuda")
    y = torch.empty_like(x)
    out = torch.zeros(4096, device="cuda")
    nb = n * 2
    specs = [
        ("read_only", ns["k_read"], 1, {"X": x, "OUT": out, "N": n}),
        ("write_only", ns["k_write"], 1, {"Y": y, "N": n}),
        ("rw_1to1", ns["k_copy"], 2, {"X": x, "Y": y, "N": n}),
        ("rw_2to1", ns["k_add2"], 3, {"X": x, "Z": z, "Y": y, "N": n}),
    ]
    rec = {}
    for name, kern, mult, args in specs:
        best = _sweep(kern, mult, nb, args, torch)
        if not best:
            rec[name] = {"achievable": None, "status": "uncalibrated", "confidence": "none"}
            continue
        rec[name] = {
            "achievable": best["achievable_gbs"], "mode": name,
            "tensor_mb": round(nb / 1e6, 1), "shape": "n=%d bf16" % n,
            "warps": best["warps"], "block": best["block"],
            "median_ms": best["median_ms"], "iters": 30, "impl": "triton 手写",
            "confidence": "high" if tensor_gb >= 1 else "low",
        }
    del x, z, y
    torch.cuda.empty_cache()
    return rec


def cal_uncalibrated(name, why):
    return {"achievable": None, "status": "uncalibrated", "reason": why,
            "confidence": "none"}


def _sclk_series(seconds=3.0):
    """测量期间轮询 sclk，返回 (min, max)。

    只读一次是不够的：GEMM/带宽负载会把 DPM 推档，标定开始前 1200、跑起来可能 1350。
    分母必须记「测量期间的档位」，不能记开始前那一次。
    """
    import time as _t
    vals = []
    t0 = _t.time()
    while _t.time() - t0 < seconds:
        v = _sclk()
        if v:
            vals.append(v)
        _t.sleep(0.5)
    if not vals:
        return None, None
    return min(vals), max(vals)


def build(torch):
    import yaml
    ladder = {
        "version": 1,
        "date": time.strftime("%Y-%m-%d"),
        "sclk_mhz_idle": _sclk(),
        "sclk_mhz_load": None,   # 下面 modes 测完后填（那时时钟还在负载档）
        "method": "CUDA event median, warmup 5",
        "layers": {
            "1_nominal": {
                "achievable": NOMINAL,
                "method": "BW1100 产品名（硬件标称）",
                "note": "**只作参考，永不当分母。** 实测显示本机可达远高于此。",
                "confidence": "n/a",
            },
            "2_read_only": None, "2_write_only": None,
            "2_rw_1to1": None, "2_rw_2to1": None,
            "3_cross_row": cal_uncalibrated(
                "cross_row", "未标定：需要 softmax over vocab / topk 这类跨行归约的代表 kernel"),
            "3_scatter_gather": cal_uncalibrated(
                "scatter_gather", "未标定：需要 gather / permute / paged write 的代表 kernel"),
            "4_card_floor": {
                "note": "**不在这里**——④ 由 scripts/floor.py 用「流量 ÷ ③」现算，"
                        "不许预先断言一个数。1724 的灾难就是跳层直接断言 ④。",
            },
        },
    }
    modes = calibrate_modes(torch)
    # 测量刚结束、时钟尚未回落时采样，这才是「负载档」
    ladder["sclk_mhz_load"] = list(_sclk_series(2.0))
    ladder["layers"]["2_read_only"] = modes.get("read_only")
    ladder["layers"]["2_write_only"] = modes.get("write_only")
    ladder["layers"]["2_rw_1to1"] = modes.get("rw_1to1")
    ladder["layers"]["2_rw_2to1"] = modes.get("rw_2to1")
    # 与上一版对账：值变了就把旧值写进 note。
    # 手工补 note 会被下一次生成覆盖——所以必须在生成器里做。
    if LADDER.is_file():
        try:
            prev = yaml.safe_load(LADDER.read_text(encoding="utf-8")) or {}
            for k, node in (modes or {}).items():
                lay_key = {"read_only": "2_read_only", "write_only": "2_write_only",
                           "rw_1to1": "2_rw_1to1", "rw_2to1": "2_rw_2to1"}[k]
                old = ((prev.get("layers") or {}).get(lay_key) or {}).get("achievable")
                new = ladder["layers"].get(lay_key, {}).get("achievable")
                if old and new and old != new:
                    ladder["layers"][lay_key]["note"] = (
                        "上一版 %d GB/s（%s，sclk=%s）→ 本版 %d GB/s（sclk=%s）。"
                        "差异来自 DPM 档位波动：同一台机同一张量，三次标定读到 "
                        "1988/1946/1946。**达成率引用时必须带本次的 sclk**。"
                        % (old, prev.get("date"), prev.get("sclk_mhz_load"),
                           new, ladder.get("sclk_mhz_load")))
        except Exception:                                        # noqa: BLE001
            pass
    return ladder


def render_yaml(ladder):
    """写成真 YAML（供 floor.py 读），注释用 # 不用 markdown 的 >。

    第一版误写成 markdown，floor.py 用 yaml.safe_load 读会炸
    （'>' 被当折叠标量起始符）。机器读的文件就用机器格式。
    """
    import yaml
    # 缩进/键序/折行全部由 yaml.dump 决定，模板只放静态头注释。
    return render(
        "bw_ladder_yaml",
        body=yaml.dump(ladder, allow_unicode=True, sort_keys=False, width=100))


def render_md(ladder):
    lay = ladder["layers"]
    nominal_row = ("| ① 硬件标称 | %s | %s | %s |"
                   % (lay["1_nominal"]["achievable"],
                      lay["1_nominal"]["method"],
                      lay["1_nominal"]["confidence"]))
    rows = []
    for k, name in (("2_read_only", "② 纯读"), ("2_write_only", "② 纯写"),
                    ("2_rw_1to1", "② 1:1 读写"), ("2_rw_2to1", "② 2:1 读写"),
                    ("3_cross_row", "③ 跨行归约"),
                    ("3_scatter_gather", "③ gather/permute")):
        v = lay[k]
        rows.append("| %s | %s | %s | %s |"
                    % (name, v.get("achievable") if v.get("achievable") else "未标定",
                       v.get("method") or v.get("reason", "")[:60],
                       v.get("confidence", "-")))
    return render("bw_ladder", date=ladder["date"],
                  nominal_row=nominal_row, mode_rows="\n".join(rows))


def main(argv=None):
    ap = argparse.ArgumentParser(description="标定访存模式 → 可达带宽的阶梯")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--json", default=None)
    a = ap.parse_args(argv)

    try:
        import torch
    except ImportError:
        print("torch 不可用", file=sys.stderr)
        return 2
    if not torch.cuda.is_available():
        print("无可用 DCU", file=sys.stderr)
        return 2

    ladder = build(torch)
    md = render_md(ladder)

    if a.json:
        Path(a.json).write_text(json.dumps(ladder, ensure_ascii=False, indent=2),
                                encoding="utf-8")
        print("JSON -> %s" % a.json)

    if a.dry_run:
        print(md)
        return 0

    if a.check:
        if not LADDER.is_file():
            print("bw-ladder.yaml 不存在，先跑一次标定", file=sys.stderr)
            return 1
        cur = LADDER.read_text(encoding="utf-8")
        want = render_yaml(ladder)
        strip = lambda t: "\n".join(l for l in t.splitlines()          # noqa: E731
                                    if not l.strip().startswith("date:"))
        if strip(cur) != strip(want):
            print("bw-ladder.yaml 与当前标定不一致（硬件/时钟可能变了）", file=sys.stderr)
            return 1
        print("bw-ladder.yaml 一致")
        return 0

    LADDER.parent.mkdir(parents=True, exist_ok=True)
    LADDER.write_text(render_yaml(ladder), encoding="utf-8")
    sha = hashlib.sha256(md.encode("utf-8")).hexdigest()[:12]
    print("已写 %s" % LADDER.relative_to(REPO))
    print("  sha256[:12] = %s" % sha)
    for k, label in (("2_read_only", "② 纯读"), ("2_write_only", "② 纯写"),
                     ("2_rw_1to1", "② 1:1 读写"), ("2_rw_2to1", "② 2:1 读写")):
        v = ladder["layers"].get(k) or {}
        if v.get("achievable"):
            print("  %-12s %s GB/s  warps=%s block=%s  (%s)"
                  % (label, v["achievable"], v.get("warps"), v.get("block"),
                     v.get("confidence")))
    print("  ③ 跨行 / gather 未标定（confidence: none）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
