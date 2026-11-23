#!/usr/bin/env python3
"""probe_pattern — 访存 run-length 扫描：拿到「这个访问模式自己的带宽上限」。

同一个 kernel 在「每行连续 256 B、行间跨 32 KB」下只能达到流式峰值的 ~60%，**那是该访问模式的
物理上限，不是实现问题**；没这个数就会一直以为「还有 40% 可优化」。纪律：任何「行内连续 X 字节」
的 kernel（tensor tile 列读取、strided gather、转置读）先跑本探针拿 pattern ceiling，再用它当
分母算达成率——用线性峰值当分母，分母就错了。

测法：固定行跨度 --stride，只改每行连续读多少字节（--runs）；每种 run 读相同总字节数
（--total-mib），GB/s 才可比；`linear` 是参照组（同 buffer 完全线性访问，同一内核测）。
输出 `run_bytes | achieved_gbs | pct_of_linear`；`--json` 机器可读。
为什么自带内核：要在「CTA 持一组行、外层按 run 推进」的顺序下测 DRAM 行激活效率；torch 单次
strided reduce 控制不了 CTA 迭代顺序，L2 会把同一行不同 run 的 16 B 读取合并，测出来就不再是
该模式的上限。故内核用 hipcc 编到缓存目录（约 2 s，复用）后 ctypes 驱动。
注意 `pct_of_linear` 的分母是 k_linear（真流式）；pattern 内核即使 run==stride 也带 tile 顺序开销，
所以 ceiling 偏保守——宁可低估上限，也不要拿一个虚高的分母说「还有空间」。

用法：python3 -I scripts/probe_pattern.py [--runs 128 256 linear] [--stride 8192]
      [--total-mib 512] [--json | --selftest]；依赖标准库 + torch + hipcc。
缺任一时**明确说无法探测**，不静默给数。退出码：0 正常；2 环境不具备或探测失败。
"""

import argparse
import ctypes
import json
import statistics
import subprocess
import sys
from pathlib import Path

DEFAULT_RUNS = ["16", "32", "64", "128", "256", "512", "1024", "2048", "linear"]
DEFAULT_STRIDE, DEFAULT_TOTAL_MIB = 32768, 256
DEFAULT_TILE_ROWS, DEFAULT_THREADS = 64, 512

KERNEL_SRC = r"""
#include <hip/hip_runtime.h>
// k_linear：参照组，完全线性 grid-stride 16 B 读。
// k_pattern：被测模式，CTA 持 tile_rows 行（工作集须 > L2），外层 kb 推进 run 起点，
//            内层线程在 (行, run 内 16 B 块) 上交错。
__global__ void __launch_bounds__(1024) k_linear(const int4* __restrict__ x, long n4,
                                                 float* __restrict__ out) {
  int4 a = make_int4(0, 0, 0, 0);
  long i = (long)blockIdx.x * blockDim.x + threadIdx.x, st = (long)gridDim.x * blockDim.x;
  for (; i < n4; i += st) { int4 v = x[i]; a.x += v.x; a.y += v.y; a.z += v.z; a.w += v.w; }
  if ((threadIdx.x & 63) == 0) out[blockIdx.x] = (float)(a.x + a.y + a.z + a.w);
}
__global__ void __launch_bounds__(1024) k_pattern(const char* __restrict__ x,
        float* __restrict__ out, int tile_rows, int row_bytes, int log_cpr, int nkb) {
  const char* base = x + (size_t)blockIdx.x * tile_rows * row_bytes;
  const int cpr = 1 << log_cpr, total = tile_rows << log_cpr;
  int4 a = make_int4(0, 0, 0, 0);
  for (int kb = 0; kb < nkb; ++kb)
    for (int i = threadIdx.x; i < total; i += blockDim.x) {
      int4 v = *(const int4*)(base + (size_t)(i >> log_cpr) * row_bytes
                 + (size_t)kb * (cpr << 4) + (size_t)(i & (cpr - 1)) * 16);
      a.x += v.x; a.y += v.y; a.z += v.z; a.w += v.w;
    }
  if ((threadIdx.x & 63) == 0) out[blockIdx.x] = (float)(a.x + a.y + a.z + a.w);
}
// 两个 host 侧入口：一个给线性参照，一个给 run 模式；都返回 hipGetLastError()。
extern "C" int pp_linear(const void* x, long bytes, void* out, int grid, int blk) {
  hipLaunchKernelGGL(k_linear, dim3((unsigned)grid), dim3((unsigned)blk), 0, 0,
                     (const int4*)x, bytes / 16, (float*)out);
  return (int)hipGetLastError();
}
extern "C" int pp_pattern(const void* x, void* out, int rows_total, int row_bytes,
                          int run_bytes, int tile_rows, int blk) {
  int grid = rows_total / tile_rows, nkb = row_bytes / run_bytes, lc = 0;
  if (grid < 1) return -1;
  while ((1 << lc) < run_bytes / 16) lc++;
  hipLaunchKernelGGL(k_pattern, dim3((unsigned)grid), dim3((unsigned)blk), 0, 0,
                     (const char*)x, (float*)out, tile_rows, row_bytes, lc, nkb);
  return (int)hipGetLastError();
}
"""


def _torch():
    """惰性导入 torch；不可用时返回 (None, 原因)。"""
    try:
        import torch
    except Exception as exc:
        return None, "没有可用的 torch：%s" % exc
    if not torch.cuda.is_available():
        return None, "torch 在，但 torch.cuda.is_available() = False（无可见 DCU）"
    return torch, None


def build_lib(build_dir=None, timeout=180):
    """编译探针内核到缓存目录；返回 (so 路径, 错误)。同一份源码只编译一次。"""
    root = Path(build_dir or (Path.home() / ".cache" / "dcu-toolkit" / "probe_pattern"))
    src, so = root / "probe_pattern.cu", root / "libprobe_pattern.so"
    try:
        root.mkdir(parents=True, exist_ok=True)
        if so.is_file() and src.is_file() and so.stat().st_mtime >= src.stat().st_mtime:
            return str(so), None
        src.write_text(KERNEL_SRC, encoding="utf-8")
        p = subprocess.run(["hipcc", "-O3", "-shared", "-fPIC", str(src), "-o", str(so)],
                           capture_output=True, text=True, timeout=timeout)
        if p.returncode != 0 or not so.is_file():
            return None, "hipcc 编译失败（rc=%d）：%s" % (
                p.returncode, (p.stderr or p.stdout or "").strip()[:400])
        return str(so), None
    except FileNotFoundError:
        return None, ("找不到 hipcc：无法编译探针内核（装 DTK 后重试；"
                      "或用 --build-dir 指向已编译好的 libprobe_pattern.so 所在目录）")
    except (OSError, subprocess.SubprocessError) as exc:
        return None, "编译探针内核失败：%s" % exc


def load_lib(torch, so_path):
    """加载内核并绑定签名；返回 (lib, out 张量)。"""
    lib = ctypes.CDLL(so_path)
    lib.pp_linear.restype = lib.pp_pattern.restype = ctypes.c_int
    lib.pp_linear.argtypes = [ctypes.c_void_p, ctypes.c_long, ctypes.c_void_p,
                              ctypes.c_int, ctypes.c_int]
    lib.pp_pattern.argtypes = [ctypes.c_void_p, ctypes.c_void_p] + [ctypes.c_int] * 5
    return lib, torch.zeros(4096, device="cuda", dtype=torch.float32)


def _report(result):
    """人读表：run_bytes | achieved_gbs | pct_of_linear + ceiling 一行。"""
    print("== 访存 run-length 扫描（stride=%d B，总量 %d MiB，tile_rows=%d，%d 线程）=="
          % tuple(result[k] for k in ("stride_bytes", "total_mib", "tile_rows", "threads")))
    print("  run_bytes | achieved GB/s | pct_of_linear")
    for r in result["records"]:
        flag = "" if r.get("stable", True) else "  caveat 抖动 >5%"
        print("  %9s | %13.1f | %5.1f%%%s" % (r["run_bytes"], r["achieved_gbs"],
                                              r["pct_of_linear"], flag))
    best = ceiling(result)
    if best:
        print("\n  本配置的 pattern ceiling = %.1f GB/s（run=%s，为线性的 %.1f%%）。"
              % (best["achieved_gbs"], best["run_bytes"], best["pct_of_linear"]))
        print("  ⇒ 算「行内连续 %s B」类 kernel 的达成率，分母用这个数，不要用线性峰值。"
              % best["run_bytes"])
    print("  注：这是该访问模式的物理上限（DRAM 行激活/扇区粒度决定），不是实现质量。")


def measure(torch, fn, reps=7, warmup=3):
    """event 计时，中位数 + MAD；hot 态，不含量测之外的分配。"""
    for _ in range(warmup + 1):            # warmup 轮 + 一次同步，清除首轮虚高
        fn()
    torch.cuda.synchronize()
    ev = [torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)]
    xs = []
    for _ in range(reps):
        ev[0].record(); fn(); ev[1].record()
        torch.cuda.synchronize()
        xs.append(ev[0].elapsed_time(ev[1]))
    med = statistics.median(xs)
    return {"median_ms": med, "mad_ms": statistics.median([abs(x - med) for x in xs])}


def parse_runs(items):
    """'16 256 linear' → [16, 256, 'linear']；run_bytes 必须是 16 B 的正整数倍。"""
    out = []
    for it in items:
        if str(it).strip().lower() == "linear":
            out.append("linear")
            continue
        v = int(it)
        if v < 16 or v % 16:              # DRAM 颗粒 16 B：非整数块会把数变成「有效带宽」
            raise SystemExit("run_bytes 必须是 16 的正整数倍：%s" % it)
        out.append(v)
    return out


def sweep(torch, lib, out, runs, stride, total_mib, tile_rows, threads, reps):
    """逐 run 测；每种 run 的总访问量都等于 buffer 大小（行数 × stride）。"""
    rows = int(total_mib) * 1024 * 1024 // stride
    if rows < tile_rows:
        raise SystemExit("行数 %d < tile_rows=%d：调大 --total-mib 或调小 --tile-rows"
                         % (rows, tile_rows))
    rows -= rows % tile_rows
    nbytes, grid = rows * stride, min(4096, max(64, rows // tile_rows))
    buf = torch.empty(nbytes, device="cuda", dtype=torch.uint8)
    try:
        lin = measure(torch, lambda: lib.pp_linear(buf.data_ptr(), nbytes, out.data_ptr(),
                                                   grid, threads), reps)
        lin_gbs = nbytes / (lin["median_ms"] * 1e6)

        def rec(run, med, note=None):
            gbs = nbytes / (med["median_ms"] * 1e6)
            d = {"run_bytes": run, "achieved_gbs": round(gbs, 1),
                 "pct_of_linear": round(gbs / lin_gbs * 100.0, 1),
                 "median_ms": round(med["median_ms"], 4), "mad_ms": round(med["mad_ms"], 4),
                 "stable": bool(med["median_ms"] and med["mad_ms"] / med["median_ms"] < .05)}
            if note:
                d["note"] = note
            return d

        recs = []
        for run in runs:
            if run == "linear":
                recs.append(rec("linear", lin, "参照组：同 buffer 完全线性访问 = 流式峰值"))
                continue
            if run > stride:
                raise SystemExit("run_bytes=%d 超过 stride=%d" % (run, stride))
            recs.append(rec(run, measure(torch, lambda: lib.pp_pattern(
                buf.data_ptr(), out.data_ptr(), rows, stride, run, tile_rows, threads), reps)))
        return {"rows": rows, "stride_bytes": stride, "total_mib": total_mib,
                "tile_rows": tile_rows, "threads": threads,
                "linear_gbs": round(lin_gbs, 1), "records": recs}
    finally:
        del buf
        torch.cuda.empty_cache()


def ceiling(result):
    """最佳的非 linear 项——这就是该配置的 pattern ceiling。"""
    recs = [r for r in result["records"] if r["run_bytes"] != "linear"]
    return max(recs, key=lambda r: r["achieved_gbs"]) if recs else None


def selftest():
    assert parse_runs(["16", "linear"]) == [16, "linear"]
    try:
        parse_runs(["15"])
        raise AssertionError("15 B 应被拒绝")
    except SystemExit:
        print("[ok] parse_runs  linear / 16 B 校验 正常")
    best = ceiling({"records": [
        {"run_bytes": 128, "achieved_gbs": 640.0, "pct_of_linear": 40.0},
        {"run_bytes": 256, "achieved_gbs": 1049.0, "pct_of_linear": 60.0},
        {"run_bytes": "linear", "achieved_gbs": 1870.0, "pct_of_linear": 100.0}]})
    assert best["run_bytes"] == 256 and best["pct_of_linear"] == 60.0, best
    print("[ok] ceiling 选中最佳非 linear 项（run=%s，%.0f%%）"
          % (best["run_bytes"], best["pct_of_linear"]))
    t, err = _torch()
    so, berr = build_lib() if t else (None, None)
    print("[ok] 环境探测：torch=%s；hipcc=%s" %
          ("可用" if t else err, "可用" if so else (berr or "未试（无 torch）")))
    print("\nselftest PASS")
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description="访存 run-length 扫描：拿到访问模式自己的带宽上限")
    ap.add_argument("--runs", nargs="+", default=list(DEFAULT_RUNS),
                    help="每行连续字节数；'linear' 为参照组")
    ap.add_argument("--stride", type=int, default=DEFAULT_STRIDE, help="行间跨度（字节）")
    ap.add_argument("--total-mib", type=int, default=DEFAULT_TOTAL_MIB,
                    help="总访问量 MiB；每种 run 读同样多，GB/s 才可比")
    ap.add_argument("--tile-rows", type=int, default=DEFAULT_TILE_ROWS,
                    help="每 CTA 持多少行（工作集需 > L2）")
    ap.add_argument("--threads", type=int, default=DEFAULT_THREADS)
    ap.add_argument("--reps", type=int, default=7, help="每个配置的计时轮数（取中位数）")
    ap.add_argument("--build-dir", help="内核编译缓存目录（默认在 ~/.cache/dcu-toolkit/）")
    ap.add_argument("--json", action="store_true", help="输出机器可读 JSON")
    ap.add_argument("--selftest", action="store_true", help="无 GPU 的自检")
    a = ap.parse_args(argv)
    if a.selftest:
        return selftest()

    runs = parse_runs(a.runs)
    torch, err = _torch()
    result, so = None, None
    if torch is not None:
        so, err = build_lib(a.build_dir)
    if so:
        try:
            lib, out = load_lib(torch, so)
            result = sweep(torch, lib, out, runs, a.stride, a.total_mib,
                           a.tile_rows, a.threads, a.reps)
            err = None
        except SystemExit as exc:
            err = str(exc)
        except Exception as exc:                 # 显存不足、内核 rc != 0 等
            err = "探测中断：%s" % exc

    code = 0 if result else 2
    if a.json:
        cfg = {k: getattr(a, k) for k in ("stride", "total_mib", "tile_rows",
                                          "threads", "reps")}  # 口径全带在输出里
        print(json.dumps({"purpose": "访问模式自己的带宽上限；用它当分母，不要用线性峰值",
                          "config": dict(cfg, runs=runs), "result": result,
                          "error": err, "ok": code == 0, "exit_code": code},
                         ensure_ascii=False, indent=2))
        return code
    if result:
        _report(result)
    else:
        print("== 访存 run-length 扫描 ==\n  ✗ 无法探测：%s" % err)
        print("  ⇒ 不输出任何数字（静默给默认值会把「物理上限」变成猜的）。")
        if so is None:      # 环境缺失才提示装什么；参数/显存问题不要误导
            print("  需要：torch + 可见 DCU + hipcc（DTK 自带）。")
    return code


if __name__ == "__main__":
    sys.exit(main())
