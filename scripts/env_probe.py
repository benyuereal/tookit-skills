#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""env_probe —— 跑任何 kernel/bench 之前，先确认环境真的能用。

## 为什么必须有它（2026-11-25 的 5 小时）

那天 FlashMLA 调试花了 5 个多小时、给出 3 次错误结论，最后发现根因是
**`cuda:N`（N>=4）指定设备号会让自定义 kernel 崩**，而
`HIP_VISIBLE_DEVICES=N` + `cuda:0` 就正常。同一张卡、同一个 .so、同一个 kernel。

这个事实**任何一次 30 秒的裸测就能发现**。而整套工具里没有一个地方
在「跑 kernel 之前先验证环境」—— `gpu_guard` 只查卡是否可测，
不查设备**指定方式**。

## 它查什么（每条都有实证）

① **每张卡真跑一次 kernel** —— 不只是 `device_count()` 或分配显存。
   `torch.cuda.mem_get_info` 显示 140GB 空闲，不代表能跑 kernel。

② **两种设备绑定方式对照**
     HIP_VISIBLE_DEVICES=N + cuda:0    vs    cuda:N
   任一崩 → 直接报「设备绑定方式有问题」，不往下走。
   **这是 2026-11-25 那 5 小时的根因，本条会在 30 秒内抓到。**

③ **构建产物可信度**：.so 是否比源码新、产物里找不找得到预期符号。
   实测踩过：补丁在文件里，编译器却从没读过（改的是 hipify 生成物）。

④ **arch 匹配**：编译目标 vs 实际设备。

## 用法

    python3 -I scripts/env_probe.py                 # 查所有卡，给出可用清单
    python3 -I scripts/env_probe.py --card 4        # 只查一张
    python3 -I scripts/env_probe.py --selftest
    python3 -I scripts/env_probe.py --json

退出码：0 = 至少一张卡可用；1 = 没有可用卡（**不要硬跑**）。
"""
import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

# 探针 kernel 的源码。故意做到最小 —— 只做一次 matmul，
# 任何复杂度都会让"崩了"这件事变得难以归因。
PROBE_SRC = r"""#include <torch/extension.h>

torch::Tensor probe_mm(torch::Tensor a, torch::Tensor b) {
  return torch::matmul(a, b);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("probe_mm", &probe_mm, "probe matmul");
}
"""

# 设备绑定方式：这两组必须都成立，否则说明绑定方式有问题
BIND_MODES = ("env", "direct")


def _run_py(code, timeout=120, env=None):
    """跑一段 Python，返回 (ok, stdout, stderr)。不抛异常。"""
    e = dict(os.environ)
    if env:
        e.update(env)
    try:
        p = subprocess.run([sys.executable, "-I", "-c", code],
                           capture_output=True, text=True, timeout=timeout,
                           env=e)
        return p.returncode == 0, p.stdout, p.stderr
    except subprocess.TimeoutExpired:
        return False, "", "超时"
    except OSError as ex:
        return False, "", str(ex)


# ── ① 单卡：能不能真跑 ────────────────────────────────────────
#
# **必须同时测两条路径。** 2026-11-26 实测：
#   纯 torch（a @ b）           在两个设备绑定方式下都正常
#   自定义 .so 加载的 kernel     cuda:N 时崩、HIP_VISIBLE_DEVICES 时正常
# 只测纯 torch 会得出「环境没问题」的错误结论 —— 第一版本脚本就是这样，
# 它报「两种方式均可」，而同一次测试里 cuda:4 + 自定义 .so 明确 dump core。
#
# 下面的代码会：编译一个最小的 torch 扩展 → 用两种方式各跑一次。
_EXT_SRC = r"""
#include <torch/extension.h>
__global__ void _probe_k(float* o) { if (threadIdx.x == 0) o[0] = 1.0f; }
torch::Tensor probe(torch::Tensor t) {
  _probe_k<<<1, 64>>>(t.data_ptr<float>());
  return t;
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("probe", &probe); }
"""

# ── 探针：必须同时测纯 torch 和自定义 kernel ──────────────────
#
# 2026-11-26 实测：纯 torch（a @ a）在两种设备绑定方式下**都正常**，
# 而自定义 .so 加载的 kernel 在 cuda:N 时崩、HIP_VISIBLE_DEVICES 时正常。
# 第一版只测纯 torch，报「两种方式均可」——**判据是假的**。
#
# 这里用 ctypes 加载一个预编译的最小 .so（Kernel 只写一个 float），
# 比 torch.utils.cpp_extension.load_inline 可靠得多（后者在 DTK 上编译失败）。
_PROBE_HIP = r"""#include <hip/hip_runtime.h>
__global__ void _probe_k(float* o) { if (threadIdx.x == 0) o[0] = 1.0f; }
extern "C" void probe_launch(void* p) { _probe_k<<<1, 64>>>((float*)p); }
"""

_CACHE = Path.home() / ".cache" / "dcu-toolkit"


def _ensure_probe_so():
    """编译（或复用）探针 .so。返回路径。"""
    _CACHE.mkdir(parents=True, exist_ok=True)
    hip = _CACHE / "probe_ext.hip"
    so = _CACHE / "probe_ext.so"
    # **只在内容变化时才写 .hip。** 无条件 write_text 会刷新它的 mtime，
    # 导致下面的「.so 比 .hip 新」永远不成立 —— 每次调用都重编一遍。
    # 2026-11-26 实测：缓存目录里只有 .hip 没有 .so，就是这个原因。
    if not hip.is_file() or hip.read_text(encoding="utf-8") != _PROBE_HIP:
        hip.write_text(_PROBE_HIP, encoding="utf-8")
    if so.is_file() and so.stat().st_mtime > hip.stat().st_mtime:
        return so
    r = subprocess.run(["hipcc", "-O3", "--offload-arch=gfx938", "-fPIC",
                        "-shared", "-o", str(so), str(hip)],
                       capture_output=True, text=True, timeout=300)
    if r.returncode != 0:
        raise RuntimeError("探针编译失败：%s" % (r.stderr or r.stdout)[:200])
    return so


_ONE_CARD = """
import ctypes, os, torch
dev = "cuda:0"
# ① 纯 torch 路径
a = torch.randn(256, 256, device=dev, dtype=torch.bfloat16)
c = a @ a
torch.cuda.synchronize()
assert torch.isfinite(c).all(), "纯 torch 结果非有限"
# ② **自定义 kernel** —— 这一步才是 2026-11-25 真正崩的东西
lib = ctypes.CDLL(os.environ["PROBE_SO"])
fn = lib.probe_launch
fn.argtypes = [ctypes.c_void_p]
fn.restype = None
t = torch.zeros(4, device=dev, dtype=torch.float32)
fn(t.data_ptr())
torch.cuda.synchronize()
assert float(t[0].item()) == 1.0, "自定义 kernel 没执行"
print("OK")
"""


def probe_one(card, build_dir=None):
    """在一张卡上真跑一次 matmul。返回 (ok, detail)。

    **用 HIP_VISIBLE_DEVICES 重映射 + cuda:0**，这是安全的方式。
    """
    ok, out, err = _run_py(
        _ONE_CARD,
        timeout=300,
        env={"HIP_VISIBLE_DEVICES": str(card),
             "PROBE_SO": str(_ensure_probe_so())})
    # 注意：这里用的是「重映射 + cuda:0」，是安全方式
    if ok and "OK" in out:
        return True, "可跑"
    msg = (err or out).strip().splitlines()
    tail = msg[-1] if msg else "无输出"
    return False, tail[:80]


# ── ② 两种绑定方式对照（本次 5 小时的根因）────────────────────
_DIRECT = _ONE_CARD.replace('"cuda:0"', '"cuda:%d"')


def probe_bind_modes(card):
    """对照两种设备指定方式。返回 (ok, detail)。

    **前提**：调用方的执行环境里没有预设 HIP_VISIBLE_DEVICES。
    如果设了（loop.py 就是这么调的：它设 env 不传 --device），
    那么进程内的 cuda:N 语义已经从物理号变成逻辑号，
    「直接指定物理卡 N」这件事根本不存在，不该判它失败。
    """
    import os
    preset = os.environ.get("HIP_VISIBLE_DEVICES")
    if preset is not None and preset.strip():
        # 已经被外层绑定 —— 只验这张卡能不能跑，不验「两种方式」
        ok_one, d_one = probe_one(card)
        if ok_one:
            return True, ("已由 HIP_VISIBLE_DEVICES=%s 绑定，卡可跑"
                          "（不适用两种方式对照）" % preset)
        return False, "已绑定 %s 但卡跑不通：%s" % (preset, d_one)

    # 以下是未被外层绑定时才走的路径。
    # 这是 **2026-11-25 那 5 小时的根因**：
    # `cuda:N`（N>=4）时自定义 kernel 崩，`HIP_VISIBLE_DEVICES` 重映射就好。
    # 纯 torch 路径两边都正常 —— 这恰恰是它难被发现的原因。
    #
    # **判据**：两种方式都必须能跑。任一崩就报告，不往下走。
    ok_env, d_env = probe_one(card)

    # 直接指定：不设 HIP_VISIBLE_DEVICES，用 cuda:card
    env_clean = {k: v for k, v in os.environ.items()
                 if k not in ("HIP_VISIBLE_DEVICES", "CUDA_VISIBLE_DEVICES")}
    code = _DIRECT % card
    e = dict(env_clean)
    e["PROBE_SO"] = str(_ensure_probe_so())
    try:
        p = subprocess.run([sys.executable, "-I", "-c", code],
                           capture_output=True, text=True, timeout=300,
                           env=e)
        ok_direct = p.returncode == 0 and "OK" in p.stdout
        d_direct = "" if ok_direct else (p.stderr or p.stdout).strip()[:80]
    except (subprocess.TimeoutExpired, OSError) as ex:
        ok_direct, d_direct = False, str(ex)[:80]

    if ok_env and ok_direct:
        return True, "两种方式均可"
    if ok_env and not ok_direct:
        return False, ("**cuda:%d 直接指定会崩，但 HIP_VISIBLE_DEVICES 重映射正常**"
                       " —— 必须用重映射（%s）" % (card, d_direct))
    if not ok_env and ok_direct:
        return False, "重映射崩但直接指定正常（少见，需查）"
    return False, "两种方式都崩：%s" % (d_env or d_direct)


# ── ③ 构建产物可信度 ────────────────────────────────────────
def check_build(src, so=None, symbols=None):
    """验证 .so 确实是从 src 编出来的、且含预期符号。

    实测踩过（D17）：改了 `dense_decode_hip.h`，但它是 hipify 生成物，
    编译器读的是 `dense_decode.h` —— 改了一整天，产物里没有那些改动。

    返回 (ok, [detail...])。
    """
    src, so = Path(src), Path(so or Path(src).with_suffix(".so"))
    out = []
    ok = True
    if not src.is_file():
        return False, ["源码不存在：%s" % src]
    if not so.is_file():
        return False, ["产物不存在：%s（需要先编译）" % so]
    if so.stat().st_mtime <= src.stat().st_mtime:
        ok = False
        out.append("✗ .so 不比源码新 —— 改了没重编")
    else:
        out.append("✓ .so 比源码新")

    if symbols:
        r = subprocess.run(["nm", "-D", str(so)], capture_output=True, text=True)
        have = r.stdout
        for s in symbols:
            if s in have:
                out.append("✓ 产物含符号 %s" % s)
            else:
                ok = False
                out.append("✗ 产物里找不到符号 %s —— 代码没进二进制" % s)
    return ok, out


# ── ④ arch 匹配 ───────────────────────────────────────────
def check_arch(expect="gfx938"):
    """编译目标与实际设备是否一致。返回 (ok, detail)。"""
    ok, out, _ = _run_py("import torch; print(torch.cuda.get_device_properties(0).gcnArchName)",
                         env={"HIP_VISIBLE_DEVICES": "0"})
    if not ok:
        return False, "读不到设备 arch"
    arch = out.strip().splitlines()[-1] if out.strip() else ""
    if expect in arch:
        return True, "实际 %s，与预期一致" % arch
    return False, "实际 %s，预期含 %s" % (arch, expect)


def check(card=None, emit=print):
    """跑全部检查。返回 (可用卡列表, 是否有问题)。"""
    problems = []

    # ④ arch 先查 —— 它最容易让后面所有结论失效
    ok_arch, d_arch = check_arch()
    emit("① arch 匹配：%s %s" % ("✓" if ok_arch else "✗", d_arch))
    if not ok_arch:
        problems.append("arch")

    # ① 扫卡
    # **不要设 HIP_VISIBLE_DEVICES=""** —— 空字符串在 HIP 下等于「看不见任何设备」，
    # 不是「不改动」。实测踩过（2026-11-26）：这样查出来 n=0，
    # 会误报成「环境不可用」。
    env_clean = {k: v for k, v in os.environ.items()
                 if k not in ("HIP_VISIBLE_DEVICES", "CUDA_VISIBLE_DEVICES")}
    try:
        p2 = subprocess.run([sys.executable, "-I", "-c",
                             "import torch; print(torch.cuda.device_count())"],
                            capture_output=True, text=True, timeout=120,
                            env=env_clean)
        ok, out = p2.returncode == 0, p2.stdout
    except (subprocess.TimeoutExpired, OSError):
        ok, out = False, ""
    n = 0
    if ok:
        try:
            n = int(out.strip().splitlines()[-1])
        except (ValueError, IndexError):
            n = 0
    if n == 0:
        emit("② 设备数：✗ 看不到任何设备")
        return [], True
    emit("② 设备数：%d" % n)

    cards = [card] if card is not None else list(range(n))
    good = []
    emit("③ 逐卡实测（含两种绑定方式对照）：")
    for c in cards:
        ok_run, d_run = probe_one(c)
        if not ok_run:
            emit("   HCU[%d] ✗ %s" % (c, d_run))
            continue
        ok_bind, d_bind = probe_bind_modes(c)
        if not ok_bind:
            emit("   HCU[%d] ⚠ %s" % (c, d_bind))
            problems.append("bind:%d" % c)
            # **绑定方式有问题时仍然算"不可用"** —— 这正是那 5 小时的教训
            continue
        emit("   HCU[%d] ✓ %s" % (c, d_bind))
        good.append(c)

    return good, bool(problems)


def selftest():
    """不依赖 GPU —— 用构造的输入验判据。"""
    # ① check_build：源码比产物新时必须报错
    with tempfile.TemporaryDirectory() as td:
        d = Path(td)
        src = d / "k.hip"
        so = d / "k.so"
        src.write_text("// x", encoding="utf-8")
        so.write_text("fake", encoding="utf-8")

        ok, out = check_build(src, so)
        assert not ok and any("不比源码新" in s for s in out), out
        print("[ok] 反向：产物不比源码新 -> 报错")

        import time
        time.sleep(0.05)
        so.write_text("fake2", encoding="utf-8")
        ok, out = check_build(src, so)
        assert ok, out
        print("[ok] 正向：产物比源码新 -> 通过")

        # 符号不存在时也必须报错
        ok, out = check_build(src, so, symbols=["____not_a_real_symbol____"])
        assert not ok and any("找不到符号" in s for s in out), out
        print("[ok] 反向：产物缺符号 -> 报错")

        missing = d / "nope.hip"
        ok, out = check_build(missing)
        assert not ok, out
        print("[ok] 反向：源码不存在 -> 报错")

    # ② 判据本身：两种方式都崩时必须判不可用
    #    用不存在的设备号，都不该崩进程，只该返回 False
    ok, d = probe_bind_modes(9999)
    assert not ok, ("不存在的设备号应判不可用，实际返回 %r" % (d,))
    print("[ok] 反向：不存在的设备 -> 判不可用（不崩进程）")

    print("\nselftest PASS（5 项，仅第 2 项需要 GPU 且应安全失败）")
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--card", type=int)
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args(argv)
    if a.selftest:
        return selftest()

    if a.json:
        cards, bad = check(emit=lambda *_: None)
        print(json.dumps({"cards": cards, "problems": bad}, ensure_ascii=False))
    else:
        cards, bad = check(a.card)
        print()
        if cards:
            print("✓ 可用卡：%s" % cards)
            if bad:
                print("⚠ 但上面有告警项 —— 报告里不要略过它们")
        else:
            print("✗ 没有可用卡 —— 不要硬跑")

    return 0 if cards else 1


if __name__ == "__main__":
    sys.exit(main())
