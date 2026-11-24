#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""forge_step —— 「诊断 → 造变体 → 编译 → 测量」里**造+编译**的那一环。

它是什么、不是什么
------------------
是：把 `kernel_scaffold` 生成的库形态骨架变成**一个已编译、已体检、可加载的变体文件**
    （`runs/forged/<name>.py` + `_kernels/<name>.so`），供 `_common.get_runner` 动态注册。
不是：决定「该造什么」。那一步是诊断（roofline / PMC / ISA），本文件只负责
    「给定模板+参数，把它变成能测的东西，并**证明**它是合法的」。

为什么必须体检（这不是可选的礼貌，是这条链路唯一的正确性保证）
------------------------------------------------------------
1. **private 必须为 0**：寄存器数组一旦溢出到私有内存，慢 3.3 倍且**不会报错**。
   实测 `VEC` 从编译期常量改成运行期变量：private 0→528B，0.165→0.538ms。
   这种退化在跑分里看起来只是「这个方向不行」，会让人**把对的改法判成错的**。
2. **threads 必须等于 THREADS**：`__launch_bounds__` 没写对时编译器会静默截断，
   grid 与每 CTA 工作量不匹配 → 漏算，同样不报错。
3. **H 必须恰好等于 THREADS*8*VEC**：不等就是越界读或漏读。

三条都是「错了也跑得动、但算错的」那一类。体检不通过就**拒绝产出变体**，
而不是产出一个待会会被测出「慢」的假变体 —— 那会污染方向预算。
"""
import json
import re
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SCAFFOLD = REPO / "scripts/kernel_scaffold.py"
INSPECT_SO = REPO / "Plugins/ops/skills/op-hip-compile/inspect_so.py"

# 体检门限。不是可调参数：这三条对应「静默算错」，没有放松的余地。
MAX_PRIVATE_BYTES = 0
MAX_VGPR = 256                      # 超了这个数就必然溢出，宁可早起不来


class ForgeStepError(RuntimeError):
    """造变体失败。**带可读原因**，且失败时不留下半成品。"""


def _run(cmd, timeout=300):
    p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    return p.returncode, (p.stdout or "") + (p.stderr or "")


def expected_vec(H, threads):
    """H 对应的 VEC；不可整除时返回 None（调用方必须报错，不能猜）。"""
    if threads * 8 == 0 or H % (threads * 8):
        return None
    vec = H // (threads * 8)
    return vec if 1 <= vec <= 8 else None


# 实测的选参依据（**不是猜的**，2026-11-24 prefill-8k / H=4096 / gfx938）：
#   THREADS=256 VEC=2  → 0.1846 ms / 1454 GB/s / util 0.865 / vgpr 38
#   THREADS=512 VEC=1  → 0.2299 ms（反而慢 24%）                    / vgpr 21
# 差别的原因：VEC=2 每线程发 2 个 uint4，访存并行度翻倍；VEC=1 时线程数虽多，
# 但每线程只有一次 16B 机会，且 512 线程的 CTA 占更多调度槽。
# 结论：**优先要 VEC≥2**，线程数够用即可。第一版 choose_params 只取「第一个能
# 整除的组合」（512 优先），于是自动生成的变体比手工版慢 24% —— 「能造」和
# 「造得对」的差距就在这里。这条依据只在一个形状上验证过，见 model_limit。
PARAM_BASIS = {
    "shape": "prefill-8k (M=8192, H=4096), bf16, gfx938 / BW1100",
    "measured": [{"THREADS": 256, "VEC": 2, "ms": 0.1846, "gbs": 1454},
                 {"THREADS": 512, "VEC": 1, "ms": 0.2299, "gbs": 1168}],
    "model_limit": "只在 H=4096 一个形状上验证过；换形状要重测，不能照搬。",
}


def choose_params(H, threads_pref=(256, 128, 512, 64)):
    """给定 H，选一组 (THREADS, WARPS, VEC)。返回 dict 或抛 ForgeStepError。

    排序依据见 PARAM_BASIS：**VEC=2 优先**（每线程 2 个 uint4 提升访存并行度），
    256 线程优先于 512（512 线程的 CTA 占更多调度槽，实测慢 24%）。
    这只是起点，不是最优值 —— 造出来之后的 sweep 才是
    （`advance` 的 EV 排序会给不同参数组合各一次机会）。
    """
    for t in threads_pref:
        v = expected_vec(H, t)
        if v is not None:
            return {"THREADS": t, "WARPS": t // 64, "VEC": v}
    raise ForgeStepError(
        "H=%d 无法匹配任何 (THREADS,VEC)：需要 H == THREADS*8*VEC，"
        "THREADS∈%s、VEC≤8。**不提供兜底**——兜底会产出一个算错的变体。"
        % (H, list(threads_pref)))


def generate(template, params, out_hip):
    """生成 .hip 源码。返回 (path, src)。"""
    out = Path(out_hip)
    out.parent.mkdir(parents=True, exist_ok=True)
    spec = ",".join("%s=%s" % (k, v) for k, v in params.items())
    rc, msg = _run([sys.executable, "-I", str(SCAFFOLD), "--template", template,
                    "--lang", "hip", "--params", spec, "--out", str(out)])
    if rc != 0 or not out.is_file():
        raise ForgeStepError("scaffold 生成失败：%s" % msg[-600:])
    return out, out.read_text(encoding="utf-8")


def compile_so(hip_path, so_path, params, arch="gfx938"):
    """编译成 .so。参数**必须**同时以 -D 传入，源码里的 @X@ 已展开但宏仍可被覆盖。"""
    so = Path(so_path)
    so.parent.mkdir(parents=True, exist_ok=True)
    cmd = ["hipcc", "-O3", "--offload-arch=%s" % arch, "-fPIC", "-shared",
           "-o", str(so), str(hip_path)]
    for k, v in params.items():
        cmd += ["-D%s=%s" % (k, v)]
    rc, msg = _run(cmd)
    if rc != 0 or not so.is_file():
        raise ForgeStepError("hipcc 失败：%s" % msg[-900:])
    return so


def inspect(so_path, threads=None):
    """读编译产物资源画像。返回 dict（vgpr/private/threads/lds/...）。"""
    cmd = [sys.executable, "-I", str(INSPECT_SO), str(so_path), "--json"]
    if threads is not None:
        cmd += ["--threads", str(threads)]
    rc, msg = _run(cmd, timeout=120)
    if rc != 0:
        raise ForgeStepError("inspect_so 失败：%s" % msg[-600:])
    try:
        data = json.loads(msg)
    except ValueError:
        raise ForgeStepError("inspect_so 输出不是 JSON：%s" % msg[:400])
    ks = data.get("kernels") or []
    if not ks:
        raise ForgeStepError("产物里没有 kernel metadata（不是合法 kernel .so？）")
    return ks[0]


def health_check(prof, expect_threads=None):
    """三条必过判据。返回 (ok, [问题...])。**问题必须非空才 ok=False**。"""
    probs = []
    priv = int(prof.get("private") or 0)
    if priv > MAX_PRIVATE_BYTES:
        probs.append(
            "private=%dB > %dB：寄存器数组溢出到私有内存。"
            "实测后果是慢 3.3 倍且**不报错**，会把对的改法判成错的。"
            "常见原因：数组索引用了运行期变量（如 `for j < vec` 而非 `j < VEC`）。"
            % (priv, MAX_PRIVATE_BYTES))
    vgpr = int(prof.get("vgpr") or 0)
    if vgpr > MAX_VGPR:
        probs.append("vgpr=%d > %d：溢出风险" % (vgpr, MAX_VGPR))
    if expect_threads is not None:
        got = int(prof.get("threads") or 0)
        if got != expect_threads:
            probs.append(
                "threads=%d != 期望 %d：`__launch_bounds__(THREADS)` 没生效。"
                "只改 host 侧 `const int threads` **不会**进 kernel metadata。"
                % (got, expect_threads))
    return (not probs), probs


_ADAPTER = '''#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""生成变体 `{name}` —— 由 scripts/forge_step.py 产出，不是手写代码。

模板：{template}
参数：{params}
来源：{hip}

**这个文件是产物**（在 runs/forged/ 下，git 忽略）。
它的成败由 loop.py 的六道门判定，不由写它的人自评。
"""
import ctypes
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
SO = HERE / "_kernels" / "{soname}"
_LIB = None


def _lib():
    global _LIB
    if _LIB is None:
        lib = ctypes.CDLL(str(SO))
        lib.{entry}.argtypes = [
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
            ctypes.c_void_p, ctypes.c_void_p,
            ctypes.c_int, ctypes.c_int, ctypes.c_float]
        lib.{entry}.restype = None
        _LIB = lib
    return _LIB


def run(x, w, r, **kw):
    M, H = x.shape
    want = {threads} * 8 * {vec}
    if H != want:
        raise ValueError(
            "变体 {name} 是按 H=%d 编译的（THREADS*8*VEC），收到 H=%d。"
            "换形状必须重新生成——不提供运行期兜底，兜底会算错。" % (want, H))
    y = torch.empty_like(x)
    ro = torch.empty_like(r)
    _lib().{entry}(x.data_ptr(), r.data_ptr(), w.data_ptr(),
                   y.data_ptr(), ro.data_ptr(), M, H,
                   float(kw.get("eps", 1e-6)))
    return y, ro
'''


def write_adapter(name, template, params, hip_rel, so_name,
                  entry="fused_rms_launch", forge_dir=None):
    """写变体 adapter。**只在体检通过后调用**。"""
    d = Path(forge_dir or (REPO / "runs" / "forged"))
    d.mkdir(parents=True, exist_ok=True)
    f = d / ("%s.py" % name)
    f.write_text(_ADAPTER.format(
        name=name, template=template,
        params=json.dumps(params, ensure_ascii=False),
        hip=hip_rel, soname=so_name, entry=entry,
        threads=params.get("THREADS"), vec=params.get("VEC")),
        encoding="utf-8")
    return f


def forge_variant(name, template, params, H, outdir, entry="fused_rms_launch",
                  arch="gfx938", adapters_dir=None, card=None):
    """完整一步：生成 → 编译 → 体检 → 写 adapter。

    体检不过就抛，**并且删掉半成品**（不留一个会被误加载的 .so）。
    返回 dict：可被 advance 写进报告的证据。

    **目录必须是同一个**：`_common.get_runner` 只认 `<卡>/runs/forged/*.py`，
    而 adapter 里的 .so 路径是相对于它自己算的（`HERE/_kernels/`）。
    第一次实现把 .so 放 /tmp、adapter 放仓库根 —— 变体生成成功但加载不到，
    且报错是「未知变体」，完全看不出是路径问题。传 card= 就两者都落在卡里。
    """
    if card is not None:
        # outdir 是 .hip 的落点；.so 恒在 adapters_dir/_kernels 下
        adapters_dir = Path(card) / "runs" / "forged"
        outdir = adapters_dir
    out = Path(outdir)
    so_dir = (Path(adapters_dir) if adapters_dir else out) / "_kernels"
    hip = out / ("%s.hip" % name)
    so = so_dir / ("%s.so" % name)
    try:
        generate(template, params, hip)
        compile_so(hip, so, params, arch=arch)
        prof = inspect(so, threads=params.get("THREADS"))
        ok, probs = health_check(prof, expect_threads=params.get("THREADS"))
        if not ok:
            raise ForgeStepError("体检不通过：\n  - " + "\n  - ".join(probs))
        adapter = write_adapter(
            name, template, params,
            hip_rel=str(hip.relative_to(REPO)) if hip.is_relative_to(REPO) else str(hip),
            so_name=so.name, entry=entry, forge_dir=adapters_dir)
    except Exception:
        # 半成品必须清掉：一个没通过体检的 .so 若留在 runs/forged/_kernels 里，
        # 同名变体下次会被 _common 直接加载 —— 体检就白做了。
        for p in (so,):
            try:
                p.unlink()
            except OSError:
                pass
        raise
    return {"name": name, "template": template, "params": params, "H": H,
            "hip": str(hip), "so": str(so), "adapter": str(adapter),
            "profile": {k: prof.get(k) for k in
                        ("vgpr", "sgpr", "private", "threads", "lds_used",
                         "vgpr_spill", "sgpr_spill")},
            "health": "ok"}


def selftest():
    """无 GPU 也要能跑的部分：参数选择与体检判据。"""
    # 4096 必须选实测最优档（VEC=2 优先），不是「第一个能整除」
    assert choose_params(4096) == {"THREADS": 256, "WARPS": 4, "VEC": 2}, choose_params(4096)
    assert choose_params(8192) == {"THREADS": 256, "WARPS": 4, "VEC": 4}
    try:
        choose_params(100)
    except ForgeStepError:
        pass
    else:
        raise AssertionError("H=100 应当报错，不许兜底")

    # 反向：每个体检判据都必须**真的会报错**
    ok, probs = health_check({"private": 528, "vgpr": 40, "threads": 256}, 256)
    assert not ok and "私有内存" in probs[0], probs
    ok, probs = health_check({"private": 0, "vgpr": 40, "threads": 128}, 256)
    assert not ok and "launch_bounds" in probs[0], probs
    ok, probs = health_check({"private": 0, "vgpr": 40, "threads": 256}, 256)
    assert ok and not probs, probs
    print("[ok] 参数选择 + 三条体检判据（含反向）")
    return 0


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--template", default="fused-add-reduce")
    ap.add_argument("--name", help="变体名")
    ap.add_argument("--H", type=int, help="目标形状的 H（用来定 THREADS/VEC）")
    ap.add_argument("--params", help="显式覆盖，如 THREADS=256,VEC=2")
    ap.add_argument("--outdir", default=str(REPO / "runs" / "forged" / "_kernels"))
    a = ap.parse_args(argv)
    if a.selftest:
        return selftest()
    if not a.name or not a.H:
        ap.error("--name 与 --H 必填")
    if a.params:
        params = dict(kv.split("=") for kv in a.params.split(","))
        params = {k: int(v) for k, v in params.items()}
    else:
        params = choose_params(a.H)
    r = forge_variant(a.name, a.template, params, a.H, a.outdir)
    print(json.dumps(r, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
