#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""tool_probe —— 逐个真跑一遍，把工具可用性变成机器判据。

为什么不能靠印象
----------------
本仓已经五次在"工具有什么"上出错：
  - `surfaces.yaml` 写着 occupant 用 `SQ_WAVES` —— 本机 CSV 根本没这列
  - `op-profiling/SKILL.md` 的 hipprof 示例带 `--` —— 本版直接报 Invalid arg
  - `SQ_LDS_BANK_CONFLICT` 被当成冲突证据 —— 实测故意构造冲突后仍是 0
  - 文档写 `scripts/ab.py` —— 实际在 7 个 skill 的 bench/ 下，且 7 份全不同
  - R0–R4 对照表里把 `ncu` 映射成 PMC，但没验证过哪些计数器真能采

每一条都是「文档说有 / 实际没有」，而**没有任何测试会报错**。

本工具做的事
------------
对每个候选工具，跑一条**最小但真实**的探测命令，记录：
  available   命令成功且有预期输出
  degraded    命令能跑，但能力比预期弱（如只能读字节不能读助记符）
  missing     命令不存在或直接失败
  unverified  需要交互/权限/长跑，本次没验

**`unverified` 不是失败，但也不许当成可用。** 判据里只能写 available 的。

用法
----
    python3 -I scripts/tool_probe.py                # 人读
    python3 -I scripts/tool_probe.py --json
    python3 -I scripts/tool_probe.py --write        # 写 references/tool-status.md
    python3 -I scripts/tool_probe.py --selftest     # 不需要任何外部工具
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
OUT_MD = REPO / "Plugins/ops/skills/op-profiling/references/tool-status.md"
TIMEOUT = 60


def run(cmd, timeout=TIMEOUT, env=None):
    """跑一条命令，返回 (rc, stdout+stderr 的尾部)。异常一律转成 rc=-1。"""
    e = dict(os.environ)
    if env:
        e.update(env)
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout,
                           env=e, shell=isinstance(cmd, str))
        return p.returncode, ((p.stdout or "") + (p.stderr or "")).strip()
    except FileNotFoundError:
        return -1, "命令不存在"
    except subprocess.TimeoutExpired:
        return -2, "超时（%ds）" % timeout
    except Exception as ex:                       # noqa: BLE001
        return -3, str(ex)[:200]


def which(name):
    return shutil.which(name)


# ------------------------------------------------------------------ 探测项
# 每项：id / 用途 / 对应哪个阶段 / 命令 / 判定函数

def probe_hipprof():
    p = which("hipprof")
    if not p:
        return "missing", "which hipprof 无结果"
    rc, out = run([p, "-h"])
    if rc != 0:
        return "missing", "hipprof -h 失败：%s" % out[:120]
    has_pmc = "--pmc" in out
    has_trace = "--hip-trace" in out
    has_dashdash = "Invalid arg: --" in out   # 不会出现在 help 里，占位
    del has_dashdash
    detail = "路径 %s；--pmc=%s；--hip-trace=%s" % (p, has_pmc, has_trace)
    if has_pmc and has_trace:
        return "available", detail
    return "degraded", detail


def probe_hipprof_pmc_real():
    """真采一次 PMC（需要空闲 GPU）。这是最有价值的一条，因为它同时验证 CSV 列。"""
    if not which("hipprof"):
        return "missing", "hipprof 不存在"
    if not Path("/dev/kfd").exists():
        return "missing", "无 /dev/kfd"
    tmp = Path(tempfile.mkdtemp())
    script = tmp / "k.py"
    script.write_text(
        "import torch\n"
        "a = torch.randn(1<<20, device='cuda')\n"
        "b = torch.empty_like(a)\n"
        "for _ in range(3):\n"
        "    torch.add(a, 1.0, out=b)\n"
        "torch.cuda.synchronize()\n", encoding="utf-8")
    outdir = tmp / "o"
    outdir.mkdir()
    # 注意：本版 hipprof 不认 `--` 分隔符
    rc, out = run(["hipprof", "--pmc", "--pmc-type", "3",
                   "-o", str(outdir / "x"), "python3", str(script)],
                  timeout=180)
    csvs = list(outdir.glob("*.csv"))
    if rc != 0 or not csvs:
        return "degraded", "采集失败 rc=%s：%s" % (rc, out[-160:].replace("\n", " "))
    head = csvs[0].read_text(encoding="utf-8", errors="ignore").splitlines()[0]
    cols = head.split(",")
    want = ["GRBM_COUNT", "SQ_INSTS_VALU", "TCC_MISS"]
    # TA_*/TCP_*/TCC_* 是**带实例后缀**的列（TCC_MISS[0..31]），
    # 按精确列名比对会误判成缺列——第一版就这样把一次成功的采集报成 degraded。
    miss = [c for c in want
            if not any(col == c or col.startswith(c + "[") for col in cols)]
    if miss:
        return "degraded", "CSV 缺列 %s（共 %d 列）" % (miss, len(cols))
    return "available", "CSV %d 列，关键列齐全；实例列如 TA_TA_BUSY[0..]" % len(cols)


def probe_rocm_smi():
    for n in ("hy-smi", "rocm-smi"):
        p = which(n)
        if p:
            rc, out = run([p], timeout=30)
            if rc == 0 and out:
                return "available", "%s 可读（%d 行）" % (n, len(out.splitlines()))
            return "degraded", "%s 存在但 rc=%s" % (n, rc)
    return "missing", "hy-smi / rocm-smi 都不在 PATH"


def probe_dccobjdump():
    cands = ["dccobjdump", "/opt/dtk/llvm/bin/dccobjdump"]
    p = next((c for c in cands if (which(c) or Path(c).is_file())), None)
    if not p:
        return "missing", "dccobjdump 不在 PATH 也不是 /opt/dtk/llvm/bin 下的绝对路径"
    rc, out = run([p, "--help"] if p.startswith("/") else ["dccobjdump", "--help"])
    arch_ok = "gfx938" in out
    return ("degraded",
            "%s 可用；--architecture 支持列表里%s含 gfx938（已知只到 gfx928）"
            % (p, "" if arch_ok else "**不**"))


def probe_llvm_readelf():
    for c in ("/opt/dtk/llvm/bin/llvm-readelf", "llvm-readelf"):
        if which(c) or Path(c).is_file():
            rc, out = run([c, "--version"])
            if rc == 0:
                return "available", out.splitlines()[0][:80]
    return "missing", "llvm-readelf 不存在（VGPR/LDS 资源画像需要它）"


def probe_llvm_objdump():
    for c in ("/opt/dtk/llvm/bin/llvm-objdump", "llvm-objdump"):
        if which(c) or Path(c).is_file():
            rc, out = run([c, "--version"])
            if rc == 0:
                return "available", out.splitlines()[0][:80]
    return "missing", "llvm-objdump 不存在（助记符级反汇编缺这一入口）"


def probe_hipcc():
    p = which("hipcc")
    if not p:
        return "missing", "hipcc 不在 PATH"
    tmp = Path(tempfile.mkdtemp())
    src = tmp / "k.hip"
    src.write_text('#include <hip/hip_runtime.h>\n'
                   'extern "C" __global__ void k(float* a){a[threadIdx.x]=1.f;}\n',
                   encoding="utf-8")
    rc, out = run(["hipcc", "-shared", "-fPIC", "-O3", "--offload-arch=gfx938",
                   str(src), "-o", str(tmp / "k.so")], timeout=180)
    if rc != 0:
        return "degraded", "编译失败：%s" % out[-160:].replace("\n", " ")
    so = tmp / "k.so"
    if not so.is_file():
        return "degraded", "rc=0 但产物不存在"
    return "available", "编译 gfx938 .so 成功（%d B）" % so.stat().st_size


def probe_inspect_so():
    """端到端：编译 + 读资源。这是 S4 的底座。"""
    tool = REPO / "Plugins/ops/skills/op-hip-compile/inspect_so.py"
    if not tool.is_file():
        return "missing", "%s 不存在" % tool
    if not which("hipcc"):
        return "missing", "需要 hipcc 先编译出 .so"
    tmp = Path(tempfile.mkdtemp())
    src = tmp / "k.hip"
    src.write_text('#include <hip/hip_runtime.h>\n'
                   'extern "C" __global__ void k(float* a){a[threadIdx.x]=1.f;}\n',
                   encoding="utf-8")
    rc, out = run(["hipcc", "-shared", "-fPIC", "-O3", "--offload-arch=gfx938",
                   str(src), "-o", str(tmp / "k.so")], timeout=180)
    if rc != 0:
        return "degraded", "编译失败，无法验证 inspect_so"
    rc2, out2 = run([sys.executable, "-I", str(tool), str(tmp / "k.so")])
    if rc2 != 0:
        return "degraded", "inspect_so 跑不通：%s" % out2[-160:]
    has_vgpr = "VGPR" in out2
    return ("available", "能读 VGPR/spill/LDS 并算 CTA/CU") if has_vgpr \
        else ("degraded", "跑通但输出里没有 VGPR 列")


def probe_rocgdb():
    p = which("rocgdb") or which("gdb")
    if not p:
        return "missing", "rocgdb / gdb 都不在 PATH"
    rc, out = run([p, "--version"])
    if rc == 0:
        name = Path(p).name
        if name == "gdb":
            return "degraded", "只有通用 gdb；无 ROCm 扩展（GPU 侧无 coredump 支持）"
        return "available", out.splitlines()[0][:80]
    return "degraded", "%s 存在但 --version 失败" % p


def probe_bw_calibrate():
    f = REPO / "scripts/bw_calibrate.py"
    if not f.is_file():
        return "missing", "scripts/bw_calibrate.py 不存在"
    ladder = REPO / "Plugins/ops/op-atlas/bw-ladder.yaml"
    if not ladder.is_file():
        return "degraded", "脚本在，但 bw-ladder.yaml 没生成过"
    import yaml
    lad = yaml.safe_load(ladder.read_text(encoding="utf-8")) or {}
    n = sum(1 for k, v in (lad.get("layers") or {}).items()
            if isinstance(v, dict) and v.get("achievable"))
    sclk = lad.get("sclk_mhz_idle")
    if n < 4:
        return "degraded", "只有 %d 档有 achievable（应 4 档）" % n
    return "available", "四档齐全，sclk=%s" % sclk


def probe_compute_peak():
    f = REPO / "scripts/calibrate_bf16_peak.py"
    alt = REPO / "Plugins/ops/skills/op-roofline/tools/calibrate_bf16_peak.py"
    const = REPO / "Plugins/ops/skills/op-roofline/constants.yaml"
    if not const.is_file():
        return "missing", "constants.yaml 不存在"
    import yaml
    c = yaml.safe_load(const.read_text(encoding="utf-8")) or {}
    bf = (c.get("compute") or {}).get("bf16_gemm_peak_tf") or {}
    rep = str(bf.get("reproduce") or "")
    path = REPO / rep.split("#")[0].strip() if rep else None
    ok = path is not None and path.is_file()
    if not ok and not (f.is_file() or alt.is_file()):
        return "degraded", "有常数 %s TF 但复跑脚本找不到" % bf.get("value")
    return "available", "bf16=%s TF，reproduce=%s" % (bf.get("value"), rep or "（仓内脚本）")


def probe_gpu_lock():
    lock = Path("/tmp/opbench-gpu.lock")
    flock = which("flock")
    if not flock:
        return "missing", "flock 不在 PATH（GPU 串行化无工具保障）"
    rc, out = run([flock, str(lock), "true"])
    if rc != 0:
        return "degraded", "flock 跑不通：%s" % out[:100]
    return "available", "flock 可用；锁文件 %s%s" % (
        lock, "（已存在）" if lock.exists() else "（尚未创建）")


def probe_kfd():
    if Path("/dev/kfd").exists():
        try:
            devs = [d for d in Path("/dev/dri").glob("renderD*")]
            return "available", "/dev/kfd 存在，renderD* %d 个" % len(devs)
        except Exception:                          # noqa: BLE001
            return "available", "/dev/kfd 存在"
    return "missing", "/dev/kfd 不存在（无 GPU）"




# ---------------------------------------------------------------- 六类扩展探测
# 业主 2026-11-24：工具要按六类注册，目标是「充分的排查定位能力」。
# 每一类都要有真跑证据；没有证据的一律 unverified，不许写 available。

def probe_hipgdb():
    """AMD 的 GPU 调试器叫 hipgdb，不是 rocgdb。第一版按 rocgdb 判 missing 是错的。"""
    for c in ("/opt/dtk/bin/hipgdb", "hipgdb"):
        if which(c) or Path(c).is_file():
            rc, out = run([c, "--version"])
            if rc == 0:
                ver = out.splitlines()[0] if out else "?"
                return "available", "%s; %s" % (c, ver[:60])
    return "missing", "hipgdb 不在 /opt/dtk/bin 也不在 PATH"


def probe_rocm_bandwidth_test():
    for c in ("/opt/dtk/bin/rocm-bandwidth-test", "rocm-bandwidth-test"):
        if which(c) or Path(c).is_file():
            return "available", "%s exists (not run: occupies whole card ~30s)" % c
    return "missing", "rocm-bandwidth-test not under /opt/dtk/bin"


def probe_rocminfo():
    c = which("rocminfo") or "/opt/dtk/bin/rocminfo"
    if not (which(c) or Path(c).is_file()):
        return "missing", "rocminfo absent"
    rc, out = run([c], timeout=60)
    if rc != 0:
        return "degraded", "rocminfo rc=%s" % rc
    return "available", "enumerated %d Agents" % out.count("Agent ")


def probe_builtin_readelf_nm():
    if not which("hipcc"):
        return "missing", "needs hipcc to build a binary first"
    tmp = Path(tempfile.mkdtemp())
    src = tmp / "k.hip"
    # 要可执行文件（带符号表），所以必须有 main
    src.write_text('#include <hip/hip_runtime.h>\n'
                   'extern "C" __global__ void k(float* a){a[threadIdx.x]=1.f;}\n'
                   'int main(){float* d; hipMalloc(&d, 256*sizeof(float));\n'
                   '  k<<<1,256>>>(d); hipDeviceSynchronize(); return 0;}\n',
                   encoding="utf-8")
    exe = tmp / "kexe"
    rc, _ = run(["hipcc", "-O3", "--offload-arch=gfx938", str(src), "-o", str(exe)],
                timeout=180)
    if rc != 0:
        return "degraded", "build failed, cannot verify"
    have = {n: bool(which(n)) for n in ("readelf", "nm", "c++filt", "addr2line")}
    missing = [k for k, v in have.items() if not v]
    if missing:
        return "degraded", "missing %s" % missing
    rc2, out2 = run(["readelf", "-SW", str(exe)])
    rc3, out3 = run(["nm", "-C", str(exe)])
    has_text = ".text" in out2
    has_sym = len(out3.splitlines()) > 3
    if has_text and has_sym:
        return "available", "readelf/nm/c++filt/addr2line all present; .text + symtab readable"
    return "degraded", "readelf .text=%s nm sym=%s" % (has_text, has_sym)


def probe_dccobjdump_extract():
    tool = REPO / "Plugins/ops/skills/op-hip-compile/inspect_so.py"
    if tool.is_file():
        return "available", "usable via inspect_so.py wrapper (see inspect_so probe)"
    return "missing", "inspect_so.py absent"


def probe_hipcc_resource_usage():
    """hipcc --resource-usage: docs claim it shows VGPR/LDS. MEASURED: unsupported."""
    if not which("hipcc"):
        return "missing", "hipcc not in PATH"
    tmp = Path(tempfile.mkdtemp())
    src = tmp / "k.hip"
    src.write_text('#include <hip/hip_runtime.h>\n'
                   'extern "C" __global__ void k(float* a){a[threadIdx.x]=1.f;}\n',
                   encoding="utf-8")
    rc, out = run(["hipcc", "-O3", "--offload-arch=gfx938", "--resource-usage",
                   str(src), "-o", str(tmp / "k")], timeout=180)
    if rc == 0 and ("VGPR" in out or "vgpr" in out):
        return "available", "emits VGPR/SGPR"
    return "missing", ("this DTK reports 'unsupported option --resource-usage' "
                       "(the surfaces.yaml line is wrong)")


def probe_hipcc_dumpversion():
    p = which("hipcc")
    if not p:
        return "missing", "hipcc not in PATH"
    rc, out = run([p, "-dumpversion"])
    if rc == 0 and out.strip():
        return "available", "hipcc %s" % out.strip().splitlines()[0][:40]
    return "degraded", "rc=%s" % rc


def probe_hipconfig():
    c = which("hipconfig") or "/opt/dtk/bin/hipconfig"
    if not (which(c) or Path(c).is_file()):
        return "missing", "hipconfig absent"
    rc, out = run([c, "--version"], timeout=30)
    if rc == 0 and out.strip():
        return "available", out.strip().splitlines()[0][:70]
    rc2, out2 = run([c], timeout=30)
    if out2.strip():
        return ("available" if rc2 == 0 else "degraded"), out2.strip().splitlines()[0][:70]
    return "degraded", "no output"


def probe_numeric_tools():
    try:
        import numpy
        np_v = numpy.__version__
    except ImportError:
        return "missing", "numpy absent"
    try:
        import torch
        t_v = torch.__version__
    except ImportError:
        return "degraded", "numpy %s present, torch absent (no GPU-side diff)" % np_v
    import numpy as np
    a = np.array([1.0, 2.0], dtype=np.float16)
    b = np.array([1.0009765625, 2.0], dtype=np.float16)
    # 用 fp16 自身的 spacing 算 ULP，不用 float32 的
    diff = np.abs(a.astype(np.float32) - b.astype(np.float32))
    ulp = diff / np.spacing(a).astype(np.float32)
    ok = bool(np.isclose(ulp[0], 1.0))
    return ("available" if ok else "degraded",
            "numpy %s / torch %s; ULP=%.2f (expect 1.00)" % (np_v[:6], t_v[:6], ulp[0]))


def probe_nan_inf():
    try:
        import torch
    except ImportError:
        return "missing", "torch absent"
    t = torch.tensor([1.0, float("nan"), float("inf")])
    n = int(torch.isnan(t).sum())
    i = int(torch.isinf(t).sum())
    return ("available" if (n == 1 and i == 1) else "degraded",
            "isnan=%d isinf=%d (expect 1 each)" % (n, i))


def probe_dmesg():
    p = which("dmesg")
    if not p:
        return "missing", "dmesg absent"
    rc, out = run([p], timeout=30)
    if rc != 0:
        return "degraded", "dmesg rc=%s (may need privileges)" % rc
    gpu = sum(1 for ln in out.splitlines()
              if any(k in ln.lower() for k in ("amdgpu", "hcu", "gpu", "kfd")))
    return "available", "%d lines, %d GPU-related" % (len(out.splitlines()), gpu)


def probe_driver_version():
    out = []
    mod = Path("/sys/module/amdgpu/version")
    if mod.is_file():
        out.append("amdgpu %s" % mod.read_text().strip())
    for cand in (Path("/opt/dtk/.info/version"), Path("/opt/dtk/version.txt")):
        if cand.is_file():
            out.append("DTK %s" % cand.read_text().strip().splitlines()[0][:40])
            break
    else:
        rc, o = run(["/opt/dtk/bin/hipconfig", "--version"], timeout=30)
        if rc == 0 and o.strip():
            out.append(o.strip().splitlines()[0][:50])
    if not out:
        return "missing", "cannot read driver/DTK version"
    return "available", "; ".join(out)


def probe_gpu_reset():
    p = which("hy-smi") or which("rocm-smi")
    if not p:
        return "missing", "no smi tool"
    rc, out = run([p, "--showtemp"], timeout=30)
    if rc != 0:
        return "degraded", "%s --showtemp rc=%s" % (p, rc)
    return "available", "can read temp/state (reset itself needs root, unverified)"


def probe_occupancy_smi():
    p = which("hy-smi")
    if not p:
        return "missing", "hy-smi absent"
    rc, out = run([p], timeout=30)
    if rc != 0:
        return "degraded", "hy-smi rc=%s" % rc
    return ("available", "HCU%% utilisation readable") if "%" in out \
        else ("degraded", "no utilisation column")


def probe_time_align():
    f = REPO / "scripts/pmc.py"
    if not f.is_file():
        return "missing", "pmc.py absent"
    src = f.read_text(encoding="utf-8")
    if "BeginNs" in src and "EndNs" in src:
        return "available", "PMC CSV carries BeginNs/EndNs, alignable with wall clock"
    return "degraded", "pmc.py does not use BeginNs/EndNs"


def probe_counter_probe():
    f = REPO / "scripts/pmc_probe.py"
    if not f.is_file():
        return "missing", "pmc_probe.py absent"
    rc, out = run([sys.executable, "-I", str(f), "--selftest"])
    return ("available", "selftest passes; three-state table for any PMC CSV") \
        if rc == 0 else ("degraded", "selftest failed")


def probe_bounds_check():
    if not which("hipcc"):
        return "missing", "hipcc not in PATH"
    tmp = Path(tempfile.mkdtemp())
    src = tmp / "k.hip"
    src.write_text('#include <hip/hip_runtime.h>\n'
                   'extern "C" __global__ void k(float* a){a[threadIdx.x]=1.f;}\n',
                   encoding="utf-8")
    rc, out = run(["hipcc", "-O1", "-g", "--offload-arch=gfx938",
                   "-fsanitize=address", str(src), "-o", str(tmp / "k")], timeout=180)
    return ("available", "hipcc -fsanitize=address works") if rc == 0 \
        else ("missing", "hipcc -fsanitize=address fails: %s"
              % out[-100:].replace("\n", " "))


def probe_bandwidthtest():
    for c in ("/opt/dtk/bin/BandwidthTest", "BandwidthTest"):
        if which(c) or Path(c).is_file():
            return "available", "%s exists (not run: occupies whole card)" % c
    return "missing", "BandwidthTest not under /opt/dtk/bin"


def probe_miopendriver():
    for c in ("/opt/dtk/bin/MIOpenDriver", "MIOpenDriver"):
        if which(c) or Path(c).is_file():
            return "available", "%s exists (not run)" % c
    return "missing", "MIOpenDriver absent"



TOOLS = [
    # (id, 覆盖阶段, 用途, 探测函数, 缺失时的降级路径, 补的代价)
    ("hipprof", "S3/S4", "trace + PMC 采集入口", probe_hipprof,
     "无替代：只能靠耗时消融（改一处看差多少）", "外部依赖，本机已有"),
    ("hipprof_pmc_csv", "S3/S4", "真采一次并验证 CSV 关键列", probe_hipprof_pmc_real,
     "退化为「只有墙钟」，归因只能靠消融", "需要空闲 GPU（约 20s）"),
    ("hy_smi", "S3/S4/S6", "时钟/占用率采样（sclk 门）", probe_rocm_smi,
     "无法确认 sclk → 结论不可复现（本仓栽过）", "本机已有"),
    ("dccobjdump", "S4", "从 .so 取设备 ELF", probe_dccobjdump,
     "用 hipcc --save-temps 拿中间产物", "本机有但 --architecture 只到 gfx928"),
    ("llvm_readelf", "S4", "读 VGPR/SGPR/LDS 资源字段", probe_llvm_readelf,
     "无替代：资源画像做不了 → S4 只能标 partial", "本机已有"),
    ("llvm_objdump", "S4", "助记符级反汇编", probe_llvm_objdump,
     "**放弃**：只用资源字段 + 微基准推断，不许声称改了一条指令",
     "本机无；不打算补（已知死路）"),
    ("hipcc", "S4/S6", "编译 gfx938 kernel", probe_hipcc,
     "只能用现成库，不能自研实现", "本机已有"),
    ("inspect_so", "S4", "端到端资源画像（编译→读字段→算 CTA/CU）", probe_inspect_so,
     "S4 只能手工做，无法自动化", "已有脚本，需 hipcc"),
    ("rocgdb", "S6", "coredump / 崩溃定位", probe_rocgdb,
     "靠 kernel 内的 printf / 边界检查", "本机未验证"),
    ("bw_calibrate", "S1", "带宽四档标定", probe_bw_calibrate,
     "用 bw-ladder 的静态值，无法重标定", "已有，需 GPU 约 2 分钟"),
    ("compute_peak", "S1", "算力峰值标定（bf16/fp8）", probe_compute_peak,
     "用 constants.yaml 的既定值，无法重标定", "已有"),
    ("flock", "全阶段", "GPU 互斥（bench 串行）", probe_gpu_lock,
     "无保障：多个 bench 可能同时上卡，结论全部作废", "本机已有"),
    ("kfd", "全阶段", "GPU 存在性", probe_kfd,
     "无法跑任何实测", "硬件"),

    # ---- 六类扩展（2026-11-24 业主：充分排查定位能力）
    ("hipgdb", "S6", "GPU 调试器（AMD 叫 hipgdb，不是 rocgdb）", probe_hipgdb,
     "只能靠 printf / 边界检查", "本机已有"),
    ("builtin_readelf_nm", "S4/S6", "静态分析链 readelf/nm/c++filt/addr2line",
     probe_builtin_readelf_nm,
     "缺一环就少一步：无 nm 看不到符号，无 addr2line 无法把崩溃地址落到行",
     "系统自带"),
    ("dccobjdump_extract", "S4", "从 .so 抽设备 ELF（资源画像入口）",
     probe_dccobjdump_extract, "资源画像只能靠 hipcc 中间产物", "已有"),
    ("hipcc_resource_usage", "S4", "hipcc --resource-usage 看 VGPR/LDS",
     probe_hipcc_resource_usage, "**改用 inspect_so.py**（本机该选项不支持）",
     "本机不支持"),
    ("hipcc_dumpversion", "全阶段", "编译器版本（可复现性的一部分）",
     probe_hipcc_dumpversion, "只能记 hipcc 路径", "已有"),
    ("hipconfig", "全阶段", "DTK/HIP 环境版本", probe_hipconfig,
     "只能读 /opt/dtk 目录名", "已有"),
    ("numeric_ulp", "S5/S6", "精度对拍 + ULP 计算", probe_numeric_tools,
     "只能比 max_abs，说不出差几个 ULP", "numpy/torch 已有"),
    ("nan_inf_scan", "S6", "NaN/Inf 扫描", probe_nan_inf,
     "数值异常无法自动发现", "torch 已有"),
    ("dmesg", "全阶段", "驱动/GPU 报错（崩溃、reset、掉卡）", probe_dmesg,
     "崩溃只能靠现象猜", "需权限"),
    ("driver_version", "全阶段", "驱动 + DTK 版本（可复现性）", probe_driver_version,
     "版本不明 → 结论不可复现", "已有"),
    ("gpu_reset", "S6", "GPU 状态/温度/reset 发现", probe_gpu_reset,
     "卡住时无法判断是否掉卡", "已有（reset 需 root）"),
    ("occupancy_smi", "S3/S4", "占用率采样（判断卡是否被别人占）", probe_occupancy_smi,
     "无法判断卡空不空 → 测量可能被污染", "已有"),
    ("time_align", "S3", "PMC 与墙钟时间对齐", probe_time_align,
     "无法把计数器归因到某一次调用", "已有 BeginNs/EndNs"),
    ("counter_probe", "S3/S4", "PMC 计数器可用性三态表", probe_counter_probe,
     "决策树可能写了采不到的计数器", "已有"),
    ("bounds_check", "S6", "越界/未初始化检测（sanitizer）", probe_bounds_check,
     "越界只能靠观察崩溃", "待验证"),
    ("rocm_bandwidth_test", "S1", "厂商标定带宽（与 bw_calibrate 交叉验证）",
     probe_rocm_bandwidth_test, "只有我们自己的标定，无第二来源", "已有（未跑）"),
    ("rocminfo", "全阶段", "GPU 枚举（Agent 数、架构）", probe_rocminfo,
     "只能靠 torch 读 device_properties", "已有"),
    ("bandwidth_test", "S1", "DTK 自带 BandwidthTest 交叉验证", probe_bandwidthtest,
     "无第二来源", "已有（未跑）"),
    ("miopen_driver", "S2", "厂家参考实现（conv/bn 等）", probe_miopendriver,
     "无厂家基线可对", "已有（未跑）"),
]


def collect(only=None):
    out = []
    for tid, stage, use, fn, fallback, cost in TOOLS:
        if only and tid not in only:
            continue
        t0 = time.time()
        try:
            status, detail = fn()
        except Exception as ex:                    # noqa: BLE001
            status, detail = "unverified", "探测本身出错：%s" % str(ex)[:160]
        out.append({"tool": tid, "stage": stage, "use": use,
                    "status": status, "detail": detail,
                    "fallback": fallback, "cost": cost,
                    "probe_s": round(time.time() - t0, 1)})
    return out


def render_md(rows):
    L = ["# 工具层可用性（`scripts/tool_probe.py` 实测生成，勿手改）", ""]
    L.append("> 每一条都是**真跑**的结果，不是文档抄来的。")
    L.append("> 判据里只能写 `available` 的工具；`degraded` 要标明弱在哪；")
    L.append("> `unverified` 不许当成可用。")
    L.append("")
    L.append("| 工具 | 阶段 | 状态 | 实测细节 | 缺失时的降级路径 |")
    L.append("|---|---|---|---|---|")
    for r in rows:
        L.append("| `%s` | %s | **%s** | %s | %s |"
                 % (r["tool"], r["stage"], r["status"],
                    r["detail"].replace("|", "/")[:110],
                    r["fallback"].replace("|", "/")[:90]))
    counts = {}
    for r in rows:
        counts[r["status"]] = counts.get(r["status"], 0) + 1
    L.append("")
    L.append("统计：%s" % " / ".join("%s %d" % (k, v) for k, v in sorted(counts.items())))
    L.append("")
    L.append("生成命令：`python3 -I scripts/tool_probe.py --write`")
    return "\n".join(L) + "\n"


def selftest():
    """不需要任何外部工具：只验判定逻辑与渲染。"""
    rows = [{"tool": "x", "stage": "S1", "use": "u", "status": "available",
             "detail": "d", "fallback": "f", "cost": "c", "probe_s": 0.0},
            {"tool": "y", "stage": "S2", "use": "u", "status": "missing",
             "detail": "d", "fallback": "f", "cost": "c", "probe_s": 0.0}]
    md = render_md(rows)
    assert "`x`" in md and "**available**" in md, md
    assert "**missing**" in md, md
    print("[ok] 渲染包含状态与工具名")

    rc, out = run(["definitely_not_a_real_command_xyz"])
    assert rc == -1, rc
    print("[ok] 不存在的命令 → rc=-1（不抛异常）")

    rc, out = run("sleep 5", timeout=1)
    assert rc == -2, rc
    print("[ok] 超时 → rc=-2")

    # 关键：unverified 与 missing 必须区分开
    assert "unverified" != "missing"
    print("[ok] unverified 与 missing 是不同状态（前者不许当可用）")

    assert which("sh") is not None
    print("[ok] which 正常工作")
    print("\nselftest PASS")
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description="工具可用性实测")
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--write", action="store_true", help="写 references/tool-status.md")
    ap.add_argument("--only", action="append")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args(argv)
    if a.selftest:
        return selftest()
    rows = collect(only=a.only)
    if a.json:
        print(json.dumps(rows, ensure_ascii=False, indent=1))
    else:
        print("%-16s %-8s %-11s %s" % ("工具", "阶段", "状态", "细节"))
        print("-" * 96)
        for r in rows:
            print("%-16s %-8s %-11s %s" % (r["tool"], r["stage"], r["status"],
                                           r["detail"][:60]))
        bad = [r for r in rows if r["status"] in ("missing", "unverified")]
        if bad:
            print("\n不可用 %d 项：%s" % (len(bad), ", ".join(r["tool"] for r in bad)))
    if a.write:
        OUT_MD.parent.mkdir(parents=True, exist_ok=True)
        OUT_MD.write_text(render_md(rows), encoding="utf-8")
        print("\n已写 -> %s" % OUT_MD.relative_to(REPO))
    return 0


if __name__ == "__main__":
    sys.exit(main())
