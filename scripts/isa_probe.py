#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""isa_probe —— 让 agent 自己能看汇编，而不是靠人肉查一次。

为什么必须有这个工具
--------------------
v5.1 §四发现：ISA 层不是「没有」，是**被文档写死了**——
`isa-census.md` 把 `library_gfx938`（不存在的目录）当真实路径，
`tool-status.md` 把 `dccobjdump` 标成 degraded（用错参数），
于是 agent 读完判定「反汇编不可用」，永远只走 PMC。
而 PMC 的三个关键计数器采不到（`SQ_LDS_BANK_CONFLICT` 恒 0、`SQ_WAVES` 缺列）。

**诊断链断在 D2→D4**：PMC 说「有 stall」，只有 ISA 说「为什么 stall」。

之前我手工跑了几条 dccobjdump 拿到 188487 B 的 .ISA —— 那是**一次性的数据**，
不是能力。下一个 kernel 来了，agent 还是不会。

本工具把那次手工流程固化成命令：输入任意 `.co` / `.so` / `.hip`，输出结构化诊断。

用法
----
    # 直接分析已有的 .co
    python3 -I scripts/isa_probe.py /opt/dtk/lib/hipblaslt/library_bw_nmz/extop_gfx938.co

    # 分析我们自己的 .so（自动抽内嵌 ELF）
    python3 -I scripts/isa_probe.py build/ship.so --kernel gemm_kernel

    # 从源码编出来再分析
    python3 -I scripts/isa_probe.py k.hip --build --arch gfx938

    # 机器可读
    python3 -I scripts/isa_probe.py <target> --json

纪律（写进输出，防止误读）
--------------------------
**静态计数 ≠ 动态权重。** 本工具回答「代码长什么样」，不回答「跑起来占多少时间」。
GEMM 主循环的 load 每 K-step 都发生，epilogue 的 store 每 tile 才一次——
静态比例 32:9 完全可能对应动态占比 <5%。
正确顺序：`isa_probe` 生成假设 → PMC 定级 → A/B 裁决。
"""
import argparse
import json
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DCC = "/opt/dtk/llvm/bin/dccobjdump"
READELF = "/opt/dtk/llvm/bin/llvm-readelf"

# 这些模式回答「有没有可攻击的形态」。每条 = (名字, 正则, 含义)
PATTERNS = [
    ("vmem_read_128", r"buffer_load_dwordx4\b", "128-bit 全局读（已用满）"),
    ("vmem_read_64", r"buffer_load_dwordx2\b", "64-bit 全局读"),
    ("vmem_read_32", r"buffer_load_dword\b", "32-bit 全局读（未向量化）"),
    ("vmem_write_128", r"buffer_store_dwordx4\b", "128-bit 全局写（已用满）"),
    ("vmem_write_64", r"buffer_store_dwordx2\b", "64-bit 全局写（还有 2× 空间）"),
    ("vmem_write_32", r"buffer_store_dword\b", "32-bit 全局写"),
    ("lsd_matrix", r"ds_read_matrix_trans_format\b", "MLS 矩阵读（走专用通路）"),
    ("lds_b128", r"ds_read_b128\b", "128-bit LDS 读"),
    ("lds_b64", r"ds_read_b(?:64|2?b64)\b", "64-bit LDS 读"),
    ("lds_scalar", r"ds_read_b32\b", "32-bit LDS 读（未向量化）"),
    ("lds_write", r"ds_write_b(?:32|64|128)\b", "LDS 写"),
    ("waitcnt_vm", r"s_waitcnt\s+vmcnt\((\d+)\)", "全局等待深度"),
    ("waitcnt_lgkm", r"s_waitcnt\s+lgkmcnt\((\d+)\)", "LDS 等待深度"),
    ("barrier", r"\bs_barrier\b", "同步屏障"),
    ("branch", r"\bs_cbranch", "条件分支（热循环内 >0 说明未完全展开）"),
    ("mma", r"v_m(?:mac|fma)_[a-z0-9_]+", "矩阵指令（MMA/MFMA）"),
    ("global_load", r"\bglobal_load\b", "用 global_load 而非 buffer_load（通常更慢）"),
    ("slc", r"\bslc\b", "流式缓存修饰符"),
]
MMA_RE = re.compile(r"(v_m(?:mac|fma)_[a-z0-9_]+)")
WAITNUM_RE = re.compile(r"s_waitcnt\s+(vmcnt|lgkmcnt)\((\d+)\)")


def _which(name, *fallbacks):
    for c in (name,) + fallbacks:
        if shutil.which(c) or Path(c).is_file():
            return c
    return None


def find_disassembler():
    for c in (DCC, "dccobjdump", "/opt/dtk/bin/dccobjdump"):
        if shutil.which(c) or Path(c).is_file():
            return c
    return None


def run(cmd, timeout=300):
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return p.returncode, (p.stdout or "") + (p.stderr or "")
    except FileNotFoundError:
        return -1, "命令不存在：%s" % cmd[0]
    except subprocess.TimeoutExpired:
        return -2, "超时"


def disassemble(target, arch="gfx938", workdir=None):
    """把 .co / .so 反汇编成 .ISA 文本。返回 (path_or_None, note)。"""
    dcc = find_disassembler()
    if not dcc:
        return None, "dccobjdump 不存在"
    tgt = Path(target)
    if not tgt.is_file():
        return None, "目标不存在：%s" % target
    wd = Path(workdir or tempfile.mkdtemp())
    wd.mkdir(parents=True, exist_ok=True)

    # dccobjdump **不管 -o**：`--extract-elf` 和反汇编产物都落在 **cwd**。
    # 所以凡是调它，必须先进 wd。原来是 extract-elf 在 chdir 之前跑，
    # 于是 `ship.so-hipv4-*.gfx938.out` 被写进仓库根目录并被误提交 7 个文件
    # （a4374b7）。这是「工具写 cwd，调用方以为是 wd」的典型事故。
    import os
    cwd0 = os.getcwd()
    try:
        os.chdir(str(wd))
        # .so 需要先抽内嵌 ELF；.co 可以直接反汇编
        if tgt.suffix == ".so":
            rc, out = run([dcc, "--extract-elf=all", "--inputs=%s" % tgt])
            elfs = sorted(wd.glob("*.co")) + sorted(wd.glob("*.hsaco"))
            if not elfs:
                # dccobjdump 抽 ELF 会 segfault，但 .ISA 仍可能落盘
                isas = sorted(wd.glob("*.ISA"))
                if isas:
                    return isas[0], "从 .so 提取（dccobjdump 报错但 .ISA 完整）"
                rc, out = run([dcc, "--inputs=%s" % tgt,
                               "--target=%s" % arch, "--recompile"])
            else:
                tgt = elfs[0]
                rc, out = run([dcc, "--inputs=%s" % tgt,
                               "--target=%s" % arch, "--recompile"])
        else:
            rc, out = run([dcc, "--inputs=%s" % tgt,
                           "--target=%s" % arch, "--recompile"])
    finally:
        os.chdir(cwd0)
    isas = sorted(wd.glob("*.ISA"))
    if isas:
        return isas[-1], ("反汇编成功" if rc == 0
                          else "dccobjdump rc=%d 但 .ISA 已落盘（已知 segfault 行为）" % rc)
    return None, "反汇编失败 rc=%s：%s" % (rc, out[-200:])


def build_from_source(src, arch="gfx938", workdir=None):
    """从 .hip/.cpp 编出 .co 再反汇编。返回 .ISA 路径或 None。"""
    hipcc = _which("hipcc", "/opt/dtk/bin/hipcc")
    if not hipcc:
        return None, "hipcc 不存在"
    wd = Path(workdir or tempfile.mkdtemp())
    wd.mkdir(parents=True, exist_ok=True)
    obj = wd / (Path(src).stem + ".hsaco")
    rc, out = run([hipcc, "--save-temps", "--offload-arch=%s" % arch,
                   "--genco", str(src), "-o", str(obj)])
    if rc != 0:
        return None, "编译失败：%s" % out[-200:]
    # hipcc --save-temps 会留下 .s 汇编
    for s in sorted(wd.glob("*.s")):
        return s, "hipcc --save-temps 产出的 .s"
    return disassemble(obj, arch, wd)


def analyze(isa_text):
    """对 ISA 文本做结构化计数。返回 dict。"""
    counts = {}
    for name, pat, _ in PATTERNS:
        counts[name] = len(re.findall(pat, isa_text))

    mma = {}
    for m in MMA_RE.finditer(isa_text):
        mma[m.group(1)] = mma.get(m.group(1), 0) + 1
    wait = {"vmcnt": {}, "lgkmcnt": {}}
    for m in WAITNUM_RE.finditer(isa_text):
        kind = m.group(1)
        d = m.group(2)
        wait[kind][d] = wait[kind].get(d, 0) + 1
    total_inst = len(re.findall(r"^\s+s_[a-z]|^\s+v_[a-z]", isa_text, re.M))
    return {"counts": counts, "mma": mma, "waitcnt": wait,
            "total_instructions": total_inst}


def screen(res):
    """按 4 条 ISA surface 的判据，给出可攻击点。**只给假设，不下结论。**"""
    c = res["counts"]
    hits = []
    # vmem-vector-width
    if c["vmem_write_32"] or c["vmem_write_64"]:
        hits.append({
            "surface": "vmem-vector-width",
            "signal": ("写回向量宽度未用满：dwordx4×%d / dwordx2×%d / dword×%d"
                       % (c["vmem_write_128"], c["vmem_write_64"], c["vmem_write_32"])),
            "hypothesis": "把写回加宽到 dwordx4 可能减少 store 指令数",
            "limit": "静态证据；store 多在 epilogue，动态占比可能 <5%。需 PMC 定级 + A/B",
        })
    # lds-read-form
    if c["lds_scalar"] and not (c["lds_b128"] or c["lsd_matrix"]):
        hits.append({
            "surface": "lds-read-form",
            "signal": "LDS 读全是 ds_read_b32（×%d），无 b64/b128/matrix" % c["lds_scalar"],
            "hypothesis": "LDS 读未向量化，可能是 bank 压力或指令数来源",
            "limit": "静态证据；需 PMC 的 SQ_INSTS_LDS 定级",
        })
    # mls-path
    if not c["lsd_matrix"]:
        hits.append({
            "surface": "mls-path",
            "signal": "无 ds_read_matrix_trans_format —— 未走 MLS 矩阵通路",
            "hypothesis": "对 GEMM 消费者，kPack=1 时可尝试矩阵读",
            "limit": "静态证据；bf16 因 kpack=2 可能被编译期拒绝",
        })
    # epilogue-expansion
    if c["branch"] > 2:
        hits.append({
            "surface": "epilogue-expansion",
            "signal": "s_cbranch × %d，可能有未展开的循环" % c["branch"],
            "hypothesis": "热循环未完全展开",
            "limit": "静态证据；需确认 branch 是否在热路径上",
        })
    # 正面结论也要报（不是只有问题才报）
    ok = []
    if c["vmem_read_128"] and not (c["vmem_read_32"] or c["vmem_read_64"]):
        ok.append("全局读已全部 128-bit（dwordx4 × %d）" % c["vmem_read_128"])
    if c["slc"] == 0:
        ok.append("未使用 slc 修饰符（与 hbird 的 -13% 结论一致）")
    if c["global_load"] == 0:
        ok.append("全走 buffer_load，无 global_load（正确做法）")
    return hits, ok


def show(res, hits, notes, as_json=False):
    if as_json:
        print(json.dumps({"analysis": res, "attack_points": hits,
                          "positives": notes.get("positives", []),
                          "provenance": notes}, ensure_ascii=False, indent=1))
        return
    print("ISA 诊断")
    print("=" * 72)
    if notes.get("isa_path"):
        print("来源    %s" % notes["isa_path"])
    if notes.get("note"):
        print("说明    %s" % notes["note"])
    print("规模    %d 条指令" % res["total_instructions"])

    print("\n访存宽度")
    c = res["counts"]
    print("  读  dwordx4×%-4d dwordx2×%-4d dword×%-4d"
          % (c["vmem_read_128"], c["vmem_read_64"], c["vmem_read_32"]))
    print("  写  dwordx4×%-4d dwordx2×%-4d dword×%-4d"
          % (c["vmem_write_128"], c["vmem_write_64"], c["vmem_write_32"]))
    print("\nLDS")
    print("  matrix×%-4d b128×%-4d b32×%-4d"
          % (c["lsd_matrix"], c["lds_b128"], c["lds_scalar"]))
    print("\n等待深度")
    for k in ("vmcnt", "lgkmcnt"):
        if res["waitcnt"][k]:
            print("  %-8s %s" % (k, " ".join(
                "(%s)×%d" % (d, n) for d, n in sorted(res["waitcnt"][k].items()))))
    if res["mma"]:
        print("\n矩阵指令")
        for k, n in sorted(res["mma"].items(), key=lambda x: -x[1])[:4]:
            print("  %-42s ×%d" % (k, n))

    if hits:
        print("\n可攻击点（**假设，不是结论**）")
        for h in hits:
            print("  [%s]" % h["surface"])
            print("    信号  %s" % h["signal"])
            print("    假设  %s" % h["hypothesis"])
            print("    限制  %s" % h["limit"])
    if notes.get("positives"):
        print("\n正面结论")
        for x in notes["positives"]:
            print("  ✓ %s" % x)
    print("\n纪律：静态计数 ≠ 动态权重。上面的比例只用于**排序假设**；")
    print("      定级要靠 PMC（动态占比），裁决要靠 A/B。")


def selftest():
    """不需要 GPU、不需要 dccobjdump：只验解析与筛查逻辑。"""
    fake = """
      buffer_load_dwordx4 v[0:3], v[4:7], s[8:11], 0
      buffer_store_dwordx2 v[0:1], v[2:3], s[4:7], 0
      ds_read_b32 v0, v1
      ds_read_b32 v2, v3
      s_waitcnt vmcnt(0)
      s_waitcnt lgkmcnt(1)
      v_mmac_f32_16x16x32_fp8_fp8 v[0:3], v[4:7], v[8:11], v[12:15]
    """
    r = analyze(fake)
    assert r["counts"]["vmem_read_128"] == 1, r["counts"]
    assert r["counts"]["vmem_write_64"] == 1
    assert r["counts"]["lds_scalar"] == 2
    assert r["waitcnt"]["vmcnt"] == {"0": 1}
    assert r["waitcnt"]["lgkmcnt"] == {"1": 1}
    assert sum(r["mma"].values()) == 1
    print("[ok] 计数：读写宽度 / LDS / waitcnt / MMA 全部解析正确")

    hits, ok = screen(r)
    ids = {h["surface"] for h in hits}
    assert "vmem-vector-width" in ids, ids
    assert "lds-read-form" in ids, ids
    assert "mls-path" in ids, ids
    print("[ok] 筛查出 3 个假设：%s" % sorted(ids))

    for h in hits:
        assert "静态" in h["limit"], h
    print("[ok] 每个假设都带「静态证据」限制")

    # 反向：读写都用满 + 走 MLS 时，不该报这两条
    good = """
      buffer_load_dwordx4 v[0:3], v[4:7], s[8:11], 0
      buffer_store_dwordx4 v[0:3], v[4:7], s[8:11], 0
      ds_read_matrix_trans_format v[0:3], s[4:7]
      ds_read_b128 v[8:11], s[12:15]
    """
    r2 = analyze(good)
    hits2, ok2 = screen(r2)
    ids2 = {h["surface"] for h in hits2}
    assert "vmem-vector-width" not in ids2, ids2
    assert "lds-read-form" not in ids2, ids2
    assert any("128-bit" in x for x in ok2), ok2
    print("[ok] 反例：读写用满 + 走 MLS 时不误报")

    print("\nselftest PASS（5 项）")
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description="ISA 诊断：反汇编 + 结构化筛查")
    ap.add_argument("target", nargs="?", help=".co / .so / .hip")
    ap.add_argument("--arch", default="gfx938")
    ap.add_argument("--build", action="store_true", help="从源码编出来再分析")
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args(argv)
    if a.selftest:
        return selftest()
    if not a.target:
        ap.error("给一个 .co / .so / .hip，或 --selftest")

    notes = {}
    if a.build or str(a.target).endswith((".hip", ".cpp", ".c")):
        path, note = build_from_source(a.target, a.arch)
    else:
        path, note = disassemble(a.target, a.arch)
    if not path:
        print("ERROR: %s" % note, file=sys.stderr)
        return 2
    notes["isa_path"] = str(path)
    notes["note"] = note
    try:
        text = Path(path).read_text(encoding="utf-8", errors="ignore")
    except OSError as e:
        print("ERROR: 读不到 %s: %s" % (path, e), file=sys.stderr)
        return 2
    res = analyze(text)
    hits, ok = screen(res)
    notes["positives"] = ok
    show(res, hits, notes, a.json)
    return 0


if __name__ == "__main__":
    sys.exit(main())
