#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""isa_probe.disassemble() 的工作目录纪律判据。

事故 a4374b7：`dccobjdump` 不认 `-o`，`--extract-elf` 和反汇编产物都写 **cwd**。
旧代码在 `os.chdir(wd)` **之前**跑 extract-elf，于是 7 个
`ship.so-hipv4-*.gfx938.out` 落进仓库根目录并被误提交。

本测试守两件事，缺一不可：
  A. 正向：调用结束后，**调用方 cwd** 里没有 `*.out` / `*.out.*` / `*.co` / `*.ISA`；
  B. 反向：同一个调用，产物必须**出现在 workdir 里**。
     只测 A 会假绿 —— 「工具根本没跑/没产物」和「产物落到别处」都满足 A。
"""
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "scripts"))

import isa_probe  # noqa: E402

# dccobjdump 的产物族（事故里的 .out 也在内，不能只查 .co/.ISA）
ARTIFACT_GLOBS = ("*.out", "*.out.*", "*.co", "*.ISA")


def _artifacts(d):
    """列出目录里的 dccobjdump 产物名（不递归）。"""
    d = Path(d)
    found = []
    for g in ARTIFACT_GLOBS:
        found.extend(p.name for p in d.glob(g))
    return sorted(found)


def _pick_target(tmpdir):
    """选一个能真正产出中间文件的输入；实在没有就 skip，绝不静默放行。"""
    candidates = [
        # 仓库内、git 忽略的 run 产物：有就用，省一次编译
        REPO / "Plugins/ops/skills/op-rmsnorm-fused/runs/forged/_kernels/fused_rms_v2_t128_v1.so",
        # DTK 自带的 code object / hsaco：不依赖仓库状态
        Path("/opt/dtk/lib/hipblaslt/library_bw_nmz/extop_gfx938.co"),
        Path("/opt/dtk/lib/hipblaslt/library_bw_nmz/Kernels.so-000-gfx938.hsaco"),
    ]
    for c in candidates:
        if c.is_file():
            return c

    hipcc = shutil.which("hipcc")
    if not hipcc and Path("/opt/dtk/bin/hipcc").is_file():
        hipcc = "/opt/dtk/bin/hipcc"
    if not hipcc:
        pytest.skip("找不到可反汇编的输入（无 .so/.co/.hsaco 且无 hipcc），判据未生效")

    src = Path(tmpdir) / "probe_target.hip"
    out = Path(tmpdir) / "probe_target.so"
    src.write_text(
        '#include <hip/hip_runtime.h>\n'
        'extern "C" __global__ void k(float* a) {\n'
        '  int i = threadIdx.x; if (i < 64) a[i] *= 2.0f;\n'
        '}\n',
        encoding="utf-8",
    )
    cp = subprocess.run([hipcc, "--offload-arch=gfx938", "-fPIC", "-shared",
                         str(src), "-o", str(out)],
                        capture_output=True, text=True)
    if cp.returncode != 0 or not out.is_file():
        pytest.skip("hipcc 编测试输入失败，判据未生效：%s" % (cp.stderr or cp.stdout)[-160:])
    return out


pytestmark = pytest.mark.skipif(
    isa_probe.find_disassembler() is None,
    reason="本机无 dccobjdump，判据未生效",
)


def test_default_workdir_keeps_cwd_clean(tmp_path, monkeypatch):
    """正向：不传 workdir 时，产物只能进 mkdtemp，绝不能留在调用方 cwd。"""
    target = _pick_target(tmp_path)
    cwd0 = os.getcwd()
    with tempfile.TemporaryDirectory() as cwd_td:
        os.chdir(cwd_td)
        try:
            # 默认 workdir 是 tempfile.mkdtemp()。把它重定向到可观测目录，
            # 这样「产物进了 mkdtemp」也能被确认，避免把「工具没跑」误判成通过。
            spool = Path(cwd_td) / "_spool"
            spool.mkdir()
            monkeypatch.setattr(isa_probe.tempfile, "mkdtemp",
                                lambda *a, **k: str(spool))
            isa_probe.disassemble(str(target), workdir=None)

            assert os.getcwd() == cwd_td, "disassemble() 没有把 cwd 还原"
            leftover = _artifacts(cwd_td)
            assert leftover == [], "dccobjdump 产物泄漏到调用方 cwd：%s" % leftover
            assert _artifacts(spool) != [], (
                "默认 workdir 里也没有产物 —— 无法区分「没泄漏」和「工具根本没产出」")
        finally:
            os.chdir(cwd0)


def test_explicit_workdir_receives_artifacts(tmp_path):
    """反向：workdir= 指向的目录必须**真的**收到产物，cwd 仍然干净。"""
    target = _pick_target(tmp_path)
    cwd0 = os.getcwd()
    with tempfile.TemporaryDirectory() as cwd_td, \
            tempfile.TemporaryDirectory() as wd_td:
        os.chdir(cwd_td)
        try:
            isa_probe.disassemble(str(target), workdir=wd_td)

            assert os.getcwd() == cwd_td, "disassemble() 没有把 cwd 还原"
            leftover = _artifacts(cwd_td)
            assert leftover == [], "dccobjdump 产物泄漏到调用方 cwd：%s" % leftover
            assert _artifacts(wd_td) != [], (
                "workdir=%s 里没有产物 —— 反向判据失效（可能根本没跑）" % wd_td)
        finally:
            os.chdir(cwd0)
