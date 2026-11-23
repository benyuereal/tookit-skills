#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""pmc_collect —— 跑一次 PMC，产出 `scripts/pmc.py` 能吃的 CSV。

**本文件只负责采集，不解析。** 解析器的唯一权威实现是 `scripts/pmc.py`
（它包含四条硬规则：稳态取 GRBM 最小行、排噪声前缀、`TA_*/TCP_*/TCC_*` 跨实例求和、
硬件常数不写死）。这里重复实现解析逻辑就等于造第二个真值源——
本项目在「同一事实多处登记」上已经栽过多次（540、49.8、INDEX 与卡片漂移）。

用法
----
    python3 -I pmc_collect.py --out /tmp/pmcout            # 采集，产 CSV
    python3 -I pmc_collect.py --out /tmp/pmcout -- python3 bench.py
    python3 -I pmc_collect.py --out /tmp/pmcout --parse    # 采集并直接交给 pmc.py

    # 只解析已有 CSV（不采集）
    python3 -I pmc_collect.py --parse-only /tmp/pmcout/x.csv --cu 64

退出码
------
  0  采集成功且找到 CSV
  1  采集跑完但没有 CSV（多半是 pmc-type 选项不支持）
  2  用法错误 / 环境缺 hipprof

为什么用 hipprof 而不是 rocprof
-------------------------------
本机（DTK 25.10/26.04）只有 `/opt/dtk/bin/hipprof`，没有 `rocprof`。
hipprof 是 DTK 的 rocprof 前端，`--pmc --pmc-type 3` 输出 CSV 表。
"""
import argparse
import glob
import os
import shutil
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "../../../../.."))
PMC_PARSER = os.path.join(REPO, "scripts", "pmc.py")

HIPPROF_CANDIDATES = ("/opt/dtk/bin/hipprof", "hipprof")


def find_hipprof():
    for c in HIPPROF_CANDIDATES:
        p = c if os.path.isabs(c) else shutil.which(c)
        if p and os.path.isfile(p) and os.access(p, os.X_OK):
            return p
    return None


def _abspathify(arg):
    """把命令里「确实存在的相对路径」转成绝对路径（因为采集时 cwd 会变）。

    只转存在的普通文件/目录，其余（选项、数值、-c 后面的代码串）原样保留。
    """
    if not arg or arg.startswith("-") or os.path.isabs(arg):
        return arg
    if os.path.exists(arg):
        return os.path.abspath(arg)
    return arg


def collect(outdir, cmd, pmc_type="3", timeout_s=900):
    """跑 hipprof --pmc，返回 (rc, stdout+stderr, [csv 路径])。"""
    hipprof = find_hipprof()
    if not hipprof:
        return 2, "找不到 hipprof（试过 %s）" % ", ".join(HIPPROF_CANDIDATES), []
    outdir = os.path.abspath(outdir)
    os.makedirs(outdir, exist_ok=True)
    # `-o` 用**相对文件名 + cwd=outdir**：实测传绝对路径时 hipprof 会静默产出 0 字节 CSV
    # （rc 仍是 0，最容易漏过）。改 cwd 后同一命令正常产出 9.5 KB。
    prefix = "pmc"
    # cwd 改成 outdir 之后，命令里的相对路径会失效 ⇒ 先把存在的路径参数绝对化
    cmd = [_abspathify(x) for x in cmd]
    argv = [hipprof, "--pmc", "--pmc-type", str(pmc_type), "-o", prefix] + cmd
    try:
        cp = subprocess.run(argv, capture_output=True, text=True, timeout=timeout_s,
                            cwd=outdir)
        log = " ".join(argv) + "\n" + (cp.stdout or "") + (cp.stderr or "")
        rc = cp.returncode
    except subprocess.TimeoutExpired:
        return 1, "采集超时（%ds）" % timeout_s, []
    csvs = sorted(glob.glob(os.path.join(outdir, "*.csv")))
    return rc, log, csvs


def parse_with_authority(csv_path, cu=None, kernel=None, row_mode="steady", as_json=False):
    """把解析工作交给 scripts/pmc.py —— 不在这里重写。"""
    if not os.path.isfile(PMC_PARSER):
        print("找不到解析器 %s" % PMC_PARSER, file=sys.stderr)
        return None
    argv = [sys.executable, "-I", PMC_PARSER, csv_path, "--row", row_mode]
    if cu is not None:
        argv += ["--cu", str(cu)]
    if kernel:
        argv += ["--kernel", kernel]
    if as_json:
        argv.append("--json")
    return subprocess.run(argv, capture_output=True, text=True)


def selftest():
    """无 GPU 也能跑的自检：验证参数组装与 0 字节检测，不真采。

    真采集验证见 `--smoke`（需要 GPU，跑一个最小 GEMM 并解析）。
    """
    ok = True
    # 1) _abspathify 只转存在的相对路径
    assert _abspathify("--pmc") == "--pmc"
    assert _abspathify("/abs/x") == "/abs/x"
    assert _abspathify("python3") == "python3"       # 不在 PATH 里的裸词原样保留
    assert os.path.isabs(_abspathify("scripts/pmc.py"))
    print("[ok] _abspathify 只转存在的相对路径")

    # 2) 找得到 hipprof 或明确报缺
    h = find_hipprof()
    print("[ok] hipprof: %s" % (h or "未找到（--smoke 会失败并报出这条路）"))

    # 3) 解析器必须存在（采集器不重复实现解析）
    assert os.path.isfile(PMC_PARSER), PMC_PARSER
    print("[ok] 解析器存在：%s（采集器不重写解析逻辑）" % os.path.relpath(PMC_PARSER, REPO))

    print("\nselftest PASS" if ok else "\nselftest FAIL")
    return 0 if ok else 1


def smoke(outdir="/tmp/pmc/smoke", cu=64):
    """真采集一次最小 GEMM 并解析 —— 证明「采集→解析」这条链是通的。"""
    script = os.path.join(outdir, "_smoke_kernel.py")
    os.makedirs(outdir, exist_ok=True)
    with open(script, "w", encoding="utf-8") as f:
        f.write("import torch\n"
                "a=torch.randn(1024,1024,device='cuda'); b=torch.randn(1024,1024,device='cuda')\n"
                "for _ in range(3): c=a@b\n"
                "torch.cuda.synchronize()\n"
                "print('smoke done', tuple(c.shape))\n")
    rc, log, csvs = collect(outdir, [sys.executable, "-I", script])
    print("# hipprof rc=%d，CSV %d 个" % (rc, len(csvs)))
    if not csvs or any(os.path.getsize(c) == 0 for c in csvs):
        print(log[-800:], file=sys.stderr)
        print("smoke 失败：没产出非空 CSV", file=sys.stderr)
        return 1
    r = parse_with_authority(csvs[0], cu=cu)
    if r is None:
        return 2
    sys.stdout.write(r.stdout)
    sys.stderr.write(r.stderr)
    return r.returncode


def main(argv=None):
    ap = argparse.ArgumentParser(description="PMC 采集（解析交给 scripts/pmc.py）")
    ap.add_argument("--out", default="/tmp/pmcout", help="CSV 落盘目录")
    ap.add_argument("--parse", action="store_true", help="采集后立刻解析")
    ap.add_argument("--parse-only", metavar="CSV", help="只解析已有 CSV，不采集")
    ap.add_argument("--cu", type=int, default=None, help="CU 数（不传则解析器打印 unknown）")
    ap.add_argument("--kernel", default=None)
    ap.add_argument("--row", default="steady", choices=["steady", "max-dur"])
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--pmc-type", default="3", help="hipprof 的导出格式；3=CSV 表")
    ap.add_argument("--selftest", action="store_true", help="无 GPU 自检")
    ap.add_argument("--smoke", action="store_true",
                    help="真采集一次最小 GEMM 并解析，证明采集→解析链通")
    a, cmd = ap.parse_known_args(argv)
    if a.selftest:
        return selftest()
    if a.smoke:
        return smoke()
    if cmd and cmd[0] == "--":
        cmd = cmd[1:]

    # ---- 只解析
    if a.parse_only:
        r = parse_with_authority(a.parse_only, a.cu, a.kernel, a.row, a.json)
        if r is None:
            return 2
        sys.stdout.write(r.stdout)
        sys.stderr.write(r.stderr)
        return r.returncode

    if not cmd:
        print("需要一条要采集的命令，例如：\n"
              "  python3 -I pmc_collect.py --out /tmp/pmcout -- python3 bench.py",
              file=sys.stderr)
        return 2

    t0 = time.time()
    rc, log, csvs = collect(a.out, cmd, a.pmc_type)
    print("# hipprof rc=%d，耗时 %.1fs" % (rc, time.time() - t0))
    if not csvs:
        print(log[-2000:], file=sys.stderr)
        print("\n没有产出 CSV。检查 --pmc-type 是否支持（本机试 3=CSV 表）。",
              file=sys.stderr)
        return 1
    empty = [c for c in csvs if os.path.getsize(c) == 0]
    if empty:
        print("采集器产出 0 字节 CSV：%s" % ", ".join(empty), file=sys.stderr)
        print("实测原因：hipprof 收到绝对路径的 -o 时会静默失败（rc 仍为 0）。", file=sys.stderr)
        return 1
    print("# CSV：%s" % "\n#      ".join(csvs))
    if not a.parse:
        print("# 解析：python3 -I %s %s --cu <CU 数>" % (PMC_PARSER, csvs[0]))
        return 0

    r = parse_with_authority(csvs[0], a.cu, a.kernel, a.row, a.json)
    if r is None:
        return 2
    sys.stdout.write(r.stdout)
    sys.stderr.write(r.stderr)
    return r.returncode


if __name__ == "__main__":
    sys.exit(main())
