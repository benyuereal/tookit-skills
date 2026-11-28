#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""gen_tool_registry —— 由 tool_probe 的**实跑结果**生成工具注册表。

为什么是生成而不是手写
----------------------
业主的硬要求：**没有 probe 证据不许写「实测可用」**。
手写注册表一定会漂：今天写了 available，明天换了 DTK 就失效，而没人会回头改。

所以这里的 pipeline 是：
    tool_probe.py --json  →  gen_tool_registry.py  →  data/ops/tool-registry.yaml

用户只要跑 `tool_probe.py --write-registry`，两者一次完成。

同时校验两条硬规则：
  1. 每条 status=available 必须有非空 probe 文本（证据）；
  2. playbooks 里引用的每个 id 必须在 tools 里存在。

用法
----
    python3 -I scripts/gen_tool_registry.py            # 生成
    python3 -I scripts/gen_tool_registry.py --check    # 与现有文件比对，不一致非 0
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "data/ops/tool-registry.yaml"
TOOL_PROBE = REPO / "scripts/tool_probe.py"

# 文本骨架放 scripts/templates/tool_registry.md；本文件只拼动态块。
# `-I` 不把脚本目录放进 sys.path（测试还按路径 importlib 加载），这里补一次。
# 本文件已有同名 `render()`，所以模板渲染器用别名导入。
_SCRIPTS = Path(__file__).resolve().parent
if str(_SCRIPTS) not in sys.path:
    sys.path.append(str(_SCRIPTS))

from template_render import render as _render_text  # noqa: E402

# 六个类别（业主指定）
CATS = {
    "runtime": ("① 运行时", "这个 kernel 在运行时到底卡在哪（周期/指令/访存/停顿）？"),
    "crash": ("② 崩溃", "它为什么崩/挂，崩在哪一行？"),
    "static": ("③ 静态", "这个二进制里有什么（符号、段、资源字段）？"),
    "build": ("④ 构建", "它被怎么编出来的，编译期能拿到什么？"),
    "numeric": ("⑤ 数值", "数值对不对，差多少？"),
    "env": ("⑥ 环境", "环境是否可复现（时钟/版本/卡占用）？"),
}
ORDER = ["runtime", "crash", "static", "build", "numeric", "env"]

CLASSIFY = {
    "hipprof": "runtime", "hipprof_pmc_csv": "runtime", "counter_probe": "runtime",
    "time_align": "runtime", "hy_smi": "runtime", "occupancy_smi": "runtime",
    "hipgdb": "crash", "rocgdb": "crash", "dmesg": "crash", "gpu_reset": "crash",
    "bounds_check": "crash",
    "llvm_readelf": "static", "llvm_objdump": "static", "builtin_readelf_nm": "static",
    "dccobjdump": "static", "dccobjdump_extract": "static",
    "hipcc": "build", "hipcc_resource_usage": "build", "hipcc_dumpversion": "build",
    "hipconfig": "build", "driver_version": "build", "inspect_so": "build",
    "numeric_ulp": "numeric", "nan_inf_scan": "numeric",
    "bw_calibrate": "env", "compute_peak": "env", "flock": "env", "kfd": "env",
    "rocm_bandwidth_test": "env", "rocminfo": "env", "bandwidth_test": "env",
    "miopen_driver": "env",
}

# 每条工具的可复跑命令（给人抄的，不是探测内部实现）
CMDS = {
    "hipprof": "hipprof --pmc --pmc-type 3 -o /tmp/pmcout/x python3 bench.py   # 注意：不认 `--`",
    "hipprof_pmc_csv": "head -1 /tmp/pmcout/x.csv | tr ',' '\\n' | head -30",
    "counter_probe": "python3 -I scripts/pmc_probe.py /tmp/pmcout/x.csv",
    "time_align": "python3 -I scripts/pmc.py /tmp/pmcout/x.csv --kernel <name> --cu 64 --json",
    "hy_smi": "hy-smi --showclocks ; hy-smi",
    "occupancy_smi": "hy-smi   # 看 HCU% 判断卡是否被别人占",
    "hipgdb": "/opt/dtk/bin/hipgdb --version",
    "rocgdb": "command -v rocgdb   # 本机没有，用 hipgdb",
    "dmesg": "dmesg | grep -iE 'amdgpu|kfd|gpu' | tail -20",
    "gpu_reset": "hy-smi --showtemp",
    "bounds_check": "hipcc -O1 -g --offload-arch=gfx938 -fsanitize=address k.hip -o k",
    "llvm_readelf": "/opt/dtk/llvm/bin/llvm-readelf --notes <so>-hipv4-amdgcn-amd-amdhsa--gfx938.out",
    "llvm_objdump": "command -v llvm-objdump   # 本机没有",
    "builtin_readelf_nm": "readelf -SW <exe> ; nm -C <exe> ; addr2line -e <exe> <addr>",
    "dccobjdump": "/opt/dtk/llvm/bin/dccobjdump --extract-elf=all --inputs=<so>",
    "dccobjdump_extract": "python3 -I Plugins/ops/skills/op-hip-compile/inspect_so.py <so>",
    "hipcc": "hipcc -shared -fPIC -O3 --offload-arch=gfx938 k.hip -o k.so",
    "hipcc_resource_usage": "hipcc --resource-usage ...   # 本机报 unsupported",
    "hipcc_dumpversion": "hipcc -dumpversion",
    "hipconfig": "hipconfig --version",
    "driver_version": "cat /sys/module/amdgpu/version ; hipconfig --version",
    "inspect_so": "python3 -I Plugins/ops/skills/op-hip-compile/inspect_so.py <so> --json",
    "numeric_ulp": "python3 -I -c \"import numpy as np; ...\"   # 见 bench 里的 compare()",
    "nan_inf_scan": "python3 -I -c \"import torch; torch.isnan(x).sum()\"",
    "bw_calibrate": "python3 -I scripts/bw_calibrate.py",
    "compute_peak": "python3 -I Plugins/ops/skills/op-roofline/tools/calibrate_bf16_peak.py",
    "flock": "flock /tmp/opbench-gpu.lock -c '<bench>'",
    "kfd": "ls /dev/kfd /dev/dri/renderD*",
    "rocm_bandwidth_test": "/opt/dtk/bin/rocm-bandwidth-test",
    "rocminfo": "rocminfo | grep -c Agent",
    "bandwidth_test": "/opt/dtk/bin/BandwidthTest",
    "miopen_driver": "/opt/dtk/bin/MIOpenDriver",
}

# playbooks：每个场景一条按顺序的工具链。引用的 id 必须都在 CLASSIFY 里。
PLAYBOOKS = [
    {
        "id": "correctness-failure",
        "when": "正确性门不过（max_abs 超容差 / cosine 掉 / bit_exact 不成立）",
        "chain": ["numeric_ulp", "nan_inf_scan", "dmesg", "builtin_readelf_nm", "hipgdb"],
        "why": "先定位数值本身（差几个 ULP、有无 NaN/Inf），再排除驱动报错，"
               "然后看符号/地址，最后才上调试器。顺序反了会花几小时在调试器里找一个 ULP 的问题。",
    },
    {
        "id": "perf-below-target",
        "when": "正确性过了但 remaining 大、或比基线慢",
        "chain": ["hy_smi", "flock", "hipprof_pmc_csv", "counter_probe", "inspect_so",
                  "time_align"],
        "why": "先确认卡没被别人占（否则数据全废），再采 PMC，"
               "用 counter_probe 确认要用的计数器真的存在，然后用 inspect_so 定驻留档，"
               "最后时间对齐把计数器归到具体调用。",
    },
    {
        "id": "crash-or-hang",
        "when": "崩、挂、或结果随机不可复现",
        "chain": ["dmesg", "gpu_reset", "bounds_check", "hipgdb", "builtin_readelf_nm"],
        "why": "先看是不是掉卡（dmesg/gpu_reset），再查越界（sanitizer），"
               "然后才用 hipgdb 看栈。跳过 dmesg 会把「掉卡」误诊成「kernel 写错」。",
    },
    {
        "id": "not-reproducible",
        "when": "同一份代码两次跑结果不同，或换机器结论变了",
        "chain": ["hy_smi", "driver_version", "hipcc_dumpversion", "hipconfig",
                  "bw_calibrate", "compute_peak"],
        "why": "先锁时钟（sclk 档位不一致是最常见原因），再钉死版本三件套，"
               "最后确认分母有没有漂（带宽/算力峰值都要重标定）。",
    },
]


def Y(s):
    """最小的 YAML 字符串输出：需要就加引号。"""
    if s is None:
        return "null"
    if isinstance(s, bool):
        return "true" if s else "false"
    if isinstance(s, (int, float)):
        return str(s)
    s = str(s)
    if not s:
        return '""'
    if any(c in s for c in ":#{}[]&*!|>'\"%@`") or s.strip() != s or "\n" in s:
        return json.dumps(s, ensure_ascii=False)
    return s


def collect():
    p = subprocess.run([sys.executable, "-I", str(TOOL_PROBE), "--json"],
                       capture_output=True, text=True)
    if p.returncode != 0:
        raise SystemExit("tool_probe 失败：%s" % (p.stderr or p.stdout)[:400])
    return json.loads(p.stdout)


def render(rows):
    """注册表文本走 scripts/templates/tool_registry.md，本函数只拼动态块。"""
    tools = []
    for cat in ORDER:
        name, q = CATS[cat]
        tools.append("")
        tools.append("  # ===== %s =====" % name)
        tools.append("  # %s" % q)
        for r in sorted([x for x in rows if CLASSIFY.get(x["tool"]) == cat],
                        key=lambda x: x["tool"]):
            t = r["tool"]
            tools.append("  - id: %s" % t)
            tools.append("    category: %s" % cat)
            tools.append("    question: %s" % Y(q))
            tools.append("    command: %s" % Y(CMDS.get(t, "")).replace("\n", " "))
            tools.append("    probe: %s" % Y(r["detail"]))
            tools.append("    status: %s" % r["status"])
            tools.append("    fallback: %s" % Y(r["fallback"]))
            tools.append("    cost: %s" % Y(r["cost"]))
            tools.append("    stage: %s" % Y(r["stage"]))

    playbooks = []
    for pb in PLAYBOOKS:
        playbooks.append("  - id: %s" % pb["id"])
        playbooks.append("    when: %s" % Y(pb["when"]))
        playbooks.append("    chain:")
        for c in pb["chain"]:
            playbooks.append("      - %s" % c)
        playbooks.append("    why: %s" % Y(pb["why"]))

    return _render_text(
        "tool_registry",
        tools_body=("\n" + "\n".join(tools)) if tools else "",
        playbooks_body=("\n" + "\n".join(playbooks)) if playbooks else "",
    )


def check_playbooks(doc_text, ids):
    """playbook 引用的工具必须存在。返回问题列表。"""
    import yaml
    d = yaml.safe_load(doc_text)
    probs = []
    for pb in d.get("playbooks") or []:
        for c in pb.get("chain") or []:
            if c not in ids:
                probs.append("playbook %s 引用了不存在的工具 %s" % (pb.get("id"), c))
    # 硬规则：available 必须有 probe
    for t in d.get("tools") or []:
        if t.get("status") == "available" and not str(t.get("probe") or "").strip():
            probs.append("工具 %s 标 available 却没有 probe 证据" % t.get("id"))
    return probs


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--out", default=str(OUT))
    a = ap.parse_args(argv)
    rows = collect()
    text = render(rows)
    ids = {r["tool"] for r in rows}
    probs = check_playbooks(text, ids)
    if probs:
        print("自检失败：")
        for p in probs:
            print("  -", p)
        return 2
    out = Path(a.out)
    if a.check:
        if not out.is_file():
            print("缺少 %s" % out)
            return 1
        if out.read_text(encoding="utf-8") != text:
            print("%s 与实跑结果不一致 —— 重跑 gen_tool_registry.py" % out)
            return 1
        print("一致（%d 个工具）" % len(rows))
        return 0
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text, encoding="utf-8")
    ok = sum(1 for r in rows if r["status"] == "available")
    print("已写 %s：%d 个工具（available %d / degraded %d / missing %d）"
          % (out, len(rows), ok,
             sum(1 for r in rows if r["status"] == "degraded"),
             sum(1 for r in rows if r["status"] == "missing")))
    print("playbooks %d 条，引用工具全部存在" % len(PLAYBOOKS))
    return 0


if __name__ == "__main__":
    sys.exit(main())
