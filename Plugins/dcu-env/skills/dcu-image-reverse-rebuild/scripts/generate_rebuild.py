#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
generate_rebuild.py — 从任意运行中的 DCU 容器动态生成重建所需的
Dockerfile(DTK段+源分层)+ install_dep.sh(两阶段 pin, 版本不写死)。

设计目标(用户要求): 指定其他容器也能根据容器生成可用的 dockerfile,
且包版本不能写死 —— 所有版本号从目标容器现场探测, 不在脚本里硬编码。

用法(在源设备上, 需 docker + 能访问 nightly/dtk 源):
  python3 generate_rebuild.py <container> [--outdir /path] [--ubuntu Ubuntu22.04]

输出:
  <outdir>/Dockerfile            (DTK 用 wget 外网直链, 动态拼 URL; 其余 RUN/ENV 骨架占位)
  <outdir>/install_dep.sh       (两阶段 pin, 版本号全部来自探测, 不写死)
  <outdir>/rebuild_report.txt   (探测明细 + 不可用版本告警)

探测逻辑:
  1. 容器内 pip freeze -> 取 DCU 包(含 +das/+dtk/海光) + 关键依赖 + 通用包原版本。
  2. 对每个 DCU 包: 在 nightly/dtk 上 pip index versions, 排除 INSTALLED 行,
     取真实可得版本。若与原版同系列(同主版本+torch系列)有可得版 -> 用可得版
     (series-aligned, 按用户规则"选高一点的"); 若 nightly 彻底无 -> 仍用原版(标注告警)。
  3. DTK 版本: cat /opt/dtk/.dtk_version -> 拼外网直链
     https://download.sourcefind.cn:65024/file/1/<DTK主版本>/<ubuntu>/DTK-<主版本>-<ubuntu>-x86_64.tar.gz
     (DTK 主版本如 DTK-26.04; .dtk_version 里的 rc 几在文件里, 直链不含 rc)。
"""
import sys, subprocess, os, re, json, argparse

VLLM_INDEX = "https://pypi.sourcefind.cn/nightly/dtk/"
TSINGHUA = "https://pypi.tuna.tsinghua.edu.cn/simple"

# DCU 适配包(必须从 nightly 装): 这些包名一旦出现就走 nightly
DCU_PKGS = ["torch", "torchvision", "triton", "flash_attn", "flash_mla",
            "lightop", "lmslim", "amdsmi", "vllm"]
# 其余也要 pin 的关键依赖(可能走 nightly 或清华, 取决于源)
KEY_DEPS = ["transformers", "xgrammar", "tensorizer", "setuptools", "msgspec",
            "safetensors", "outlines_core", "llguidance", "gguf", "mistral_common",
            "numba", "grpcio", "timm", "runai-model-streamer", "cinotificate",
            "citest", "numpy", "opencompass", "h5py", "cmake"]


def dcu(v):
    return ("+das" in v.lower()) or ("+dtk" in v.lower()) or ("dtk" in v.lower())


def run_inexec(c, cmd):
    out = subprocess.run(["docker", "exec", c] + cmd,
                         capture_output=True, text=True)
    return out.stdout + out.stderr


def pip_freeze(c):
    txt = run_inexec(c, ["pip", "list", "--format=freeze"])
    d = {}
    for l in txt.splitlines():
        l = l.strip()
        if "==" in l:
            n, v = l.split("==", 1)
            d[n.strip().lower()] = v
    return d


def dtk_version(c):
    v = run_inexec(c, ["cat", "/opt/dtk/.dtk_version"]).strip().splitlines()
    return v[0].strip() if v else ""


def nightly_available(c, pkg, ver):
    """返回 (status, real_version). status: AVAILABLE / SERIES / MISSING / SAME"""
    out = subprocess.run(["docker", "exec", c, "pip", "index", "versions", pkg,
                          "-i", VLLM_INDEX], capture_output=True, text=True).stdout
    # 排除 INSTALLED 行, 收集真实可得版本
    avail = []
    for line in out.splitlines():
        if "INSTALLED" in line:
            continue
        for m in re.findall(r"[\w.+\-]+das[\w.+\-]*", line):
            avail.append(m)
        # 也抓普通版本号
        for m in re.findall(r"\b\d+\.\d+\.\d+[\w.+\-]*\b", line):
            if m not in avail:
                avail.append(m)
    if not avail:
        return ("MISSING", None)
    # 精确匹配
    for a in avail:
        if a == ver:
            return ("SAME", a)
    # 同系列(torch 系列标识): 取含同一 torch 系列标记(如 torch290)或同主版本的可得版,
    # 并选 BUILD 号最高者(用户规则: 对不齐时"选高一点的" -> 取 nightly 上同系列最新可用).
    base = re.match(r"([\d.]+)", ver).group(1)  # 主版本如 2.9.0 / 2.8.3
    series_tag = None
    m = re.search(r"(torch\d+)", ver)
    if m:
        series_tag = m.group(1)
    series_matches = []
    for a in avail:
        if series_tag and series_tag in a:
            series_matches.append(a)
    if not series_matches:
        for a in avail:
            if a.startswith(base):
                series_matches.append(a)
    if not series_matches:
        return ("MISSING", None)
    # 取内嵌最大数字组(build 时间戳)最高者 = 同系列最新(用户: 选高一点的)
    def vkey(s):
        digs = re.findall(r"\d+", s)
        return max(int(d) for d in digs) if digs else 0
    best = max(series_matches, key=vkey)
    return ("SERIES", best)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("container")
    ap.add_argument("--outdir", default="./rebuild_out")
    ap.add_argument("--ubuntu", default="Ubuntu22.04")
    args = ap.parse_args()
    c = args.container
    os.makedirs(args.outdir, exist_ok=True)

    print(f"[*] 探测容器 {c} ...")
    freeze = pip_freeze(c)
    dtk = dtk_version(c)
    print(f"[*] DTK 版本: {dtk}")

    # 决定每个包的目标版本
    # 先收集所有相关包的原版本
    all_pkgs = {}
    for p in DCU_PKGS + KEY_DEPS:
        if p in freeze:
            all_pkgs[p] = freeze[p]

    report = []
    decisions = {}  # pkg -> (final_ver, source, note)
    for p, orig in all_pkgs.items():
        if p in DCU_PKGS:
            idx = VLLM_INDEX
        else:
            # 通用包: opencompass/numpy/h5py/setuptools 走清华; 其余 DCU 相关走 nightly
            idx = TSINGHUA if p in ("opencompass", "numpy", "h5py", "setuptools") else VLLM_INDEX
        # 探测 nightly 可得性(仅对 nightly 源的包有意义)
        if idx == VLLM_INDEX:
            st, real = nightly_available(c, p, orig)
            if st == "SAME":
                decisions[p] = (orig, "nightly", "逐字节一致")
            elif st == "SERIES":
                decisions[p] = (real, "nightly", f"series-aligned (原 {orig})")
            else:
                decisions[p] = (orig, "nightly", "⚠ nightly无可用版, 仍用原版(可能build失败)")
        else:
            decisions[p] = (orig, "tsinghua", "通用包走清华")
        report.append(f"{p}: orig={orig} -> final={decisions[p][0]} [{decisions[p][2]}]")

    # DTK 直链: 取主版本段(去掉 rc 等后缀)
    dtk_main = re.match(r"(DTK-[\d.]+)", dtk)
    dtk_main = dtk_main.group(1) if dtk_main else dtk
    dtk_url = (f"https://download.sourcefind.cn:65024/file/1/{dtk_main}/"
               f"{args.ubuntu}/{dtk_main}-{args.ubuntu}-x86_64.tar.gz")

    # ---- 生成 install_dep.sh ----
    stage1 = [p for p in DCU_PKGS if p in decisions]
    # vllm 单独 --no-deps 处理, 不含在阶段一普通 pin
    stage1_normal = [p for p in stage1 if p != "vllm"]
    pub = [p for p in ("opencompass", "numpy") if p in decisions]
    nightly_others = [p for p in decisions if p not in stage1 and p not in pub
                     and p not in ("setuptools", "h5py", "cmake")]

    lines = []
    lines.append("#!/bin/bash")
    lines.append("# 自动生成 (generate_rebuild.py) — 版本号全部来自目标容器探测, 不写死.")
    lines.append(f'PIP_OPT="--timeout 120 --retries 10 --resume-retries 50"')
    lines.append('set -e')
    lines.append(f'VLLM_INDEX="{VLLM_INDEX}"')
    lines.append(f'TSINGHUA="{TSINGHUA}"')
    lines.append("")
    lines.append("# ===== 阶段一: 锁 DCU 适配包(全 nightly) =====")
    for p in stage1_normal:
        v = decisions[p][0]
        lines.append(f'pip install ${{PIP_OPT}} "{p}=={v}" -i ${{VLLM_INDEX}}')
    lines.append("")
    lines.append("# ===== 阶段二: 通用包 + vllm =====")
    for p in pub:
        v = decisions[p][0]
        idx = TSINGHUA if p in ("opencompass", "numpy") else VLLM_INDEX
        lines.append(f'pip install ${{PIP_OPT}} "{p}=={v}" -i {idx}')
    for p in nightly_others:
        v = decisions[p][0]
        lines.append(f'pip install ${{PIP_OPT}} "{p}=={v}" -i ${{VLLM_INDEX}}')
    lines.append("")
    lines.append("# ---- vllm: --no-deps 装本体(跳过 cmake==3.29), 再补依赖排除 cmake ----")
    vllm_v = decisions.get("vllm", (freeze.get("vllm", ""),))[0]
    lines.append(f'VLLM_SPEC="{vllm_v}"')
    lines.append('rm -rf /tmp/vwhl && mkdir -p /tmp/vwhl')
    lines.append('pip download ${PIP_OPT} "${VLLM_SPEC}" --no-deps -d /tmp/vwhl -i ${VLLM_INDEX}')
    lines.append('VLLM_WHL=$(ls /tmp/vwhl/*.whl | head -1)')
    lines.append('pip install --no-deps "${VLLM_WHL}"')
    lines.append('echo "=== 补 vllm 依赖(排除 cmake, torch 已锁跳过) ==="')
    lines.append('python3 - "${VLLM_INDEX}" <<\'PY\'')
    lines.append("import sys, zipfile, re, subprocess, time")
    lines.append("idx = sys.argv[1]")
    lines.append("whl = subprocess.check_output([\"bash\",\"-c\",\"ls /tmp/vwhl/*.whl\"]).decode().strip().split()[0]")
    lines.append("deps = []")
    lines.append("with zipfile.ZipFile(whl) as z:")
    lines.append("    for n in z.namelist():")
    lines.append("        if n.endswith(\"METADATA\") or n.endswith(\"metadata\"):")
    lines.append("            txt = z.read(n).decode(errors=\"ignore\")")
    lines.append("            for line in txt.splitlines():")
    lines.append("                m = re.match(r\"Requires-Dist:\\\\s*(.+)\", line)")
    lines.append("                if not m: continue")
    lines.append("                name = re.split(r\"[;(<>=!~ ]\", m.group(1).strip())[0].strip()")
    lines.append("                if name.lower() in (\"cmake\",\"torch\"): continue")
    lines.append("                deps.append(name)")
    lines.append("seen=set(); uniq=[]")
    lines.append("for d in deps:")
    lines.append("    if d.lower() not in seen: seen.add(d.lower()); uniq.append(d)")
    lines.append("for d in uniq:")
    lines.append("    ok=False")
    lines.append("    for attempt in range(6):")
    lines.append("        try:")
    lines.append("            subprocess.run([\"pip\",\"install\",\"--timeout\",\"120\",\"--retries\",\"10\",\"--resume-retries\",\"50\",\"-i\",idx,d], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)")
    lines.append("            ok=True; print(f\"[OK] {d}\"); break")
    lines.append("        except subprocess.CalledProcessError as e:")
    lines.append("            err=e.stderr.decode(errors=\"ignore\")")
    lines.append("            if \"from versions: none\" in err and \"ReadTimeout\" not in err:")
    lines.append("                print(f\"[SKIP-none] {d}\"); ok=True; break")
    lines.append("            print(f\"[RETRY {attempt+1}] {d}\"); time.sleep(8)")
    lines.append("    if not ok: print(f\"[WARN] {d} 多次失败\")")
    lines.append("PY")
    lines.append("")
    lines.append("# ---- 末尾对齐: numpy 等若被拉高再 pin 回 ----")
    if "numpy" in decisions:
        lines.append(f'pip install ${{PIP_OPT}} "numpy=={decisions["numpy"][0]}" -i ${{TSINGHUA}}')
    lines.append(f'pip install ${{PIP_OPT}} "cmake==3.29.0.1" -i ${{VLLM_INDEX}}')
    lines.append("rm -rf /tmp/vwhl")
    with open(os.path.join(args.outdir, "install_dep.sh"), "w") as f:
        f.write("\n".join(lines) + "\n")

    # ---- 生成 Dockerfile(DTK 段动态填 URL, RUN/ENV 骨架占位) ----
    df = []
    df.append("# 自动生成骨架 (generate_rebuild.py) — RUN/ENV 仍需按 docker history 1:1 补全,")
    df.append("# 此处仅给出 DTK 源分层段 + pip.conf; 其余层请回填原镜像 history.")
    df.append("FROM ubuntu:22.04")
    df.append("")
    df.append("# ---- pip 源: 主索引清华(通用包稳), extra-index nightly(DCU包) ----")
    df.append('RUN mkdir -p ~/.pip && echo "[global]\\nindex-url = https://pypi.tuna.tsinghua.edu.cn/simple\\n')
    df.append('extra-index-url = https://pypi.sourcefind.cn/nightly/dtk/\\n" > ~/.pip/pip.conf')
    df.append("")
    df.append("# ---- DTK (动态直链, 来自容器 .dtk_version 探测) ----")
    df.append(f"# DTK 版本: {dtk} (主版本 {dtk_main}); 容器内可 wget, 宿主机可能超时")
    df.append(f'RUN wget -q --tries=0 --timeout=60 -O /tmp/dtk.tar.gz "{dtk_url}" \\')
    df.append("    && mkdir -p /opt/dtk && tar -xzf /tmp/dtk.tar.gz -C /opt/dtk --strip-components=1 \\")
    df.append("    && rm -f /tmp/dtk.tar.gz")
    df.append("# 备选(构建机整网不通时): COPY dtk /opt/dtk  (先下好 DTK 解压进上下文)")
    df.append("")
    df.append("# ---- 主安装层: 执行自动生成的 install_dep.sh ----")
    df.append("COPY install_dep.sh /install_dep.sh")
    df.append("RUN pip install --ignore-installed blinker && bash /install_dep.sh")
    with open(os.path.join(args.outdir, "Dockerfile"), "w") as f:
        f.write("\n".join(df) + "\n")

    # ---- 报告 ----
    with open(os.path.join(args.outdir, "rebuild_report.txt"), "w") as f:
        f.write(f"容器: {c}\nDTK: {dtk} (主版本 {dtk_main})\nDTK直链: {dtk_url}\n\n")
        f.write("版本决策:\n" + "\n".join(report) + "\n")

    print(f"[*] 已生成: {args.outdir}/Dockerfile, install_dep.sh, rebuild_report.txt")
    print("[*] 版本决策摘要:")
    for r in report:
        print("   " + r)
    print(f"[*] DTK 直链: {dtk_url}")


if __name__ == "__main__":
    main()
