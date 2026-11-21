#!/bin/bash
# DCU 容器反推安装脚本模板 —— 对应原镜像里的 /install_dep.sh
# 用法: Dockerfile 里 `COPY install_dep.sh /install_dep.sh` + `RUN bash /install_dep.sh`
#
# 本模板固化本次实战结论:
#   1) vllm 用 --no-deps 装预编译 wheel, 再从 METADATA 提取依赖(排除 cmake)逐个补装
#      —— 规避 vllm 元数据 cmake==3.29 死结(源无 3.29.0, 只有 3.29.0.1)
#   2) 所有 pip install 加 --timeout 120 --retries 10 抗 sourcefind 偶发下载中断
#   3) DCU 定制包 + vllm 关键依赖**全部写死精确 pin**(版本号从原容器 pip freeze 提取,
#      并确认 nightly/dtk 上"非 INSTALLED 真实可得"再写). 切勿让 pip 解析最新版.
#   4) opencompass 等 sourcefind /open/ 下载不稳的公开包: 从原容器 docker cp 出
#      site-packages 目录 + dist-info, 由 Dockerfile 直接 COPY 进 dist-packages (不要 pip install <目录>)
#
# 注意: 下列 pin 值是 <device-B> 上 vllm015-0409 原镜像的实测值, 换别的镜像必须重新从原容器
#       pip freeze 提取并核对 nightly/dtk 可得性, 不可照搬.

PIP_OPT="--timeout 120 --retries 10"
set -e
VLLM_INDEX="https://pypi.sourcefind.cn/nightly/dtk/"

# ---- opencompass 已由 Dockerfile COPY 进 dist-packages (零网络), 此处不装 ----

# ---- DCU 评测辅助包: 精确 pin 回原镜像版本; --no-deps 避免触发 opencompass 下载 ----
pip install ${PIP_OPT} --no-deps "citest==0.0.8" -i ${VLLM_INDEX}
pip install ${PIP_OPT} --no-deps "cinotificate==0.0.3" -i ${VLLM_INDEX}
pip install ${PIP_OPT} "amdsmi==24.5.3+02cbffb.dirty" -i ${VLLM_INDEX}
pip install ${PIP_OPT} "numpy==1.25.0" -i ${VLLM_INDEX}

# ---- vllm: --no-deps 装本体(跳过 cmake==3.29 约束), 再补其余依赖(排除 cmake) ----
VLLM_SPEC="vllm==0.15.1+das.opt1.alpha.dtk2604.torch290.2604081832.gbcb2ba"
rm -rf /tmp/vwhl && mkdir -p /tmp/vwhl
pip download ${PIP_OPT} "${VLLM_SPEC}" --no-deps -d /tmp/vwhl -i ${VLLM_INDEX}
VLLM_WHL=$(ls /tmp/vwhl/*.whl | head -1)
pip install --no-deps "${VLLM_WHL}"
echo "=== vllm 本体已装, 开始补其余依赖(排除 cmake, 逐包安装带超时重试) ==="
python3 - "${VLLM_INDEX}" <<'PY'
import sys, zipfile, re, subprocess, time
idx = sys.argv[1]
whl = subprocess.check_output(["bash","-c","ls /tmp/vwhl/*.whl"]).decode().strip().split()[0]
deps = []
with zipfile.ZipFile(whl) as z:
    for n in z.namelist():
        if n.endswith("METADATA") or n.endswith("metadata"):
            txt = z.read(n).decode(errors="ignore")
            for line in txt.splitlines():
                m = re.match(r"Requires-Dist:\s*(.+)", line)
                if not m:
                    continue
                name = re.split(r"[;\(<>=!~ ]", m.group(1).strip())[0].strip()
                if name.lower() == "cmake":
                    continue
                deps.append(name)
seen=set(); uniq=[]
for d in deps:
    if d.lower() not in seen:
        seen.add(d.lower()); uniq.append(d)
print("需安装依赖(排除 cmake, 去重):", uniq)
for d in uniq:
    ok=False
    for attempt in range(6):
        try:
            subprocess.run(["pip","install","--timeout","120","--retries","10","-i",idx,d],
                           check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            ok=True
            print(f"[OK] {d}")
            break
        except subprocess.CalledProcessError as e:
            err=e.stderr.decode(errors="ignore")
            if "from versions: none" in err and "ReadTimeout" not in err:
                print(f"[SKIP-none] {d} 不可用, 跳过")
                ok=True
                break
            print(f"[RETRY {attempt+1}] {d} 失败, 重试... ({err[:120]})")
            time.sleep(8)
    if not ok:
        print(f"[WARN] {d} 多次重试仍失败, 继续(不阻断整体)")
PY

# ---- 版本对齐: 逐包补依赖时 nightly/dtk 会选到不兼容最新版, 此处全部 pin 回原容器精确版本 ----
pip install ${PIP_OPT} "torch==2.9.0+das.opt1.dtk2604.20260331.g4e3c1e7" -i ${VLLM_INDEX}
pip install ${PIP_OPT} "torchvision==0.24.0+das.opt1.dtk2604.torch290.2604091152.g66069c" -i ${VLLM_INDEX}
pip install ${PIP_OPT} "triton==3.3.0+das.opt2.dtk2604.torch290.20260331.g31542e" -i ${VLLM_INDEX}
pip install ${PIP_OPT} "flash_attn==2.8.3+das.opt1.dtk2604.torch290.2607280958.gebb4be" -i ${VLLM_INDEX}
pip install ${PIP_OPT} "flash_mla==1.2.0+das.opt1.dtk2604.torch290.2605281449.ge3aed6" -i ${VLLM_INDEX}
pip install ${PIP_OPT} "lightop==0.6.0+das.dtk2604.torch290.2604091144.ge8aebd" -i ${VLLM_INDEX}
pip install ${PIP_OPT} "lmslim==0.3.1+das.opt4.dtk2604.torch290.2604091256.gafc115" -i ${VLLM_INDEX}
pip install ${PIP_OPT} "numpy==1.25.0" -i ${VLLM_INDEX}
pip install ${PIP_OPT} "transformers==5.2.0" -i ${VLLM_INDEX}
pip install ${PIP_OPT} "xgrammar==0.1.29" -i ${VLLM_INDEX}
pip install ${PIP_OPT} "tensorizer==2.10.1" -i ${VLLM_INDEX}
pip install ${PIP_OPT} "runai-model-streamer==0.15.3" -i ${VLLM_INDEX}
pip install ${PIP_OPT} "setuptools==79.0.1" -i ${VLLM_INDEX}
pip install ${PIP_OPT} "msgspec==0.21.0" -i ${VLLM_INDEX}
pip install ${PIP_OPT} "safetensors==0.7.0" -i ${VLLM_INDEX}
pip install ${PIP_OPT} "outlines_core==0.2.11" -i ${VLLM_INDEX}
pip install ${PIP_OPT} "llguidance==1.3.0" -i ${VLLM_INDEX}
pip install ${PIP_OPT} "gguf==0.18.0" -i ${VLLM_INDEX}
pip install ${PIP_OPT} "mistral_common==1.11.0" -i ${VLLM_INDEX}
pip install ${PIP_OPT} "numba==0.61.2" -i ${VLLM_INDEX}
pip install ${PIP_OPT} "grpcio==1.80.0" -i ${VLLM_INDEX}
pip install ${PIP_OPT} "timm==1.0.26" -i ${VLLM_INDEX}
pip install ${PIP_OPT} "cmake==3.29.0.1" -i ${VLLM_INDEX}
rm -rf /tmp/vwhl
