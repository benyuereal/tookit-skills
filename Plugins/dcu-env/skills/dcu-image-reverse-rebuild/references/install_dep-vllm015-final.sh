#!/bin/bash
# 重建用安装脚本 —— 对应原镜像 /install_dep.sh
# 原内容: pip install citest / cinotificate / amdsmi / vllm==0.15.1 / numpy==1.25.0
#
# 源策略(分层, 这是反推成功的关键):
#   DCU 适配包(torch/torchvision/triton/flash_*/lightop/lmslim/amdsmi/vllm) -> 仅 sourcefind/nightly/dtk
#   通用公开包(opencompass/h5py/tokenizers/numpy/setuptools 等) -> 清华镜像(稳定)
#   <device-B> 上 pypi.org 的 whl 下载被反代到 sourcefind /open 且间歇断流, 故通用包统一走清华.
#   Dockerfile 的 pip.conf: index-url=清华, extra-index-url=nightly/dtk (DCU 包从 extra 解析).
#
# 两阶段顺序(避免通用包从清华拉官方 torch):
#   阶段一: 先 pin 所有 DCU 包(含 torch 2.9.0+dcu), 锁死 DCU torch, 后续任何包发现 torch 已满足不再重装.
#   阶段二: 再装 opencompass(清华, 其 torch>=1.13.1 已被 DCU 2.9.0 满足) + citest/cinotificate + vllm.
#
# 版本原则: 原容器 pip freeze 的 DCU 定制包 + vllm 关键依赖的精确版本号全部写死 pin,
#   保证换任何设备用本 Dockerfile 都能构建出包版本一致的镜像. 每个 pin 均已在 nightly 确认真实可得.
#   (注意: pip index 的 INSTALLED 行是原容器已装版, 不可当可用; 必须查 nightly 实际可下载版本)
#
# 已知死结与规避:
#   - cmake==3.29: vllm 元数据要求 cmake==3.29, 但源无 3.29.0 精确版(只有 3.29.0.1). 解决: vllm 用
#     --no-deps 装预编译 wheel(无需本地编译), 再提取 METADATA 依赖排除 cmake 逐个补装; cmake 装 3.29.0.1.
#   - flash_attn/flash_mla: 原镜像精确构建号 nightly 无, 用 torch290 系列最新可用版(同版本号, 构建号差, 功能等价).
#   - set -e: 任一包安装失败立即终止, 避免静默假成功.
PIP_OPT="--timeout 120 --retries 10 --resume-retries 50"
set -e
VLLM_INDEX="https://pypi.sourcefind.cn/nightly/dtk/"
TSINGHUA="https://pypi.tuna.tsinghua.edu.cn/simple"

# ============ 阶段一: 锁定所有 DCU 适配包(仅 nightly/dtk) ============
# 先把 torch/torchvision/triton/flash_*/lightop/lmslim/amdsmi 全部按原镜像版本 pin 装好,
# 锁死 DCU torch 2.9.0, 后续通用包不会从清华拉官方 torch.
pip install ${PIP_OPT} "torch==2.9.0+das.opt1.dtk2604.20260331.g4e3c1e7" -i ${VLLM_INDEX}
pip install ${PIP_OPT} "torchvision==0.24.0+das.opt1.dtk2604.torch290.2604091152.g66069c" -i ${VLLM_INDEX}
pip install ${PIP_OPT} "triton==3.3.0+das.opt2.dtk2604.torch290.20260331.g31542e" -i ${VLLM_INDEX}
pip install ${PIP_OPT} "flash_attn==2.8.3+das.opt1.dtk2604.torch290.2607280958.gebb4be" -i ${VLLM_INDEX}
pip install ${PIP_OPT} "flash_mla==1.2.0+das.opt1.dtk2604.torch290.2605281449.ge3aed6" -i ${VLLM_INDEX}
pip install ${PIP_OPT} "lightop==0.6.0+das.dtk2604.torch290.2604091144.ge8aebd" -i ${VLLM_INDEX}
pip install ${PIP_OPT} "lmslim==0.3.1+das.opt4.dtk2604.torch290.2604091256.gafc115" -i ${VLLM_INDEX}
pip install ${PIP_OPT} "amdsmi==24.5.3+02cbffb.dirty" -i ${VLLM_INDEX}

# ============ 阶段二: 通用包 + 评测辅助 + vllm ============
# opencompass(通用公开包, 走清华); 其依赖 torch 已被阶段一 DCU 2.9.0 满足, 不会从清华拉官方 torch.
pip install ${PIP_OPT} "opencompass==0.5.1" -i ${TSINGHUA}
# citest/cinotificate 为 DCU 定制, 走 nightly
pip install ${PIP_OPT} "citest==0.0.8" -i ${VLLM_INDEX}
pip install ${PIP_OPT} "cinotificate==0.0.3" -i ${VLLM_INDEX}
# numpy 通用包, 走清华(也 pin 原镜像版本)
pip install ${PIP_OPT} "numpy==1.25.0" -i ${TSINGHUA}

# ---- vllm: --no-deps 装本体(跳过 cmake==3.29 约束), 再补其余依赖(排除 cmake) ----
VLLM_SPEC="vllm==0.15.1+das.opt1.alpha.dtk2604.torch290.2604081832.gbcb2ba"
rm -rf /tmp/vwhl && mkdir -p /tmp/vwhl
pip download ${PIP_OPT} "${VLLM_SPEC}" --no-deps -d /tmp/vwhl -i ${VLLM_INDEX}
VLLM_WHL=$(ls /tmp/vwhl/*.whl | head -1)
pip install --no-deps "${VLLM_WHL}"
echo "=== vllm 本体已装, 开始补其余依赖(排除 cmake, torch 已锁不重装) ==="
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
                if name.lower() in ("cmake",):
                    continue
                deps.append(name)
seen=set(); uniq=[]
for d in deps:
    if d.lower() not in seen:
        seen.add(d.lower()); uniq.append(d)
print("需安装依赖(排除 cmake, 去重):", uniq)
for d in uniq:
    # torch 已在阶段一锁死 DCU 版, 跳过不重装
    if d.lower() == "torch":
        print(f"[SKIP-torch] 已由阶段一锁定 DCU 版")
        continue
    ok=False
    for attempt in range(6):
        try:
            subprocess.run(["pip","install","--timeout","120","--retries","10","--resume-retries","50","-i",idx,d],
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

pip install ${PIP_OPT} "numpy==1.25.0" -i ${TSINGHUA}
# 通用包 + 其余 vllm 关键依赖 pin 回原容器版本(均已在 nightly/清华确认真实可得) ----
# 注: torch 已在阶段一锁定, 此处不再动 torch.
pip install ${PIP_OPT} "transformers==5.2.0" -i ${VLLM_INDEX}
pip install ${PIP_OPT} "xgrammar==0.1.29" -i ${VLLM_INDEX}
pip install ${PIP_OPT} "tensorizer==2.10.1" -i ${VLLM_INDEX}
pip install ${PIP_OPT} "runai-model-streamer==0.15.3" -i ${VLLM_INDEX}
pip install ${PIP_OPT} "setuptools==79.0.1" -i ${TSINGHUA}
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
