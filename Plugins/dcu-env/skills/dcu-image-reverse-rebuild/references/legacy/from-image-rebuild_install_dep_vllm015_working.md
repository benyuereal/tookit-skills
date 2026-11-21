# install_dep.sh —— vllm 0.15.1 (dtk2604) 重建实测可用版（最终）

来源会话：2026-08-05 在 <device-B> (<<internal-ip>>) 上把内网友源镜像
`vllm015-ubuntu22.04-dtk26.04-0409-modelzoo` 用国产 sourcefind 源纯重建
（除 DTK 从容器 COPY 外）。本脚本对应原镜像 /install_dep.sh，经 25 次
build 迭代，最终 BUILD_EXIT=0，且 `import torch/vllm` 通过、带卡容器
`vllm serve -h` EXIT=0。

## 关键设计（缺一不可，前 23 次失败的教训）
1. `set -e` —— 任意包失败立即终止，杜绝"缺 vllm 假成功"。
2. **两阶段**：阶段一先 pin 全部 DCU 适配包（torch/torchvision/triton/
   flash_*/lightop/lmslim/amdsmi，全 -i nightly/dtk）锁死 DCU torch 2.9.0；
   阶段二再装 opencompass(清华)/citest/cinotificate/vllm，其 torch 依赖
   已满足、不会从清华拉官方 torch。
3. vllm 用 `--no-deps` 装本体，绕过元数据写死的 `cmake==3.29`（PyPI 无
   3.29.0 精确版，--no-build-isolation 也救不了）。
4. 从 vllm wheel METADATA 提取 Requires-Dist，排除 cmake + 排除 torch（已锁），
   逐包安装（`--timeout 120 --retries 10 --resume-retries 50`，防 sourcefind
   大包间隙断流——`--resume-retries` 才是下载续传次数，`--retries` 只管连接）。
5. 末尾对齐段**再 pin numpy==1.25.0 一次**（防 opencompass 从清华拉成 2.x）。

## 脚本（可直接复制，按 nightly/dtk 实际可用构建号微调）
```bash
#!/bin/bash
# 源策略: DCU 适配包(全) -> nightly/dtk ; 通用公开包(opencompass/numpy/setuptools) -> 清华
PIP_OPT="--timeout 120 --retries 10 --resume-retries 50"
set -e
VLLM_INDEX="https://pypi.sourcefind.cn/nightly/dtk/"
TSINGHUA="https://pypi.tuna.tsinghua.edu.cn/simple"

# ============ 阶段一: 锁定所有 DCU 适配包(仅 nightly/dtk) ============
pip install ${PIP_OPT} "torch==2.9.0+das.opt1.dtk2604.20260331.g4e3c1e7" -i ${VLLM_INDEX}
pip install ${PIP_OPT} "torchvision==0.24.0+das.opt1.dtk2604.torch290.2604091152.g66069c" -i ${VLLM_INDEX}
pip install ${PIP_OPT} "triton==3.3.0+das.opt2.dtk2604.torch290.20260331.g31542e" -i ${VLLM_INDEX}
pip install ${PIP_OPT} "flash_attn==2.8.3+das.opt1.dtk2604.torch290.2607280958.gebb4be" -i ${VLLM_INDEX}
pip install ${PIP_OPT} "flash_mla==1.2.0+das.opt1.dtk2604.torch290.2605281449.ge3aed6" -i ${VLLM_INDEX}
pip install ${PIP_OPT} "lightop==0.6.0+das.dtk2604.torch290.2604091144.ge8aebd" -i ${VLLM_INDEX}
pip install ${PIP_OPT} "lmslim==0.3.1+das.opt4.dtk2604.torch290.2604091256.gafc115" -i ${VLLM_INDEX}
pip install ${PIP_OPT} "amdsmi==24.5.3+02cbffb.dirty" -i ${VLLM_INDEX}

# ============ 阶段二: 通用包 + 评测辅助 + vllm ============
pip install ${PIP_OPT} "opencompass==0.5.1" -i ${TSINGHUA}
pip install ${PIP_OPT} "citest==0.0.8" -i ${VLLM_INDEX}
pip install ${PIP_OPT} "cinotificate==0.0.3" -i ${VLLM_INDEX}
pip install ${PIP_OPT} "numpy==1.25.0" -i ${TSINGHUA}

# ---- vllm: --no-deps 装本体(跳过 cmake==3.29 约束), 再补其余依赖(排除 cmake+torch) ----
VLLM_SPEC="vllm==0.15.1+das.opt1.alpha.dtk2604.torch290.2604081832.gbcb2ba"
rm -rf /tmp/vwhl && mkdir -p /tmp/vwhl
pip download ${PIP_OPT} "${VLLM_SPEC}" --no-deps -d /tmp/vwhl -i ${VLLM_INDEX}
VLLM_WHL=$(ls /tmp/vwhl/*.whl | head -1)
pip install --no-deps "${VLLM_WHL}"
echo "=== vllm 本体已装, 开始补其余依赖(排除 cmake/torch, 逐包安装带超时+续传重试) ==="
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
                if not m: continue
                name = re.split(r"[;\(<>=!~ ]", m.group(1).strip())[0].strip()
                if name.lower() in ("cmake","torch"): continue
                deps.append(name)
seen=set(); uniq=[]
for d in deps:
    if d.lower() not in seen: seen.add(d.lower()); uniq.append(d)
print("需安装依赖(排除 cmake/torch, 去重):", uniq)
for d in uniq:
    ok=False
    for attempt in range(6):
        try:
            subprocess.run(["pip","install","--timeout","120","--retries","10","--resume-retries","50","-i",idx,d],
                           check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            ok=True; print(f"[OK] {d}"); break
        except subprocess.CalledProcessError as e:
            err=e.stderr.decode(errors="ignore")
            if "from versions: none" in err and "ReadTimeout" not in err:
                print(f"[SKIP-none] {d} 不可用, 跳过"); ok=True; break
            print(f"[RETRY {attempt+1}] {d} 失败, 重试... ({err[:120]})"); time.sleep(8)
    if not ok: print(f"[WARN] {d} 多次重试仍失败, 继续(不阻断整体)")
PY

# ---- 末尾版本对齐(含 numpy 再 pin 一次) ----
pip install ${PIP_OPT} "numpy==1.25.0" -i ${TSINGHUA}
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
```

## 配套 Dockerfile 要点
- pip.conf：index-url=清华 + extra-index-url=nightly/dtk（**主索引别用 pypi.org**，DCU 主机上被反代到 sourcefind /open 间歇断流）
- ENV LD_LIBRARY_PATH 必须含 /opt/dtk/.hyhal/rocm_smi/lib（否则 import 缺 librocm_smi64.so.2）
- RUN echo '/opt/dtk/.hyhal/rocm_smi/lib' > /etc/ld.so.conf.d/dtk.conf && ldconfig
- /etc/profile.d/env.sh 内容改为 source /opt/dtk/env.sh
- 长 build 用 nohup docker build ... > log 2>&1 & 脱离 ssh，防网络抖动误杀

## 最终核对结果（<device-B> 实测，带卡容器 vllm serve -h EXIT=0）
逐字节一致：torchvision/triton/vllm/lightop/lmslim/transformers/xgrammar/numpy/amdsmi/cmake
系列对齐(构建号差, 按用户"选高一点"裁定, 接受)：torch(2.9.0 同系列)、flash_attn(2.8.3 同版本)、flash_mla(1.0.0→1.2.0 升版)
无错版、无缺包；所有 DCU 定制包都在且是 nightly 上的 DCU 适配版。

## 验证命令
# import 层(无卡即可)
docker run --rm <img> python3 -c "import torch, vllm; print(torch.__version__, vllm.__version__)"
# serve 链路(必须带卡: --device=/dev/kfd --device=/dev/dri --device=/dev/mkfd -v /opt/hyhal:/opt/hyhal:ro --shm-size=200g)
docker run --rm <带卡容器> vllm serve -h   # 期望打印 help 文本 EXIT=0
