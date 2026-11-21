#!/bin/bash
# 已知可用的 install_dep.sh 模板（DCU vllm0.15.1-dtk2604 镜像逆向重建）
# 解决死结: cmake==3.29 缺失(--no-deps 绕) / import torch 缺库(Dockerfile ENV LD_LIBRARY_PATH 含 rocm_smi/lib) / torch+triton+numpy+transformers+xgrammar 版本漂移
# 配合 Dockerfile: 主装层 pip.conf 用 pypi.org 主索引 + nightly/dtk extra-index
set -e
VLLM_INDEX="https://pypi.sourcefind.cn/nightly/dtk/"

pip install citest -i ${VLLM_INDEX}
pip install cinotificate -i ${VLLM_INDEX}
pip install amdsmi==24.5.3+02cbffb.dirty -i ${VLLM_INDEX}
pip install numpy==1.25.0 -i ${VLLM_INDEX}

# ---- vllm: --no-deps 装本体(绕过 cmake==3.29 硬约束), 再补其余依赖(排除 cmake) ----
VLLM_SPEC="vllm==0.15.1+das.opt1.alpha.dtk2604.torch290.2604081832.gbcb2ba"
rm -rf /tmp/vwhl && mkdir -p /tmp/vwhl
pip download "${VLLM_SPEC}" --no-deps -d /tmp/vwhl -i ${VLLM_INDEX}
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
    for attempt in range(4):
        try:
            subprocess.run(["pip","install","--timeout","120","-i",idx,d],
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
            time.sleep(5)
    if not ok:
        print(f"[WARN] {d} 多次重试仍失败, 继续(不阻断整体)")
PY
# ---- 版本对齐: 覆盖 nightly 可能拉错的最新版, 锁回 vllm0.15.1(torch290) 要求 ----
# torch/triton/numpy 是硬依赖; transformers/xgrammar 逐包补依赖会装成 4.57.6/0.2.5,
# 而 vllm import 要求 transformers==5.2.0 / xgrammar==0.1.29, 不锁会 import 失败。
pip install "torch==2.9.0+das.opt1.dtk2604.2607281120.g8d7e41" -i ${VLLM_INDEX}
pip install "triton==3.3.0+das.opt2.dtk2604.torch290.20260331.g31542e" -i ${VLLM_INDEX}
pip install "numpy==1.25.0" -i ${VLLM_INDEX}
pip install "transformers==5.2.0" -i ${VLLM_INDEX}
pip install "xgrammar==0.1.29" -i ${VLLM_INDEX}
rm -rf /tmp/vwhl
