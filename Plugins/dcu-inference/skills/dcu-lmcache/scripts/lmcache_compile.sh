#!/usr/bin/env bash
# LMCache 在海光 DCU 上的编译安装（基于实测文档）
# 对应版本映射：LMCache 0.3.9 <-> vLLM 0.9.2；0.4.1 <-> vLLM 0.15.1；0.4.2 <-> vLLM 0.18.1
# 注意：同版本在不同容器/镜像的 DAS commit 非固定，以实际发布版为准。
set -euo pipefail

# 选对应分支（示例为 0.3.9，对应 vLLM 0.9.2）
LMCAHE_BRANCH="${LMCAHE_BRANCH:-v0.3.9}"
git clone -b "$LMCAHE_BRANCH" https://github.com/LMCache/LMCache
cd LMCache

# 海光 DCU 编译关键：BUILD_WITH_HIP=1 + hipcc + 指定 gfx936
# 注意：PYTORCH_ROCM_ARCH 按目标卡型填（gfx936 为 BW100 系；K100/Z100 系按实际 gfx 填）
pip uninstall setuptools-scm
PYTORCH_ROCM_ARCH="gfx936" \
TORCH_DONT_CHECK_COMPILER_ABI=1 \
CXX=hipcc \
BUILD_WITH_HIP=1 \
python3 -m pip install --no-build-isolation --no-dependencies -e . \
  -i https://pypi.tuna.tsinghua.edu.cn/simple

pip install aiofile -i https://pypi.tuna.tsinghua.edu.cn/simple
echo "OK: LMCache 编译安装完成 (branch=$LMCAHE_BRANCH)"
