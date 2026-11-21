#!/bin/bash
# DTK 26.04 + PyTorch 2.1.0 编译环境变量（在 pytorch 容器内 source 本文件）
# 来源: 内部文档《DCU PyTorch框架迁移适配与算子验证实践》
set -a
# ── ROCm/HIP 路径 ──
export ROCM_PATH=/opt/dtk
export ROCM_HOME=/opt/dtk
export HIP_PATH=/opt/dtk/hip
export PATH=/opt/dtk/bin:/opt/dtk/hip/bin:/opt/dtk/llvm/bin:$PATH
export LD_LIBRARY_PATH=/opt/dtk/lib:/opt/dtk/lib64:/opt/dtk/hip/lib:/opt/dtk/miopen/lib:/opt/dtk/rocblas/lib:$LD_LIBRARY_PATH

# ── PyTorch 构建开关 ──
export USE_ROCM=1          # 启用 ROCm/HIP 后端
export USE_CUDA=0          # 禁用 CUDA
export USE_NINJA=1         # 使用 ninja 加速编译
export BUILD_TEST=0        # 跳过测试，加速构建
export USE_DISTRIBUTED=1   # 启用分布式支持
export USE_MKLDNN=1        # 启用 MKL-DNN
export BUILD_NVFUSER=0     # 必须关闭！DTK 26 下 nvfuser 编译失败
export USE_CUSPARSELT=0    # 关闭 cuSPARSELt

# ── DCU 架构（BW100 = gfx936；其他卡用 rocminfo 确认）──
export PYTORCH_ROCM_ARCH=gfx936
export HCC_AMDGPU_TARGET=gfx936
export AMDGPU_TARGETS=gfx936

# ── 编译并行度与 ABI ──
export MAX_JOBS=240
export _GLIBCXX_USE_CXX11_ABI=1
export GLIBCXX_USE_CXX11_ABI=1

# ── 输出版本号 ──
export PYTORCH_BUILD_VERSION=2.1.0
export PYTORCH_BUILD_NUMBER=11
set +a
echo "[env] DTK/PyTorch build env exported (PYTORCH_ROCM_ARCH=$PYTORCH_ROCM_ARCH)"
