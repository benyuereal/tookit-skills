#!/bin/bash
# =============================================================================
# DCU 物理机环境激活脚本 — 复刻镜像
#   <internal_registry>/dcu/admin/base/custom:sglang0.5.12-ubuntu22.04-dtk26.04-py3.10-20260620
#
# 用法:  source /public/dcu-env/activate_dcu_env.sh
# 退出:  conda deactivate
#
# 环境布局（按实际拷贝位置调整）:
#   /public/miniconda3/envs/sglang0512   conda py3.10 环境（含镜像全部 200+ 包）
#   /opt/dtk -> /public/dtk -> /public/dtk-26.04   完整 DTK 26.04（从镜像拷出，17G）
#   /public/dcu-env/opt/{ucx,mpi,hwloc} DTK 配套组件（从镜像拷出）
#   /opt/hyhal                          物理机 DCU 驱动（系统自带，勿动）
# =============================================================================

# ---------- 1. DTK 26.04 环境变量（对齐镜像 env） ----------
# 关键：ROCM_PATH 必须指向目录名为 "dtk" 的路径（/opt/dtk 软链 -> /public/dtk -> /public/dtk-26.04）。
# Triton 的 path_to_rocm_llvm() 靠 rocm_path.name=="dtk" 判断用 aillvm(clang-18) 还是 llvm(clang-17)。
# 若直接指向 /public/dtk-26.04（name="dtk-26.04"）会误选 llvm/dcc 的 clang-17，导致 Triton kernel 编译失败。
export DTKROOT=/opt/dtk
export ROCM_PATH=$DTKROOT
export HIP_PATH=$ROCM_PATH/hip
export HYHAL_PATH=/opt/hyhal
export DTK_VERSION=2604
export AMDGPU_TARGETS="gfx906;gfx926;gfx928;gfx936;gfx938"
export MIOPEN_FIND_MODE=3

# LD_LIBRARY_PATH 必须显式列出（镜像内库 RUNPATH 写死 /opt/dtk，物理机该路径指向 25.04，
# LD_LIBRARY_PATH 优先级高于 RUNPATH，可正确覆盖）
export LD_LIBRARY_PATH=$DTKROOT/dcc/gcvm/lib:$DTKROOT/hip/lib:$DTKROOT/llvm/lib:$DTKROOT/lib:$DTKROOT/lib64:/opt/hyhal/lib:/opt/hyhal/lib64:$DTKROOT/dushmem/lib:$DTKROOT/opencl/lib:/public/dcu-env/opt/ucx/lib:/public/dcu-env/opt/mpi/lib:/public/dcu-env/opt/hwloc/lib:$LD_LIBRARY_PATH

export PATH=$DTKROOT/bin:$DTKROOT/llvm/bin:$DTKROOT/hip/bin:$DTKROOT/hip/bin/hipify:/opt/hyhal/bin:$DTKROOT/opencl/bin:/public/dcu-env/opt/ucx/bin:/public/dcu-env/opt/mpi/bin:/public/dcu-env/opt/hwloc/bin:$PATH

export C_INCLUDE_PATH=$DTKROOT/dcc/gcvm/include:$DTKROOT/include:/opt/hyhal/include:$DTKROOT/llvm/include:$DTKROOT/opencl/include
export CPLUS_INCLUDE_PATH=$DTKROOT/dcc/gcvm/include:$DTKROOT/include:/opt/hyhal/include:$DTKROOT/llvm/include:$DTKROOT/dushmem/include:$DTKROOT/opencl/include
export LIBRARY_PATH=/opt/hyhal/lib:$DTKROOT/opencl/lib:$DTKROOT/dushmem/lib
export CMAKE_PREFIX_PATH=/opt/hyhal:

# ---------- 2. conda 环境（镜像全部 Python 包） ----------
source /public/miniconda3/etc/profile.d/conda.sh
conda activate sglang0512

echo "============================================================"
echo " DCU 环境已激活（复刻 sglang0.5.12-dtk26.04-py3.10 镜像）"
echo "   DTK:      $DTKROOT"
echo "   Python:   $(which python)"
echo "   退出:     conda deactivate"
echo "============================================================"
