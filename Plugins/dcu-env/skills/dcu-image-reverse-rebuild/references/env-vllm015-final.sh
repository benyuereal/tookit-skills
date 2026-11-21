#!/bin/sh
# DTK 环境注入 —— 对应原镜像 /etc/profile.d/env.sh
# 原镜像通过该文件加载 /opt/dtk/env.sh 补全 DTK 运行时库路径
# (含 /opt/dtk/.hyhal/rocm_smi/lib, 即 librocm_smi64.so.2 所在, 否则 import torch 缺库)
# 同时保留 hwloc/mpi 等 PATH/LD_LIBRARY_PATH (DTK env 已含, 此处兜底追加)
if [ -f /opt/dtk/env.sh ]; then
    . /opt/dtk/env.sh
fi
export PATH=/opt/hwloc/bin:/opt/mpi/bin:/opt/ucx/bin:${PATH}
export LD_LIBRARY_PATH=/opt/hwloc/lib:/opt/mpi/lib:/opt/ucx/lib:${LD_LIBRARY_PATH}
