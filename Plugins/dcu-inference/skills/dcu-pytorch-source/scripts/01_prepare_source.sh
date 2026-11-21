#!/bin/bash
# 获取 PyTorch 2.1.0 (DAS 分支) 源码，初始化子模块并 HIP 化。
# 用法: bash 01_prepare_source.sh /path/to/pytorch
set -e
PT=${1:?"用法: bash 01_prepare_source.sh <pytorch源码目录>"}
cd "$PT"
git checkout "${PT_BRANCH:-your-2.1.0-das-branch}"
git submodule sync --recursive
git submodule update --init --recursive
echo "[*] submodule 状态（不应大量出现 '-' 前缀）:"
git submodule status | head
echo "[*] 执行 HIP 化 (build_amd.py) —— 'skipped, already hipified' 属正常现象"
python tools/amd_build/build_amd.py
echo "[ok] 源码准备完成"
