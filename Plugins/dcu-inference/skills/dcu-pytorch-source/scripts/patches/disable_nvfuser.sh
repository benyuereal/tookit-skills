#!/bin/bash
# 问题四修复: 直接关闭 nvfuser（LibTorch 不依赖此组件）。
# 优先环境变量；若已 cmake 配置过则用缓存方式。
# 用法: bash patches/disable_nvfuser.sh [pytorch源码目录]
set -e
export BUILD_NVFUSER=0
PT=${1:-}
if [ -n "$PT" ] && [ -f "$PT/build/CMakeCache.txt" ]; then
  sed -i 's/^BUILD_NVFUSER:BOOL=.*/BUILD_NVFUSER:BOOL=OFF/' "$PT/build/CMakeCache.txt"
  echo "[ok] 已改写 CMakeCache.txt 关闭 nvfuser"
fi
echo "[ok] BUILD_NVFUSER=0 已设置（环境变量）"
