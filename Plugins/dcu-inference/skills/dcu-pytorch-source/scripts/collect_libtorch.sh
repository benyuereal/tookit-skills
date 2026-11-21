#!/bin/bash
# 从编译产物提取独立可部署的 LibTorch 运行时：
#   - 收集 .so 动态库 (torch/lib/*.so*)
#   - 收集 include 头文件 (torch/include 完整目录)
#   - patchelf 修正 RPATH 为 $ORIGIN（否则跨环境 libc10.so not found）
# 用法: bash collect_libtorch.sh <pytorch源码目录> [输出目录 默认 /path/to/torch-2.1.0]
set -e
PT=${1:?"用法: bash collect_libtorch.sh <pytorch源码目录> [输出目录]"}
PREFIX=${2:-/path/to/torch-2.1.0}
rm -rf "$PREFIX"; mkdir -p "$PREFIX/lib" "$PREFIX/include"

# 1) 收集 so（含版本化符号链接）
cp -a "$PT"/torch/lib/*.so* "$PREFIX/lib/"
echo "[*] 关键库校验:"
ls -lh "$PREFIX/lib/" | grep -E "libtorch|libc10|libtorch_hip" || true

# 2) 收集 include（保留目录结构/符号链接）
cp -a "$PT"/torch/include/. "$PREFIX/include/"
for d in at c10 torch torch/csrc/api/include; do
  [ -d "$PREFIX/include/$d" ] || echo "[warn] 缺 include/$d"
done

# 3) 修正 RPATH 为 $ORIGIN（同目录依赖，可移植）
echo "[*] 修正 RPATH ..."
for so in "$PREFIX"/lib/*.so; do
  [ -f "$so" ] || continue
  patchelf --set-rpath '$ORIGIN' "$so"
done
echo "[ok] libtorch.so rpath = $(patchelf --print-rpath "$PREFIX/lib/libtorch.so" 2>/dev/null)"
echo "[done] LibTorch 运行时位于 $PREFIX (include/ + lib/)"
