#!/bin/bash
# 修复 DTK 26.04 的 version-dev，使其满足 LoadHIP.cmake 的 x.y.z 正则。
# 必须在 cmake 配置前执行。
set -e
ROCM_H=/opt/dtk/include/rocm_version.h
INFO=/opt/dtk/.info/version-dev
echo "[*] 读取真实 ROCm 版本 (MAJOR.MINOR.PATCH) ..."
MAJ=$(grep -oP "ROCM_VERSION_MAJOR \K[0-9]+" "$ROCM_H")
MIN=$(grep -oP "ROCM_VERSION_MINOR \K[0-9]+" "$ROCM_H")
PAT=$(grep -oP "ROCM_VERSION_PATCH \K[0-9]+" "$ROCM_H")
VER="${MAJ}.${MIN}.${PAT}"
echo "[*] 检测到 ROCm ${VER}"
[ -f "$INFO" ] && cp -a "$INFO" "$INFO.bak"
echo "${VER}-26.04" > "$INFO"
echo "[ok] 已写入 $INFO : $(cat "$INFO")"
