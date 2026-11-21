#!/bin/bash
# 问题三修复: DTK 26 / ROCm 6.x 的 hipsparseDestroy* 已改为 const 签名，
# 但旧版宏逻辑在 ROCM_VERSION>=50200 时打开了非 const 分支，导致编译错误。
# 本脚本在 aten/src/ATen/hip/HIPSparse.h 末尾追加正确的宏定义：
#   ROCM_VERSION>=60000 -> AT_USE_CUSPARSE_CONST_DESCRIPTORS=1
#   ROCm 5.2<=v<6.0     -> AT_USE_HIPSPARSE_GENERIC_52_API=1
# 用法: bash patches/patch_hipsparse_h.sh /path/to/pytorch
set -e
PT=${1:?"用法: bash patches/patch_hipsparse_h.sh <pytorch源码目录>"}
F="$PT/aten/src/ATen/hip/HIPSparse.h"
[ -f "$F" ] || { echo "[err] 找不到 $F"; exit 1; }
if grep -q "DCU DTK26 patch" "$F"; then
  echo "[skip] 已包含 DCU DTK26 patch，跳过"
  exit 0
fi
cp -a "$F" "$F.bak"
cat >> "$F" <<'EOF'

// === DCU DTK26 patch: ROCm 6.x+ const descriptor (对齐 CUDA 12) ===
#if defined(USE_ROCM) && ROCM_VERSION >= 60000
#define AT_USE_CUSPARSE_NON_CONST_DESCRIPTORS() 0
#define AT_USE_CUSPARSE_CONST_DESCRIPTORS() 1
#endif
#if defined(USE_ROCM) && ROCM_VERSION >= 50200 && ROCM_VERSION < 60000
#define AT_USE_HIPSPARSE_GENERIC_52_API() 1
#else
#define AT_USE_HIPSPARSE_GENERIC_52_API() 0
#endif
// === end DCU DTK26 patch ===
EOF
echo "[ok] 已追加宏定义到 $F，touch 相关头文件触发重编"
