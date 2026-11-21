#!/bin/bash
# 问题二修复: DTK 26 设备属性 maxThreadsDim/maxGridSize 类型变为 uint32_t[3]，
# 源码仍按旧 HIP 声明为 int*，导致编译错误。将这几处声明改为 auto 推断。
# 用法: bash patches/patch_maxThreadsDim.sh /path/to/pytorch
set -e
PT=${1:?"用法: bash patches/patch_maxThreadsDim.sh <pytorch源码目录>"}
cd "$PT"
for f in \
  aten/src/ATen/native/hip/UpSampleBilinear2d.hip \
  aten/src/ATen/native/hip/UpSampleNearest2d.hip \
  aten/src/ATen/native/hip/AdaptiveAveragePooling.hip \
  aten/src/ATen/native/hip/DilatedMaxPool2d.hip
do
  echo "[patch] $f"
  sed -i \
    -e 's/int\* maxThreadsDim =/auto maxThreadsDim =/g' \
    -e 's/int\* maxGridSize =/auto maxGridSize =/g' \
    "$f"
done
echo "[ok] maxThreadsDim/maxGridSize 已改为 auto。增量编译会自动重编相关目标。"
