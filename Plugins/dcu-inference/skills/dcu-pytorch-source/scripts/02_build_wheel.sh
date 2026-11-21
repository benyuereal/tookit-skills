#!/bin/bash
# 增量编译 PyTorch 2.1.0 wheel。
# 注意: 严禁随意执行 setup.py clean（会清掉中间产物导致 2-4h 全量重编）。
#       遇编译错误修复后直接重跑本脚本，ninja 自动增量。
# 用法: bash 02_build_wheel.sh /path/to/pytorch [wheel输出目录]
set -e
PT=${1:?"用法: bash 02_build_wheel.sh <pytorch源码目录> [wheel输出目录]"}
WHEELS=${2:-/path/to/pytorch-wheels}
cd "$PT"
mkdir -p "$WHEELS"
python setup.py bdist_wheel -d "$WHEELS" 2>&1 | tee /path/to/pytorch-build.log
echo "[ok] wheel 位于 $WHEELS"
