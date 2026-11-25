#!/bin/sh
# 编译融合 kernel（gfx938）。BLOCK_ROWS=2 是实测最优（4 行因寄存器压力退化）。
set -e
cd "$(dirname "$0")"
hipcc -O3 --offload-arch=gfx938 -fPIC -shared -DBLOCK_ROWS=2 \
      -o mr2.so norm_act_swiglu_mr.hip
echo "built mr2.so (BLOCK_ROWS=2)"
