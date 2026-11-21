# =============================================================================
# 实战 native INT8 MoE kernel 编译脚本（K100 AI / gfx928 / vLLM 0.18.1 / DTK 26.04）
# -----------------------------------------------------------------------------
# 来源：本地复现工程 <local_repro_pkg>_2026-08-07.gz
#        native_ext/build_k100_channelwise_int8_moe.py
# 安全说明：纯构建脚本，无密码 / 密钥 / 凭据。
# 功能：编译 k100_channelwise_int8_moe.hip 为 .so；关键动作是 patch AITER 的
#       moe_w8a8_block_wise_hip.h 模板，把 final scale 实现改为 true per-output-channel
#       （group_size_n==1），否则多 channel scale 被塌缩成标量会算错。
# 脱敏说明（原文件含以下绝对路径，已替换为占位，使用前按真实环境回填）：
#   - ROOT = /data/qwen36-k100-int8-opt  ->  <PROJECT_ROOT>（你的项目目录）
#   - AITER 模板路径 build/aiter-gfx928-int8native/build/module_moe_c_kernel/build/srcs
#     ->  <AITER_BUILD>/srcs 与 /include（AITER 在容器内的构建产物目录）
#   - /usr/local/lib/python3.10/dist-packages/aiter_meta/3rdparty/composable_kernel/...
#     ->  <VLLM_SITE_PACKAGES>/aiter_meta/3rdparty/composable_kernel/{include,library/include}
#   - native_ext 输出：<PROJECT_ROOT>/native_ext/k100_channelwise_int8_moe.so
# 使用前提：须在 DTK 26.04 容器内运行（需 hipcc / torch.utils.cpp_extension），
#           PYTORCH_ROCM_ARCH=gfx928；且 AITER 源码已就位。
# =============================================================================

#!/usr/bin/env python3
from __future__ import annotations

import os
import shutil
from pathlib import Path

from torch.utils.cpp_extension import load

ROOT = Path("<PROJECT_ROOT>")
SRC = ROOT / "native_ext/k100_channelwise_int8_moe.hip"
AITER = ROOT / "<AITER_BUILD>"
OUT = ROOT / "native_ext/build"
OUT.mkdir(parents=True, exist_ok=True)
GENERATED = ROOT / "native_ext/generated"
GENERATED.mkdir(parents=True, exist_ok=True)

# Patch the AITER template for true per-output-channel scales. The upstream
# source loads several channel scales but collapses them to one scalar, which
# is only valid when 128 output channels share a scale. For GROUP_SIZE_N=1,
# keep activation scaling in the accumulation and apply the exact channel
# weight scale at the output lane.
header_src = AITER / "srcs/moe_w8a8_block_wise_hip.h"
header_text = header_src.read_text()
old_final = "final_scales[m] = expert_scales_groups_float * input_scales_float[m];"
new_final = """if constexpr (group_size_n == 1) {
                    final_scales[m] = input_scales_float[m];
                  } else {
                    final_scales[m] = expert_scales_groups_float * input_scales_float[m];
                  }"""
final_index = header_text.rfind(old_final)
if final_index < 0:
    raise RuntimeError("final scale pattern not found")
header_text = (
    header_text[:final_index]
    + new_final
    + header_text[final_index + len(old_final):]
)
old_store = """res_float[m][n_idx*(WARP_N/16) + i][n_loop][j] = res[m][n_idx*(WARP_N/16) + i][n_loop][j] /* * expert_scales_groups_input[m] * expert_scales_groups[0][0].float_array[0] */;"""
new_store = """float channel_weight_scale = 1.0f;
                             if constexpr (group_size_n == 1) {
                               const int output_channel = n_idx*32 + offset_n + i*16 + j*4*(WARP_N/16);
                               channel_weight_scale = expert_scales[output_channel * (size_k/group_size_k)];
                             }
                             res_float[m][n_idx*(WARP_N/16) + i][n_loop][j] =
                                 res[m][n_idx*(WARP_N/16) + i][n_loop][j] * channel_weight_scale;"""
store_index = header_text.rfind(old_store)
if store_index < 0:
    raise RuntimeError("output scale pattern not found")
header_text = (
    header_text[:store_index]
    + new_store
    + header_text[store_index + len(old_store):]
)
(GENERATED / "moe_w8a8_block_wise_hip.h").write_text(header_text)

os.environ.setdefault("PYTORCH_ROCM_ARCH", "gfx928")
os.environ.setdefault("TORCH_EXTENSIONS_DIR", str(OUT))

module = load(
    name="k100_channelwise_int8_moe",
    sources=[str(SRC)],
    extra_include_paths=[
        str(GENERATED),
        str(AITER / "srcs"),
        str(AITER / "include"),
        "<VLLM_SITE_PACKAGES>/aiter_meta/3rdparty/composable_kernel/include",
        "<VLLM_SITE_PACKAGES>/aiter_meta/3rdparty/composable_kernel/library/include",
    ],
    extra_cflags=["-O3", "-std=c++20"],
    extra_cuda_cflags=[
        "-O3",
        "-std=c++20",
        "-DDTK_ENV",
        "-fno-gpu-rdc",
    ],
    with_cuda=True,
    verbose=True,
)

src_so = Path(module.__file__)
dst_so = ROOT / "native_ext/k100_channelwise_int8_moe.so"
shutil.copy2(src_so, dst_so)
print(dst_so)
