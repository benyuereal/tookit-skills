---
name: dcu-lightop-moe
description: LightOp MoE 融合算子多量化路径与适配索引
version: 1.0.0
author: DCU Engineer
license: MIT
---

# LightOp MoE 算子深度索引（dcu-lightop-moe）

> 从 `dcu-lightop-source` 析出的 MoE 专题。总览/构建/DTK 矩阵见 `dcu-lightop-source`；本文聚焦 LightOp 的 MoE 实现、落点、机制、与 AITER MoE 的对照。
> 源码基准：`<local_src_cache>/lightop-master/`（海光自研融合算子库）。

## §0 定位：LightOp MoE 是什么（vs AITER MoE）

LightOp 的 MoE 是**海光自研**实现（非 AMD ROCm 移植），与 `dcu-aiter-moe` 是**两套独立实现**，选型时按 `dcu-lightop-source` §0 的对照表决定改哪个。

| 维度 | LightOp MoE（本文） | AITER MoE（dcu-aiter-moe） |
|---|---|---|
| 架构 | 多量化路径直接实现 + asm 后端，无"四后端自动调度框架" | 有 `get_aiter_moe_config` 四后端（moe_c/asm/triton/ck）调度框架 |
| 量化覆盖 | w8a8 / w4a8 / w16a16 + Marlin(w4a16/w4a8/fp8) | 同量化类型，但经统一选择器分发 |
| token 重排 | **permute/unpermute 机制**（`deepgemm_moe_permute.cu`、`groupgemm_permute_unpermute.cu`、`moe_align_block_size*.cu`） | 外部 `aiter_moe_shfl_weight` shuffle（**必做，否则静默错误**） |
| vLLM 衔接 | 自带 `vllm_compat/fused_moe_cache.py` + `_lmslim_native/layers/fused_moe` | 经 `dcu-vllm-op-replace` 6 步接入 |
| 专用变体 | `moe_fused_gate_kimi_k2`（Kimi K2）、`moe_fused_gate_sqrtsoftplus`、`moe_fuse_gate_per_gourp_topk` | 通用 fused gate |

## §1 量化路径与实现落点（动手改时按此定位）

**所有 MoE GEMM 的 pybind 入口都在 `csrc/export.cpp`（159–188 行）**——改 C++ 接口签名从这里看：
- `moe_gemm_w8a8` / `moe_gemm_marlin_w8a8` / `moe_gemm_marlin_w8a8_fp8` / `moe_gemm_marlin_w8a8_blockwise_fp8`（159–162）
- `moe_gemm_w4a8`（163）、`moe_gemm_w16a16`（164）
- asm 变体：`moe_w8a8_asm` / `moe_marlin_w8a8_asm` / `moe_marlin_w8a8_fp8_asm`（182–184）、`moe_w4a8_asm` / `moe_w4a8_marlin_asm`（185–186）、`moe_groupgemm_w4a8_asm` / `moe_groupgemm_marlin_w4a8_asm`（187–188）

**kernel 实现落点（按路径改）**：
| 量化/后端 | csrc 文件 | 说明 |
|---|---|---|
| w8a8 通用 | `csrc/moew8a8/`（目录） | w8a8 融合 MoE GEMM |
| w4a8 | `csrc/moew4a8/`、`csrc/asm_cu/asm_moe_w4a8.cu`、`asm_moe_w4a8_marlin.cu` | 含 Marlin 变体 |
| w16a16 | `csrc/moew16a16/`、`csrc/asm_cu/asm_moe_w16a16.cu`(+marlin) | BF16 直通 |
| Marlin w4a16 | `csrc/marlinw4a16/`（`moe_w4a16.cu`、`awq_marlin_repack.cu`、`marlin.h`） | W4A16 Marlin 权重重排+GEMM |
| Marlin w4a8 | `csrc/marlinw4a8/`（同结构） | |
| Marlin fp8 | `csrc/marlinw4fp8/`（目录） | |
| asm 后端全套 | `csrc/asm_cu/asm_moe_*`（w8a8/w4a8/w16a16 + marlin） | 汇编 kernel |
| 通用 fused gate | `csrc/moe_fused_gate.cu` | 门控融合（含 Kimi/sqrtsoftplus 变体文件同目录） |
| Kimi-K2 专用 | `csrc/moe_fused_gate_kimi_k2.cu` | K2 模型特化 |
| sqrtsoftplus | `csrc/moe_fused_gate_sqrtsoftplus.cu` | 激活变体 |
| per-group topk | `csrc/moe_fuse_gate_per_gourp_topk.cu` | 分组 topk 门控 |
| SwiGLU 动态量化 | `csrc/moe_swiglu_dynamic_quant.cu` | 门控+动态量化融合 |
| align_sum | `csrc/moe_align_sum_kernels.cu` | token 聚合 |

## §2 token 重排机制（踩坑重点，区别于 AITER）

LightOp **不用** AITER 那种"调用前必须 shuffle 权重"的模式，而是用 **permute/unpermute**：
- `csrc/deepgemm_moe_permute.cu`：GEMM 前按专家对 token 重排（替代外部 shuffle）
- `csrc/groupgemm_permute_unpermute.cu` + `csrc/groupgemm_permute_unpermute` 逻辑：GEMM 后 unpermute 还原
- `csrc/moe_align_block_size*.cu`（及 `python/lightop/_lmslim_native/vllm_compat/moe_align_block_size.py`）：block 对齐（类似 AITER 的 moe_align_block_size）

**踩坑**：
- 若要手写新的 MoE 路径，**必须走 permute→GEMM→unpermute**，不要模仿 AITER 的"外部 shuffle 权重"思路（两库机制不同，混用会 layout 错）。
- `moe_align_block_size` 参数需和 `configs/MOE_BLOCKINT4_*` 调优表的 block 维度一致，否则性能下降或越界。
- 改 `export.cpp` 的 MoE 函数签名时，Python 侧 `python/lightop/moe/` 的对应绑定也要同步（pybind 名称须一致）。

## §3 Python 接口与 vLLM 兼容层

- **Python 接口**：`python/lightop/moe/`（MoE 主目录）、`python/lightop/moe_fused_gate.py`、`moe_align_sum.py`、`moe_fuse_gate_per_gourp_topk.py`、`fuse_topk_transform.py`
- **vLLM 兼容层（重要）**：`python/lightop/_lmslim_native/vllm_compat/fused_moe_cache.py` + `python/lightop/_lmslim_native/layers/fused_moe/` —— LightOp 设计上就考虑了接 vLLM，改 MoE 后优先走这条兼容路径验证，而非从头写 `dcu-vllm-op-replace` 6 步。
- **测试入口**：`test/run_moe_tests.py`、`test/moe_*`（moe_w8a8 / moe_w4a8 / moe_w16a16 / moe_marlin / moe_groupgemm_w4a8 / test_fused_moe_gate / test_kimi_k2_moe_fused_gate / test_moe_align_block_size* / test_moe_swiglu_dynamic_quant 等）
  ```bash
  cd test && python3 run_moe_tests.py        # 跑全部 MoE 测试
  python3 test/moe_w8a8/test_*.py           # 单测某路径
  ```

## §4 调优表（分 gfx，换卡必换）

- 位置：`configs/MOE_BLOCKINT4_*.json`
- 命名规则：`MOE_BLOCKINT4_CUDA_MARLIN_<M>_<N>_gfx{936,938}_{CU}.json`（含非 Marlin 的 `BW200` 变体）
- 例：`MOE_BLOCKINT4_CUDA_MARLIN_1024_7168_gfx936_80.json`、`..._gfx938_72.json` —— 不同 gfx + 不同 CU 数有独立调优
- **换 gfx（936↔938）必须换对应 json**；编译/运行时不匹配会 fallback 或性能劣化。

## §5 与 vLLM / AITER 的衔接

- **优先 vLLM 兼容层**：LightOp 自带 `vllm_compat/fused_moe_cache.py`，改 MoE 后先在此验证接 vLLM，比 `dcu-vllm-op-replace` 6 步更直接（见 dcu-lightop-source §6）。
- **与 AITER MoE 对照**：两库 MoE 独立实现，选型见 `dcu-lightop-source` §0；注意 `aiter_hip_common.h` 是 LightOp↔AITER 共享头（改 HIP 公共定义需双向确认）。
- **开发规范 / 验证闭环**：`dcu-kernel-dev`（HIP 语法/编译）、`dcu-op-optimization` + `dcu-perf-profiling`（profile→定位→改→验证）。
- **vLLM 原生 MoE 对照**：vLLM 0.15.1 有 `csrc/moe/moe_fused_gate.cu`（dcu-vllm-source §csrc），可与 LightOp `moe_fused_gate.cu` 对照异同。

## §6 验证入口（实测命令）

```bash
# 安装（含 MoE kernel，按 gfx 指定）
PYTORCH_ROCM_ARCH='gfx936;gfx938' python3 setup.py install
python -c "import lightop; print(lightop.__version__)"

# MoE 测试
cd test && python3 run_moe_tests.py
# 或单路径
python3 test/moe_w8a8/<具体测试脚本>.py
python3 test/moe_marlin/<具体测试脚本>.py
```

## 使用须知
- 经验仅供参考，不保证修复。**先核对**：gfx 型号、DTK 版本、pytorch 版本、报错原文。LightOp 无公开上游，以 DTK 矩阵 + 实际编译为准（dcu-lightop-source §3）。
- 改 MoE kernel 后必须重编 `setup.py install`；`PYTORCH_ROCM_ARCH` 与 gfx 强相关，**换 gfx 必须重编**（旧 DTK 去 928）。
- 新 MoE 路径必须走 permute→GEMM→unpermute（§2），不要套用 AITER 的 shuffle 思路。
- 高危操作（重装驱动/刷 vbios）需谨慎，确认带外恢复通道、优先测试机验证（dcu-ontrack-experience 使用须知）。
