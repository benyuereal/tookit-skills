---
name: dcu-deepgemm-source
description: DeepGEMM通用GEMM库DCU源码结构与适配
version: 1.0.0
author: DCU Engineer
license: MIT
kind: knowledge
---

# DeepGEMM 源码适配索引（dcu-deepgemm-source）

> 基于 `<local_src_cache>/deepgemm-main/`（海光 DeepGEMM，基于 DeepSeek 开源 DeepGEMM 移植+改造）。README 实测：专注通用/分组 GEMM，支持 FP8/INT8/BF16，MoE 分组场景。
> 落地标准见 `dcu-inference-infra-skill-builder/references/source-skill-landing-standard.md`（五要素：文件级+函数行号+pybind+测试+调优表）。

## §0 定位：DeepGEMM 是什么、与 AITER/LightOp/vLLM 的关系

DeepGEMM = 海光**基于 DeepSeek 开源 DeepGEMM 移植到 DCU（HIP）+ 改造**的专用 GEMM 库。专注 **grouped GEMM**（普通 + MoE 分组），支持 FP8/INT8/BF16。

| 维度 | DeepGEMM（本文） | AITER / LightOp 的 GEMM 部分 |
|---|---|---|
| 范围 | **纯 GEMM 专精**（grouped/普通，含 MoE 分组） | 大而全算子库中的 GEMM 子项 |
| 上游 | 有（DeepSeek `deepseek-ai/DeepGEMM`，CUDA 原版）；海光移植 Hip + 加 INT8/perChannel/W6 | AITER=AMD ROCm 上游；LightOp=海光自研无上游 |
| 路径细分 | masked / contiguous / low-latency 三套 × 量化类型，layout 区分细 | 各自一套 |
| 选型建议 | 要**精细控制 GEMM 路径/layout（masked vs contiguous vs low-latency、W6 packed、perChannel）**时优先 DeepGEMM | 只要标准 MoE GEMM 接入走 AITER/LightOp 即可 |

**链路**（你"写适配优化算子"）：DeepGEMM 取/改 GEMM kernel → `dcu-kernel-dev`（HIP 规范）→ `dcu-vllm-op-replace`（接入 vLLM 6 步）。vLLM 0.15/0.18 的 `csrc/moe/`、`csrc/quantization/` 已有 GEMM kernel，DeepGEMM 是更底层的 GEMM 实现来源。

## §1 源码地图（实测目录）

```
deepgemm-main/
├── deepgemm/              # Python 包（对外 wrapper 接口）
│   ├── gemm.py             # fp8_gemm_nt（普通 FP8 GEMM 入口）
│   ├── m_group_gemm.py / m_group_gemm_nt_masked.py / m_group_gemm_nt_contiguous.py  # 分组 GEMM 三路径
│   ├── mqa_logits.py       # MQA logits（paged_mqa_logits / get_paged_mqa_logits_metadata）
│   ├── w8a16_linear.py      # W8A16 线性层
│   ├── moe_marlin_prefill.py  # MoE Marlin prefill
│   ├── tf32_hc_pernorm_gemm.py  # TF32 per-norm GEMM（海光 HC 特化）
│   ├── config.py / config_deepgemm.py / configs/  # 调优配置
│   ├── print_size.py       # DEEPGEMM_PRINT_OP_SIZE 调试支持
│   └── op.*                # 历史底层接口（兼容/ASM 对比/内部测试，非稳定公开）
├── csrc/
│   ├── deepgemm_ops.cpp    # pybind 总注册（所有 C++ 接口入口）
│   ├── py_itfs_cu/         # HIP 接口实现（对应上游 JIT/CUDA 的 DCU 移植）
│   └── include/            # 头文件
├── op_test/                # 精确测试脚本（见 §4）
├── doc/                    # deepgemm API 接口文档.md
├── 3rdparty/               # hipblaslt-install-dtk{2504,2604}.tar.gz（DTK 绑定 hipBLASLt）
└── setup.py / get_version.py
```

## §2 构建与安装（海光实测）

- **DTK 支持**：dtk24.04 / dtk25.04 / dtk26.04（README 实证）。
- **依赖 hipBLASLt**：`3rdparty/hipblaslt-install-dtkXXXX.tar.gz`（按 DTK 版本选对应包解压到系统）。
- **安装**：`python3 setup.py bdist_wheel`（打 whl）；或 develop 安装。
- **gfx 特殊路径**：
  - `gfx92a`：`PYTORCH_ROCM_ARCH=gfx92a python3 setup.py bdist_wheel`（显式指定）
  - **FP8 限制**：fp8 的某些路径（masked/contiguous）**仅 ARCH>=GFX938**；gfx936 的 fp8 路径受限（见 `op_test/test_deepgemm_contiguous_w8a8_int8_gfx936.py` 专用测试）
- **调试环境变量**（重要）：`DEEPGEMM_PRINT_OP_SIZE=1` 打印算子 size；`DEEPGEMM_PRINT_OP_SIZE_FILTER=w8a8` 过滤；`DEEPGEMM_PRINT_OP_SIZE_STATS=1` 打印 masked_m sum/min/max（触发 GPU 同步，仅调试）。详见 `doc/算子Size打印使用说明.md`。

## §3 海光适配要点（有上游，但文件级差异小）

> **开源锚点**：官方开源版 **`deepgemm-das`**（`github.com/HYGON-AI/deepgemm-das`，`main` 分支）上游基线 `deepseek-ai/DeepGEMM` commit **`559d79fb`**，MIT。开源版把 csrc 按架构分了目录：`csrc/arch/gfx92a`、`csrc/arch/gfx936`、`csrc/arch/gfx938`、`csrc/arch/gfx936_gfx938_shared`——**找某个 gfx 专属实现（HIPC、W6、shuffle-unique）先进对应 arch 目录**，不是在全库搜同名文件。下面的本地实测以发布包为基准，两边的接口名一致、目录布局以开源版为准。

> **方法论**：DeepGEMM **有公开上游**（DeepSeek `deepseek-ai/DeepGEMM`，CUDA 原版）。实测 `diff -rq 上游 海光版` **仅 3 个顶层文件不同**（`.gitignore` / `README.md` / `setup.py`）——因为海光把上游 CUDA（.cu + JIT）**同名移植为 HIP（.cpp + py_itfs_cu/）**，保持文件名结构，故文件级看不到"新增文件"，差异在**内容级**。这与 AITER（csrc 大面积改同名文件）同属"内容级改造"类，但 DeepGEMM 上游本就小、且 README 已详尽列出差异，故适配点以 README 5 点 + DTK/gfx 矩阵为准，不必纠缠 diff 行数。

**README 明示的 5 点海光改造**（相对官方 DeepGemm）：
1. 增加 **INT8** 类型支持（官方原版无）
2. input 与 weight 均使用 **perChannel 量化**
3. INT8/FP8 的 group GEMM 权重 layout 按路径区分：**masked 历史路径用 marlin/marlin2 layout**；**contiguous / low-latency 路径推荐 W6 packed layout**（由 `pack_int8_weight_enk_to_w6_low_latency` 生成）
4. contiguous 接口 input 按 **256 倍数对齐**
5. `k_group_contiguous` kernel **暂不支持**

**DTK/gfx 矩阵**：dtk24.04~26.04；gfx92a 特殊编译；fp8 路径需 gfx938（gfx936 fp8 受限）。
**与 LightOp 耦合**：LightOp `csrc/groupgemm_*` / `moew8a8` 等 GEMM 实现与 DeepGEMM 功能重叠——选型见 §5；改 GEMM 前确认两库哪个被框架实际调用。

## §4 GEMM 路径索引（实现落点速查，按落点标准）

| 路径/算子 | Python 接口 | csrc 落点 | 测试脚本（精确） | 说明 |
|---|---|---|---|---|
| 普通 FP8 GEMM | `deepgemm/gemm.py:fp8_gemm_nt` | `csrc/py_itfs_cu/`（fp8 gemm 实现） | `op_test/test_fp8_gemm.py` | FP8 nt 布局 |
| 分组 masked W8A8 | `deepgemm/m_group_gemm_nt_masked.py:m_grouped_w8a8_gemm_nt_masked` | `csrc/py_itfs_cu/` | `op_test/test_deepgemm_masked_w8a8_fp8.py` / `_w8a8_int8.py` / `_w8a8_int8_gfx936.py` / `_w8a8_fp8_tensor_wise.py` | marlin/marlin2 layout |
| 分组 masked W4A8 | `m_grouped_w4a8_gemm_nt_masked` | 同上 | `op_test/test_deepgemm_masked_w4a8.py` | |
| **分组 masked W4A8 · HIPC** | `m_grouped_w4a8_gemm_nt_masked_hipc` | `csrc/arch/gfx936`、`gfx938`（开源版仓） | 对应 `_hipc` 测试 | **权重须先经 `pack_w4a8_moe_hipc_weight` 转 HIPC 专用 layout**；与 marlin 版是两条入口，布局不可互换 |
| **分组 masked W8A8 · gfx936 专用** | `m_grouped_w8a8_gemm_nt_masked_gfx936` | `csrc/arch/gfx936` | `op_test/test_deepgemm_masked_w8a8_int8_gfx936.py` | 权重用 **W6 packed**，由 `pack_w8a8_gfx936_weight_to_w6` 生成；gfx936 上走它而不是通用入口 |
| 分组 masked BF16 | `m_grouped_bf16_gemm_nt_masked` | 同上 | `op_test/test_deepgemm_masked_w16a16_bf16.py` | 不涉及量化重排 |
| 分组 masked FP8 | `m_grouped_fp8_gemm_nt_masked` | 同上 | `op_test/test_deepgemm_masked_w8a8_fp8.py` | 仅 ARCH>=GFX938 |
| 分组 contiguous W8A8/INT8/FP8/BF16 | `m_group_gemm_nt_contiguous.py:*` | 同上 | `op_test/test_deepgemm_contiguous_w8a8_fp8*.py` / `test_deepgemm_contiguous_w8a8_int8*.py` / `test_deepgemm_contiguous_w8a8_int8_gfx936.py` | 推荐 W6 packed；input 256 对齐 |
| 分组 low-latency W8A8/FP8 | `m_grouped_fp8_gemm_nt_masked_ll`（及 `_ll_tensor_wise` 变体） | 同上 | 对应 `_ll` 测试（见 op_test/ 命名规律） | W6 packed；仅覆盖 N/K shape；`_ll_tensor_wise` 是 scale 粒度不同的另一入口 |
| **分组 contiguous INT8（shuffle 开关）** | `m_grouped_i8_gemm_nt_contiguous` | `csrc/arch/gfx936` / `gfx92a` / `gfx938` | 对应 `_i8_contiguous` 测试 | `shuffle_unique=1`（默认）走 shuffle-unique/HIPC；`0` 在 gfx936/gfx92a 走 ASM；**gfx938 不区分**（两值同一实现） |
| **普通 FP8 GEMM（hipBLASLt 后端）** | `deepgemm/gemm.py:fp8_gemm` | 转调 hipBLASLt | `op_test/test_fp8_gemm.py` | 不吃 DeepGEMM 的 layout 约束，但也不吃它的调优表 |
| **einsum / bmm** | `einsum` / `fp8_einsum` | 同上 | — | **BF16 的 einsum/bmm 走 hipBLASLt；FP8 落到 `fp8_gemm`** ——两条路的后端不同，性能不可直接对比 |
| W8A16 线性 | `deepgemm/w8a16_linear.py` | 同上 | `op_test/test_w8a16_linear.py` / `test_w4a16_gw32_masked.py` | |
| MQA logits | `deepgemm/mqa_logits.py:mqa_logits` / `paged_mqa_logits` / `get_paged_mqa_logits_metadata` | `csrc/py_itfs_cu/`（mqa 实现） | `op_test/test_mqa_logits.py` / `test_paged_mqa_logits.py` | 与 AITER `mqa_logits_kernels.cu` 同类 |
| MoE Marlin prefill | `deepgemm/moe_marlin_prefill.py` | 同上 | — | MoE Marlin 预填充 |
| TF32 per-norm GEMM | `deepgemm/tf32_hc_pernorm_gemm.py` | 同上 | `op_test/test_tf32_hc_pernorm_gemm.py` | 海光 HC 特化 |

**pybind 总入口**：`csrc/deepgemm_ops.cpp`（所有 C++ 接口签名注册在此；改 C++ 接口看这里）。
**调优配置**：`deepgemm/configs/`（按 gfx/shape 分）；`deepgemm/config_deepgemm.py`。

## §5 与 AITER / LightOp / vLLM 的 GEMM 选型

- **DeepGEMM vs AITER GEMM**：AITER 的 GEMM 集成在 MoE 流程里（`dcu-aiter-moe` §1 多量化路径）；DeepGEMM 是独立专精 GEMM 库，路径/layout 控制更细（masked/contiguous/low-latency + W6 packed + perChannel）。**要精细控制 GEMM layout/路径时优先 DeepGEMM**；只要标准接入走 AITER。
- **DeepGEMM vs LightOp GEMM**：LightOp 的 `moew8a8`/`moew4a8`/`marlinw4a16` 自带 GEMM 实现（`dcu-lightop-moe` §1），与 DeepGEMM 功能重叠。选型依据框架实际调用路径；改前确认两库哪个被用。
- **与 vLLM 衔接**：vLLM 0.15/0.18 的 `csrc/moe/`、`csrc/quantization/` 已有 GEMM kernel；DeepGEMM 可作为更底层 GEMM 实现来源，经 `dcu-vllm-op-replace` 6 步替换接入。
- **调试**：优先用 `DEEPGEMM_PRINT_OP_SIZE` 环境变量定位 size/shape 问题（§2），比直接看 kernel 快。

## §6 验证入口（实测命令）

```bash
# 安装（按 gfx / DTK）
PYTORCH_ROCM_ARCH=gfx936;gfx938 python3 setup.py bdist_wheel   # 或 gfx92a 单独指定
# 验证某个 GEMM 路径
cd op_test && python3 test_deepgemm_contiguous_w8a8_fp8.py
python3 test_deepgemm_masked_w8a8_int8_gfx936.py    # gfx936 专用

# 调试 size 打印
DEEPGEMM_PRINT_OP_SIZE=1 DEEPGEMM_PRINT_OP_SIZE_FILTER=w8a8 python3 op_test/test_deepgemm_contiguous_w8a8_fp8.py
```

## §7 何时选它 / 何时别选

- **选它**：**grouped GEMM**（MoE 每专家 M 很小、专家数多）——它是为「token 按专家分组」这个形状设计的，三条路径覆盖 token 的三种组织方式。
- **选它**：需要按布局/架构精细选入口——W4A8 要 HIPC、gfx936 要 W6、decode 要 low-latency；这些入口别处没有。
- **选它**：要把调优沉淀成表——masked 的 MODE 表与 contiguous 的 size-aware 阈值都是加表项，不用改代码。
- **选它**：普通 fp8/einsum 想直接借 hipBLASLt（`fp8_gemm` / `einsum`）——但要记得这条路径不吃 DeepGEMM 的调优表。
- **别选（结构性吃亏）**：稠密大 GEMM。它的专精是分组与量化布局；单个 (M,N,K) 大矩阵走 hipBLASLt 或 hytlass 的深流水线更合适，用 DeepGEMM 只是多一层 wrapper。
- **别选**：attention / 融合链 / 逐元素——不覆盖；也不要以它为起点改出 attention（KV gather 的瓶颈不在 GEMM）。
- **别选**：要求一套入口跨 gfx936/938 不分叉。`gfx936` 专属 W6 入口与通用入口是两条线，强行统一就是 layout 混用（静默错，不是报错）。
- **结构性吃亏的一点**：fp8 masked 仅在 gfx938 及以后，gfx936 上走 fp8 masked 是「能跑但不是这条路」；且 contiguous 的 k_group kernel 不支持——这两处是已知缺口，不是调参空间。

## 使用须知
- 经验仅供参考，不保证修复。**先核对**：gfx 型号（fp8 需 gfx938）、DTK 版本（24.04/25.04/26.04）、pytorch 版本、报错原文。DeepGEMM 有上游（DeepSeek DeepGEMM），但海光为同名 Hip 移植，文件级差异小——以 README 5 点差异 + DTK 矩阵 + 实际编译为准。
- 权重 layout 陷阱：masked 路径用 marlin layout，contiguous/low-latency 用 W6 packed（由 `pack_int8_weight_enk_to_w6_low_latency` 生成）——**混用 layout 会静默错误**，务必按路径选对。
- contiguous input 需 256 对齐；`k_group_contiguous` 暂不支持。
- 改 GEMM kernel 后必须重编 wheel；gfx 变更需重编。
- 高危操作（重装驱动/刷 vbios）需谨慎，确认带外恢复通道、优先测试机验证（dcu-ontrack-experience 使用须知）。
