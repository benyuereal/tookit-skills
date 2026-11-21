---
name: dcu-deepgemm-arch
description: DeepGEMM三路径架构与MODE调优机制：masked/contiguous/ll选型与layout陷阱
version: 1.0.0
author: DCU Engineer
license: MIT
---

# DeepGEMM 三路径架构与调优机制（dcu-deepgemm-arch）

> 从 `<local_src_cache>/deepgemm-main` 源码实证提炼的**架构与机制**专题。源码地图/构建/路径索引表见 `dcu-deepgemm-source`；本文回答"三路径怎么选、MODE 怎么自动选、layout 为什么不能混"。
> 实证样本：`deepgemm/m_group_gemm_nt_masked.py`、`m_group_gemm_nt_contiguous.py`、`m_group_gemm.py`（low-latency + W6 pack）、`config_deepgemm.py`、`configs/w8a8_fp8_contiguous.py`、`csrc/include/low_latency_fp8_masked.h`。

## §0 总纲：DeepGEMM 的核心抽象是"三路径 × 量化类型"

DeepGEMM 不是"一个 grouped GEMM"，而是**三条独立路径**，每条路径的 token 组织方式、权重 layout、调优机制都不同：

| 路径 | token 组织 | 权重 layout | 调优机制 | 典型场景 |
|---|---|---|---|---|
| **masked** | `masked_m`（每专家 token 数）+ `expected_m_per_group`，token 不物理重排 | **marlin/marlin2**（3D [E,N,K] 量化权重） | MODE 表（EXPECTED_M 取 2 的幂 + 分 CU 数 dict + JSON 覆盖） | MoE prefill 通用 |
| **contiguous** | `m_indices`（int32，每 token 的专家号），token **物理按专家连续排列** | **W6 packed**（6D [E,K/64,N/16,4,16,16]） | size-aware 形状表（(E,N,K)→padded-M 阈值，两档配置） | MoE 大 batch、权重已 W6 打包 |
| **low-latency**（`*_ll`） | 同 masked（`masked_m` + `expected_m_per_group`） | **W6 packed**（同 contiguous） | `cu` 参数 +（fp8 变体）overlap/signal，无 MODE | decode 低延迟、与通信 overlap |

**一句话选型**：token 没物理重排 → masked；token 已按专家 permute 成连续块 → contiguous；decode 小 M 要低延迟/overlap → low-latency。

## §1 三路径接口与参数（源码实证）

### 1.1 masked（`m_group_gemm_nt_masked.py`）
- 接口：`m_grouped_{w8a8,w4a8,fp8,bf16}_gemm_nt_masked(a, b, d, masked_m, expected_m_per_group, config)`
- `a = (input, a_scale)`、`b = (b_qweight, b_scale)`，input 是 [E, M, K] 3D（每专家一段）
- `masked_m`：每专家实际 token 数；`expected_m_per_group`：期望值（决定 MODE）
- fp8 变体返回 `(block_m, threshold)` 供上层调度；支持 `enable_overlap` + `signal`
- **fp8 masked 仅 ARCH>=GFX938**（gfx936 走 int8 专用测试路径）

### 1.2 contiguous（`m_group_gemm_nt_contiguous.py`）
- 接口：`m_grouped_{i8,fp8,bf16}_gemm_nt_contiguous(a, b, output, m_indices, config)`
- `m_indices`：int32，**每 token 的专家号**（token 已物理 permute 成按专家连续）
- **input 需 256 倍数对齐**（README 改造点 4）
- W6 校验：`b_qweight.dim() != 6` 直接 raise（错误信息里写明期望 layout 和生成函数）
- 变体：`_nopad`（带 `token_per_expert`）、`_tensor_wise`（b_scale 必须 [E] 或 [E,1]）
- `k_group_contiguous` kernel **暂不支持**（README 改造点 5）

### 1.3 low-latency（`m_group_gemm.py` 的 `*_ll`）
- 接口：`m_grouped_{w8a8,fp8}_gemm_nt_masked_ll(a, b, d, masked_m, expected_m_per_group, config, *, block_wise=False, cu=128)`
- **注意参数差异**：`enable_overlap` + `signal`（与通信 overlap 的握手）**只有 fp8 ll 变体有**（`m_grouped_fp8_gemm_nt_masked_ll` / `_ll_tensor_wise`）；**w8a8 ll 没有**这两个参数
- 权重**必须**是 W6 6D（docstring 明示 "b[0] must be 6-D packed weights from pack_int8_weight_enk_to_w6_low_latency"）
- scale：默认 2D per-token/per-channel（[E,M]/[E,N]，3D 带尾 1 会自动 squeeze）；`block_wise=True` 走 block-wise scale
- kernel 层（`low_latency_fp8_masked.h`）：`kGroupGemm_aggressive` / `kGroupGemm` / `kGroupGemmBlockWise(_)` 四个 `__global__`，均 `__launch_bounds__(256, 1)`

## §2 MODE 调优选择机制（核心，源码逐行）

### 2.1 masked 路径：EXPECTED_M 取 2 的幂 + 分 CU dict
`deepgemm_masked_config(m, size_k)`（lru_cache）：
1. 特判：`size_k==7168 and m<=12 → 41`；`size_k==2048 and m<=12 → 33`（cuda 选择，目前只支持 gfx938）
2. `m<16 → adapt_m=16`，否则 `round_up_to_next_power_of_2(m)`
3. 查 `EXPECTED_M{adapt_m}` 键 → `config_deepgemm.gemm_deepgemm_masked_config` 的 dict
4. dict 按 **CU 数**分表：`DEEPGEMM_GPU_CUS` 环境变量（默认 80），支持 80/72/64/48 四档，当前四档值都是 1002
5. 查不到 → 回退 1000（i8fp8 变体回退 1002）

**MODE → block 映射**（`get_deepgemm_block_m/n`）：
- MODE 1000 → block_m 256；1001 → 128；1002 → 64
- block_n 恒 256（MODE 1000-1002）

**JSON 覆盖层**（`get_groupgemm_w8a8_mode`）：优先 `get_deepgemm_groupgemm_w8a8_config(E, M, N, K, device_name, num_cus)` 读 JSON 调优表（device_name 取 `gcnArchName`），status 成功且含 MODE 就用；否则回退上面的 dict 逻辑。**加新 shape 调优 = 往 JSON 表加条目，不改代码。**

### 2.2 contiguous 路径：size-aware 形状表（两档配置）
`select_fp8_contig_hipc_config(size_m, size_n, size_k, num_groups)`：
1. 形状表 `FP8_CONTIG_HIPC_M_THRESHOLDS`，key=(E,N,K)：
   - 默认 (16,2048,4096) → 阈值 6400（gfx938 上 10 个随机 token layout 调出）
   - (24,1536,4096) → 12288（BW1000B 上 3 轮随机 layout 调出）
2. `size_m <= 阈值 → FP8_CONTIG_HIPC_OPTIMIZED = (64, 256, 2)`（bm64/nloop2）
3. 否则 → `FP8_CONTIG_HIPC_V75 = (128, 256, 4)`（bm128/nloop4）
- 设计目标（注释原话）："Keep runtime overhead tiny: one shape-table lookup and one padded-M comparison"——**运行时只查一次表 + 一次比较**

**两个 resolve 函数**（config 为 None 或 auto 时）：
- `_resolve_block_m(size_m, mode)`：mode=0/auto → m%64==0 取 64，m%32==0 取 32，否则 raise；mode=1000 → 32；合法值 (32,64,128)
- `_resolve_n_loop_num(size_n, requested)`：0=auto 从 {4,2,1} 取**最大的**满足 `N % (128*candidate)==0` 的；显式值不整除则向下回退，再不行任意合法值，全不行 raise

### 2.3 low-latency 路径：无 MODE，靠 cu + overlap
- `config` 参数保留未用（`del config`）
- 调优维度是 `cu`（默认 128）和 `block_wise`/`enable_overlap`
- 权重 W6 打包在**加载期一次性**完成，运行时零调优开销

## §3 权重 layout 机制（踩坑重灾区）

**两套 layout，按路径严格区分，混用 = 静默错误**：

| layout | 形状 | 生成 | 用于 |
|---|---|---|---|
| marlin/marlin2 | 3D [E,N,K] 量化权重 | 上游 marlin repack | **masked** 路径 |
| W6 packed | 6D [E, K/64, N/16, 4, 16, 16] | `pack_int8_weight_enk_to_w6_low_latency` | **contiguous + low-latency** |

**W6 打包实现**（`_logical_int8_weight_2d_to_w6`）：
```
[E,N,K] → reshape(E, N/n_in, n_in, K/(k_in*4), 4, k_in)
        → permute(0, 3, 1, 4, 2, 5) → contiguous
```
- 约束：`N % _CUSTOMER_N_TILE == 0` 且 `K % _CUSTOMER_K_PACK == 0`，不满足直接 raise
- 入口校验：`weight_enk.dim() != 3` raise；contiguous 接口侧 `b_qweight.dim() != 6` raise（双向校验，layout 错在入口就炸，不会静默）

**经验**：
- 权重打包是**加载期一次性**操作（模型加载时 pack 一次，推理复用），不要放进推理热路径
- 同一份权重要同时服务 masked 和 contiguous/ll 时，**两套 layout 各存一份**，不要运行时互转
- 换路径 = 换 layout = 换打包函数，三者绑定，改一个必须同步另两个

## §4 选路径 / 写接入的 checklist

1. **token 组织**：没物理重排（只有每专家计数）→ masked；已 permute 成按专家连续 → contiguous；decode 小 M 低延迟 → low-latency
2. **layout 匹配**：masked 用 marlin 3D；contiguous/ll 用 W6 6D（`pack_int8_weight_enk_to_w6_low_latency` 加载期打包）
3. **量化类型**：fp8 masked 需 gfx938；gfx936 走 int8（有专用测试 `test_deepgemm_contiguous_w8a8_int8_gfx936.py`）
4. **对齐**：contiguous input 256 倍数对齐；W6 要求 N%N_TILE==0、K%K_PACK==0
5. **scale 形状**：ll 默认 per-channel 2D（[E,M]/[E,N]）；block_wise=True 才用 3D block scale；tensor_wise 变体 b_scale 必须 [E] 或 [E,1]
6. **调优**：masked 加 shape → 往 JSON 表加条目（`get_deepgemm_groupgemm_w8a8_config`）；contiguous 加 shape → 往 `FP8_CONTIG_HIPC_M_THRESHOLDS` 加 (E,N,K) 阈值（需实测调出）
7. **overlap**：prefill 大 batch 用 contiguous；decode 与通信 overlap 用 **fp8 ll** 的 `enable_overlap`+`signal`（w8a8 ll 无此参数）
8. **不支持**：`k_group_contiguous` 暂不支持；grouped GEMM 路径不覆盖 fp8 权重 + 16bit 激活组合（W8A16 只有 `w8a16_linear` 线性层，非 grouped）
9. **调试**：`DEEPGEMM_PRINT_OP_SIZE=1`（+`_FILTER=w8a8` 过滤）先定位 size/shape，比直接看 kernel 快

## §5 源码落点（去哪看）

- 三路径 python 接口：`deepgemm/m_group_gemm_nt_masked.py`（186 行）/ `m_group_gemm_nt_contiguous.py`（261 行）/ `m_group_gemm.py`（413 行，含 ll + W6 pack）
- MODE 表：`deepgemm/config_deepgemm.py`（分 CU dict + block_m/n 映射）
- contiguous 形状表：`deepgemm/configs/w8a8_fp8_contiguous.py`（阈值表 + 两档配置 + resolve 函数）
- ll kernel：`csrc/include/low_latency_fp8_masked.h`（4 个 `__global__` kGroupGemm*）
- pybind 总入口：`csrc/deepgemm_ops.cpp`
- 源码地图/构建/测试索引：`dcu-deepgemm-source`
- 与 AITER/LightOp 的 GEMM 选型：`dcu-deepgemm-source` §5

## 使用须知
- 经验仅供参考，不保证修复。**先核对**：gfx 型号（fp8 需 gfx938）、DTK 版本（24.04/25.04/26.04）、CU 数（`DEEPGEMM_GPU_CUS`）、报错原文。
- layout 混用是**静默错误**高危点：masked 传 W6 或 contiguous 传 marlin，入口 dim 校验能拦住一部分，但 scale 形状错（如 ll 传 3D scale 给 per-channel 路径）可能不炸只算错——接入时按 §4 checklist 逐项对。
- 改 GEMM kernel 后必须重编 wheel；gfx 变更需重编。
- 开发规范/编译调试：`dcu-kernel-dev`；profile 定位：`dcu-op-optimization` + `dcu-perf-profiling`。
