---
name: op-gemm-fp8
description: "FP8 GEMM（含 batched/分组）在 DCU gfx938 上的实现选型与坑：tilelang（MatrixCorePreshuffle、warp_size=64、k_pack=2）与 hipBLASLt(_scaled_mm) 两条路，FP8/BF8 MFMA intrinsic 限制、fp8 dtype 实测判据（float8_e4m3fn vs fnuz、512³ GEMM 的 max_abs 必须为 0）、scale 布局、以及 testing/amd 中 mfma fp8 跑不通的现状。触发场景：FP8 GEMM 编译失败/精度异常/性能不达标、tilelang fp8 写法、fn 还是 fnuz 选不准、三角选择（tilelang vs hipBLASLt vs aiter）。不适用：MoE 的 routed/shared 两条路径与融合判据走 op-moe-w8a8；手写 HIP 的编译 flag 与 VGPR 读法走 op-hip-compile；容差与 floor 字段怎么写走 op-contract。"
version: 0.1.0
author: DCU Toolkit Contributors
license: MIT
kind: card
---

# FP8 GEMM（op-gemm-fp8）· draft

> 状态：`draft`。选型与坑已整理，bench/tests 待落地。

## 一、是什么

`out[M,N] = A[M,K] @ B[K,N]`，A/B 为 FP8（e4m3），累加 fp32，输出 bf16/fp16。变体包括 batched（每 expert/每组一套权重）与 grouped（按 token 分组）两类。

**关键参数是 M**：
- `M ≤ 32`：权重读取主导，贴 HBM 带宽地板，GEMM 本身几乎没有优化空间。
- `M ≥ 128`：算力主导，tile 形状与 k_pack 才有意义。

## 二、变体矩阵

| 变体 | 来源 | 适用 | 状态 |
|---|---|---|---|
| `hipblaslt` | `torch._scaled_mm`（hipBLASLt 0.10） | 通用大 M | **gfx938 上无 FP8 方案**（实测不可用） |
| `tilelang` | tilelang-hygon `examples/gemm_fp8/example_tilelang_gemm_amd_fp8_preshuffle.py` | 大 M、可定制 epilogue | 可用（需 `target="hip"`、`warp_size=64`） |
| `tilelang-rs` | 同上 `gemm_type=rs`（寄存器 A） | A 驻留寄存器的场景 | 待评 |
| `aiter` | aiter GEMM 库 | 生产 | 待评 |
| `hand-hip` | 手写 HIP + MFMA intrinsic | 特殊形状 | 兜底 |

## 三、gfx938 的 FP8 事实（写代码前需要知道）

- gfx938 **有** FP8/BF8 MFMA intrinsic：`__builtin_amdgcn_mfma_f32_32x32x16_fp8_fp8`、`_16x16x32_bf8_bf8`、`smfmac_f32_16x16x64_*`。
- 裸 `cvt_pk_fp8_f32` 需要 `fp8-insts` 目标特性——**gfx938 不提供**，必须改用 HIP 的 `cast_to_f8` / `cast_from_f8`。
- **fp8 dtype 必须实测确认，不能按「ROCm 就是 fnuz」的旧说法写**：本机组合（gfx938 / 当前 DTK+torch）`float8_e4m3fn` 逐位精确；`float8_e4m3fnuz` 呈整体比例偏差。「ROCm 上 `determine_fp8_type()` 返回 fnuz（gfx950 除外）」这条在本机**不成立**——量化/反量化代码以实测 dtype 为准。`contract.yaml` 的 `dtypes` 与 `layout_constraints` 仍写着 e4m3/fnuz，以本节判据为准（待同步）。
- `k_pack`：1 = RDNA/gfx936，2 = CDNA 系（本机取 2）。
- `warp_size` 必须显式 64；shfl 默认 32 会越界返回自身值导致归约错。
- 权威现状：`tilelang/testing/python/amd` 里 **mfma BW1000 fp8 跑不通**；`examples/gemm` 正常。

### fp8 dtype 判据：fn vs fnuz（实测，2026-11-13）

**现象**：同一份 fp8 量化 + GEMM 代码，`float8_e4m3fn` 对 fp32 参考逐位精确；换成 `float8_e4m3fnuz` 后 cosine 仍是 1.0，但 `max_abs` 明显——表现为**整体比例偏差**，不是随机误差。
**机制**：fn 与 fnuz 的指数偏置/零值编码不同，同一组 bit 解出来的数值差一个整体比例；这个差异表达不成某个可调 scale，所以逐元素缩放补不回来。
**判据**：拿一个 **512³** 的 fp8 GEMM 对 `A.float() @ B.float().T` 比，**`max_abs` 必须为 0**。不为 0 就**换 dtype**，不要调 scale。`cosine=1.0` 不作为通过条件（比例偏差下它照样是 1.0）。

## 四、MLS 路径的架构门

tilelang 的 MLS（Matrix Load Store）路径只在 `{gfx938, gfx92a, gfx946}` 开放（`src/target/utils.cc` 的 `GetHcuArchString()`）。gfx938 在白名单内，可用；但换到 gfx936 会 ICHECK 崩溃。普通 GEMM（ldmatrix + mmac）不受此限。

MLS 还有一条 dtype 前置条件：`kPack=1`。bf16 kernel 因 gather 向量化宽度必须是 `kpack=2`，所以 **bf16 用不了 MLS**（编译期报 `MatrixLoad dst Gemm consumer must have kPack=1, got 2`）；判据见 `op-roofline` 的 R7。

## 五、判据

- 正确性：vs fp32 参考，容差见 `contract.yaml`；FP8 需要**逐元素 scale 反量化后**再比，不能直接比量化后的整数。
- **dtype 门（先于容差门）**：fp8 结果对 `A.float() @ B.float().T` 的 `max_abs` 必须为 0（从 512³ 起测）；不为 0 是 dtype 选错，不是容差问题，换 dtype 而不是调 scale。
- 不变量：`determinism` + `shape_tiling_invariance`（不同 tile 配置结果一致；FP8 下用容差而非 bit-exact，累加顺序会变）。
- 性能：达成率按「M 大小决定是带宽型还是算力型」分别算地板，见 `floor.basis`。

## 六、已知坑

| 坑 | 现象 | 规避 |
|---|---|---|
| `_scaled_mm` 报无解决方案 | gfx938 上 hipBLASLt 没有 FP8 | 走 tilelang 或 aiter |
| 只设 `target` 不设 `warp_size` | 结果错但不报错 | 显式 `warp_size=64` |
| 按「ROCm=fnuz」选 dtype 或用 scale 追偏差 | cosine 1.0 但 max_abs 大（整体比例偏差） | 本机用 `float8_e4m3fn`；512³ 对 fp32 参考 `max_abs` 必须 0，不为 0 换 dtype |
| 用 `alloc_shared` 存精度敏感量 | 结果随机 | 改 `alloc_local` |
| 动态 M 用 Python 变量 | 编译期不识别 | `T.dynamic('M','int32')` |
| 内核缓存命中跳过 lowering | 改了源码不生效 | `TILELANG_DISABLE_CACHE=1` |
| 改了生成源码想打补丁 | 重编译即失效 | 用 `TILELANG_OVERRIDE_DEVICE_SOURCE` |
| 拿低 M 的带宽达成率推高 M | 结论相反 | 按 M 分档分别判 |

## 七、待补与深引用

`contract.yaml` 已就位；bench 驱动与 adapter `_common.py` 待写（对应门均已声明 `required: false`，补齐后改 `true`）。首个目标：跑通 `example_tilelang_gemm_amd_fp8_preshuffle.py` 并在 gfx938 上给出「编译 / 精度 / 达成率」三件事，再推进到 `measured`。

- tilelang 源码索引与已知问题：`dcu-operators/dcu-tilelang-source`、`dcu-kernelwiki`；kernel 开发方法学与 FP8 轴：`dcu-operators/dcu-kernel-dev` §15；优化闭环：`dcu-perf/dcu-op-optimization`
