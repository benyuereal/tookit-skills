# op-gemm-fp8 台账

> 绝对数字进 `~/.config/dcu-toolkit/kb/op-opt/`；本文件只留相对倍数、机制与判定。

## 一、当前状态

**draft**。契约、形状档位、变体矩阵、adapter、A/B 适配器均已就绪；
缺的是**本机安装 TileLang(hcu) / aiter 的 FP8 GEMM 后端**——门会 FAIL 并说明原因。

## 二、已确立的事实（来自 contract.yaml，有实测支撑）

| 事实 | 机制 | 影响 |
|---|---|---|
| gfx938 上 hipBLASLt 无 FP8 方案 | 实测确认 | `torch._scaled_mm` 这条路在本平台不可用 |
| fp8 在 ROCm 是 **fnuz** 语义（gfx950 除外） | 平台 ISA 约定 | 量化代码不能照搬 NVIDIA e4m3 |
| 裸 `cvt_pk_fp8_f32` 需 fp8-insts | gfx938 不提供该目标特性 | 必须用 `cast_to_f8` / `cast_from_f8` |
| tilelang 必须显式 `warp_size=64`、`target=hip` | 默认 32 会越界返回自身值 | 漏了会得到错误结果而非报错 |
| `k_pack=2` 是 CDNA 系取向 | 硬件 gather 向量化宽度 | bf16 用不了 MLS 路径 |
| M 分档是两个瓶颈类型 | M≤32 带宽型 / M≥128 算力型 | 不可用同一地板判据 |

## 三、待验证（open_questions）

1. `example_tilelang_gemm_amd_fp8_preshuffle.py` 在 gfx938 上的编译/精度/达成率
2. `k_pack=2` 与 `BLOCK_N` 的最优组合

## 四、复现方式

```bash
# 正确性门（tiny 档，adapter 自检；真实档留给真机 bench）
python3 scripts/opcheck.py correctness \
    --contract Plugins/ops/skills/op-gemm-fp8/contract.yaml \
    --shape tiny --variants tilelang

# A/B（需先装后端）
python3 Plugins/ops/skills/op-gemm-fp8/bench/ab.py \
    --variants tilelang --shape decode-m32 --rounds 5
```

## 五、纪律

- 参考实现用 `torch.mm`（fp32），**不用 `_scaled_mm`**——后者是被测对象。
- 分母按档位选：decode 档用带宽 1724 GB/s；prefill 档用 fp8 算力峰值。
  两档混用会把结论判反。
- `tiny` 档只用于验证门是否通，**绝不用于性能结论**。

## 2026-11-23：原生 fp8 路径验证（结论：存在，但只有 nominal 的 8.6%）

**问题**：本机有没有原生 fp8 MMA 路径？还是所有自称 fp8 的实现都在做 dequant 回退？

### 三条证据

| 路径 | 结果 |
|---|---|
| `torch._scaled_mm`（hipBLASLt） | **FAIL**：`could not find valid hipblaslt solution` |
| `aiter.gemm_a8w8_opus`（kernelId=auto） | **OK**：48.2 TF（4096³） |
| **同库同形状** fp8/bf16 速率比 | **1.92x** |

### 判定

```
同一份矩阵形状、同一个库（aiter）：
  gemm_a8w8_opus  (fp8)   2.8768 ms = 47.8 TF
  gemm_a16w16_opus (bf16) 5.5829 ms = 24.6 TF
  ⇒ 算力比 1.94x（复测 1.92x）
```

**若走 dequant 回退，fp8 输入会先反量化成 bf16 再跑同一套 MMA，
两边耗时应相同（比值 ≈1.0）。实测 1.92~1.94x 说明是原生 fp8 双速率通路。**

### 结论

1. **hipBLASLt 在 gfx938 上确实没有 fp8 方案** —— 这条原判断成立。
2. **但本机有原生 fp8 路径**（aiter opus），不是 dequant 回退。
3. 它的水平是 **48~50 TF，只有 fp8 nominal(560) 的 0.086**。
   ⇒ **方向不是「找原生路径」（已存在），而是「这条实现为什么这么低」。**

### 一条必须记住的判据

**判「是否原生 fp8」必须同库同形状对照。** 跨库比（aiter 的 fp8 对 hipBLASLt 的 bf16）
会把**库的差距**记成**精度的差距** —— 两库的 bf16 本身差 8.8x。
2026-11-23 早前正是这样得出「fp8 比 bf16 慢 5.3x」的错误结论。

### 未测定

- aiter opus 为什么只有 48 TF（缺 PMC 归因）
- tilelang / hand-hip 两条路径（本机未跑通）
- 是否存在比 opus 更快的 fp8 实现
