# mls_requires_kpack1

> **状态：✅ measured**
> **实测**：bf16 的 `k_pack=2` 与 MLS（matrix load store）**互斥**，编译期直接报错
> 复现：`Plugins/ops/skills/op-hip-compile/references/isa-census.md` §二
> **发现于**：`mqa-logits` / `gemm-fp8`，2026-11-13

## 问题

想用 MLS（`ds_read_m32x32_b8` 一类矩阵读原语，比逐元素 LDS 读快得多）时，
bf16 路径会编译失败：

```
MatrixLoad dst Gemm consumer must have kPack=1, got 2
```

## 机制

gfx938 的 MMA 指令对 dtype 有不同的 `k_pack` 约束：

| dtype | MMA 指令 | 宽度 (M×N×K) | `k_pack` | MLS 可用 |
|---|---|---|---|---|
| bf16 | `v_mmac_f32_16x16x16_bf16` | 16×16×**16** | **2** | ❌ |
| fp8 (e4m3) | `v_mmac_f32_16x16x32_fp8_fp8` | 16×16×**32** | **1** | ✅ |

bf16 的 K 维是 16，为了凑满 32 的搬运行宽，硬件要求 `kPack=2`（一次打包两个 K 段）。
而 MLS 原语是**按 32 宽**设计的，只接受 `kPack=1`。

两者在指令层面不可兼得。

## 做法

**三个选项，各有代价**：

1. **bf16 放弃 MLS**，用 `ds_read_format_tensor_a/b` + `k_pack=2`
   —— 本项目 kernel 走这条路，micro GEMM 逐位对 fp32 参考通过
2. **降到 `k_pack=1` 换 MLS**
   —— 但 gather 宽度被砍半，实测在 `mqa-logits` 上净负
3. **改用 fp8**（如果精度允许）
   —— fp8 天然 `k_pack=1`，MLS 可用，8192³ 达 432 TF

## 实测

| 路径 | k_pack | MLS | 结果 |
|---|---|---|---|
| bf16 + format_tensor | 2 | ❌ | 可用，固定宽度 16 |
| bf16 + MLS | 1 | ✅ | 编译通过但 gather 宽度砍半，净负 |
| fp8 + MLS | 1 | ✅ | 432 TF（8192³） |

## 什么时候不适用

- **fp8 / int8 路径**：它们本来就是 `k_pack=1`，这条约束不存在
- **不用 MLS 的 kernel**：`k_pack=2` 反而更好（打包更宽）

## 反例/陷阱

**MLS 是「双边」的，只给一边挂会静默算错。**

实测（2026-11-13，MQA logits）：A 和 B 都要 `matrix_load`，
只给一边挂时**编译通过、无报错、无 inf**，但 `cos` 掉到 0.90/1.00。

这是本仓记录的第三类「静默算错」：
- `private_spill_trap`（慢 3.3 倍但算对）
- 这条（算错但不报错）
- 以及 `reg_resident_vs_lds` 的反例（多行反而退化）

**另外**：`matrix_load` 的目标区域必须是**静态形状**。
`q_sh[i*H:(i+1)*H]` 这种动态子切片会报
`matrix_load dst last-2 region must be static`。

## 关联

- `Plugins/ops/skills/op-hip-compile/references/isa-census.md` —— 完整普查表
- [`reg_resident_vs_lds`](reg_resident_vs_lds.md) —— 什么时候根本不需要 LDS
