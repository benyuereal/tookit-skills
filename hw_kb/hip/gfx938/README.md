# HIP / gfx938 机制库

> 平台：海光 DCU BW1100（gfx938，64 CU，8MB L2，64KB LDS/CU，wavefront 64）
> 编译：`hipcc -O3 --offload-arch=gfx938`

## 已收录

| 机制 | 状态 | 一句话 |
|---|---|---|
| [`reg_resident_vs_lds`](reg_resident_vs_lds.md) | ✅ | 行数据留寄存器，LDS 只放归约表 —— **6.9x 差距** |
| [`private_spill_trap`](private_spill_trap.md) | ✅ | 数组索引编译期非确定 → 溢出私有内存 → **慢 3.3x 且不报错** |
| [`vec16_uint4`](vec16_uint4.md) | ✅ | 16B 向量访存是上 1000 GB/s 的必要条件 |
| [`warp_reduce_shfl`](warp_reduce_shfl.md) | ✅ | shuffle + 静态 LDS 表，屏障从 8 次降到 2 次 |
| [`lds_bank_swizzle`](lds_bank_swizzle.md) | ❌ **refuted** | 解决了真问题（9.6 冲突/指令）但**只快 0.36%** |

## 待收录（有实测但还没整理）

- `sto_global_cache_modifier` —— `.cs` / `.cg` 在 DCU 上的实际效果（`_common.py` 注释说 `.cs` 不支持）
- `grid_tail_discard` —— 小 M 时 grid 填不满 64 CU 的处理
- `multi_row_cta` —— 多行/CTA 的收益与 vgpr 压力的平衡点

## 平台差异（写 documento 前必看）

| 项 | gfx938 | NVIDIA |
|---|---|---|
| wavefront | **64** | 32 |
| LDS/CU | 64 KB | 228 KB (H100) |
| LDS bank | 32 | 32 |
| 向量宽度 | 16B (uint4) | 16B |

**wavefront 64 是最容易踩的差异**：所有 `o = 32` 起步的 shuffle 循环、
所有 `lane / 32` 的写法都要重新检查。
