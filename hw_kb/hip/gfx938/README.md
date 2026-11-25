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
| [`mls_requires_kpack1`](mls_requires_kpack1.md) | ✅ | bf16 的 k_pack=2 与 MLS 互斥；MLS 只挂一边会**静默算错** |
| [`slc_cache_hint`](slc_cache_hint.md) | ❌ **refuted** | slc 提示慢 13%；但「纯 load」比 descriptor 慢 18% —— 三者关系是重点 |

## 待收录（有线索但缺独立实测）

| 候选 | 缺什么证据 |
|---|---|
| `grid_tail_occupancy` | 小 M 时 grid 填不满 64 CU 的实测数字（只有推理，没测过） |
| `prefetch_depth_isa` | `s_waitcnt` 深度与实测耗时的对应关系 |
| `epilogue_overlap` | 尾部展开与下一轮计算重叠的实际收益 |

**没有独立实测的不写进来** —— 这是本库的纪律。
「看起来该有用」不足以进 hw_kb，那和猜测没有区别。

## 平台差异（写 documento 前必看）

| 项 | gfx938 | NVIDIA |
|---|---|---|
| wavefront | **64** | 32 |
| LDS/CU | 64 KB | 228 KB (H100) |
| LDS bank | 32 | 32 |
| 向量宽度 | 16B (uint4) | 16B |

**wavefront 64 是最容易踩的差异**：所有 `o = 32` 起步的 shuffle 循环、
所有 `lane / 32` 的写法都要重新检查。
