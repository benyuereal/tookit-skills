# vec16_uint4

> **状态：✅ measured**
> **实测**：16B 向量读写是达到 >1000 GB/s 的必要条件
> 复现：三张卡的 kernel 都用它
> **发现于**：`op-rmsnorm-fused` / `fused-add-rmsnorm` / `norm-act-swiglu`

## 问题

逐元素（1 个 bf16 = 2B）访存时，每条 load/store 指令只搬 2 字节。
H=4096 时需要 2048 条指令，指令发射成为瓶颈，带宽上不去。

## 机制

gfx938 的 `GLOBAL_LOAD_DWORDX4` 一次搬 128 bit = **16 字节 = 8 个 bf16**。
用 `uint4` 类型转换让编译器生成向量指令：

```c
// ❌ 逐元素：每条指令 2B
for (int i = 0; i < H; ++i) y[i] = ...;

// ✅ 向量：每条指令 16B
uint4 v = *(const uint4*)(x + off);   // 8 个 bf16
bf16* p = (bf16*)&v;
```

## 做法

```c
bf16 xv[VEC * 8];
#pragma unroll
for (int j = 0; j < VEC; ++j) {
  const int off = (threadIdx.x + j * THREADS) * 8;
  *(uint4*)&xv[j*8] = *(const uint4*)(xr + off);   // 一次 8 个
}
```

**对齐要求**：`off` 必须是 8 的倍数（16B 对齐）。
`(threadIdx.x + j*THREADS) * 8` 天然满足。

## 实测

三张卡上的一致结论：去掉向量化后带宽掉到 200-400 GB/s 量级；
加上后达 1100-1450 GB/s。

| 卡 | 向量化后 | 地板占比 |
|---|---|---|
| rmsnorm-fused | 1454 GB/s | 87% |
| fused-add-rmsnorm | 1438 GB/s | 83% |
| norm-act-swiglu | 1194 GB/s | 69% |

## 什么时候不适用

- **H 不是 8 的倍数**：尾部要走标量路径（本仓的模板直接拒绝这种形状，不做兜底）
- **访问不连续**（gather/scatter）：向量化要求连续 16B

## 反例/陷阱

`hip_bfloat16` 的数组直接 `*(uint4*)` 转换要求**地址 16B 对齐**。
PyTorch 分配的 tensor 起点是对齐的，但**切片后的视图可能不是**。
用之前先确认 `data_ptr() % 16 == 0`。

## 关联

- [`reg_resident_vs_lds`](reg_resident_vs_lds.md)
