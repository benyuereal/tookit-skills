# reg_resident_vs_lds

> **状态：✅ measured**
> **实测**：寄存器驻留 **1454 GB/s** vs 整行 LDS 中转 **212 GB/s**（6.9x）@ `event_isolated`
> 复现：`Plugins/ops/skills/op-rmsnorm-fused/results/forged-vs-lightop.json`
> **发现于**：`op-rmsnorm-fused` / `fused-add-rmsnorm` / `norm-act-swiglu`（**三张卡各发现一次**）

## 问题

行内归约算子（RMSNorm / LayerNorm / softmax）的常见写法是：
**整行读进 LDS → 归约 → 从 LDS 读回写**。

H=4096 时每行 16KB，这个写法把「2 读 1 写」变成了 **4 次往返**。

## 机制

LDS（64KB/CU）是**片上但不在寄存器**的存储。走 LDS 的代价：

```
HBM ──读──▶ 寄存器 ──写──▶ LDS ──读──▶ 寄存器 ──写──▶ HBM
                 ↑                    ↑
            多一次 LDS 写        多一次 LDS 读
```

而寄存器驻留只需要：

```
HBM ──读──▶ 寄存器（全程在寄存器）──写──▶ HBM
```

**LDS 只该放跨 warp 的归约表**（WARPS×4 字节），不该放行数据。

## 做法

```c
// 每线程持 VEC*8 个 float（编译期常量！见 private_spill_trap）
float xf[VEC * 8];
float acc = 0.f;
#pragma unroll
for (int j = 0; j < VEC; ++j) {
  const int off = (threadIdx.x + j * THREADS) * 8;
  *(uint4*)&xv[j*8] = *(const uint4*)(xr + off);   // 16B 向量读
  *(uint4*)&rv[j*8] = *(const uint4*)(rr + off);
}
#pragma unroll
for (int i = 0; i < VEC*8; ++i) {
  xf[i] = float(xv[i]) + float(rv[i]);
  acc += xf[i] * xf[i];
}
// 归约只过 WARPS*4 字节的静态 LDS 表
__shared__ float warp_part[WARPS];
```

## 实测

| 实现 | 耗时 | 带宽 | 达成率 |
|---|---|---|---|
| scaffold 默认骨架（整行 LDS） | — | **212 GB/s** | 13% |
| 寄存器驻留（1 行/CTA） | 148.0 µs | 1133 GB/s | 66% |
| 寄存器驻留（2 行/CTA） | **140.5 µs** | 1194 GB/s | 69% |
| lightop（厂商，同语义） | 255.2 µs | 具体见卡 | — |

延迟 LDS 版本比寄存器驻留慢 **6.9 倍**。

## 什么时候不适用

- **H 很大**（> 8192）：每线程的寄存器装不下整行，必须回退到分块 + 在线归约
- **需要跨行操作的算子**（如 column-wise norm）：行数据本来就要交换，LDS 不可避免
- **归约本身很简单时**（H ≤ 512）：LDS 的开销占比小，差别不明显

## 反例/陷阱

**行数不是越多越好。** `norm-act-swiglu` 上实测：

| BLOCK_ROWS | 耗时 | 地板占比 | vgpr |
|---|---|---|---|
| 1 | 148.0 µs | 66% | 39 |
| **2** | **140.5 µs** | **69%** | 57 |
| 4 | 153.3 µs | 63% | **84** |

4 行因寄存器压力（vgpr 84）反而退化。**多行摊薄 barrier 的收益，在某个点被寄存器成本吃掉。**

## 关联

- [`private_spill_trap`](private_spill_trap.md) —— 寄存器驻留的前提是数组索引编译期确定
- [`warp_reduce_shfl`](warp_reduce_shfl.md) —— 归约怎么走
- [`vec16_uint4`](vec16_uint4.md) —— 访存宽度
