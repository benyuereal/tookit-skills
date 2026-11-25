# warp_reduce_shfl

> **状态：✅ measured**
> **实测**：warp shuffle + 静态 LDS 表，比树状 `__syncthreads` 归约少 1-2 次屏障
> 复现：三张卡的 kernel
> **发现于**：`op-rmsnorm-fused`

## 问题

行内归约（sum / mean / max）需要把 256 个线程各自的部分和合并成一个值。
朴素写法是**每级都 `__syncthreads`** 的树状归约：log₂(256) = 8 次屏障。

每次 `__syncthreads` 都是全局停顿，且**归约延迟藏不住**（每 CTA 只搬 16KB）。

## 机制

两级归约：

```
① warp 内（64 lane）：__shfl_down 6 轮，(64 → 1)，**不需要屏障**
② 跨 warp（WARPS 个）：写静态 LDS 表，warp0 汇总，**只需 2 次 __syncthreads**
```

**gfx938 的 wavefront 是 64，不是 32**（NVIDIA 是 32）。写 `o = 32` 起始的
shuffle 循环会漏掉 lane 32-63。

## 做法

```c
__shared__ float warp_part[WARPS];       // 静态！动态 LDS 不进 metadata，读不到

// ① warp 内：**从 32 开始，但 wavefront 是 64**
//    __shfl_down 在 wavefront 内自动跨 64 lane，所以 32 起步是对的
#pragma unroll
for (int o = 32; o > 0; o >>= 1) acc += __shfl_down(acc, o);

// ② 跨 warp：每 warp 的 lane 0 写表
if (lane == 0) warp_part[warp] = acc;
__syncthreads();
if (warp == 0 && lane < WARPS) {
  float s = 0.f;
  #pragma unroll
  for (int i = 0; i < WARPS; ++i) s += warp_part[i];
  // 或再 shuffle 一轮
}
__syncthreads();
```

## 实测

| 归约方式 | 屏障次数 | 备注 |
|---|---|---|
| 树状全量 | 8 | 朴素写法 |
| **shuffle + 静态 LDS** | **2** | 本仓所有 kernel 用这个 |

`norm-act-swiglu` 上，归约方式从树状改成 shuffle 后，从 66% 地板提到 69%
（配合 2 行/CTA 才到这个数）。

## 什么时候不适用

- **归约元素数 < 64**：shuffle 的 6 轮多过直接串行
- **需要跨 CTA 归约**：shuffle 只在 wavefront 内，跨 CTA 必须走全局内存 + 二次 kernel

## 反例/陷阱

**动态 LDS（`extern __shared__`）不进 kernel metadata**，所以
`inspect_so.py` 读不到它的用量。

**静态声明的 `__shared__ float warp_part[WARPS]` 可以读到** —— 这也是选静态的理由之一：
资源画像可回读，能验证「LDS 用量确实是 WARPS*4 字节」而不是「大约是」。

## 关联

- [`reg_resident_vs_lds`](reg_resident_vs_lds.md) —— LDS 只放归约表，不放行数据
