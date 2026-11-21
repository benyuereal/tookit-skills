# DCU≠NVIDIA 硬约束 + persistent 融合实测（2026-10-20，dcu-bw150 / gfx936 / DTK 26.04）

> 来源：对 SGLang Low-Latency Mode（sgl-amd-mono，LMSYS 2026-10-18）在 DCU 上的最小复现
> （材料在 `dcu-minrep/`）。以下为真机 kernel 级实测结论，非读文推断；性能数字随硬件/DTK 变化。

## 每 block 线程上限 = 256（gfx936 实测）

设 `blockDim=1024` 直接 VMFault：
```
Launch params (1024,1,1) are larger than launch bounds (256) for kernel ...
```
写 DCU kernel 用 **128 / 256**，别超。

## 宏定义务必加括号

`#define N (1 << 20)`（而非 `#define N 1 << 20`）。否则派生宏（如 `P (N/SEG)`）因运算符优先级错乱
（`1 << 20/SEG` 被解析成 `1 << (20/SEG)`），导致数组 size 与 kernel 内 index 不一致 → VMFault。

## 跨卡能力边界（2026-10-20 dcu-bw150 实测）

- `hipMemcpyPeer` 可用（`CanAccessPeer=1`，跨卡数据拷贝正确）。
- **`hipIpcOpenMemHandle`（IPC 把对端显存映射成本 kernel 内可直接访问的指针）在 DTK 26.04 失败**（返回
  `invalid argument`）。→ 跨卡 all-reduce 无法像 sgl-amd-mono 那样在单 kernel 内直接原子写对端显存，
  只能走 `hipMemcpyPeer`（多流/多 kernel）或 RCCL。
- `cooperative launch` 支持（单设备=1，多设备=1）→ grid sync（跨 block barrier）可用。

## "融合不一定更快"（边界：naive 实现 + kernel 级测量）

把多个小算子融合成单次 persistent 启动，若用 `__syncthreads` 做阶段同步，**反而可能比拆成多 kernel 慢**
（实测 speedup 0.92–0.96，即融合慢 4–8%）。原因：
1. DCU 的 kernel 间 launch 边界空闲本来就小（2–4 µs），省 1 次 launch 收益有限；
2. `__syncthreads` barrier 同步开销更大（每行约 11 次 barrier）。
tag 数据就绪同步（原子+threadfence+自旋）能正确工作（结果与拆开版逐元素一致），但**异步重叠收益
在该场景下不显著**。**别默认"融合就提速"——先测 DCU 上 launch 边界空闲量级再决定。**

**可复用预判**：这类"整个 decoder 层合并成 1–2 次 persistent 启动 + 内核内同步"的上游 kernel 文章，
在 DCU 上**"能实现、正确"，但"不能指望拿到文章宣称的性能收益"**。真正评估需放到贴近真实 decoder 层
（复杂依赖 + 长计算链 + 大量 token 流水线）的端到端路径测。
