# slc_cache_hint

> **状态：❌ refuted**（作为「通用加速手段」被证伪，但不是完全无效）
> **实测**：`buffer_load ... slc` 对 gate_up 慢 13%、对 qkv 打平，且 **bit-exact**
> 复现：`Plugins/ops/skills/op-hip-compile/references/isa-census.md` §四点五
> **发现于**：`hbird-gemm`，2026-11-23

## 问题

L2/SLC 缓存提示（`.cs` 流式 / `.cg` 全局 / `slc` descriptor）看起来是「无副作用的
性能开关」—— 只是告诉硬件「这份数据用完就不会再读」。

实测结果：**它会变慢，而且没有任何警告。**

## 机制

gfx938 的访存可以带缓存策略位。`slc`（system level cache）提示告诉硬件
把数据标记为流式，不要污染 L2。

但**流式标记也会关掉一些本地优化**（如 L1 的预取合并）。
对访存模式规整的 kernel，这个代价大于收益。

## 做法

```c
// 带 slc descriptor（实测更慢）
buffer_load_dwordx4 v[0:3], v[4:7], s[8:11], 0 offen slc
```

**结论：不要默认加。** 只有在「确实确认 L2 被污染、且这份数据确实不复用」时才试。

## 实测

| 配置 | gate_up | qkv | 精度 |
|---|---|---|---|
| 带 `slc` descriptor | **0.868×（慢 13%）** | 打平 | **bit-exact** |
| 不带 descriptor（纯 load） | 比 descriptor 慢 18% | — | — |

**注意第二行**：不带 descriptor 不是最好的 —— 说明**该用 descriptor，但别加 `slc`**。
这三者的关系是本条最有用的部分。

## 什么时候不适用

- **确实不复用的流式数据**（如一次性的 embedding 查表）：可能有效
- **L2 容量远小于工作集时**：污染问题真实存在，这时才值得试

## 反例/陷阱

**这条是「对照组设计」的教科书案例。**

如果只测「加 slc vs 什么都不加」，会得出「slc 慢 13%」；
再加第三组「用普通 descriptor 但不加 slc」，才发现真正的最优是第三组
（比 slc 快 13%，比纯 load 快 18%）。

**两类配置的对比会漏掉最优解。** 这就是 `ablate.py` 为什么要双向测
（单独加 / 单独减），以及 ForgeKernel 的 lever-pricing 为什么要
「加一个 / 减一个 / 全加」三组。

## 关联

- [`lds_bank_swizzle`](lds_bank_swizzle.md) —— 另一条「解决了真问题但收益极小」的反例
- `Plugins/ops/skills/op-hip-compile/references/isa-census.md` §四点五
