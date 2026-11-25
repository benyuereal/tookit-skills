# private_spill_trap

> **状态：✅ measured**
> **实测**：`private` 0B → 528B，0.165ms → 0.538ms（**慢 3.3 倍**）@ `event_isolated`
> 复现：`Plugins/ops/skills/op-rmsnorm-fused/runs/forged/_kernels/fused_rms_v2.hip`
> **发现于**：`op-rmsnorm-fused`，2026-11-24

## 问题

在 HIP kernel 里用**局部数组**存中间结果（如每线程 16 个 float）时，
如果数组索引是**运行期变量**，编译器无法把它放进寄存器，会整块溢出到私有内存。

**最要命的是它不报错** —— kernel 跑得动、算得对、就是慢 3.3 倍。
在跑分里看起来只是「这个方向不行」，会让人把**对的改法判成错的**。

## 机制

VGPR（向量通用寄存器）的索引必须是**编译期可确定**的。
`for (int j = 0; j < vec; ++j)` 里 `vec` 是运行期变量 → 编译器不能展开 →
数组退化成私有的 scratch 内存（走 L1/L2，延迟高一个数量级）。

```c
// ❌ 运行期上界 → 溢出
const int vec = ept / 8;              // 运行期算出
for (int j = 0; j < vec; ++j) xv[j*8] = ...

// ✅ 编译期常量 → 留在寄存器
#define VEC 2
#pragma unroll
for (int j = 0; j < VEC; ++j) xv[j*8] = ...
```

## 怎么做

1. **所有影响数组大小的量走 `-D` 宏**，不用函数参数
2. 循环加 `#pragma unroll`
3. **编译后必须验 `private`**：

```bash
python3 -I Plugins/ops/skills/op-hip-compile/inspect_so.py <so> --json | grep private
# 必须是 0
```

## 实测

| 配置 | private | 耗时 | 相对 |
|---|---|---|---|
| `VEC` 编译期常量 | 0 B | 0.165 ms | 1.00x |
| `VEC` 运行期变量 | 528 B | 0.538 ms | **3.26x 慢** |

## 什么时候不适用

- 数组很小（≤4 个元素）且索引简单：编译器可能仍能放进寄存器
- 用 `__shared__` 显式声明的（那是 LDS，不是私有内存，另一回事）

## 反例/陷阱

**这不是「性能问题」，是「错误归因风险」**。
慢 3.3 倍的 kernel 看起来像「这个优化方向无效」，实际是编译配置错了。

现在 `scripts/forge_step.py` 的 `health_check()` 把 `private > 0` 设为**硬失败**，
拒绝产出变体 —— 就是防这件事。
