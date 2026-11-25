# hw_kb —— 硬件机制知识库（算子无关）

> 借鉴 ForgeKernel 的 `hw_kb/`。**它 200 条，我们从 0 开始。**
> 但我们的每一条都是本机实测的，不是抄来的。

## 为什么需要它

现在每张新卡都要重新踩一遍同样的坑：

| 卡 | 重新发现的东西 |
|---|---|
| `rmsnorm-fused` | 整行 LDS 往返把带宽打到 212 GB/s |
| `fused-add-rmsnorm` | 同上 —— **第二次发现** |
| `norm-act-swiglu` | 同上 —— **第三次发现** |

**这是纯浪费。** 知识应该沉在这里，一次认证、所有算子受益。

## 目录结构

```
hw_kb/
├── README.md              本文件（规则）
├── TEMPLATE.md            单条机制的模板
├── hip/gfx938/            HIP 路径（我们的主路径）
├── triton/gfx938/         Triton 路径
└── tilelang/gfx938/       TileLang 路径
```

**分树规则同 ForgeKernel**：一条机制的认证不跨语言迁移 ——
HIP 上的结论不能直接套到 Triton（Triton 控不了寄存器排布）。

## 每条必须满足的格式

见 [`TEMPLATE.md`](TEMPLATE.md)。三条硬要求：

1. **认证状态必填**：`✅ measured` / `⬜ unverified` / `❌ refuted`
2. **带实测值**：数字 + 口径（`event_isolated` / `back_to_back`）+ 复现路径
3. **算子无关**：讲「怎么用某个硬件特性」，不讲「某个算子怎么写」

## 与 ForgeKernel 的差别

| | ForgeKernel | 我们 |
|---|---|---|
| 认证方式 | GPU 探针（六条 contract 全 PASS） | 实测 + 复现脚本 |
| 分母口径 | 同 job confirmation ratio | **三层口径**（必须写明是哪层） |
| 反例也收 | 未见 | **收**（`lds_bank_swizzle` 就是反例：解决了真问题但只快 0.36%） |

**反例我们特意收**：一条「修了但没用」的机制，比十条「有效」的机制更能防止浪费时间。
