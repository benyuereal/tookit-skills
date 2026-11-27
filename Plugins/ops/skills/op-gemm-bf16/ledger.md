### 轮 1 · 建工坊（2026-11-26）

**做法**：从卡片出发建 contract / shapes / _common.py / candidates /
roofline / bench。

**状态**（无决策·建模，不计入轮次）：
有 2 个能跑的变体（ref-fp32 / torch-mm），但**都不是竞品**。
能验正确性，不能出有意义的 B 项。

**这张卡的特殊性**：
它是**选型题不是优化题**。目标不是写一个比厂商快的 GEMM（不可能），
是搞清「给定形状该用哪个库、上限多少」。
所以 comparison 的正确形态是「库 A vs 库 B」，不是「自研 vs 厂商」。

**历史 B 项只有 8 分为什么是对的**：
原 comparison 是 F.linear vs torch.mm，两个都是 torch 路径，
没有厂商库参与。B=8 如实反映了这个事实。

**下一步**：
  1. 验 reference 正确性
  2. 接上 lightop.gemm（已确认符号存在，签名待核对）
  3. 接上 hipBLASLt 直调并扫参 —— 这才是本卡最该做的对比

### 轮 2 · 接竞品：lightop.gemm 不可用，改用 aiter

**做法**：
1. 先接 `lightop.gemm`（符号存在，`inspect.signature` 是 `(a, b)`）
2. 发现结果错误，改接 `aiter.gemm_a16w16_opus`

**实测（2026-11-27）**：

`lightop.gemm` 在所有 M>=24 的形状上都错：
```
M=24    误差 4.614309
M=64    误差 4.872625
M=128   误差 3.823497
M=1024  误差 1.000000
M=8192  误差 4.218176
```
**两种调用方式（`gemm(A,B.t())` / `gemm(A,B)`）都错。**
此前的记录说「M>=16 可用」是错的 —— 实际是「M>=24 全错」，方向记反了。

`aiter.gemm_a16w16_opus` 可用，但**入参要转置**：
```
gemm(A, B)     误差 1.362659  错
gemm(A, B.T)   误差 0.003665  对
```

**结论**（refute lightop / adopt aiter）：
`lightop.gemm` 本机不可用，标 blocked。改用 aiter 当竞品。

**踩到的坑**：两次都是「库要求 B 转置，我们传了 B[K,N]」。
误差 1.3~4.8 这个量级是**形状对得上、数值全错**的典型症状 ——
和 `wrong-variable` 故障面同族。

### 轮 3 · 三形状实测：torch.mm 全面占优

**做法**：prefill-8k / decode-m32 / decode-m1 三形状，
event_isolated 3 轮，torch.mm vs aiter vs ref-fp32。

**实测**：

| 形状 | torch.mm | aiter | 差距 |
|---|---|---|---|
| prefill-8k (M=8192) | **0.9374 ms** | 10.9784 ms | torch 快 **11.7x** |
| decode-m32 (M=32) | **0.0771 ms** | 0.2832 ms | torch 快 **3.7x** |
| decode-m1 (M=1) | 0.0694 ms | — | — |

ref-fp32（fp32 参考）作为正确性锚点：7.1338 ms。
torch-mm 相对 ref 的三形状 geomean = 0.3239。

**结论**（adopt torch.mm）：
本机 bf16 GEMM 的答案是**用 `torch.mm`，别自己折腾**。
aiter 在大 M 小 M 都更慢；lightop 不可用。

**这本身就是选型题要的结论** —— 不是「我们优化了多少」，
是「给定形状该用哪个库」。`comparison` 里 aiter 作 competitor，
哪怕它更慢也合法：**记下「aiter 在此形状更慢」正是 atlas 的价值。**

**未做的**：
- hipBLASLt 直调扫参（torch.mm 只走默认启发式，手工扫可能还有 5-15%）
- deepgemm 对比（仓库已 clone 未构建）
