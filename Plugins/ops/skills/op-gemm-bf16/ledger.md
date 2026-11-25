## 轮 1 · 建工坊（2026-11-26）

**做法**：从卡片出发建 contract / shapes / _common.py / candidates /
roofline / bench。

**状态**（无决策·建模）：
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
