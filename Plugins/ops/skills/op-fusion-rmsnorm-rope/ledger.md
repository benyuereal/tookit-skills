## 轮 1 · 建工坊（2026-11-26）

**做法**：从 `fusion-rmsnorm-rope` 卡片出发，建 contract / shapes /
candidates / roofline / bench / _common.py。

**状态**（无决策·建模）：
只建了骨架，**还没有跑过任何测量**。
reference 已写，但没有可竞速的候选，所以本卡当前：
  ✅ 能验正确性
  ❌ 不能出性能结论
  ❌ 不能算 S 项（三形状都没有可跑的候选）

**为什么先建不先跑**：
巡检单的发现是对的 —— 53 张卡里只有 3 个「能开工的工坊」。
E4 推不动的真原因是这个，不是 GPU。
建工坊是纯 CPU 活，GPU 满的时候正好做。

**下一步**：
  1. 跑 `bench/ab.py --variants ref-heavy --shape tiny` 验通链路
  2. 实现 torch-unfused 作为对照组
  3. 核对 lightop 是否有 rope 融合算子（**不要假设**）
