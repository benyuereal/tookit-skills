## 轮 1 · 建工坊（2026-11-26）

**做法**：从卡片出发建 contract / shapes / _common.py / candidates /
roofline / bench。

**状态**（无决策·建模）：
只有 reference，没有可竞速的候选。能验正确性，不能出性能结论。

**与 op-fused-add-rmsnorm 的关系**：
本卡是它的超集（多一趟量化）。那张卡的寄存器驻留方案
（`(x+residual)` 留在寄存器）在这里同样适用，且量化可以在寄存器里做，
不需要额外读一遍 —— 这是本卡最值得做的方向。

**下一步**：
  1. 验 reference 正确性（尤其量化部分：clamp 顺序）
  2. 核对 lightop 是否有量化融合算子
  3. 实现 hand-hip-3in1
