## 轮 1 · 建工坊（2026-11-26）

**做法**：从 `mqa-logits` 卡片出发建 contract / shapes / _common.py /
candidates / roofline / bench。

**状态**（无决策·建模）：
只建了骨架，**还没有跑过任何测量**。只有 reference，没有竞品候选。

**为什么先建它**：
`optimize-glm53/OPERATOR-DIFF-V2-V3.md` 实测显示，DSA 打分
BF16 -> INT8 是 v2->v3 最大的单项收益：
  prefill  `mqa_logits_128x128x32`  35.0% -> 4.9%，单次 6.6x
  decode   `paged_mqa_logits`       32.5% -> 6.7%，单次 6.6x

**建的时候就发现一个疑点**（写进 roofline.json 的 quant_ceiling）：
bf16 -> int8 只省一半带宽，理论上限 2.0x。**6.6x 差 3.3 倍无法用带宽解释。**
可能是叠加了分块/实现差异，也可能 INT8 版根本没算全。
**在验证之前，6.6x 不能当目标。**

**下一步**：
  1. 跑 `bench/ab.py --shape tiny --variants ref-heavy` 验通链路
  2. 验 reference 正确性（尤其 relu 位置：点积之后、乘 W 之前）
  3. 实现 bf16-baseline 作为对照
  4. 核对 lightop 的 mqa_int8 接口（**不要凭名字猜**）
