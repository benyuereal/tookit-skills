# elementwise

逐元素 / norm / 激活 / 量化融合。memory-bound，收益来源单一：省 HBM 往返。

## 跨家族攻击面

见 `references/surfaces.yaml`（op-optimize）。本文件只放**这一族特有的**攻击面。

（待补：本族特有的结构性陷阱与判据。）
