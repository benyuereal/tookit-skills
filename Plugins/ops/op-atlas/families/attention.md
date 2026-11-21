# attention

Q/K/V 参与计算。瓶颈通常在 KV 搬运与片上归约，不在 GEMM 主体。

## 跨家族攻击面

见 `references/surfaces.yaml`（op-optimize）。本文件只放**这一族特有的**攻击面。

（待补：本族特有的结构性陷阱与判据。）
