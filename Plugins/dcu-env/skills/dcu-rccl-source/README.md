# dcu-rccl-source — 海光 DTK RCCL 源码分析：PCIe 混合链路优化

海光 DTK RCCL（custom-rccl fork）的 PCIe 混合链路优化源码级地图：三个核心概念（linkTopoType/p2pType/p2pMemType）、Collective vs P2P 路径与 connIndex 分流、三个关键环境变量（NCCL_SIMPLE_CHANNELS / RCCL_COLL_XHCL_CHANNEL_NUM / RCCL_P2P_XHCL_CHANNEL_NUM）的生效条件/作用/协同、源码级定位（init.cc:1672 / p2p.cc 的 RCCL_PARAM 与关键逻辑）。

## 目录

```
dcu-rccl-source/
└── SKILL.md   # agent 向：完整机制解析 + 源码片段 + 调优指向
```

## 快速要点

- 机制：限制非 PCIe 路径的 channel 数量 + 控制 buffer 放置（普通 shareable vs PCIe link memory）
- 三个环境变量只在**单节点且有 xHCL** 时生效（多节点/纯 PCIe 退化 MAXCHANNELS）；NCCL_SIMPLE_CHANNELS 仅 gfx936 / 多节点 gfx938 生效
- 源码片段基于 DTK 26.04 custom-rccl，行号可能漂移，需拉海光 fork 实时核验
- 不存性能数字；调优效果用 dcu-rccl-test 实测

> 详细内容（源码定位、connIndex 切换、阈值逻辑）见 `SKILL.md`。
