# dcu-deepep-source — DeepEP DCU 适配源码索引

DeepEP（MoE 专家并行 dispatch/combine 通信库）在海光 DCU 上的源码结构与适配：Buffer 类 API 精确落点、low-latency 路由策略、DTK 链接（ldd 实测），附多节点带宽测试方法与实测参考线。

## 目录

```
dcu-deepep-source/
└── SKILL.md   # agent 向：源码地图 + Buffer/路由策略 + 多机测试方法
```

## 快速用法 / 要点

- 查「DeepEP 某个机制在源码哪里」→ §1 源码地图 + §2/§3 精确落点（buffer.py / low_latency_policy.py）。
- 多机 low-latency 测试：run_test.sh 用法、topo.config 卡网映射、参考线数据与基线环境核对都在 §8。
- 跨节点 RDMA 网络环境搭建转 `dcu-roce-network`；性能分析转 `dcu-perf-profiling`。

> 详细内容见 `SKILL.md`。