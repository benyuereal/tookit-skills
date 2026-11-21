# dcu-training-frameworks-source — DCU 训练框架适配合集索引

五大训练框架在海光 DCU 上的适配形态索引：deepspeed（wheel 带 14 个 .hip 源码）、megatron-core（纯 Python 内容级适配）、dcu-megatron（海光 patch 层，重点）、apex（27 个 .so fused kernel HIP 移植）、transformer-engine。

## 目录

```
dcu-training-frameworks-source/
└── SKILL.md   # agent 向：五包总览 + 各包适配点 + DTK 链接汇总
```

## 快速用法 / 要点

- 先看 §0 五包总览定位框架，再进对应小节。
- dcu-megatron 的 patch 适配层是重点（§3），deepspeed 的 wheel 内 .hip 源码可直接查（§1）。
- 训练性能问题转 `dcu-perf-profiling` / `dcu-framework-profilers`。

> 详细内容见 `SKILL.md`。