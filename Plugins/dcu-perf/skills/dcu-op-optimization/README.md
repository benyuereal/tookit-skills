# dcu-op-optimization — 算子优化标准闭环

DCU 算子优化标准闭环方法论：profile → 定位热点 → 改（HIP kernel）→ 验证正确性 → 验证性能。适用于「某算子耗时异常/想榨干某 kernel」场景，与具体推理框架解耦。强调强制闭环、量化对比、留痕，禁止「感觉快了」式结论。

## 目录

```
dcu-op-optimization/
└── SKILL.md
```

## 快速用法 / 要点

- 性能量化动作见 dcu-perf-profiling
- 改完必须回测正确性 + 性能，且量化对比（同条件加速比）
- 自定义算子替换流程见 dcu-vllm-op-replace

> 本 skill 详细内容（命令模板、适配点索引、Pitfalls）见 `SKILL.md`。
