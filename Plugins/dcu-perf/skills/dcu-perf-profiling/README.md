# dcu-perf-profiling — 性能剖析方法论（rocprof/rocTracer）

DCU 性能剖析方法论：rocprof / rocTracer 指标约定、命令模板、瓶颈判定。定义「如何量化一个算子」的标准动作（耗时/带宽/算力利用率/加速比），不存任何性能数字。与 dcu-hipprof（hipprof 系）互补。

## 目录

```
dcu-perf-profiling/
└── SKILL.md
```

## 快速用法 / 要点

- kernel 耗时排序：rocprof --timestamp on --stats <app>
- 带 HW 计数器：rocprof -i profile_metric.txt <app>
- 瓶颈判定（Memory/Compute/Latency-bound）见 SKILL.md 表

> 本 skill 详细内容（命令模板、适配点索引、Pitfalls）见 `SKILL.md`。
