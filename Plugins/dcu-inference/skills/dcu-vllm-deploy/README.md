# dcu-vllm-deploy — vLLM 部署调优经验速查

DCU 上 vLLM 部署与调优经验速查：镜像/容器启动约束、关键启动参数、已知问题、性能调优入口。配合 dcu-vllm-source-0.18 与 dcu-perf-profiling 使用。含「经验仅供参考、先核对设备+版本+报错」须知。

## 目录

```
dcu-vllm-deploy/
├── SKILL.md
├── references
└── templates
```

## 快速用法 / 要点

- 启动参数 / 容器约束 → SKILL.md 关键启动参数段
- 已知问题排查 → references/ 与 SKILL.md 已知问题
- 调优入口 → 先 perf-profiling 量化再改

> 本 skill 详细内容（命令模板、适配点索引、Pitfalls）见 `SKILL.md`。
