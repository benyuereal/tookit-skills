# dcu-inference-infra — 推理 infra 子域总入口/路由器

DCU 推理 infra 子域总入口/路由器：覆盖框架适配（vLLM 等，按大版本拆分 0.18/0.26）、算子优化、性能剖析、部署调优。按用户意图分发到 dcu-vllm-source-* / dcu-op-optimization / dcu-perf-profiling / dcu-vllm-deploy 等。

## 目录

```
dcu-inference-infra/
└── SKILL.md
```

## 快速用法 / 要点

- 用户要查 vLLM/SGLang/PyTorch 适配、算子替换、部署调优 → 由本 skill 路由
- 本身只做分发与边界说明，细节在对应子 skill
- 新增子域 skill 见 dcu-inference-infra-skill-builder

> 本 skill 详细内容（命令模板、适配点索引、Pitfalls）见 `SKILL.md`。
