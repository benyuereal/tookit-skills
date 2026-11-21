# dcu-migraphx-perf — MIGraphX 推理引擎性能优化最佳实践

海光 DCU 上 MIGraphX 推理引擎的性能优化最佳实践：测量 / 批处理 / 序列化 / 精度 / 融合 / TensorCore /
多框架集成 / 精度调试 / 环境配置。

## 核心内容
- 性能测量（perf_test 等）与读报告
- 批处理、流式、序列化、精度（FP16/INT8）、自动调优
- 算子融合、TensorCore 对齐、多框架集成（ORT/torch_migraphx）
- 自定义算子、环境配置

## 目录
```
dcu-migraphx-perf/
├── SKILL.md              优化方法论（10 节）
└── scripts/             ORT/torch_migraphx/自定义算子 示例
```

## 快速用法
MIGraphX 推理提速 / 读 perf 报告 / 精度量化时读 `SKILL.md`。
