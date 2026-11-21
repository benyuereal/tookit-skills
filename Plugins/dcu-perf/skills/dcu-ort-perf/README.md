# dcu-ort-perf — ONNX Runtime 推理引擎性能优化最佳实践

海光 DCU 上 ONNX Runtime 推理引擎的性能优化最佳实践：perf_test 测量 / 批处理 / FP16 / EP 选择 / 图优化 /
多线程 / io_binding 零拷贝 / 内存复用 / 多实例提 QPS / 自定义算子 / 环境配置。

## 核心内容
- 性能测量指标与 onnxruntime_perf_test 工具
- 模型性能优化：批处理、FP16、EP 选择（ROCm/MIGraphX）、图优化、多线程
- 客户全链路：io_binding 零拷贝、内存复用、多实例
- 自定义算子（Python + C++ 完整案例）、环境配置

## 目录
```
dcu-ort-perf/
├── SKILL.md                优化方法论（章节式）
└── scripts/               多个示例脚本（fp16/io_binding/内存复用/custom_op 等）
```

## 快速用法
ORT 推理提速 / 读 perf_test 报告 / 多实例提 QPS 时读 `SKILL.md`。
