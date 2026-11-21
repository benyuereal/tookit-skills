# dcu-vllm-trace-analysis — vLLM V1 torch.profiler trace 分析

解析 vLLM V1 torch.profiler trace（.pt.trace.json[.gz]），定位 prefill/decode 阶段，按类别
（gemm/w8a16/MoE/attn/triton…）汇总 kernel 统计并产出 xlsx。

## 核心内容
- 解析 trace 并按类别汇总 kernel 耗时占比
- 区分 prefill / decode 阶段
- 产出 xlsx 便于进一步分析

## 目录
```
dcu-vllm-trace-analysis/
├── SKILL.md               分析流程
└── scripts/analyze_trace.py  trace 解析脚本
```

## 快速用法
已有 trace.json 想看各类 kernel 耗时占比时跑本 skill；与 `llm-torch-profiler-analysis` 互补。
