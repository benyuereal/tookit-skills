# dcu-vllm-runtime-patch — vLLM 运行时 monkey-patch 注入专用 kernel

DCU 上 vLLM 运行时 monkey-patch 注入专用 kernel 的标准接入范式：环境变量开关 + 形状表 + 上游 fallback
安全网 + bit-identical 一致性门禁。不改 vLLM 源码即可做单卡/单并发 decode 优化。

## 核心内容
- **范式**：sitecustomize.py + PYTHONPATH 注入，环境变量控制开关
- **形状表**：按 (M,K,N) 精确形状分发，其余 fallback 上游
- **门禁**：bit-identical / relative_l2=0 / SHA 一致性校验
- **references/**（自包含）：K100 AI Qwen3.6 W8A8 实战完整材料（R180/R184 补丁 + 启动脚本 + INT8 MoE kernel + 编译脚本 + 调优 JSON）

## 目录
```
dcu-vllm-runtime-patch/
├── SKILL.md                        接入范式
├── scripts/                       sitecustomize 骨架 + 形状探针
└── references/                    K100 AI Qwen3.6 W8A8 实战完整材料
```

## 快速用法
单卡/单并发 decode 优化、不改源码注入专用 kernel 时读 `SKILL.md`。
