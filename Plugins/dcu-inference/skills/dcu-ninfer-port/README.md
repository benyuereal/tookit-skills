# dcu-ninfer-port — NInfer → DCU(gfx936) 移植

把 NInfer 推理框架移植到海光 DCU（gfx936）：SIMT-first 策略 + 4-layer 量化。

## 核心内容
- **SIMT-first**：先按 SIMT 语义移植，规避 fine-grained 依赖
- **4-layer quant**：量化分层方法

## 目录
```
dcu-ninfer-port/
├── SKILL.md                  移植策略 + 步骤
└── references/w8g32-conversion.md  W8G32 量化转换
```

## 快速用法
NInfer 移植 DCU / 量化方案选型时读 `SKILL.md`。
