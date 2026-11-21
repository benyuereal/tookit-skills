# dcu-engine-precision-localization — 自定义 C++ 推理引擎 DCU 精度定位

自定义 C++ 推理引擎在 DCU 上「能跑通但输出是垃圾」时，把精度问题定位到具体算子。

## 核心内容
- 定位方法：从引擎输出异常反查/隔离到具体算子层
- 参考案例：ninfer 引擎 Qwen3.8-27B 精度异常定位全程

## 目录
```
dcu-engine-precision-localization/
├── SKILL.md                              定位方法论
└── references/ninfer_qwen3_8_27b_case.md  Qwen3.8-27B 实例
```

## 快速用法
引擎跑通但对不齐/输出垃圾时读 `SKILL.md`；同类算子对齐问题参考 case 文档。
