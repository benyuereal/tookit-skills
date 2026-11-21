# dcu-aiter-source — AITER 算子库源码 / 海光适配

AITER（AI Tensor Engine for ROCm）算子库在海光 DCU 上的源码结构、构建、海光适配点、四后端调度机制与算子族索引。基于 aiter-0.1.3-das184.dtk2604（DAS 海光适配版）。是写 DCU 算子适配/优化的底层实现索引。

## 目录

```
dcu-aiter-source/
├── SKILL.md
└── references
    └── aiter-operator-reference.md   # 海光自写中文算子文档（785 行，MoE/Norm/Activation/Quant/MQA 参数表与调用树）
```

## 快速用法 / 要点

- 找某算子在 AITER 源码的落点 → 算子族索引
- 四后端（moe_c/asm/triton/ck）调度机制见 SKILL.md
- 构建/海光适配点见对应段；参考见 references/

> 本 skill 详细内容（命令模板、适配点索引、Pitfalls）见 `SKILL.md`。
